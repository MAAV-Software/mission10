use defmt::{info, warn};
use dw3000_ng::hl::SendTime;
use dw3000_ng::time::Instant as RadioInstant;
use dw3000_ng::{DW3000, FastCommand};
use embassy_nrf::gpio::{Level, Output, OutputDrive};
use embassy_nrf::spim;
use embassy_time::{Delay, Timer};
use embedded_hal_bus::spi::ExclusiveDevice;
use mission10_uwb_protocol::{
    Diagnostic, FRAME_LEN, Frame, MessageKind, REPLY_DELAY_US, RadioToHost, ResponderTimestamps,
    TIMESTAMP_MASK, delayed_tx_time, distance_metres,
};

use crate::Irqs;
use crate::board::{
    FALLBACK_ANTENNA_DELAY, INITIATOR_REPLY_DELAY_US, OWN_ADDRESS, PEER_ADDRESS, PEER_INDEX,
    RadioHardware, RadioWait, prepare_rx, prepare_tx, radio_config, recoverable_receive_error,
    wait_receive, wait_send,
};
use crate::host::publish;

fn publish_error(diagnostic: Diagnostic) {
    warn!("radio diagnostic={=u8}", diagnostic as u8);
    publish(RadioToHost::Error { diagnostic });
}

fn publish_health(wait: &RadioWait<'_>, ranges: &mut u32) {
    *ranges = ranges.wrapping_add(1);
    if *ranges % 32 == 0 {
        publish(wait.counters().host_message());
    }
}

pub async fn run(hw: RadioHardware) {
    // DW_RST is open-drain. Driving high disconnects the nRF output and lets
    // the module pull-up release reset.
    let mut reset = Output::new(hw.reset, Level::High, OutputDrive::Standard0Disconnect1);
    reset.set_low();
    Timer::after_millis(2).await;
    reset.set_high();
    Timer::after_millis(5).await;

    let mut spi_config = spim::Config::default();
    spi_config.frequency = spim::Frequency::M4;
    let spi = spim::Spim::new(hw.spi, Irqs, hw.sck, hw.miso, hw.mosi, spi_config);
    let cs = Output::new(hw.cs, Level::High, OutputDrive::Standard);
    let spi_device = match ExclusiveDevice::new(spi, cs, Delay) {
        Ok(device) => device,
        Err(_) => panic!("failed to create DW3000 SPI device"),
    };
    let mut wait = RadioWait::new(hw.irq);

    let mut radio = DW3000::new(spi_device);
    let id = match radio.ll().dev_id().read().await {
        Ok(id) => id,
        Err(_) => panic!("failed to read DW3000 device ID"),
    };
    publish(RadioToHost::RadioId {
        ridtag: id.ridtag(),
        model: id.model(),
        version: id.ver(),
        revision: id.rev(),
    });

    let tx_power = radio.read_otp(0x011).await.unwrap_or(0);
    let antenna = radio.read_otp(0x01a).await.unwrap_or(0);
    let xtal = radio.read_otp(0x01e).await.unwrap_or(0);
    let otp_revision = radio.read_otp(0x01f).await.unwrap_or(0);
    publish(RadioToHost::Otp {
        tx_power,
        antenna,
        xtal,
        revision: otp_revision,
    });

    let radio = match radio.init().await {
        Ok(radio) => radio,
        Err(_) => panic!("DW3000 initialization failed"),
    };
    let mut radio = match radio.config(radio_config(), Delay).await {
        Ok(radio) => radio,
        Err(_) => panic!("DW3000 radio configuration failed"),
    };

    let otp_tx_delay = antenna as u16;
    let otp_rx_delay = (antenna >> 16) as u16;
    let use_fallback = cfg!(feature = "engineering-sample")
        || otp_tx_delay == 0
        || otp_rx_delay == 0
        || antenna == u32::MAX;
    let (rx_delay, tx_delay) = if use_fallback {
        (FALLBACK_ANTENNA_DELAY, FALLBACK_ANTENNA_DELAY)
    } else {
        (otp_rx_delay, otp_tx_delay)
    };
    if radio.set_antenna_delay(rx_delay, tx_delay).await.is_err() {
        panic!("failed to set antenna delay");
    }
    if tx_power != 0
        && tx_power != u32::MAX
        && radio
            .ll()
            .tx_power()
            .write(|w| w.value(tx_power))
            .await
            .is_err()
    {
        panic!("failed to set calibrated transmit power");
    }
    publish(RadioToHost::Ready {
        role: if cfg!(feature = "initiator") { 0 } else { 1 },
        rx_delay,
        tx_delay,
    });
    publish(wait.counters().host_message());

    let config = radio_config();
    let mut rx_buf = [0_u8; 128];
    let mut range_count = 0_u32;

    if cfg!(feature = "initiator") {
        'transaction: loop {
            prepare_tx(&mut radio).await.expect("prepare POLL transmit");
            let poll = Frame::poll(OWN_ADDRESS);
            let mut sending = match radio
                .send_raw(poll.as_bytes(), SendTime::Now, &config)
                .await
            {
                Ok(sending) => sending,
                Err(_) => panic!("failed to start POLL"),
            };
            let poll_tx = match wait_send(&mut sending, &mut wait).await {
                Ok(timestamp) => timestamp,
                Err(error) => {
                    publish_error(error);
                    panic!("POLL failed");
                }
            };
            radio = match sending.finish_sending().await {
                Ok(radio) => radio,
                Err(_) => panic!("failed to finish POLL"),
            };

            prepare_rx(&mut radio)
                .await
                .expect("prepare POLL_ACK receive");
            let mut receiving = match radio.receive(config).await {
                Ok(receiving) => receiving,
                Err(_) => panic!("failed to receive POLL_ACK"),
            };
            let (length, poll_ack_rx) =
                match wait_receive(&mut receiving, &mut rx_buf, &mut wait).await {
                    Ok(message) => message,
                    Err(error) if recoverable_receive_error(error) => {
                        warn!("restarting exchange after POLL_ACK receive error");
                        publish_error(error);
                        wait.recovered();
                        radio = match receiving.finish_receiving().await {
                            Ok(radio) => radio,
                            Err(_) => panic!("failed to abort POLL_ACK receive"),
                        };
                        Timer::after_millis(10).await;
                        continue 'transaction;
                    }
                    Err(error) => {
                        publish_error(error);
                        panic!("fatal POLL_ACK receive error");
                    }
                };
            radio = match receiving.finish_receiving().await {
                Ok(radio) => radio,
                Err(_) => panic!("failed to finish POLL_ACK receive"),
            };
            if length < FRAME_LEN {
                publish_error(Diagnostic::ShortPollAck);
                continue;
            }
            let ack = match Frame::decode(&rx_buf[..FRAME_LEN]) {
                Ok(frame) => frame,
                Err(_) => {
                    publish_error(Diagnostic::FrameDecode);
                    continue;
                }
            };
            publish(RadioToHost::Rx {
                kind: ack.kind() as u8,
            });
            if ack.kind() != MessageKind::PollAck || ack.source() != PEER_ADDRESS {
                continue;
            }

            let range_at_value = delayed_tx_time(poll_ack_rx.value(), INITIATOR_REPLY_DELAY_US);
            let predicted_range_tx = range_at_value.wrapping_add(tx_delay as u64) & TIMESTAMP_MASK;
            let range = Frame::range(
                OWN_ADDRESS,
                poll_tx.value(),
                poll_ack_rx.value(),
                predicted_range_tx,
            );
            let schedule_now = radio
                .sys_time()
                .await
                .expect("read system time before RANGE");
            let schedule_target = (range_at_value >> 8) as u32;
            info!(
                "RANGE schedule now={=u32:08x} target={=u32:08x} remaining={=u32}",
                schedule_now,
                schedule_target,
                schedule_target.wrapping_sub(schedule_now)
            );
            prepare_tx(&mut radio)
                .await
                .expect("prepare RANGE transmit");
            let range_at = RadioInstant::new(range_at_value).expect("masked 40-bit timestamp");
            let mut sending = match radio
                .send_raw(range.as_bytes(), SendTime::Delayed(range_at), &config)
                .await
            {
                Ok(sending) => sending,
                Err(_) => panic!("failed to start delayed RANGE"),
            };
            if let Err(error) = wait_send(&mut sending, &mut wait).await {
                publish_error(error);
                panic!("delayed RANGE failed");
            }
            radio = match sending.finish_sending().await {
                Ok(radio) => radio,
                Err(_) => panic!("failed to finish RANGE"),
            };

            prepare_rx(&mut radio)
                .await
                .expect("prepare RANGE_REPORT receive");
            let mut receiving = match radio.receive(config).await {
                Ok(receiving) => receiving,
                Err(_) => panic!("failed to receive RANGE_REPORT"),
            };
            let (length, _) = match wait_receive(&mut receiving, &mut rx_buf, &mut wait).await {
                Ok(message) => message,
                Err(error) if recoverable_receive_error(error) => {
                    warn!("restarting exchange after RANGE_REPORT receive error");
                    publish_error(error);
                    wait.recovered();
                    radio = match receiving.finish_receiving().await {
                        Ok(radio) => radio,
                        Err(_) => panic!("failed to abort RANGE_REPORT receive"),
                    };
                    Timer::after_millis(10).await;
                    continue 'transaction;
                }
                Err(error) => {
                    publish_error(error);
                    panic!("fatal RANGE_REPORT receive error");
                }
            };
            radio = match receiving.finish_receiving().await {
                Ok(radio) => radio,
                Err(_) => panic!("failed to finish RANGE_REPORT receive"),
            };
            if length < FRAME_LEN {
                publish_error(Diagnostic::ShortRangeReport);
                continue;
            }
            let report = match Frame::decode(&rx_buf[..FRAME_LEN]) {
                Ok(frame) => frame,
                Err(_) => {
                    publish_error(Diagnostic::FrameDecode);
                    continue;
                }
            };
            publish(RadioToHost::Rx {
                kind: report.kind() as u8,
            });
            if report.kind() == MessageKind::RangeReport && report.source() == PEER_ADDRESS {
                if let Some(millimetres) = report.distance_mm() {
                    publish(RadioToHost::Range {
                        peer: PEER_INDEX,
                        millimetres,
                    });
                    publish_health(&wait, &mut range_count);
                }
            }
            Timer::after_millis(100).await;
        }
    }

    loop {
        prepare_rx(&mut radio).await.expect("prepare POLL receive");
        let mut receiving = match radio.receive(config).await {
            Ok(receiving) => receiving,
            Err(_) => panic!("failed to enter receive mode"),
        };
        let (length, poll_rx) = loop {
            match wait_receive(&mut receiving, &mut rx_buf, &mut wait).await {
                Ok(message) => break message,
                Err(error) if recoverable_receive_error(error) => {
                    warn!("recovering DW3000 receive error");
                    publish_error(error);
                    wait.recovered();
                    if receiving.fast_cmd(FastCommand::CMD_CLR_IRQS).await.is_err()
                        || receiving.fast_cmd(FastCommand::CMD_RX).await.is_err()
                    {
                        panic!("failed to re-arm DW3000 receiver");
                    }
                }
                Err(error) => {
                    publish_error(error);
                    panic!("fatal DW3000 receive error");
                }
            }
        };
        radio = match receiving.finish_receiving().await {
            Ok(radio) => radio,
            Err(_) => panic!("failed to finish DW3000 receive"),
        };
        if length < FRAME_LEN {
            publish_error(Diagnostic::ShortPoll);
            continue;
        }
        let poll = match Frame::decode(&rx_buf[..FRAME_LEN]) {
            Ok(frame) => frame,
            Err(_) => {
                publish_error(Diagnostic::FrameDecode);
                continue;
            }
        };
        publish(RadioToHost::Rx {
            kind: poll.kind() as u8,
        });
        if poll.kind() != MessageKind::Poll || poll.source() != PEER_ADDRESS {
            continue;
        }

        let ack_at = delayed_tx_time(poll_rx.value(), REPLY_DELAY_US);
        let ack_at = RadioInstant::new(ack_at).expect("masked 40-bit timestamp");
        prepare_tx(&mut radio)
            .await
            .expect("prepare POLL_ACK transmit");
        let ack = Frame::poll_ack(OWN_ADDRESS);
        let mut sending = match radio
            .send_raw(ack.as_bytes(), SendTime::Delayed(ack_at), &config)
            .await
        {
            Ok(sending) => sending,
            Err(_) => panic!("failed to start delayed POLL_ACK"),
        };
        let poll_ack_tx = match wait_send(&mut sending, &mut wait).await {
            Ok(timestamp) => timestamp,
            Err(error) => {
                publish_error(error);
                panic!("delayed POLL_ACK failed");
            }
        };
        radio = match sending.finish_sending().await {
            Ok(radio) => radio,
            Err(_) => panic!("failed to finish POLL_ACK"),
        };

        prepare_rx(&mut radio).await.expect("prepare RANGE receive");
        let mut receiving = match radio.receive(config).await {
            Ok(receiving) => receiving,
            Err(_) => panic!("failed to receive RANGE"),
        };
        let (length, range_rx) = loop {
            match wait_receive(&mut receiving, &mut rx_buf, &mut wait).await {
                Ok(message) => break message,
                Err(error) if recoverable_receive_error(error) => {
                    warn!("recovering RANGE receive error");
                    publish_error(error);
                    wait.recovered();
                    if receiving.fast_cmd(FastCommand::CMD_CLR_IRQS).await.is_err()
                        || receiving.fast_cmd(FastCommand::CMD_RX).await.is_err()
                    {
                        panic!("failed to re-arm RANGE receiver");
                    }
                }
                Err(error) => {
                    publish_error(error);
                    panic!("fatal RANGE receive error");
                }
            }
        };
        radio = match receiving.finish_receiving().await {
            Ok(radio) => radio,
            Err(_) => panic!("failed to finish RANGE receive"),
        };
        if length < FRAME_LEN {
            publish_error(Diagnostic::ShortRange);
            continue;
        }
        let range = match Frame::decode(&rx_buf[..FRAME_LEN]) {
            Ok(frame) => frame,
            Err(_) => {
                publish_error(Diagnostic::FrameDecode);
                continue;
            }
        };
        publish(RadioToHost::Rx {
            kind: range.kind() as u8,
        });
        if range.kind() != MessageKind::Range || range.source() != PEER_ADDRESS {
            continue;
        }
        let Some(initiator_timestamps) = range.timestamps() else {
            continue;
        };
        let Some(distance) = distance_metres(
            initiator_timestamps,
            ResponderTimestamps {
                poll_rx: poll_rx.value(),
                poll_ack_tx: poll_ack_tx.value(),
                range_rx: range_rx.value(),
            },
        ) else {
            publish_error(Diagnostic::InvalidDistance);
            continue;
        };
        let millimetres = (distance * 1_000.0 + 0.5) as u32;
        prepare_tx(&mut radio)
            .await
            .expect("prepare RANGE_REPORT transmit");
        let report = Frame::range_report(OWN_ADDRESS, millimetres);
        let mut sending = match radio
            .send_raw(report.as_bytes(), SendTime::Now, &config)
            .await
        {
            Ok(sending) => sending,
            Err(_) => panic!("failed to start RANGE_REPORT"),
        };
        if let Err(error) = wait_send(&mut sending, &mut wait).await {
            publish_error(error);
            panic!("RANGE_REPORT failed");
        }
        radio = match sending.finish_sending().await {
            Ok(radio) => radio,
            Err(_) => panic!("failed to finish RANGE_REPORT"),
        };
        publish(RadioToHost::Range {
            peer: PEER_INDEX,
            millimetres,
        });
        publish_health(&wait, &mut range_count);
    }
}
