use defmt::warn;
use dw3000_ng::hl::SendTime;
use dw3000_ng::time::Instant as RadioInstant;
use dw3000_ng::{DW3000, FastCommand};
use embassy_embedded_hal::SetConfig;
use embassy_nrf::gpio::{Level, Output, OutputDrive};
use embassy_nrf::spim;
use embassy_time::{Delay, Timer};
use embedded_hal_bus::spi::ExclusiveDevice;
use mission10_uwb_protocol::air::{
    self, AIR_FRAME_MAX_NO_FCS, AirEnvelope, AirMessage, DecodedAirFrame,
};
use mission10_uwb_protocol::host::{Diagnostic, OperatingMode, RadioToHost};
use mission10_uwb_protocol::{
    Destination, NodeAddress, REPORT_TURNAROUND_US, ResponderTimestamps, TIMESTAMP_MASK,
    delayed_tx_time, distance_metres, scheduled_tx_matches,
};

use crate::Irqs;
use crate::board::{
    FALLBACK_ANTENNA_DELAY, INITIATOR_INTER_EXCHANGE_GUARD_MS, OWN_ADDRESS, PEER_ADDRESS,
    REPLY_DELAY_US, RESPONSE_TIMEOUT_US, RadioHardware, RadioWait, diagnostic, finish_receiving,
    finish_sending, prepare_rx, prepare_tx, radio_config, recoverable_receive_error,
    reset_after_radio_failure, wait_receive, wait_send,
};
use crate::host::{latest_ego_state, publish, transport_counters, update_radio_counters};

const PHY_FCS_LEN: usize = 2;

fn publish_error(diagnostic: Diagnostic) {
    warn!("radio diagnostic={=u8}", diagnostic as u8);
    publish(RadioToHost::Error { diagnostic });
}

fn publish_health(wait: &RadioWait<'_>, ranges: &mut u32) {
    *ranges = ranges.wrapping_add(1);
    if (*ranges).is_multiple_of(32) {
        let mut counters = wait.counters().health_counters();
        counters.merge_transport(transport_counters());
        update_radio_counters(counters);
        publish(RadioToHost::Health {
            request_id: 0,
            counters,
        });
    }
}

const RETAINED_RESET_REPORT_MS: u64 = 3_000;
const RETAINED_RESET_REPORT_INTERVAL_MS: u64 = 750;

fn publish_reset_diagnostics(
    previous_radio_failure: Option<Diagnostic>,
    previous_watchdog_reset: bool,
) {
    if let Some(diagnostic) = previous_radio_failure {
        publish_error(diagnostic);
        publish_error(Diagnostic::RadioReset);
    }
    if previous_watchdog_reset {
        publish_error(Diagnostic::WatchdogReset);
    }
}

async fn report_previous_reset(
    wait: &mut RadioWait<'_>,
    previous_radio_failure: Option<Diagnostic>,
    previous_watchdog_reset: bool,
) {
    let reset_was_retained = previous_radio_failure.is_some() || previous_watchdog_reset;
    if !reset_was_retained {
        return;
    }

    // Repeat through the stable USB window: CDC may become configured and
    // consume a one-shot event before the host process has opened the tty.
    // Four radio-failure pairs exactly fit the diagnostic queue if no host is
    // attached, so this reporting path cannot create queue drops by itself.
    for _ in 0..RETAINED_RESET_REPORT_MS / RETAINED_RESET_REPORT_INTERVAL_MS {
        publish_reset_diagnostics(previous_radio_failure, previous_watchdog_reset);
        wait.pet_watchdog();
        Timer::after_millis(RETAINED_RESET_REPORT_INTERVAL_MS).await;
    }
    wait.pet_watchdog();
}

pub async fn run(hw: RadioHardware) {
    let mut wait = RadioWait::new(hw.irq, hw.watchdog);
    report_previous_reset(
        &mut wait,
        hw.previous_radio_failure,
        hw.previous_watchdog_reset,
    )
    .await;

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
        Err(_) => reset_after_radio_failure(Diagnostic::Unknown),
    };
    let mut radio = DW3000::new(spi_device);
    let id = match radio.ll().dev_id().read().await {
        Ok(id) => id,
        Err(_) => reset_after_radio_failure(Diagnostic::Spi),
    };
    publish(RadioToHost::RadioId {
        ridtag: id.ridtag(),
        model: id.model(),
        version: id.ver(),
        revision: id.rev(),
    });

    let tx_power = radio
        .read_otp(0x011)
        .await
        .unwrap_or_else(|_| reset_after_radio_failure(Diagnostic::Spi));
    let antenna = radio
        .read_otp(0x01a)
        .await
        .unwrap_or_else(|_| reset_after_radio_failure(Diagnostic::Spi));
    let xtal = radio
        .read_otp(0x01e)
        .await
        .unwrap_or_else(|_| reset_after_radio_failure(Diagnostic::Spi));
    let otp_revision = radio
        .read_otp(0x01f)
        .await
        .unwrap_or_else(|_| reset_after_radio_failure(Diagnostic::Spi));
    publish(RadioToHost::Otp {
        tx_power,
        antenna,
        xtal,
        revision: otp_revision,
    });

    let radio = match radio.init().await {
        Ok(radio) => radio,
        Err(error) => reset_after_radio_failure(diagnostic(&error)),
    };
    let mut radio = match radio.config(radio_config(), Delay).await {
        Ok(radio) => radio,
        Err(error) => reset_after_radio_failure(diagnostic(&error)),
    };

    // The DW3000 must be initialized on a slow SPI clock. Once it is in its
    // configured IDLE_PLL state, both it and nRF52833 SPIM3 support 32 MHz.
    let mut fast_spi_config = spim::Config::default();
    fast_spi_config.frequency = spim::Frequency::M32;
    radio
        .ll()
        .bus()
        .bus_mut()
        .set_config(&fast_spi_config)
        .unwrap_or_else(|_| reset_after_radio_failure(Diagnostic::Unknown));

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
        reset_after_radio_failure(Diagnostic::Spi);
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
        reset_after_radio_failure(Diagnostic::Spi);
    }
    publish(RadioToHost::Ready {
        mode: OperatingMode::Native,
        rx_delay,
        tx_delay,
        node_address: OWN_ADDRESS,
    });
    let initial_counters = wait.counters().health_counters();
    update_radio_counters(initial_counters);
    publish(RadioToHost::Health {
        request_id: 0,
        counters: initial_counters,
    });

    let config = radio_config();
    let mut rx_buf = [0_u8; AIR_FRAME_MAX_NO_FCS + PHY_FCS_LEN];
    let mut range_count = 0_u32;
    let mut mac_sequence = 0_u8;
    let mut next_exchange_id = 0_u16;
    // Preparation-path SPI errors get three attempts between completed ranges.
    // Errors from APIs that consume Ready or occur mid-transfer reset at once
    // because the driver cannot guarantee a safe typestate for re-entry.
    let mut spi_errors_since_range = 0_u8;

    if cfg!(feature = "initiator") {
        'transaction: loop {
            let exchange_id = next_exchange_id;
            next_exchange_id = next_exchange_id.wrapping_add(1);
            if let Err(error) = prepare_tx(&mut radio).await {
                recover_exchange(error, &mut wait, &mut spi_errors_since_range);
                Timer::after_millis(10).await;
                continue 'transaction;
            }
            let poll = encode_air(
                &mut mac_sequence,
                PEER_ADDRESS,
                AirEnvelope::new(
                    exchange_id,
                    AirMessage::Poll {
                        state: latest_ego_state(),
                    },
                ),
            )
            .unwrap_or_else(|error| reset_after_radio_failure(error));
            let mut sending = match radio.send_raw(poll.bytes(), SendTime::Now, &config).await {
                Ok(sending) => sending,
                Err(error) => reset_after_radio_failure(diagnostic(&error)),
            };
            let poll_tx = match wait_send(&mut sending, &mut wait).await {
                Ok(timestamp) => timestamp,
                Err(error) => {
                    reset_after_radio_failure(error);
                }
            };
            radio = finish_sending(sending, &mut wait).await;

            if let Err(error) = prepare_rx(&mut radio, Some(RESPONSE_TIMEOUT_US)).await {
                recover_exchange(error, &mut wait, &mut spi_errors_since_range);
                Timer::after_millis(10).await;
                continue 'transaction;
            }
            let mut receiving = match radio.receive(config).await {
                Ok(receiving) => receiving,
                Err(error) => reset_after_radio_failure(diagnostic(&error)),
            };
            let (length, poll_ack_rx) =
                match wait_receive(&mut receiving, &mut rx_buf, &mut wait).await {
                    Ok(message) => message,
                    Err(error) if recoverable_receive_error(error) => {
                        warn!("restarting exchange after POLL_ACK receive error");
                        publish_error(error);
                        wait.recovered();
                        radio = finish_receiving(receiving, &mut wait).await;
                        Timer::after_millis(10).await;
                        continue 'transaction;
                    }
                    Err(error) => {
                        reset_after_radio_failure(error);
                    }
                };
            radio = finish_receiving(receiving, &mut wait).await;
            let response = match decode_air(&rx_buf, length) {
                Ok(frame) => frame,
                Err(error) => {
                    publish_error(error);
                    continue;
                }
            };
            publish_rx(&response.envelope.message);
            let (poll_rx, response_tx, peer_state) = match response.envelope.message {
                AirMessage::Response {
                    poll_rx,
                    response_tx,
                    state,
                } if response.source == PEER_ADDRESS
                    && response.envelope.exchange_id == exchange_id =>
                {
                    (poll_rx, response_tx, state)
                }
                _ => {
                    publish_error(Diagnostic::UnexpectedAir);
                    continue;
                }
            };
            publish(RadioToHost::PeerState {
                peer: response.source,
                exchange_id,
                state: peer_state,
            });

            let final_at_value = delayed_tx_time(poll_ack_rx.value(), REPLY_DELAY_US);
            let predicted_final_tx = final_at_value.wrapping_add(tx_delay as u64) & TIMESTAMP_MASK;
            let final_frame = encode_air(
                &mut mac_sequence,
                PEER_ADDRESS,
                AirEnvelope::new(
                    exchange_id,
                    AirMessage::Final {
                        poll_tx: poll_tx.value(),
                        response_rx: poll_ack_rx.value(),
                        final_tx: predicted_final_tx,
                    },
                ),
            )
            .unwrap_or_else(|error| reset_after_radio_failure(error));
            if let Err(error) = prepare_tx(&mut radio).await {
                recover_exchange(error, &mut wait, &mut spi_errors_since_range);
                Timer::after_millis(10).await;
                continue 'transaction;
            }
            let final_at = RadioInstant::new(final_at_value)
                .unwrap_or_else(|| reset_after_radio_failure(Diagnostic::InvalidTimestamp));
            let mut sending = match radio
                .send_raw(final_frame.bytes(), SendTime::Delayed(final_at), &config)
                .await
            {
                Ok(sending) => sending,
                Err(error) => reset_after_radio_failure(diagnostic(&error)),
            };
            let actual_final_tx = match wait_send(&mut sending, &mut wait).await {
                Ok(timestamp) => timestamp,
                Err(error) => {
                    recover_exchange(error, &mut wait, &mut spi_errors_since_range);
                    radio = finish_sending(sending, &mut wait).await;
                    Timer::after_millis(INITIATOR_INTER_EXCHANGE_GUARD_MS).await;
                    continue 'transaction;
                }
            };
            radio = finish_sending(sending, &mut wait).await;
            if !scheduled_tx_matches(predicted_final_tx, actual_final_tx.value()) {
                publish_error(Diagnostic::DelayedSendTooLate);
                Timer::after_millis(INITIATOR_INTER_EXCHANGE_GUARD_MS).await;
                continue 'transaction;
            }

            if let Err(error) = prepare_rx(&mut radio, Some(RESPONSE_TIMEOUT_US)).await {
                recover_exchange(error, &mut wait, &mut spi_errors_since_range);
                Timer::after_millis(10).await;
                continue 'transaction;
            }
            let mut receiving = match radio.receive(config).await {
                Ok(receiving) => receiving,
                Err(error) => reset_after_radio_failure(diagnostic(&error)),
            };
            let (length, _report_rx) =
                match wait_receive(&mut receiving, &mut rx_buf, &mut wait).await {
                    Ok(message) => message,
                    Err(error) if recoverable_receive_error(error) => {
                        warn!("restarting exchange after REPORT receive error");
                        publish_error(error);
                        wait.recovered();
                        radio = finish_receiving(receiving, &mut wait).await;
                        Timer::after_millis(10).await;
                        continue 'transaction;
                    }
                    Err(error) => {
                        reset_after_radio_failure(error);
                    }
                };
            radio = finish_receiving(receiving, &mut wait).await;
            let report = match decode_air(&rx_buf, length) {
                Ok(frame) => frame,
                Err(error) => {
                    publish_error(error);
                    continue;
                }
            };
            publish_rx(&report.envelope.message);
            let final_rx = match report.envelope.message {
                AirMessage::Report {
                    final_rx,
                    status: 0,
                } if report.source == PEER_ADDRESS
                    && report.envelope.exchange_id == exchange_id =>
                {
                    final_rx
                }
                _ => {
                    publish_error(Diagnostic::UnexpectedAir);
                    continue;
                }
            };
            let Some(distance) = distance_metres(
                [poll_tx.value(), poll_ack_rx.value(), predicted_final_tx],
                ResponderTimestamps {
                    poll_rx,
                    poll_ack_tx: response_tx,
                    range_rx: final_rx,
                },
            ) else {
                publish_error(Diagnostic::InvalidDistance);
                continue;
            };
            publish(RadioToHost::Range {
                peer: report.source,
                exchange_id,
                range_event_time_dtu: actual_final_tx.value(),
                millimetres: millimetres(distance),
                rssi_cdbm: i16::MIN,
                quality_flags: 0,
            });
            publish_health(&wait, &mut range_count);
            spi_errors_since_range = 0;
            Timer::after_millis(INITIATOR_INTER_EXCHANGE_GUARD_MS).await;
        }
    }

    'responder: loop {
        if let Err(error) = prepare_rx(&mut radio, None).await {
            recover_exchange(error, &mut wait, &mut spi_errors_since_range);
            Timer::after_millis(10).await;
            continue 'responder;
        }
        let mut receiving = match radio.receive(config).await {
            Ok(receiving) => receiving,
            Err(error) => reset_after_radio_failure(diagnostic(&error)),
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
                        reset_after_radio_failure(Diagnostic::Spi);
                    }
                }
                Err(error) => {
                    reset_after_radio_failure(error);
                }
            }
        };
        radio = finish_receiving(receiving, &mut wait).await;
        let poll = match decode_air(&rx_buf, length) {
            Ok(frame) => frame,
            Err(error) => {
                publish_error(error);
                continue;
            }
        };
        publish_rx(&poll.envelope.message);
        let peer_state = match poll.envelope.message {
            AirMessage::Poll { state } if poll.source == PEER_ADDRESS => state,
            _ => {
                publish_error(Diagnostic::UnexpectedAir);
                continue;
            }
        };
        let exchange_id = poll.envelope.exchange_id;
        publish(RadioToHost::PeerState {
            peer: poll.source,
            exchange_id,
            state: peer_state,
        });

        let response_at_value = delayed_tx_time(poll_rx.value(), REPLY_DELAY_US);
        let predicted_response_tx =
            response_at_value.wrapping_add(tx_delay as u64) & TIMESTAMP_MASK;
        let response_at = RadioInstant::new(response_at_value)
            .unwrap_or_else(|| reset_after_radio_failure(Diagnostic::InvalidTimestamp));
        if let Err(error) = prepare_tx(&mut radio).await {
            recover_exchange(error, &mut wait, &mut spi_errors_since_range);
            Timer::after_millis(10).await;
            continue 'responder;
        }
        let response = encode_air(
            &mut mac_sequence,
            poll.source,
            AirEnvelope::new(
                exchange_id,
                AirMessage::Response {
                    poll_rx: poll_rx.value(),
                    response_tx: predicted_response_tx,
                    state: latest_ego_state(),
                },
            ),
        )
        .unwrap_or_else(|error| reset_after_radio_failure(error));
        let mut sending = match radio
            .send_raw(response.bytes(), SendTime::Delayed(response_at), &config)
            .await
        {
            Ok(sending) => sending,
            Err(error) => reset_after_radio_failure(diagnostic(&error)),
        };
        let actual_response_tx = match wait_send(&mut sending, &mut wait).await {
            Ok(timestamp) => timestamp,
            Err(error) => {
                recover_exchange(error, &mut wait, &mut spi_errors_since_range);
                radio = finish_sending(sending, &mut wait).await;
                continue 'responder;
            }
        };
        radio = finish_sending(sending, &mut wait).await;
        if !scheduled_tx_matches(predicted_response_tx, actual_response_tx.value()) {
            publish_error(Diagnostic::DelayedSendTooLate);
            continue 'responder;
        }

        if let Err(error) = prepare_rx(&mut radio, Some(RESPONSE_TIMEOUT_US)).await {
            recover_exchange(error, &mut wait, &mut spi_errors_since_range);
            Timer::after_millis(10).await;
            continue 'responder;
        }
        let mut receiving = match radio.receive(config).await {
            Ok(receiving) => receiving,
            Err(error) => reset_after_radio_failure(diagnostic(&error)),
        };
        let (length, final_rx) = match wait_receive(&mut receiving, &mut rx_buf, &mut wait).await {
            Ok(message) => message,
            Err(error) if recoverable_receive_error(error) => {
                warn!("restarting exchange after FINAL receive error");
                publish_error(error);
                wait.recovered();
                radio = finish_receiving(receiving, &mut wait).await;
                continue 'responder;
            }
            Err(error) => {
                reset_after_radio_failure(error);
            }
        };
        radio = finish_receiving(receiving, &mut wait).await;
        let final_frame = match decode_air(&rx_buf, length) {
            Ok(frame) => frame,
            Err(error) => {
                publish_error(error);
                continue;
            }
        };
        publish_rx(&final_frame.envelope.message);
        let (poll_tx, response_rx, final_tx) = match final_frame.envelope.message {
            AirMessage::Final {
                poll_tx,
                response_rx,
                final_tx,
            } if final_frame.source == poll.source
                && final_frame.envelope.exchange_id == exchange_id =>
            {
                (poll_tx, response_rx, final_tx)
            }
            _ => {
                publish_error(Diagnostic::UnexpectedAir);
                continue;
            }
        };
        let Some(distance) = distance_metres(
            [poll_tx, response_rx, final_tx],
            ResponderTimestamps {
                poll_rx: poll_rx.value(),
                poll_ack_tx: predicted_response_tx,
                range_rx: final_rx.value(),
            },
        ) else {
            publish_error(Diagnostic::InvalidDistance);
            continue;
        };
        if let Err(error) = prepare_tx(&mut radio).await {
            recover_exchange(error, &mut wait, &mut spi_errors_since_range);
            Timer::after_millis(10).await;
            continue 'responder;
        }
        let report = encode_air(
            &mut mac_sequence,
            poll.source,
            AirEnvelope::new(
                exchange_id,
                AirMessage::Report {
                    final_rx: final_rx.value(),
                    status: 0,
                },
            ),
        )
        .unwrap_or_else(|error| reset_after_radio_failure(error));
        let report_at_value = delayed_tx_time(final_rx.value(), REPORT_TURNAROUND_US);
        let report_at = RadioInstant::new(report_at_value)
            .unwrap_or_else(|| reset_after_radio_failure(Diagnostic::InvalidTimestamp));
        let mut sending = match radio
            .send_raw(report.bytes(), SendTime::Delayed(report_at), &config)
            .await
        {
            Ok(sending) => sending,
            Err(error) => reset_after_radio_failure(diagnostic(&error)),
        };
        if let Err(error) = wait_send(&mut sending, &mut wait).await {
            reset_after_radio_failure(error);
        }
        radio = finish_sending(sending, &mut wait).await;
        publish(RadioToHost::Range {
            peer: poll.source,
            exchange_id,
            range_event_time_dtu: final_rx.value(),
            millimetres: millimetres(distance),
            rssi_cdbm: i16::MIN,
            quality_flags: 0,
        });
        publish_health(&wait, &mut range_count);
        spi_errors_since_range = 0;
    }
}

fn encode_air(
    mac_sequence: &mut u8,
    destination: NodeAddress,
    envelope: AirEnvelope,
) -> Result<air::EncodedAirFrame, Diagnostic> {
    let sequence = *mac_sequence;
    *mac_sequence = (*mac_sequence).wrapping_add(1);
    air::encode(
        OWN_ADDRESS,
        Destination::Node(destination),
        sequence,
        &envelope,
    )
    .map_err(|_| Diagnostic::MalformedAir)
}

fn decode_air(buffer: &[u8], received_length: usize) -> Result<DecodedAirFrame, Diagnostic> {
    let Some(frame_length) = received_length.checked_sub(PHY_FCS_LEN) else {
        return Err(Diagnostic::MalformedAir);
    };
    if frame_length > AIR_FRAME_MAX_NO_FCS || frame_length > buffer.len() {
        return Err(Diagnostic::MalformedAir);
    }
    air::decode(&buffer[..frame_length], OWN_ADDRESS).map_err(|_| Diagnostic::MalformedAir)
}

fn publish_rx(message: &AirMessage) {
    publish(RadioToHost::Rx {
        kind: match message {
            AirMessage::Poll { .. } => 0,
            AirMessage::Response { .. } => 1,
            AirMessage::Final { .. } => 2,
            AirMessage::Report { .. } => 3,
        },
    });
}

fn millimetres(distance: f64) -> u32 {
    (distance * 1_000.0 + 0.5).clamp(0.0, u32::MAX as f64) as u32
}

fn recover_exchange(
    diagnostic: Diagnostic,
    wait: &mut RadioWait<'_>,
    spi_errors_since_range: &mut u8,
) {
    publish_error(diagnostic);
    wait.recovered();
    if diagnostic == Diagnostic::Spi {
        *spi_errors_since_range = (*spi_errors_since_range).saturating_add(1);
        if *spi_errors_since_range >= 3 {
            reset_after_radio_failure(diagnostic);
        }
    } else {
        *spi_errors_since_range = 0;
    }
}
