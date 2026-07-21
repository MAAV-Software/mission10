#![allow(missing_docs)]

extern crate alloc;

use alloc::collections::VecDeque;

use dw1000_rs::protocol::{
    decode_poll_targets, decode_range_timings, detect_frame_kind, encode_discovery_blink,
    encode_poll_ack, encode_range_report, encode_ranging_init, encode_schedule_sync, parse_frame,
    Frame, FrameKind, PollTarget, RangeReportPayload, RangeTiming,
};
use dw1000_rs::ranging::RangingRadio;
use dw1000_rs::{
    DeviceIdentity, DwTime, Error, Eui64, PanId, RangingConfig, RangingEvent, RangingNode, Role,
    RxFrame, RxOptions, ShortAddress, SignalMetrics, SysStatus, Timestamps, TxOptions,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MockError;

#[derive(Debug, Clone)]
struct QueuedFrame {
    bytes: Vec<u8>,
    timestamp: DwTime,
    metrics: SignalMetrics,
}

#[derive(Debug, Clone, Default)]
struct MockRadio {
    rx_frames: VecDeque<QueuedFrame>,
    transmitted: Vec<(Vec<u8>, TxOptions)>,
    receive_modes: Vec<RxOptions>,
    next_timestamps: VecDeque<Timestamps>,
    delayed_times: VecDeque<DwTime>,
}

impl MockRadio {
    fn push_rx(&mut self, bytes: &[u8], timestamp: DwTime, metrics: SignalMetrics) {
        self.rx_frames.push_back(QueuedFrame {
            bytes: bytes.to_vec(),
            timestamp,
            metrics,
        });
    }

    fn last_tx(&self) -> &[u8] {
        &self.transmitted.last().unwrap().0
    }
}

impl RangingRadio<MockError, MockError> for MockRadio {
    fn start_receive(&mut self, options: RxOptions) -> Result<(), Error<MockError, MockError>> {
        self.receive_modes.push(options);
        Ok(())
    }

    fn transmit(
        &mut self,
        frame: &[u8],
        options: TxOptions,
    ) -> Result<(), Error<MockError, MockError>> {
        self.transmitted.push((frame.to_vec(), options));
        Ok(())
    }

    fn read_frame<'a>(
        &mut self,
        buffer: &'a mut [u8],
    ) -> Result<RxFrame<'a>, Error<MockError, MockError>> {
        let queued = self.rx_frames.pop_front().unwrap();
        buffer[..queued.bytes.len()].copy_from_slice(&queued.bytes);
        Ok(RxFrame {
            bytes: &buffer[..queued.bytes.len()],
            timestamp: queued.timestamp,
            metrics: queued.metrics,
            status: SysStatus::EMPTY,
        })
    }

    fn read_timestamps(&mut self) -> Result<Timestamps, Error<MockError, MockError>> {
        Ok(self.next_timestamps.pop_front().unwrap())
    }

    fn compute_delayed_time(
        &mut self,
        _delay: DwTime,
    ) -> Result<DwTime, Error<MockError, MockError>> {
        Ok(self.delayed_times.pop_front().unwrap())
    }
}

fn identity(short: u16, eui: [u8; 8]) -> DeviceIdentity {
    DeviceIdentity::new(
        PanId::new(0xDECA),
        ShortAddress::new(short),
        Eui64::new(eui),
    )
}

fn metrics() -> SignalMetrics {
    SignalMetrics {
        receive_power_dbm: -82.0,
        first_path_power_dbm: -84.0,
        quality: 1.5,
    }
}

#[test]
fn single_tag_anchor_exchange_completes() {
    let tag_identity = identity(0x1234, [0x7D, 0x00, 0x22, 0xEA, 0x82, 0x60, 0x3B, 0x9C]);
    let anchor_identity = identity(0x4321, [0x82, 0x17, 0x5B, 0xD5, 0xA9, 0x9A, 0xE2, 0x9C]);
    let mut tag = RangingNode::<4>::new(Role::Tag, RangingConfig::new(tag_identity));
    let mut anchor = RangingNode::<4>::new(Role::Anchor, RangingConfig::new(anchor_identity));
    let mut tag_radio = MockRadio::default();
    let mut anchor_radio = MockRadio::default();

    tag.start(&mut tag_radio, 0).unwrap();
    anchor.start(&mut anchor_radio, 0).unwrap();

    tag.tick(&mut tag_radio, 80).unwrap();
    anchor_radio.push_rx(tag_radio.last_tx(), DwTime::from_ticks(10), metrics());
    let blink_event = anchor
        .on_rx(&mut anchor_radio, 81, &mut [0u8; 127])
        .unwrap();
    assert!(matches!(blink_event, Some(RangingEvent::BlinkReceived(_))));

    tag_radio.push_rx(anchor_radio.last_tx(), DwTime::from_ticks(20), metrics());
    let init_event = tag.on_rx(&mut tag_radio, 82, &mut [0u8; 127]).unwrap();
    assert_eq!(
        init_event,
        Some(RangingEvent::RangingInitReceived(ShortAddress::new(0x4321)))
    );

    tag.tick(&mut tag_radio, 160).unwrap();
    tag.tick(&mut tag_radio, 240).unwrap();
    tag_radio.next_timestamps.push_back(Timestamps {
        tx: DwTime::from_ticks(100),
        rx: DwTime::zero(),
        system: DwTime::zero(),
    });
    tag.on_tx_done(&mut tag_radio).unwrap();

    anchor_radio.push_rx(tag_radio.last_tx(), DwTime::from_ticks(120), metrics());
    anchor
        .on_rx(&mut anchor_radio, 241, &mut [0u8; 127])
        .unwrap();
    anchor_radio.next_timestamps.push_back(Timestamps {
        tx: DwTime::from_ticks(180),
        rx: DwTime::zero(),
        system: DwTime::zero(),
    });
    anchor.on_tx_done(&mut anchor_radio).unwrap();

    tag_radio.push_rx(anchor_radio.last_tx(), DwTime::from_ticks(200), metrics());
    tag_radio.delayed_times.push_back(DwTime::from_ticks(260));
    tag.on_rx(&mut tag_radio, 242, &mut [0u8; 127]).unwrap();
    tag_radio.next_timestamps.push_back(Timestamps {
        tx: DwTime::from_ticks(260),
        rx: DwTime::zero(),
        system: DwTime::zero(),
    });
    tag.on_tx_done(&mut tag_radio).unwrap();

    anchor_radio.push_rx(tag_radio.last_tx(), DwTime::from_ticks(280), metrics());
    let anchor_event = anchor
        .on_rx(&mut anchor_radio, 243, &mut [0u8; 127])
        .unwrap();
    let Some(RangingEvent::RangeUpdated(anchor_snapshot)) = anchor_event else {
        panic!("expected anchor range update");
    };

    tag_radio.push_rx(anchor_radio.last_tx(), DwTime::from_ticks(300), metrics());
    let tag_event = tag.on_rx(&mut tag_radio, 244, &mut [0u8; 127]).unwrap();
    let Some(RangingEvent::RangeUpdated(tag_snapshot)) = tag_event else {
        panic!("expected tag range update");
    };

    let expected_range = DwTime::from_ticks(20).as_meters();
    assert!((anchor_snapshot.range_m - expected_range).abs() < 1e-6);
    assert!((tag_snapshot.range_m - expected_range).abs() < 1e-6);
}

#[test]
fn secondary_anchor_observes_coordinator_sync_without_adding_a_peer() {
    let coordinator_identity = identity(0x3344, [0xB1, 0x4A, 0x7C, 0x00, 0x11, 0x22, 0x33, 0x44]);
    let secondary_identity = identity(0x3345, [0xB1, 0x4A, 0x7C, 0x00, 0x11, 0x22, 0x33, 0x45]);
    let mut secondary = RangingNode::<4>::new(Role::Anchor, RangingConfig::new(secondary_identity));
    let mut radio = MockRadio::default();
    let mut sync = [0u8; 127];
    let len = encode_schedule_sync(
        1,
        coordinator_identity.short_address,
        ShortAddress::BROADCAST,
        80,
        1,
        250,
        &mut sync,
    )
    .unwrap();

    secondary.start(&mut radio, 0).unwrap();
    radio.push_rx(&sync[..len], DwTime::from_ticks(10), metrics());

    assert_eq!(
        secondary.on_rx(&mut radio, 81, &mut [0u8; 127]).unwrap(),
        Some(RangingEvent::ScheduleSyncReceived(
            coordinator_identity.short_address
        ))
    );
    assert_eq!(secondary.peers().count(), 0);
    assert!(radio.transmitted.is_empty());
}

#[test]
fn discovery_response_uses_the_configured_anchor_slot_delay() {
    let tag_identity = identity(0x3400, [1, 2, 3, 4, 5, 6, 7, 8]);
    let anchor_identity = identity(0x3345, [9, 10, 11, 12, 13, 14, 15, 16]);
    let mut config = RangingConfig::new(anchor_identity);
    config.schedule.anchor_slot = 1;
    config.schedule.discovery_slot_spacing_us = 12_000;
    let mut anchor = RangingNode::<4>::new(Role::Anchor, config);
    let mut radio = MockRadio::default();
    let mut blink = [0u8; 127];
    let blink_len =
        encode_discovery_blink(0, tag_identity.eui, tag_identity.short_address, &mut blink)
            .unwrap();

    anchor.start(&mut radio, 0).unwrap();
    radio.push_rx(&blink[..blink_len], DwTime::from_ticks(10), metrics());
    anchor.on_rx(&mut radio, 1, &mut [0u8; 127]).unwrap();

    assert_eq!(
        radio.transmitted[0].1.delayed_time,
        Some(DwTime::from_micros(12_000.0))
    );
}

#[test]
fn four_anchor_exchange_completes() {
    let tag_identity = identity(0x1234, [0x7D, 0x00, 0x22, 0xEA, 0x82, 0x60, 0x3B, 0x9C]);
    let anchor_identities = [
        identity(0x4001, [0x10, 0x17, 0x5B, 0xD5, 0xA9, 0x9A, 0xE2, 0x91]),
        identity(0x4002, [0x20, 0x17, 0x5B, 0xD5, 0xA9, 0x9A, 0xE2, 0x92]),
        identity(0x4003, [0x30, 0x17, 0x5B, 0xD5, 0xA9, 0x9A, 0xE2, 0x93]),
        identity(0x4004, [0x40, 0x17, 0x5B, 0xD5, 0xA9, 0x9A, 0xE2, 0x94]),
    ];
    let mut tag_config = RangingConfig::new(tag_identity);
    tag_config.reply_delay_us = 12_000;
    tag_config.schedule.session_timeout_ms = 200;
    let mut tag = RangingNode::<4>::new(Role::Tag, tag_config);
    let mut anchors = [
        RangingNode::<4>::new(Role::Anchor, RangingConfig::new(anchor_identities[0])),
        RangingNode::<4>::new(Role::Anchor, RangingConfig::new(anchor_identities[1])),
        RangingNode::<4>::new(Role::Anchor, RangingConfig::new(anchor_identities[2])),
        RangingNode::<4>::new(Role::Anchor, RangingConfig::new(anchor_identities[3])),
    ];
    let mut tag_radio = MockRadio::default();
    let mut anchor_radios = [
        MockRadio::default(),
        MockRadio::default(),
        MockRadio::default(),
        MockRadio::default(),
    ];

    tag.start(&mut tag_radio, 0).unwrap();
    for (index, anchor) in anchors.iter_mut().enumerate() {
        anchor.start(&mut anchor_radios[index], 0).unwrap();
    }

    tag.tick(&mut tag_radio, 80).unwrap();
    let blink = tag_radio.last_tx().to_vec();
    for (index, anchor) in anchors.iter_mut().enumerate() {
        anchor_radios[index].push_rx(&blink, DwTime::from_ticks(10 + index as i64), metrics());
        anchor
            .on_rx(
                &mut anchor_radios[index],
                81 + index as u32,
                &mut [0u8; 127],
            )
            .unwrap();
    }
    for (index, radio) in anchor_radios.iter().enumerate() {
        tag_radio.push_rx(
            radio.last_tx(),
            DwTime::from_ticks(20 + index as i64),
            metrics(),
        );
        assert!(matches!(
            tag.on_rx(&mut tag_radio, 90 + index as u32, &mut [0u8; 127])
                .unwrap(),
            Some(RangingEvent::RangingInitReceived(_))
        ));
    }

    tag.tick(&mut tag_radio, 160).unwrap();
    tag_radio.next_timestamps.push_back(Timestamps {
        tx: DwTime::from_ticks(100),
        rx: DwTime::zero(),
        system: DwTime::zero(),
    });
    tag.on_tx_done(&mut tag_radio).unwrap();
    let poll = tag_radio.last_tx().to_vec();
    for (index, anchor) in anchors.iter_mut().enumerate() {
        anchor_radios[index].push_rx(&poll, DwTime::from_ticks(120 + index as i64), metrics());
        anchor
            .on_rx(
                &mut anchor_radios[index],
                170 + index as u32,
                &mut [0u8; 127],
            )
            .unwrap();
        anchor_radios[index].next_timestamps.push_back(Timestamps {
            tx: DwTime::from_ticks(180 + index as i64),
            rx: DwTime::zero(),
            system: DwTime::zero(),
        });
        anchor.on_tx_done(&mut anchor_radios[index]).unwrap();
    }

    tag_radio.delayed_times.push_back(DwTime::from_ticks(260));
    for (index, radio) in anchor_radios.iter().enumerate() {
        tag_radio.push_rx(
            radio.last_tx(),
            DwTime::from_ticks(200 + index as i64),
            metrics(),
        );
        tag.on_rx(&mut tag_radio, 190 + index as u32, &mut [0u8; 127])
            .unwrap();
    }
    assert_eq!(
        detect_frame_kind(tag_radio.last_tx()).unwrap(),
        FrameKind::Range
    );

    tag_radio.next_timestamps.push_back(Timestamps {
        tx: DwTime::from_ticks(260),
        rx: DwTime::zero(),
        system: DwTime::zero(),
    });
    tag.on_tx_done(&mut tag_radio).unwrap();
    let range = tag_radio.last_tx().to_vec();
    for (index, anchor) in anchors.iter_mut().enumerate() {
        anchor_radios[index].push_rx(&range, DwTime::from_ticks(280 + index as i64), metrics());
        assert!(matches!(
            anchor
                .on_rx(
                    &mut anchor_radios[index],
                    220 + index as u32,
                    &mut [0u8; 127]
                )
                .unwrap(),
            Some(RangingEvent::RangeUpdated(_))
        ));
    }

    let mut reports = 0;
    for (index, radio) in anchor_radios.iter().enumerate() {
        tag_radio.push_rx(
            radio.last_tx(),
            DwTime::from_ticks(300 + index as i64),
            metrics(),
        );
        if matches!(
            tag.on_rx(&mut tag_radio, 260 + index as u32, &mut [0u8; 127])
                .unwrap(),
            Some(RangingEvent::RangeUpdated(_))
        ) {
            reports += 1;
        }
    }
    assert_eq!(reports, 4);
}

#[test]
fn tick_prunes_inactive_peers() {
    let tag_identity = identity(0x1234, [0, 1, 2, 3, 4, 5, 6, 7]);
    let anchor_identity = identity(0x4321, [8, 9, 10, 11, 12, 13, 14, 15]);
    let mut tag = RangingNode::<4>::new(Role::Tag, RangingConfig::new(tag_identity));
    let mut anchor = RangingNode::<4>::new(Role::Anchor, RangingConfig::new(anchor_identity));
    let mut tag_radio = MockRadio::default();
    let mut anchor_radio = MockRadio::default();

    tag.start(&mut tag_radio, 0).unwrap();
    anchor.start(&mut anchor_radio, 0).unwrap();
    tag.tick(&mut tag_radio, 80).unwrap();
    anchor_radio.push_rx(tag_radio.last_tx(), DwTime::from_ticks(10), metrics());
    anchor
        .on_rx(&mut anchor_radio, 81, &mut [0u8; 127])
        .unwrap();
    tag_radio.push_rx(anchor_radio.last_tx(), DwTime::from_ticks(20), metrics());
    tag.on_rx(&mut tag_radio, 82, &mut [0u8; 127]).unwrap();

    let event = tag.tick(&mut tag_radio, 400).unwrap();
    assert_eq!(
        event,
        Some(RangingEvent::PeerInactive(ShortAddress::new(0x4321)))
    );
}

#[test]
fn anchor_recovers_after_inactivity_and_accepts_rediscovery() {
    let tag_identity = identity(0x1234, [0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17]);
    let anchor_identity = identity(0x4321, [0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27]);
    let mut tag_config = RangingConfig::new(tag_identity);
    let mut anchor_config = RangingConfig::new(anchor_identity);
    tag_config.reset_period_ms = 100;
    anchor_config.reset_period_ms = 100;

    let mut tag = RangingNode::<4>::new(Role::Tag, tag_config);
    let mut anchor = RangingNode::<4>::new(Role::Anchor, anchor_config);
    let mut tag_radio = MockRadio::default();
    let mut anchor_radio = MockRadio::default();

    tag.start(&mut tag_radio, 0).unwrap();
    anchor.start(&mut anchor_radio, 0).unwrap();

    tag.tick(&mut tag_radio, 80).unwrap();
    anchor_radio.push_rx(tag_radio.last_tx(), DwTime::from_ticks(10), metrics());
    anchor
        .on_rx(&mut anchor_radio, 81, &mut [0u8; 127])
        .unwrap();
    tag_radio.push_rx(anchor_radio.last_tx(), DwTime::from_ticks(20), metrics());
    tag.on_rx(&mut tag_radio, 82, &mut [0u8; 127]).unwrap();

    tag.tick(&mut tag_radio, 160).unwrap();
    anchor_radio.push_rx(tag_radio.last_tx(), DwTime::from_ticks(30), metrics());
    anchor
        .on_rx(&mut anchor_radio, 161, &mut [0u8; 127])
        .unwrap();

    let anchor_timeout = anchor.tick(&mut anchor_radio, 300).unwrap();
    assert_eq!(
        anchor_timeout,
        Some(RangingEvent::PeerInactive(ShortAddress::new(0x1234)))
    );
    let tag_timeout = tag.tick(&mut tag_radio, 301).unwrap();
    assert_eq!(
        tag_timeout,
        Some(RangingEvent::PeerInactive(ShortAddress::new(0x4321)))
    );

    tag.tick(&mut tag_radio, 380).unwrap();
    anchor_radio.push_rx(tag_radio.last_tx(), DwTime::from_ticks(40), metrics());
    let blink_event = anchor
        .on_rx(&mut anchor_radio, 381, &mut [0u8; 127])
        .unwrap();
    assert!(matches!(blink_event, Some(RangingEvent::BlinkReceived(_))));

    tag_radio.push_rx(anchor_radio.last_tx(), DwTime::from_ticks(50), metrics());
    let init_event = tag.on_rx(&mut tag_radio, 382, &mut [0u8; 127]).unwrap();
    assert_eq!(
        init_event,
        Some(RangingEvent::RangingInitReceived(ShortAddress::new(0x4321)))
    );

    tag.tick(&mut tag_radio, 460).unwrap();
    anchor_radio.push_rx(tag_radio.last_tx(), DwTime::from_ticks(60), metrics());
    assert!(anchor
        .on_rx(&mut anchor_radio, 461, &mut [0u8; 127])
        .is_ok());
}

#[test]
fn recover_link_preserves_peer_and_next_poll_is_accepted_without_rediscovery() {
    let tag_identity = identity(0x1234, [0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37]);
    let anchor_identity = identity(0x4321, [0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47]);
    let mut tag = RangingNode::<4>::new(Role::Tag, RangingConfig::new(tag_identity));
    let mut anchor = RangingNode::<4>::new(Role::Anchor, RangingConfig::new(anchor_identity));
    let mut tag_radio = MockRadio::default();
    let mut anchor_radio = MockRadio::default();

    tag.start(&mut tag_radio, 0).unwrap();
    anchor.start(&mut anchor_radio, 0).unwrap();

    tag.tick(&mut tag_radio, 80).unwrap();
    anchor_radio.push_rx(tag_radio.last_tx(), DwTime::from_ticks(10), metrics());
    anchor
        .on_rx(&mut anchor_radio, 81, &mut [0u8; 127])
        .unwrap();
    tag_radio.push_rx(anchor_radio.last_tx(), DwTime::from_ticks(20), metrics());
    let init_event = tag.on_rx(&mut tag_radio, 82, &mut [0u8; 127]).unwrap();
    assert_eq!(
        init_event,
        Some(RangingEvent::RangingInitReceived(ShortAddress::new(0x4321)))
    );

    let receive_calls_before = anchor_radio.receive_modes.len();
    assert_eq!(anchor.peers().count(), 1);
    anchor.recover_link(&mut anchor_radio, 120).unwrap();
    assert_eq!(anchor.peers().count(), 1);
    assert_eq!(anchor_radio.receive_modes.len(), receive_calls_before + 1);
    assert_eq!(
        anchor_radio.receive_modes.last().copied(),
        Some(RxOptions {
            delayed_time: None,
            permanent: true,
        })
    );

    tag.tick(&mut tag_radio, 160).unwrap();
    anchor_radio.push_rx(tag_radio.last_tx(), DwTime::from_ticks(30), metrics());
    let tx_count_before = anchor_radio.transmitted.len();
    let event = anchor
        .on_rx(&mut anchor_radio, 161, &mut [0u8; 127])
        .unwrap();
    assert_eq!(event, None);
    assert_eq!(anchor_radio.transmitted.len(), tx_count_before + 1);
    assert_eq!(
        detect_frame_kind(anchor_radio.last_tx()).unwrap(),
        FrameKind::PollAck
    );
}

#[test]
fn tag_ignores_poll_ack_for_other_destination() {
    let tag_identity = identity(0x1234, [1, 2, 3, 4, 5, 6, 7, 8]);
    let anchor_identity = identity(0x4321, [9, 10, 11, 12, 13, 14, 15, 16]);
    let mut tag = RangingNode::<4>::new(Role::Tag, RangingConfig::new(tag_identity));
    let mut radio = MockRadio::default();

    tag.start(&mut radio, 0).unwrap();
    tag.tick(&mut radio, 80).unwrap();

    let mut init = [0u8; 127];
    let init_len = encode_ranging_init(
        0,
        anchor_identity.short_address,
        tag_identity.eui,
        &mut init,
    )
    .unwrap();
    radio.push_rx(&init[..init_len], DwTime::from_ticks(20), metrics());
    let init_event = tag.on_rx(&mut radio, 81, &mut [0u8; 127]).unwrap();
    assert_eq!(
        init_event,
        Some(RangingEvent::RangingInitReceived(
            anchor_identity.short_address
        ))
    );

    tag.tick(&mut radio, 160).unwrap();
    radio.next_timestamps.push_back(Timestamps {
        tx: DwTime::from_ticks(100),
        rx: DwTime::zero(),
        system: DwTime::zero(),
    });
    tag.on_tx_done(&mut radio).unwrap();

    let tx_count_before = radio.transmitted.len();
    let mut wrong_poll_ack = [0u8; 127];
    let wrong_poll_ack_len = encode_poll_ack(
        1,
        anchor_identity.short_address,
        ShortAddress::new(0x7777),
        &mut wrong_poll_ack,
    )
    .unwrap();
    radio.push_rx(
        &wrong_poll_ack[..wrong_poll_ack_len],
        DwTime::from_ticks(200),
        metrics(),
    );
    assert_eq!(tag.on_rx(&mut radio, 161, &mut [0u8; 127]).unwrap(), None);
    assert_eq!(radio.transmitted.len(), tx_count_before);

    radio.delayed_times.push_back(DwTime::from_ticks(260));
    let mut poll_ack = [0u8; 127];
    let poll_ack_len = encode_poll_ack(
        1,
        anchor_identity.short_address,
        tag_identity.short_address,
        &mut poll_ack,
    )
    .unwrap();
    radio.push_rx(
        &poll_ack[..poll_ack_len],
        DwTime::from_ticks(200),
        metrics(),
    );
    assert_eq!(tag.on_rx(&mut radio, 162, &mut [0u8; 127]).unwrap(), None);
    assert_eq!(
        detect_frame_kind(radio.last_tx()).unwrap(),
        FrameKind::Range
    );
}

#[test]
fn stale_poll_ack_is_ignored_while_waiting_for_range_report() {
    let tag_identity = identity(0x1234, [17, 18, 19, 20, 21, 22, 23, 24]);
    let anchor_identity = identity(0x4321, [25, 26, 27, 28, 29, 30, 31, 32]);
    let mut tag = RangingNode::<4>::new(Role::Tag, RangingConfig::new(tag_identity));
    let mut radio = MockRadio::default();

    tag.start(&mut radio, 0).unwrap();
    tag.tick(&mut radio, 80).unwrap();

    let mut init = [0u8; 127];
    let init_len = encode_ranging_init(
        0,
        anchor_identity.short_address,
        tag_identity.eui,
        &mut init,
    )
    .unwrap();
    radio.push_rx(&init[..init_len], DwTime::from_ticks(20), metrics());
    tag.on_rx(&mut radio, 81, &mut [0u8; 127]).unwrap();

    tag.tick(&mut radio, 160).unwrap();
    radio.next_timestamps.push_back(Timestamps {
        tx: DwTime::from_ticks(100),
        rx: DwTime::zero(),
        system: DwTime::zero(),
    });
    tag.on_tx_done(&mut radio).unwrap();

    radio.delayed_times.push_back(DwTime::from_ticks(260));
    let mut poll_ack = [0u8; 127];
    let poll_ack_len = encode_poll_ack(
        1,
        anchor_identity.short_address,
        tag_identity.short_address,
        &mut poll_ack,
    )
    .unwrap();
    radio.push_rx(
        &poll_ack[..poll_ack_len],
        DwTime::from_ticks(200),
        metrics(),
    );
    tag.on_rx(&mut radio, 161, &mut [0u8; 127]).unwrap();
    radio.next_timestamps.push_back(Timestamps {
        tx: DwTime::from_ticks(260),
        rx: DwTime::zero(),
        system: DwTime::zero(),
    });
    tag.on_tx_done(&mut radio).unwrap();

    let tx_count_before = radio.transmitted.len();
    radio.push_rx(
        &poll_ack[..poll_ack_len],
        DwTime::from_ticks(200),
        metrics(),
    );
    assert_eq!(tag.on_rx(&mut radio, 162, &mut [0u8; 127]).unwrap(), None);
    assert_eq!(radio.transmitted.len(), tx_count_before);

    let mut range_report = [0u8; 127];
    let range_report_len = encode_range_report(
        2,
        anchor_identity.short_address,
        tag_identity.short_address,
        RangeReportPayload {
            poll_received: DwTime::from_ticks(120),
            poll_ack_sent: DwTime::from_ticks(180),
            range_received: DwTime::from_ticks(280),
            receive_power_dbm: -82.0,
        },
        &mut range_report,
    )
    .unwrap();
    radio.push_rx(
        &range_report[..range_report_len],
        DwTime::from_ticks(300),
        metrics(),
    );
    assert!(matches!(
        tag.on_rx(&mut radio, 163, &mut [0u8; 127]).unwrap(),
        Some(RangingEvent::RangeUpdated(_))
    ));
}

#[test]
fn missing_poll_ack_times_out_and_ranges_only_acknowledged_anchor() {
    let tag_identity = identity(0x1234, [33, 34, 35, 36, 37, 38, 39, 40]);
    let anchor_a = identity(0x4001, [41, 42, 43, 44, 45, 46, 47, 48]);
    let anchor_b = identity(0x4002, [49, 50, 51, 52, 53, 54, 55, 56]);
    let mut config = RangingConfig::new(tag_identity);
    config.schedule.session_timeout_ms = 20;
    let mut tag = RangingNode::<4>::new(Role::Tag, config);
    let mut radio = MockRadio::default();

    tag.start(&mut radio, 0).unwrap();
    tag.tick(&mut radio, 80).unwrap();

    for anchor in [anchor_a, anchor_b] {
        let mut init = [0u8; 127];
        let init_len =
            encode_ranging_init(0, anchor.short_address, tag_identity.eui, &mut init).unwrap();
        radio.push_rx(&init[..init_len], DwTime::from_ticks(20), metrics());
        tag.on_rx(&mut radio, 81, &mut [0u8; 127]).unwrap();
    }

    tag.tick(&mut radio, 160).unwrap();
    radio.next_timestamps.push_back(Timestamps {
        tx: DwTime::from_ticks(100),
        rx: DwTime::zero(),
        system: DwTime::zero(),
    });
    tag.on_tx_done(&mut radio).unwrap();

    let mut poll_ack = [0u8; 127];
    let poll_ack_len = encode_poll_ack(
        1,
        anchor_a.short_address,
        tag_identity.short_address,
        &mut poll_ack,
    )
    .unwrap();
    radio.push_rx(
        &poll_ack[..poll_ack_len],
        DwTime::from_ticks(200),
        metrics(),
    );
    tag.on_rx(&mut radio, 161, &mut [0u8; 127]).unwrap();

    radio.delayed_times.push_back(DwTime::from_ticks(260));
    tag.tick(&mut radio, 181).unwrap();
    let Frame::Range { payload, .. } = parse_frame(radio.last_tx()).unwrap() else {
        panic!("expected range frame");
    };
    let mut timings = [RangeTiming {
        short_address: ShortAddress::new(0),
        poll_sent: DwTime::zero(),
        poll_ack_received: DwTime::zero(),
        range_sent: DwTime::zero(),
    }; 4];
    let count = decode_range_timings(payload, &mut timings).unwrap();
    assert_eq!(count, 1);
    assert_eq!(timings[0].short_address, anchor_a.short_address);
}

#[test]
fn two_tags_use_distinct_tdma_slots() {
    let mut slot0_config = RangingConfig::new(identity(0x3400, [1, 1, 1, 1, 1, 1, 1, 1]));
    slot0_config.schedule.tag_slot = 0;
    slot0_config.schedule.tag_slot_count = 2;
    slot0_config.schedule.tag_slot_ms = 100;

    let mut slot1_config = RangingConfig::new(identity(0x3401, [2, 2, 2, 2, 2, 2, 2, 2]));
    slot1_config.schedule.tag_slot = 1;
    slot1_config.schedule.tag_slot_count = 2;
    slot1_config.schedule.tag_slot_ms = 100;

    let mut slot0 = RangingNode::<4>::new(Role::Tag, slot0_config);
    let mut slot1 = RangingNode::<4>::new(Role::Tag, slot1_config);
    let mut slot0_radio = MockRadio::default();
    let mut slot1_radio = MockRadio::default();

    slot0.start(&mut slot0_radio, 0).unwrap();
    slot1.start(&mut slot1_radio, 0).unwrap();

    slot0.tick(&mut slot0_radio, 80).unwrap();
    slot1.tick(&mut slot1_radio, 80).unwrap();
    assert_eq!(slot0_radio.transmitted.len(), 1);
    assert_eq!(slot1_radio.transmitted.len(), 0);

    slot0.tick(&mut slot0_radio, 160).unwrap();
    slot1.tick(&mut slot1_radio, 160).unwrap();
    assert_eq!(slot0_radio.transmitted.len(), 1);
    assert_eq!(slot1_radio.transmitted.len(), 1);
}

#[test]
fn broadcast_poll_schedules_four_anchors_without_overflow() {
    let tag_identity = identity(0x1234, [33, 34, 35, 36, 37, 38, 39, 40]);
    let anchors = [
        identity(0x4001, [41, 42, 43, 44, 45, 46, 47, 48]),
        identity(0x4002, [49, 50, 51, 52, 53, 54, 55, 56]),
        identity(0x4003, [57, 58, 59, 60, 61, 62, 63, 64]),
        identity(0x4004, [65, 66, 67, 68, 69, 70, 71, 72]),
    ];
    let mut config = RangingConfig::new(tag_identity);
    config.reply_delay_us = 12_000;

    let mut tag = RangingNode::<4>::new(Role::Tag, config);
    let mut radio = MockRadio::default();

    tag.start(&mut radio, 0).unwrap();
    tag.tick(&mut radio, 80).unwrap();

    for (index, anchor) in anchors.iter().enumerate() {
        let mut init = [0u8; 127];
        let init_len =
            encode_ranging_init(0, anchor.short_address, tag_identity.eui, &mut init).unwrap();
        radio.push_rx(
            &init[..init_len],
            DwTime::from_ticks(20 + index as i64),
            metrics(),
        );
        tag.on_rx(&mut radio, 81 + index as u32, &mut [0u8; 127])
            .unwrap();
    }

    tag.tick(&mut radio, 160).unwrap();
    let Frame::Poll { payload, .. } = parse_frame(radio.last_tx()).unwrap() else {
        panic!("expected poll frame");
    };
    let mut targets = [PollTarget {
        short_address: ShortAddress::new(0),
        reply_delay_us: 0,
    }; 4];
    let count = decode_poll_targets(payload, &mut targets).unwrap();
    assert_eq!(count, 4);
    assert_eq!(
        targets.map(|target| target.reply_delay_us),
        [12_000, 36_000, 60_000, 84_000]
    );
}
