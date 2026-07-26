use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use dw1000_rs::registers::status;
use dw1000_rs::{DelayedTransmit, DwTime, RxOptions, SysStatus, TxOptions};
use mission10_uwb_protocol::air::{self, AIR_FRAME_MAX_NO_FCS, AirEnvelope, AirMessage};
use mission10_uwb_protocol::{
    Destination, EgoState, NodeAddress, REPORT_TURNAROUND_US, ResponderTimestamps, TIMESTAMP_MASK,
    distance_metres, scheduled_tx_matches,
};

use crate::board::RadioHardware;

const MAX_IRQ_WAIT: Duration = Duration::from_millis(100);
const MIN_IRQ_WAIT: Duration = Duration::from_micros(50);
const MAX_STATUS_DRAIN_PASSES: usize = 8;
const RX_EVENT_BITS: u64 = status::RX_TERMINAL_EVENTS.0;
const RX_ERROR_BITS: u64 = status::RX_ERROR_EVENTS.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecoveryAction {
    Rearm,
    ResetReceiverThenRearm,
}

const fn recovery_action(status_value: SysStatus) -> RecoveryAction {
    let no_reset_errors = status::RX_PREAMBLE_TIMEOUT.0 | status::RX_SFD_TIMEOUT.0;
    if status_value.0 & RX_ERROR_BITS & !no_reset_errors == 0 {
        RecoveryAction::Rearm
    } else {
        RecoveryAction::ResetReceiverThenRearm
    }
}

#[derive(Debug, Clone)]
pub struct RangingConfig {
    address: NodeAddress,
    peers: Vec<NodeAddress>,
    poll_period: Duration,
    reply_delay_us: u32,
    response_timeout: Duration,
}

impl RangingConfig {
    pub fn new(
        address: NodeAddress,
        peers: Vec<NodeAddress>,
        poll_period: Duration,
        reply_delay_us: u32,
        response_timeout: Duration,
    ) -> Result<Self> {
        if peers.is_empty() {
            bail!("at least one peer is required");
        }
        if peers.len() > mission10_uwb_protocol::host::MAX_PEERS {
            bail!(
                "at most {} peers are supported",
                mission10_uwb_protocol::host::MAX_PEERS
            );
        }
        if peers.contains(&address) {
            bail!("node {address} cannot range with itself");
        }
        if peers
            .iter()
            .enumerate()
            .any(|(position, peer)| peers[..position].contains(peer))
        {
            bail!("peer addresses must be unique");
        }
        if poll_period.is_zero() {
            bail!("poll period must be positive");
        }
        if reply_delay_us == 0 {
            bail!("reply delay must be positive");
        }
        if response_timeout.is_zero() {
            bail!("response timeout must be positive");
        }
        Ok(Self {
            address,
            peers,
            poll_period,
            reply_delay_us,
            response_timeout,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RangeMeasurement {
    pub receiver: NodeAddress,
    pub source: NodeAddress,
    pub distance_metres: f64,
    pub sequence: u64,
    pub range_event_time_dtu: u64,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct RangingStats {
    pub polls_sent: u64,
    pub ranges: u64,
    pub timeouts: u64,
    pub rx_errors: u64,
    pub invalid_frames: u64,
    pub wrong_peer_frames: u64,
    pub unexpected_frames: u64,
    pub unexpected_tx_events: u64,
    pub scheduled_tx_misses: u64,
    pub explicit_rx_arms: u64,
    pub receiver_resets: u64,
}

#[derive(Debug, Clone, Copy)]
enum ExchangeState {
    Idle,
    SendingPoll {
        peer: NodeAddress,
        exchange_id: u16,
    },
    AwaitResponse {
        peer: NodeAddress,
        exchange_id: u16,
        poll_tx: u64,
    },
    SendingResponse {
        peer: NodeAddress,
        exchange_id: u16,
        poll_rx: u64,
        response_tx: u64,
    },
    AwaitFinal {
        peer: NodeAddress,
        exchange_id: u16,
        poll_rx: u64,
        response_tx: u64,
    },
    SendingFinal {
        peer: NodeAddress,
        exchange_id: u16,
        poll_tx: u64,
        poll_rx: u64,
        response_tx: u64,
        response_rx: u64,
        final_tx: u64,
    },
    AwaitReport {
        peer: NodeAddress,
        exchange_id: u16,
        poll_tx: u64,
        poll_rx: u64,
        response_tx: u64,
        response_rx: u64,
        final_tx: u64,
        final_event_time: u64,
    },
    SendingReport,
}

impl ExchangeState {
    fn is_idle(self) -> bool {
        matches!(self, Self::Idle)
    }
}

pub struct Ranger {
    hardware: RadioHardware,
    config: RangingConfig,
    state: ExchangeState,
    deadline: Option<Instant>,
    next_poll: Instant,
    initiator_peers: Vec<NodeAddress>,
    initiator_cursor: usize,
    next_exchange_id: u16,
    mac_sequence: u8,
    sequence: u64,
    stats: RangingStats,
}

impl Ranger {
    pub fn new(hardware: RadioHardware, config: RangingConfig) -> Self {
        let initiator_peers = config
            .peers
            .iter()
            .copied()
            .filter(|peer| *peer > config.address)
            .collect();
        Self {
            hardware,
            config,
            state: ExchangeState::Idle,
            deadline: None,
            next_poll: Instant::now(),
            initiator_peers,
            initiator_cursor: 0,
            next_exchange_id: 0,
            mac_sequence: 0,
            sequence: 0,
            stats: RangingStats::default(),
        }
    }

    pub fn device_id(&self) -> u32 {
        self.hardware.device_id()
    }

    pub fn peers(&self) -> &[NodeAddress] {
        &self.config.peers
    }

    pub fn run(
        &mut self,
        stop: &AtomicBool,
        duration: Option<Duration>,
        mut on_range: impl FnMut(RangeMeasurement),
    ) -> Result<RangingStats> {
        self.arm_receive(RecoveryAction::Rearm)?;
        let started = Instant::now();
        while !stop.load(Ordering::Relaxed)
            && duration.is_none_or(|duration| started.elapsed() < duration)
        {
            let now = Instant::now();
            let status = self.read_status()?;
            if status.0 & (status::TX_FRAME_SENT.0 | RX_EVENT_BITS) != 0 {
                self.service_status(status, &mut on_range)?;
                continue;
            }
            if self.deadline.is_some_and(|deadline| now >= deadline) {
                self.stats.timeouts += 1;
                self.recover_exchange(RecoveryAction::Rearm)?;
                continue;
            }
            if let Some(peer) = self.peer_to_poll(now) {
                self.send_poll(peer)?;
                continue;
            }

            let wait = self.next_wait(now);
            if !self.hardware.irq.wait(wait)? {
                continue;
            }
            // An edge only promises that status changed. Re-read it on the next
            // pass so every SPI access remains serialized in this thread.
        }
        Ok(self.stats)
    }

    fn peer_to_poll(&mut self, now: Instant) -> Option<NodeAddress> {
        if !self.state.is_idle() || now < self.next_poll || self.initiator_peers.is_empty() {
            return None;
        }
        let peer = self.initiator_peers[self.initiator_cursor];
        self.initiator_cursor = (self.initiator_cursor + 1) % self.initiator_peers.len();
        Some(peer)
    }

    fn next_wait(&self, now: Instant) -> Duration {
        let mut wait = MAX_IRQ_WAIT;
        if let Some(deadline) = self.deadline {
            wait = wait.min(deadline.saturating_duration_since(now));
        }
        if self.state.is_idle() && !self.initiator_peers.is_empty() {
            wait = wait.min(self.next_poll.saturating_duration_since(now));
        }
        wait.max(MIN_IRQ_WAIT)
    }

    fn read_status(&mut self) -> Result<SysStatus> {
        self.hardware
            .radio
            .read_sys_status()
            .map_err(|error| anyhow::anyhow!("read DW1000 status: {error:?}"))
    }

    fn service_status(
        &mut self,
        mut status_value: SysStatus,
        on_range: &mut impl FnMut(RangeMeasurement),
    ) -> Result<()> {
        for _ in 0..MAX_STATUS_DRAIN_PASSES {
            if status_value.contains(status::TX_FRAME_SENT) {
                self.handle_tx_done()?;
                self.clear_events(status::TX_FRAME_SENT)?;
            }

            if status_value.0 & RX_EVENT_BITS != 0 {
                if status_value.0 & RX_ERROR_BITS != 0
                    || !status_value.contains(status::RX_FRAME_READY)
                {
                    self.stats.rx_errors += 1;
                    self.clear_rx_events(status_value)?;
                    self.recover_exchange(recovery_action(status_value))?;
                } else {
                    let mut bytes = [0u8; AIR_FRAME_MAX_NO_FCS];
                    let received = self
                        .hardware
                        .radio
                        .read_frame_raw(&mut bytes)
                        .map_err(|error| anyhow::anyhow!("read DW1000 frame: {error:?}"));
                    self.clear_rx_events(status_value)?;
                    match received {
                        Ok(frame) => {
                            let radio_time = ticks(frame.timestamp);
                            self.handle_frame(frame.bytes, radio_time, on_range)?;
                        }
                        Err(_) => {
                            self.stats.rx_errors += 1;
                            self.recover_exchange(RecoveryAction::ResetReceiverThenRearm)?;
                        }
                    }
                }
            }

            status_value = self.read_status()?;
            if status_value.0 & (status::TX_FRAME_SENT.0 | RX_EVENT_BITS) == 0 {
                return Ok(());
            }
        }
        bail!("DW1000 IRQ status did not drain after {MAX_STATUS_DRAIN_PASSES} passes")
    }

    fn handle_tx_done(&mut self) -> Result<()> {
        match self.state {
            ExchangeState::SendingPoll { peer, exchange_id } => {
                self.state = ExchangeState::AwaitResponse {
                    peer,
                    exchange_id,
                    poll_tx: self.tx_timestamp()?,
                };
            }
            ExchangeState::SendingResponse {
                peer,
                exchange_id,
                poll_rx,
                response_tx,
            } => {
                let actual_response_tx = self.tx_timestamp()?;
                if !scheduled_tx_matches(response_tx, actual_response_tx) {
                    self.stats.scheduled_tx_misses += 1;
                    self.recover_exchange(RecoveryAction::Rearm)?;
                    return Ok(());
                }
                self.state = ExchangeState::AwaitFinal {
                    peer,
                    exchange_id,
                    poll_rx,
                    response_tx,
                };
            }
            ExchangeState::SendingFinal {
                peer,
                exchange_id,
                poll_tx,
                poll_rx,
                response_tx,
                response_rx,
                final_tx,
            } => {
                let actual_final_tx = self.tx_timestamp()?;
                if !scheduled_tx_matches(final_tx, actual_final_tx) {
                    self.stats.scheduled_tx_misses += 1;
                    self.recover_exchange(RecoveryAction::Rearm)?;
                    return Ok(());
                }
                self.state = ExchangeState::AwaitReport {
                    peer,
                    exchange_id,
                    poll_tx,
                    poll_rx,
                    response_tx,
                    response_rx,
                    final_tx,
                    final_event_time: actual_final_tx,
                };
                // Report has an explicit turnaround. Re-arm here so its RX
                // attempt starts from cleared status and synchronized buffers.
                self.arm_receive(RecoveryAction::Rearm)?;
            }
            ExchangeState::SendingReport => {
                self.state = ExchangeState::Idle;
                self.deadline = None;
                self.arm_receive(RecoveryAction::Rearm)?;
            }
            ExchangeState::Idle
            | ExchangeState::AwaitResponse { .. }
            | ExchangeState::AwaitFinal { .. }
            | ExchangeState::AwaitReport { .. } => {
                self.stats.unexpected_tx_events += 1;
            }
        }
        Ok(())
    }

    fn tx_timestamp(&mut self) -> Result<u64> {
        self.hardware
            .radio
            .read_timestamps()
            .map(|timestamps| ticks(timestamps.tx))
            .map_err(|error| anyhow::anyhow!("read DW1000 TX timestamp: {error:?}"))
    }

    fn handle_frame(
        &mut self,
        bytes: &[u8],
        rx_time: u64,
        on_range: &mut impl FnMut(RangeMeasurement),
    ) -> Result<()> {
        let frame = match air::decode(bytes, self.config.address) {
            Ok(frame) => frame,
            Err(_) => {
                self.stats.invalid_frames += 1;
                self.arm_receive(RecoveryAction::Rearm)?;
                return Ok(());
            }
        };
        let Some(source) = self.peer_for_address(frame.source) else {
            self.stats.wrong_peer_frames += 1;
            self.arm_receive(RecoveryAction::Rearm)?;
            return Ok(());
        };

        match (self.state, frame.envelope.message) {
            (ExchangeState::Idle, AirMessage::Poll { .. }) => {
                self.send_response(source, frame.envelope.exchange_id, rx_time)?;
            }
            (
                ExchangeState::AwaitResponse {
                    peer,
                    exchange_id,
                    poll_tx,
                },
                AirMessage::Response {
                    poll_rx,
                    response_tx,
                    ..
                },
            ) if peer == source && exchange_id == frame.envelope.exchange_id => {
                self.send_final(source, exchange_id, poll_tx, poll_rx, response_tx, rx_time)?;
            }
            (
                ExchangeState::AwaitFinal {
                    peer,
                    exchange_id,
                    poll_rx,
                    response_tx,
                },
                AirMessage::Final {
                    poll_tx,
                    response_rx,
                    final_tx,
                },
            ) if peer == source && exchange_id == frame.envelope.exchange_id => {
                let Some(distance) = distance_metres(
                    [poll_tx, response_rx, final_tx],
                    ResponderTimestamps {
                        poll_rx,
                        poll_ack_tx: response_tx,
                        range_rx: rx_time,
                    },
                ) else {
                    self.stats.invalid_frames += 1;
                    self.recover_exchange(RecoveryAction::Rearm)?;
                    return Ok(());
                };
                self.send_report(source, exchange_id, rx_time)?;
                self.emit(source, distance, rx_time, on_range);
            }
            (
                ExchangeState::AwaitReport {
                    peer,
                    exchange_id,
                    poll_tx,
                    poll_rx,
                    response_tx,
                    response_rx,
                    final_tx,
                    final_event_time,
                },
                AirMessage::Report {
                    final_rx,
                    status: 0,
                },
            ) if peer == source && exchange_id == frame.envelope.exchange_id => {
                let Some(distance) = distance_metres(
                    [poll_tx, response_rx, final_tx],
                    ResponderTimestamps {
                        poll_rx,
                        poll_ack_tx: response_tx,
                        range_rx: final_rx,
                    },
                ) else {
                    self.stats.invalid_frames += 1;
                    self.recover_exchange(RecoveryAction::Rearm)?;
                    return Ok(());
                };
                self.state = ExchangeState::Idle;
                self.deadline = None;
                self.arm_receive(RecoveryAction::Rearm)?;
                self.emit(source, distance, final_event_time, on_range);
            }
            _ => {
                self.stats.unexpected_frames += 1;
                self.arm_receive(RecoveryAction::Rearm)?;
            }
        }
        Ok(())
    }

    fn send_poll(&mut self, peer: NodeAddress) -> Result<()> {
        let exchange_id = self.next_exchange_id;
        self.next_exchange_id = self.next_exchange_id.wrapping_add(1);
        let frame = self.encode(
            peer,
            AirEnvelope::new(
                exchange_id,
                AirMessage::Poll {
                    state: EgoState::default(),
                },
            ),
        )?;
        self.hardware
            .radio
            .transmit(
                frame.bytes(),
                TxOptions {
                    delayed_time: None,
                    wait_for_response: true,
                },
            )
            .map_err(|error| anyhow::anyhow!("send POLL: {error:?}"))?;
        self.state = ExchangeState::SendingPoll { peer, exchange_id };
        self.deadline = Some(Instant::now() + self.config.response_timeout);
        self.next_poll += self.config.poll_period;
        let now = Instant::now();
        if self.next_poll < now {
            self.next_poll = now;
        }
        self.stats.polls_sent += 1;
        Ok(())
    }

    fn send_response(&mut self, peer: NodeAddress, exchange_id: u16, poll_rx: u64) -> Result<()> {
        let planned = self.planned_tx()?;
        let response_tx = ticks(planned.timestamp().value());
        let frame = self.encode(
            peer,
            AirEnvelope::new(
                exchange_id,
                AirMessage::Response {
                    poll_rx,
                    response_tx,
                    state: EgoState::default(),
                },
            ),
        )?;
        self.hardware
            .radio
            .transmit_at(frame.bytes(), planned, true)
            .map_err(|error| anyhow::anyhow!("send delayed RESPONSE: {error:?}"))?;
        self.state = ExchangeState::SendingResponse {
            peer,
            exchange_id,
            poll_rx,
            response_tx,
        };
        self.deadline = Some(Instant::now() + self.config.response_timeout);
        Ok(())
    }

    fn send_final(
        &mut self,
        peer: NodeAddress,
        exchange_id: u16,
        poll_tx: u64,
        poll_rx: u64,
        response_tx: u64,
        response_rx: u64,
    ) -> Result<()> {
        let planned = self.planned_tx()?;
        let final_tx = ticks(planned.timestamp().value());
        let frame = self.encode(
            peer,
            AirEnvelope::new(
                exchange_id,
                AirMessage::Final {
                    poll_tx,
                    response_rx,
                    final_tx,
                },
            ),
        )?;
        self.hardware
            .radio
            .transmit_at(frame.bytes(), planned, false)
            .map_err(|error| anyhow::anyhow!("send delayed FINAL: {error:?}"))?;
        self.state = ExchangeState::SendingFinal {
            peer,
            exchange_id,
            poll_tx,
            poll_rx,
            response_tx,
            response_rx,
            final_tx,
        };
        self.deadline = Some(Instant::now() + self.config.response_timeout);
        Ok(())
    }

    fn send_report(&mut self, peer: NodeAddress, exchange_id: u16, final_rx: u64) -> Result<()> {
        let planned = self
            .hardware
            .radio
            .schedule_delayed_transmit(DwTime::from_micros(REPORT_TURNAROUND_US as f32))
            .map_err(|error| anyhow::anyhow!("compute REPORT timestamp: {error:?}"))?;
        let frame = self.encode(
            peer,
            AirEnvelope::new(
                exchange_id,
                AirMessage::Report {
                    final_rx,
                    status: 0,
                },
            ),
        )?;
        self.hardware
            .radio
            .transmit_at(frame.bytes(), planned, false)
            .map_err(|error| anyhow::anyhow!("send delayed REPORT: {error:?}"))?;
        self.state = ExchangeState::SendingReport;
        self.deadline = Some(Instant::now() + self.config.response_timeout);
        Ok(())
    }

    fn encode(&mut self, peer: NodeAddress, envelope: AirEnvelope) -> Result<air::EncodedAirFrame> {
        let sequence = self.mac_sequence;
        self.mac_sequence = self.mac_sequence.wrapping_add(1);
        air::encode(
            self.config.address,
            Destination::Node(peer),
            sequence,
            &envelope,
        )
        .map_err(|error| anyhow::anyhow!("encode native air frame: {error:?}"))
    }

    fn planned_tx(&mut self) -> Result<DelayedTransmit> {
        self.hardware
            .radio
            .schedule_delayed_transmit(DwTime::from_micros(self.config.reply_delay_us as f32))
            .map_err(|error| anyhow::anyhow!("compute delayed TX timestamp: {error:?}"))
    }

    fn recover_exchange(&mut self, action: RecoveryAction) -> Result<()> {
        self.state = ExchangeState::Idle;
        self.deadline = None;
        self.arm_receive(action)
    }

    fn arm_receive(&mut self, action: RecoveryAction) -> Result<()> {
        if action == RecoveryAction::ResetReceiverThenRearm {
            self.hardware
                .radio
                .reset_receiver()
                .map_err(|error| anyhow::anyhow!("reset DW1000 receiver: {error:?}"))?;
            self.stats.receiver_resets += 1;
        }
        self.hardware
            .radio
            .start_receive(RxOptions {
                delayed_time: None,
                permanent: false,
            })
            .map_err(|error| anyhow::anyhow!("arm DW1000 receiver: {error:?}"))?;
        self.stats.explicit_rx_arms += 1;
        Ok(())
    }

    fn clear_events(&mut self, events: SysStatus) -> Result<()> {
        self.hardware
            .radio
            .clear_events(events)
            .map_err(|error| anyhow::anyhow!("clear DW1000 status 0x{:x}: {error:?}", events.0))
    }

    fn clear_rx_events(&mut self, status_value: SysStatus) -> Result<()> {
        self.clear_events(SysStatus(status_value.0 & status::RX_CLEAR_EVENTS.0))
    }

    fn peer_for_address(&self, address: NodeAddress) -> Option<NodeAddress> {
        self.config
            .peers
            .iter()
            .copied()
            .find(|peer| *peer == address)
    }

    fn emit(
        &mut self,
        source: NodeAddress,
        distance: f64,
        range_event_time_dtu: u64,
        on_range: &mut impl FnMut(RangeMeasurement),
    ) {
        let measurement = RangeMeasurement {
            receiver: self.config.address,
            source,
            distance_metres: distance,
            sequence: self.sequence,
            range_event_time_dtu,
        };
        self.sequence += 1;
        self.stats.ranges += 1;
        on_range(measurement);
    }
}

fn ticks(time: DwTime) -> u64 {
    (time.get_timestamp() as u64) & TIMESTAMP_MASK
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranging_config_accepts_multiple_addressed_initiator_peers() {
        let node0 = NodeAddress::new(0).unwrap();
        let node1 = NodeAddress::new(1).unwrap();
        let node2 = NodeAddress::new(2).unwrap();
        let timing = Duration::from_millis(10);

        assert!(RangingConfig::new(node0, vec![], timing, 2_000, timing).is_err());
        assert!(RangingConfig::new(node0, vec![node0], timing, 2_000, timing).is_err());
        assert!(RangingConfig::new(node0, vec![node1, node1], timing, 2_000, timing).is_err());
        assert!(RangingConfig::new(node0, vec![node1, node2], timing, 2_000, timing).is_ok());
        assert!(
            RangingConfig::new(
                node0,
                vec![
                    node1,
                    node2,
                    NodeAddress::new(3).unwrap(),
                    NodeAddress::new(4).unwrap(),
                ],
                timing,
                2_000,
                timing,
            )
            .is_err()
        );
    }

    #[test]
    fn ranging_config_rejects_zero_timing() {
        let node0 = NodeAddress::new(0).unwrap();
        let node1 = NodeAddress::new(1).unwrap();
        let timing = Duration::from_millis(10);

        assert!(RangingConfig::new(node0, vec![node1], Duration::ZERO, 2_000, timing).is_err());
        assert!(RangingConfig::new(node0, vec![node1], timing, 0, timing).is_err());
        assert!(RangingConfig::new(node0, vec![node1], timing, 2_000, Duration::ZERO).is_err());
    }

    #[test]
    fn receiver_recovery_is_derived_from_terminal_status() {
        assert_eq!(
            recovery_action(status::RX_PREAMBLE_TIMEOUT),
            RecoveryAction::Rearm
        );
        assert_eq!(
            recovery_action(status::RX_SFD_TIMEOUT),
            RecoveryAction::Rearm
        );
        for status_value in [
            status::RX_HEADER_ERROR,
            status::RX_FRAME_CHECK_ERROR,
            status::RX_REED_SOLOMON_ERROR,
            status::RX_TIMEOUT,
            status::LDE_ERROR,
            status::RX_OVERRUN,
            status::FRAME_FILTER_REJECTION,
        ] {
            assert_eq!(
                recovery_action(status_value),
                RecoveryAction::ResetReceiverThenRearm
            );
        }
        assert_eq!(
            recovery_action(status::RX_SFD_TIMEOUT | status::RX_FRAME_CHECK_ERROR),
            RecoveryAction::ResetReceiverThenRearm
        );
    }

    #[test]
    fn receive_event_masks_cover_irq_terminal_and_clear_latches() {
        for event in [
            status::RX_FRAME_READY,
            status::RX_PREAMBLE_TIMEOUT,
            status::RX_SFD_TIMEOUT,
            status::RX_OVERRUN,
        ] {
            assert_ne!(RX_EVENT_BITS & event.0, 0);
        }

        for latch in [
            status::RX_PREAMBLE_DETECTED,
            status::RX_SFD_DETECTED,
            status::LDE_DONE,
            status::RX_HEADER_DETECTED,
            status::RX_FRAME_GOOD,
        ] {
            assert_ne!(status::RX_CLEAR_EVENTS.0 & latch.0, 0);
        }
        assert_eq!(RX_EVENT_BITS & status::LDE_DONE.0, 0);
    }
}
