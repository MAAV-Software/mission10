use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use dw1000_rs::registers::status;
use dw1000_rs::{DwTime, RxOptions, SysStatus, TxOptions};
use mission10_uwb_protocol::dw1000_bench::{FRAME_LEN, Frame, MessageKind};
use mission10_uwb_protocol::{ResponderTimestamps, TIMESTAMP_MASK, distance_metres};

use crate::board::RadioHardware;
use crate::{NodeIndex, address_for};

const MAX_IRQ_WAIT: Duration = Duration::from_millis(100);
const MIN_IRQ_WAIT: Duration = Duration::from_micros(50);
const MAX_STATUS_DRAIN_PASSES: usize = 8;
// DX_TIME has 512-DTU scheduling granularity.
const MAX_SCHEDULED_TX_ERROR_DTU: u64 = 512;

const RX_EVENT_BITS: u64 = status::RX_FRAME_READY.0
    | status::RX_FRAME_GOOD.0
    | status::RX_FRAME_CHECK_ERROR.0
    | status::RX_REED_SOLOMON_ERROR.0
    | status::RX_TIMEOUT.0
    | status::LDE_DONE.0
    | status::LDE_ERROR.0
    | status::RX_HEADER_ERROR.0;
const RX_ERROR_BITS: u64 = status::RX_FRAME_CHECK_ERROR.0
    | status::RX_REED_SOLOMON_ERROR.0
    | status::RX_TIMEOUT.0
    | status::LDE_ERROR.0
    | status::RX_HEADER_ERROR.0;

#[derive(Debug, Clone)]
pub struct RangingConfig {
    index: NodeIndex,
    peers: Vec<NodeIndex>,
    poll_period: Duration,
    reply_delay_us: u32,
    response_timeout: Duration,
}

impl RangingConfig {
    pub fn new(
        index: NodeIndex,
        peers: Vec<NodeIndex>,
        poll_period: Duration,
        reply_delay_us: u32,
        response_timeout: Duration,
    ) -> Result<Self> {
        if peers.is_empty() {
            bail!("at least one peer is required");
        }
        if peers.contains(&index) {
            bail!("node {index} cannot range with itself");
        }
        if peers
            .iter()
            .enumerate()
            .any(|(position, peer)| peers[..position].contains(peer))
        {
            bail!("peer indices must be unique");
        }
        if peers.iter().filter(|peer| **peer > index).count() > 1 {
            bail!("the destination-free bench frame permits only one initiated peer");
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
            index,
            peers,
            poll_period,
            reply_delay_us,
            response_timeout,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RangeMeasurement {
    pub receiver: NodeIndex,
    pub source: NodeIndex,
    pub distance_metres: f64,
    pub sequence: u64,
    pub completed_at_dtu: u64,
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
}

#[derive(Debug, Clone, Copy)]
enum ExchangeState {
    Idle,
    SendingPoll {
        peer: NodeIndex,
    },
    AwaitPollAck {
        peer: NodeIndex,
        poll_tx: u64,
    },
    SendingPollAck {
        peer: NodeIndex,
        poll_rx: u64,
    },
    AwaitRange {
        peer: NodeIndex,
        poll_rx: u64,
        poll_ack_tx: u64,
    },
    SendingRange {
        peer: NodeIndex,
        range_tx: u64,
    },
    AwaitRangeReport {
        peer: NodeIndex,
    },
    SendingRangeReport,
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
    initiator_peer: Option<NodeIndex>,
    sequence: u64,
    stats: RangingStats,
}

impl Ranger {
    pub fn new(hardware: RadioHardware, config: RangingConfig) -> Self {
        let initiator_peer = config
            .peers
            .iter()
            .copied()
            .find(|peer| *peer > config.index);
        Self {
            hardware,
            config,
            state: ExchangeState::Idle,
            deadline: None,
            next_poll: Instant::now(),
            initiator_peer,
            sequence: 0,
            stats: RangingStats::default(),
        }
    }

    pub fn device_id(&self) -> u32 {
        self.hardware.device_id()
    }

    pub fn peers(&self) -> &[NodeIndex] {
        &self.config.peers
    }

    pub fn run(
        &mut self,
        stop: &AtomicBool,
        duration: Option<Duration>,
        mut on_range: impl FnMut(RangeMeasurement),
    ) -> Result<RangingStats> {
        self.arm_receive()?;
        let started = Instant::now();
        while !stop.load(Ordering::Relaxed)
            && duration.is_none_or(|duration| started.elapsed() < duration)
        {
            let now = Instant::now();
            if self.deadline.is_some_and(|deadline| now >= deadline) {
                self.stats.timeouts += 1;
                self.recover_exchange()?;
                continue;
            }
            if let Some(peer) = self.peer_to_poll(now) {
                self.send_poll(peer)?;
                continue;
            }

            let status = self.read_status()?;
            if status.0 & (status::TX_FRAME_SENT.0 | RX_EVENT_BITS) != 0 {
                self.service_status(status, &mut on_range)?;
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

    fn peer_to_poll(&self, now: Instant) -> Option<NodeIndex> {
        if self.state.is_idle() && now >= self.next_poll {
            self.initiator_peer
        } else {
            None
        }
    }

    fn next_wait(&self, now: Instant) -> Duration {
        let mut wait = MAX_IRQ_WAIT;
        if let Some(deadline) = self.deadline {
            wait = wait.min(deadline.saturating_duration_since(now));
        }
        if self.state.is_idle() && self.initiator_peer.is_some() {
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
                    self.clear_events(SysStatus(status_value.0 & RX_EVENT_BITS))?;
                    self.recover_exchange()?;
                } else {
                    let mut bytes = [0u8; FRAME_LEN];
                    let received = self
                        .hardware
                        .radio
                        .read_frame_raw(&mut bytes)
                        .map_err(|error| anyhow::anyhow!("read DW1000 frame: {error:?}"));
                    self.clear_events(SysStatus(status_value.0 & RX_EVENT_BITS))?;
                    match received {
                        Ok(frame) => {
                            let len = frame.bytes.len();
                            let radio_time = ticks(frame.timestamp);
                            if len != FRAME_LEN {
                                self.stats.invalid_frames += 1;
                                self.arm_receive()?;
                            } else {
                                self.handle_frame(&bytes, radio_time, on_range)?;
                            }
                        }
                        Err(_) => {
                            self.stats.rx_errors += 1;
                            self.recover_exchange()?;
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
            ExchangeState::SendingPoll { peer } => {
                self.state = ExchangeState::AwaitPollAck {
                    peer,
                    poll_tx: self.tx_timestamp()?,
                };
            }
            ExchangeState::SendingPollAck { peer, poll_rx } => {
                self.state = ExchangeState::AwaitRange {
                    peer,
                    poll_rx,
                    poll_ack_tx: self.tx_timestamp()?,
                };
            }
            ExchangeState::SendingRange { peer, range_tx } => {
                if timestamp_error(range_tx, self.tx_timestamp()?) > MAX_SCHEDULED_TX_ERROR_DTU {
                    self.stats.scheduled_tx_misses += 1;
                }
                self.state = ExchangeState::AwaitRangeReport { peer };
            }
            ExchangeState::SendingRangeReport => {
                self.state = ExchangeState::Idle;
                self.deadline = None;
                self.arm_receive()?;
            }
            ExchangeState::Idle
            | ExchangeState::AwaitPollAck { .. }
            | ExchangeState::AwaitRange { .. }
            | ExchangeState::AwaitRangeReport { .. } => {
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
        bytes: &[u8; FRAME_LEN],
        rx_time: u64,
        on_range: &mut impl FnMut(RangeMeasurement),
    ) -> Result<()> {
        let frame = match Frame::decode(bytes) {
            Ok(frame) => frame,
            Err(_) => {
                self.stats.invalid_frames += 1;
                self.arm_receive()?;
                return Ok(());
            }
        };
        let Some(source) = self.peer_for_address(frame.source()) else {
            self.stats.wrong_peer_frames += 1;
            self.arm_receive()?;
            return Ok(());
        };

        match (self.state, frame.kind()) {
            (ExchangeState::Idle, MessageKind::Poll) => {
                self.send_poll_ack(source, rx_time)?;
            }
            (ExchangeState::AwaitPollAck { peer, poll_tx }, MessageKind::PollAck)
                if peer == source =>
            {
                self.send_range(source, poll_tx, rx_time)?;
            }
            (
                ExchangeState::AwaitRange {
                    peer,
                    poll_rx,
                    poll_ack_tx,
                },
                MessageKind::Range,
            ) if peer == source => {
                let Some(initiator) = frame.timestamps() else {
                    self.stats.invalid_frames += 1;
                    self.recover_exchange()?;
                    return Ok(());
                };
                let Some(distance) = distance_metres(
                    initiator,
                    ResponderTimestamps {
                        poll_rx,
                        poll_ack_tx,
                        range_rx: rx_time,
                    },
                ) else {
                    self.stats.invalid_frames += 1;
                    self.recover_exchange()?;
                    return Ok(());
                };
                self.send_range_report(distance)?;
                self.emit(source, distance, rx_time, on_range);
            }
            (ExchangeState::AwaitRangeReport { peer }, MessageKind::RangeReport)
                if peer == source =>
            {
                let Some(distance_mm) = frame.distance_mm() else {
                    self.stats.invalid_frames += 1;
                    self.recover_exchange()?;
                    return Ok(());
                };
                self.state = ExchangeState::Idle;
                self.deadline = None;
                self.arm_receive()?;
                self.emit(source, f64::from(distance_mm) / 1000.0, rx_time, on_range);
            }
            _ => {
                self.stats.unexpected_frames += 1;
                self.arm_receive()?;
            }
        }
        Ok(())
    }

    fn send_poll(&mut self, peer: NodeIndex) -> Result<()> {
        let frame = Frame::poll(address_for(self.config.index));
        self.hardware
            .radio
            .transmit(
                frame.as_bytes(),
                TxOptions {
                    delayed_time: None,
                    wait_for_response: true,
                },
            )
            .map_err(|error| anyhow::anyhow!("send POLL: {error:?}"))?;
        self.state = ExchangeState::SendingPoll { peer };
        self.deadline = Some(Instant::now() + self.config.response_timeout);
        self.next_poll += self.config.poll_period;
        let now = Instant::now();
        if self.next_poll < now {
            self.next_poll = now;
        }
        self.stats.polls_sent += 1;
        Ok(())
    }

    fn send_poll_ack(&mut self, peer: NodeIndex, poll_rx: u64) -> Result<()> {
        let planned = self.planned_tx()?;
        let frame = Frame::poll_ack(address_for(self.config.index));
        self.hardware
            .radio
            .transmit_at(frame.as_bytes(), planned, true)
            .map_err(|error| anyhow::anyhow!("send delayed POLL_ACK: {error:?}"))?;
        self.state = ExchangeState::SendingPollAck { peer, poll_rx };
        self.deadline = Some(Instant::now() + self.config.response_timeout);
        Ok(())
    }

    fn send_range(&mut self, peer: NodeIndex, poll_tx: u64, poll_ack_rx: u64) -> Result<()> {
        let planned = self.planned_tx()?;
        let range_tx = ticks(planned);
        let frame = Frame::range(
            address_for(self.config.index),
            poll_tx,
            poll_ack_rx,
            range_tx,
        );
        self.hardware
            .radio
            .transmit_at(frame.as_bytes(), planned, true)
            .map_err(|error| anyhow::anyhow!("send delayed RANGE: {error:?}"))?;
        self.state = ExchangeState::SendingRange { peer, range_tx };
        self.deadline = Some(Instant::now() + self.config.response_timeout);
        Ok(())
    }

    fn send_range_report(&mut self, distance: f64) -> Result<()> {
        let millimetres = (distance * 1000.0).round().clamp(0.0, u32::MAX as f64) as u32;
        let frame = Frame::range_report(address_for(self.config.index), millimetres);
        self.hardware
            .radio
            .transmit(frame.as_bytes(), TxOptions::default())
            .map_err(|error| anyhow::anyhow!("send RANGE_REPORT: {error:?}"))?;
        self.state = ExchangeState::SendingRangeReport;
        self.deadline = Some(Instant::now() + self.config.response_timeout);
        Ok(())
    }

    fn planned_tx(&mut self) -> Result<DwTime> {
        self.hardware
            .radio
            .compute_delayed_time(DwTime::from_micros(self.config.reply_delay_us as f32))
            .map_err(|error| anyhow::anyhow!("compute delayed TX timestamp: {error:?}"))
    }

    fn recover_exchange(&mut self) -> Result<()> {
        self.state = ExchangeState::Idle;
        self.deadline = None;
        self.arm_receive()
    }

    fn arm_receive(&mut self) -> Result<()> {
        self.hardware
            .radio
            .start_receive(RxOptions {
                delayed_time: None,
                permanent: false,
            })
            .map_err(|error| anyhow::anyhow!("arm DW1000 receiver: {error:?}"))
    }

    fn clear_events(&mut self, events: SysStatus) -> Result<()> {
        self.hardware
            .radio
            .clear_events(events)
            .map_err(|error| anyhow::anyhow!("clear DW1000 status 0x{:x}: {error:?}", events.0))
    }

    fn peer_for_address(&self, address: [u8; 2]) -> Option<NodeIndex> {
        self.config
            .peers
            .iter()
            .copied()
            .find(|peer| address_for(*peer) == address)
    }

    fn emit(
        &mut self,
        source: NodeIndex,
        distance: f64,
        completed_at_dtu: u64,
        on_range: &mut impl FnMut(RangeMeasurement),
    ) {
        let measurement = RangeMeasurement {
            receiver: self.config.index,
            source,
            distance_metres: distance,
            sequence: self.sequence,
            completed_at_dtu,
        };
        self.sequence += 1;
        self.stats.ranges += 1;
        on_range(measurement);
    }
}

fn ticks(time: DwTime) -> u64 {
    (time.get_timestamp() as u64) & TIMESTAMP_MASK
}

fn timestamp_error(expected: u64, actual: u64) -> u64 {
    let forward = actual.wrapping_sub(expected) & TIMESTAMP_MASK;
    let backward = expected.wrapping_sub(actual) & TIMESTAMP_MASK;
    forward.min(backward)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delayed_timestamp_error_wraps_and_chooses_the_short_path() {
        assert_eq!(timestamp_error(TIMESTAMP_MASK - 10, 9), 20);
        assert_eq!(timestamp_error(1_000, 1_512), 512);
    }

    #[test]
    fn ranging_config_rejects_ambiguous_peer_sets() {
        let node0 = NodeIndex::new(0).unwrap();
        let node1 = NodeIndex::new(1).unwrap();
        let timing = Duration::from_millis(10);

        assert!(RangingConfig::new(node0, vec![], timing, 2_000, timing).is_err());
        assert!(RangingConfig::new(node0, vec![node0], timing, 2_000, timing).is_err());
        assert!(RangingConfig::new(node0, vec![node1, node1], timing, 2_000, timing).is_err());
        assert!(
            RangingConfig::new(
                node0,
                vec![node1, NodeIndex::new(2).unwrap()],
                timing,
                2_000,
                timing,
            )
            .is_err()
        );
    }

    #[test]
    fn ranging_config_rejects_zero_timing() {
        let node0 = NodeIndex::new(0).unwrap();
        let node1 = NodeIndex::new(1).unwrap();
        let timing = Duration::from_millis(10);

        assert!(RangingConfig::new(node0, vec![node1], Duration::ZERO, 2_000, timing).is_err());
        assert!(RangingConfig::new(node0, vec![node1], timing, 0, timing).is_err());
        assert!(RangingConfig::new(node0, vec![node1], timing, 2_000, Duration::ZERO).is_err());
    }
}
