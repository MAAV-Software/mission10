use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Result, bail};
use dw1000_rs::registers::status;
use dw1000_rs::{DelayedTransmit, DwTime, DxTimeDeadline, OnAirTimestamp, RxOptions, SysStatus};
use mission10_uwb_protocol::air::{
    self, AIR_FRAME_MAX_NO_FCS, AirEnvelope, AirMessage, REPORT_STATUS_CLOSE,
};
use mission10_uwb_protocol::scheduler::{
    ExchangeId, FlightRoster, MAX_PAIRS, Pair, PairMacState, PairRole, classify_range, due_pair,
    next_due_us, pair_index,
};
use mission10_uwb_protocol::{
    DTU_PER_US, Destination, EgoState, NodeAddress, REPORT_TURNAROUND_US, ResponderTimestamps,
    TIMESTAMP_MASK, delayed_tx_time, distance_metres, scheduled_tx_matches, wrapping_delta,
};

use crate::DEFAULT_ANTENNA_DELAY;
use crate::board::RadioHardware;

const MAX_IRQ_WAIT: Duration = Duration::from_millis(100);
const MIN_IRQ_WAIT: Duration = Duration::from_micros(50);
const MAX_STATUS_DRAIN_PASSES: usize = 8;
const POLL_PROGRAM_LEAD_US: u64 = 1_000;
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
    reply_delay_us: u32,
    response_timeout: Duration,
}

impl RangingConfig {
    pub fn new(
        address: NodeAddress,
        peers: Vec<NodeAddress>,
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
        if reply_delay_us == 0 {
            bail!("reply delay must be positive");
        }
        if response_timeout.is_zero() {
            bail!("response timeout must be positive");
        }
        if response_timeout > Duration::from_millis(10) {
            bail!("response timeout must not exceed 10 ms");
        }
        FlightRoster::new(address, &peers)
            .ok_or_else(|| anyhow::anyhow!("addresses do not form a valid fixed roster"))?;
        Ok(Self {
            address,
            peers,
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
    pub mission_event_time_us: Option<u64>,
    pub mission_generation: Option<u32>,
    pub mission_time_error_us: u32,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct RangingStats {
    pub radio_reinitializations: u64,
    pub polls_sent: u64,
    pub polls_received: u64,
    pub responses_sent: u64,
    pub responses_received: u64,
    pub finals_sent: u64,
    pub finals_received: u64,
    pub reports_sent: u64,
    pub reports_received: u64,
    pub exchange_completions: u64,
    pub contention_backoffs: u64,
    pub ranges: u64,
    pub timeouts: u64,
    pub tx_completion_timeouts: u64,
    pub rx_completion_timeouts: u64,
    pub rx_errors: u64,
    pub invalid_frames: u64,
    pub wrong_peer_frames: u64,
    pub unexpected_frames: u64,
    pub unexpected_tx_events: u64,
    pub scheduled_tx_misses: u64,
    pub explicit_rx_arms: u64,
    pub receiver_resets: u64,
}

impl RangingStats {
    pub fn merge(&mut self, other: Self) {
        self.radio_reinitializations += other.radio_reinitializations;
        self.polls_sent += other.polls_sent;
        self.polls_received += other.polls_received;
        self.responses_sent += other.responses_sent;
        self.responses_received += other.responses_received;
        self.finals_sent += other.finals_sent;
        self.finals_received += other.finals_received;
        self.reports_sent += other.reports_sent;
        self.reports_received += other.reports_received;
        self.exchange_completions += other.exchange_completions;
        self.contention_backoffs += other.contention_backoffs;
        self.ranges += other.ranges;
        self.timeouts += other.timeouts;
        self.tx_completion_timeouts += other.tx_completion_timeouts;
        self.rx_completion_timeouts += other.rx_completion_timeouts;
        self.rx_errors += other.rx_errors;
        self.invalid_frames += other.invalid_frames;
        self.wrong_peer_frames += other.wrong_peer_frames;
        self.unexpected_frames += other.unexpected_frames;
        self.unexpected_tx_events += other.unexpected_tx_events;
        self.scheduled_tx_misses += other.scheduled_tx_misses;
        self.explicit_rx_arms += other.explicit_rx_arms;
        self.receiver_resets += other.receiver_resets;
    }
}

#[derive(Debug, Clone, Copy)]
enum ExchangeState {
    Idle,
    SendingPoll {
        peer: NodeAddress,
        exchange_id: ExchangeId,
        poll_tx: u64,
    },
    AwaitResponse {
        peer: NodeAddress,
        exchange_id: ExchangeId,
        poll_tx: u64,
    },
    SendingResponse {
        peer: NodeAddress,
        exchange_id: ExchangeId,
        poll_rx: u64,
        response_tx: u64,
    },
    AwaitFinal {
        peer: NodeAddress,
        exchange_id: ExchangeId,
        poll_rx: u64,
        response_tx: u64,
    },
    SendingFinal {
        peer: NodeAddress,
        exchange_id: ExchangeId,
        poll_tx: u64,
        poll_rx: u64,
        response_tx: u64,
        response_rx: u64,
        final_tx: u64,
    },
    AwaitReport {
        peer: NodeAddress,
        exchange_id: ExchangeId,
        poll_tx: u64,
        poll_rx: u64,
        response_tx: u64,
        response_rx: u64,
        final_tx: u64,
        final_event_time: u64,
    },
    SendingReport {
        peer: NodeAddress,
        exchange_id: ExchangeId,
    },
}

impl ExchangeState {
    fn is_idle(self) -> bool {
        matches!(self, Self::Idle)
    }
}

pub struct Ranger {
    hardware: RadioHardware,
    config: RangingConfig,
    roster: FlightRoster,
    pair_states: [PairMacState; MAX_PAIRS],
    state: ExchangeState,
    deadline: Option<Instant>,
    active_pair: Option<Pair>,
    started_at: Instant,
    next_exchange_id: u16,
    mission_generation: u32,
    last_received_exchange:
        [Option<(ExchangeId, Instant)>; mission10_uwb_protocol::host::MAX_PEERS],
    mac_sequence: u8,
    sequence: u64,
    stats: RangingStats,
}

impl Ranger {
    pub fn new(hardware: RadioHardware, config: RangingConfig) -> Self {
        let roster = FlightRoster::new(config.address, &config.peers)
            .expect("validated ranging configuration must form a roster");
        let boot_nonce = std::process::id()
            ^ unix_time_us().unwrap_or_default() as u32
            ^ u32::from(config.address.get()).rotate_left(16);
        let mut pair_states = [PairMacState::new(boot_nonce); MAX_PAIRS];
        for (index, pair) in roster.pairs().enumerate() {
            pair_states[index].initialize(0, pair);
        }
        Self {
            hardware,
            config,
            roster,
            pair_states,
            state: ExchangeState::Idle,
            deadline: None,
            active_pair: None,
            started_at: Instant::now(),
            next_exchange_id: 0,
            mission_generation: boot_nonce,
            last_received_exchange: [None; mission10_uwb_protocol::host::MAX_PEERS],
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

    pub fn stats(&self) -> RangingStats {
        self.stats
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
                self.record_deadline_phase();
                self.recover_exchange(RecoveryAction::Rearm)?;
                continue;
            }
            if self.service_schedule()? {
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

    fn service_schedule(&mut self) -> Result<bool> {
        if !self.state.is_idle() {
            return Ok(false);
        }
        let now_us = self.monotonic_us();
        for state in &mut self.pair_states {
            state.expire(now_us);
        }
        let Some(pair) = due_pair(self.roster, self.config.address, now_us, &self.pair_states)
        else {
            return Ok(false);
        };
        let exchange_id = ExchangeId::new(self.next_exchange_id);
        self.next_exchange_id = self.next_exchange_id.wrapping_add(1);
        let index = pair_index(self.roster, pair).expect("due pair has scheduler state");
        self.pair_states[index].begin_attempt(now_us);
        self.active_pair = Some(pair);
        self.send_poll(pair.responder, exchange_id)?;
        Ok(true)
    }

    fn next_wait(&self, now: Instant) -> Duration {
        let mut wait = MAX_IRQ_WAIT;
        if let Some(deadline) = self.deadline {
            wait = wait.min(deadline.saturating_duration_since(now));
        }
        if self.state.is_idle()
            && let Some(next_due) = next_due_us(self.roster, self.config.address, &self.pair_states)
        {
            wait = wait.min(Duration::from_micros(
                next_due.saturating_sub(self.monotonic_us()),
            ));
        }
        wait.max(MIN_IRQ_WAIT)
    }

    fn monotonic_us(&self) -> u64 {
        u64::try_from(self.started_at.elapsed().as_micros()).unwrap_or(u64::MAX)
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
            ExchangeState::SendingPoll {
                peer,
                exchange_id,
                poll_tx,
            } => {
                let actual_poll_tx = self.tx_timestamp()?;
                if !scheduled_tx_matches(poll_tx, actual_poll_tx) {
                    self.stats.scheduled_tx_misses += 1;
                    self.record_timeout();
                    self.recover_exchange(RecoveryAction::Rearm)?;
                    return Ok(());
                }
                self.state = ExchangeState::AwaitResponse {
                    peer,
                    exchange_id,
                    poll_tx: actual_poll_tx,
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
                    self.record_timeout();
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
                    self.record_timeout();
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
            ExchangeState::SendingReport { peer, exchange_id } => {
                let _ = (peer, exchange_id);
                self.state = ExchangeState::Idle;
                self.deadline = None;
                self.active_pair = None;
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
            (ExchangeState::Idle, AirMessage::Poll { .. })
                if self.poll_is_valid(source, frame.envelope.exchange_id) =>
            {
                self.stats.polls_received += 1;
                self.active_pair = Pair::new(self.config.address, source);
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
                self.stats.responses_received += 1;
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
                self.stats.finals_received += 1;
                let Some(distance) = distance_metres(
                    [poll_tx, response_rx, final_tx],
                    ResponderTimestamps {
                        poll_rx,
                        response_tx,
                        final_rx: rx_time,
                    },
                ) else {
                    self.stats.invalid_frames += 1;
                    self.recover_exchange(RecoveryAction::Rearm)?;
                    return Ok(());
                };
                let pair = Pair::new(self.config.address, source)
                    .expect("validated peers always form a pair");
                let index =
                    pair_index(self.roster, pair).expect("validated peers always have pair state");
                let close = classify_range(
                    self.pair_states[index].is_close(self.monotonic_us()),
                    millimetres(distance),
                );
                self.pair_states[index].record_success(self.monotonic_us(), close);
                self.send_report(source, exchange_id, rx_time, close)?;
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
                AirMessage::Report { final_rx, status },
            ) if peer == source && exchange_id == frame.envelope.exchange_id => {
                self.stats.reports_received += 1;
                let Some(distance) = distance_metres(
                    [poll_tx, response_rx, final_tx],
                    ResponderTimestamps {
                        poll_rx,
                        response_tx,
                        final_rx,
                    },
                ) else {
                    self.stats.invalid_frames += 1;
                    self.recover_exchange(RecoveryAction::Rearm)?;
                    return Ok(());
                };
                self.state = ExchangeState::Idle;
                self.deadline = None;
                self.observe_range(source, status & REPORT_STATUS_CLOSE != 0);
                self.emit(source, distance, final_event_time, on_range);
                self.arm_receive(RecoveryAction::Rearm)?;
            }
            _ => {
                self.stats.unexpected_frames += 1;
                self.arm_receive(RecoveryAction::Rearm)?;
            }
        }
        Ok(())
    }

    fn observe_range(&mut self, peer: NodeAddress, close: bool) {
        let pair =
            Pair::new(self.config.address, peer).expect("validated peers always form a pair");
        let index = pair_index(self.roster, pair).expect("validated peers always have pair state");
        self.pair_states[index].record_success(self.monotonic_us(), close);
        self.active_pair = None;
    }

    fn poll_is_valid(&mut self, source: NodeAddress, exchange_id: ExchangeId) -> bool {
        let Some(pair) = Pair::new(source, self.config.address) else {
            return false;
        };
        if pair.role(self.config.address) != (PairRole::Responder { peer: source })
            || !self.roster.contains(source)
        {
            return false;
        }
        let Some(peer_index) = self.config.peers.iter().position(|peer| *peer == source) else {
            return false;
        };
        let now = Instant::now();
        if self.last_received_exchange[peer_index].is_some_and(|(previous, received_at)| {
            previous == exchange_id
                && now.saturating_duration_since(received_at) < Duration::from_millis(20)
        }) {
            return false;
        }
        self.last_received_exchange[peer_index] = Some((exchange_id, now));
        true
    }

    fn send_poll(&mut self, peer: NodeAddress, exchange_id: ExchangeId) -> Result<()> {
        let planned = self
            .hardware
            .radio
            .schedule_delayed_transmit(DwTime::from_micros(POLL_PROGRAM_LEAD_US as f32))
            .map_err(|error| anyhow::anyhow!("compute delayed Poll timestamp: {error:?}"))?;
        let poll_tx = ticks(planned.timestamp().value());
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
            .transmit_at(frame.bytes(), planned, true)
            .map_err(|error| anyhow::anyhow!("send delayed Poll: {error:?}"))?;
        self.state = ExchangeState::SendingPoll {
            peer,
            exchange_id,
            poll_tx,
        };
        self.deadline = Some(Instant::now() + self.config.response_timeout);
        self.stats.polls_sent += 1;
        Ok(())
    }

    fn send_response(
        &mut self,
        peer: NodeAddress,
        exchange_id: ExchangeId,
        poll_rx: u64,
    ) -> Result<()> {
        let planned = self.planned_tx_after(poll_rx, self.config.reply_delay_us);
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
        self.stats.responses_sent += 1;
        Ok(())
    }

    fn send_final(
        &mut self,
        peer: NodeAddress,
        exchange_id: ExchangeId,
        poll_tx: u64,
        poll_rx: u64,
        response_tx: u64,
        response_rx: u64,
    ) -> Result<()> {
        let planned = self.planned_tx_after(response_rx, self.config.reply_delay_us);
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
        self.stats.finals_sent += 1;
        Ok(())
    }

    fn send_report(
        &mut self,
        peer: NodeAddress,
        exchange_id: ExchangeId,
        final_rx: u64,
        close: bool,
    ) -> Result<()> {
        let planned = self.planned_tx_after(final_rx, REPORT_TURNAROUND_US);
        let frame = self.encode(
            peer,
            AirEnvelope::new(
                exchange_id,
                AirMessage::Report {
                    final_rx,
                    status: if close { REPORT_STATUS_CLOSE } else { 0 },
                },
            ),
        )?;
        self.hardware
            .radio
            .transmit_at(frame.bytes(), planned, false)
            .map_err(|error| anyhow::anyhow!("send delayed REPORT: {error:?}"))?;
        self.state = ExchangeState::SendingReport { peer, exchange_id };
        self.deadline = Some(Instant::now() + self.config.response_timeout);
        self.stats.reports_sent += 1;
        Ok(())
    }

    fn encode(&mut self, peer: NodeAddress, envelope: AirEnvelope) -> Result<air::EncodedAirFrame> {
        self.encode_destination(Destination::Node(peer), envelope)
    }

    fn encode_destination(
        &mut self,
        destination: Destination,
        envelope: AirEnvelope,
    ) -> Result<air::EncodedAirFrame> {
        let sequence = self.mac_sequence;
        self.mac_sequence = self.mac_sequence.wrapping_add(1);
        air::encode(self.config.address, destination, sequence, &envelope)
            .map_err(|error| anyhow::anyhow!("encode native air frame: {error:?}"))
    }

    fn planned_tx_after(&self, received_at: u64, delay_us: u32) -> DelayedTransmit {
        let deadline = delayed_tx_time(received_at, delay_us);
        let on_air = deadline.wrapping_add(u64::from(DEFAULT_ANTENNA_DELAY)) & TIMESTAMP_MASK;
        DelayedTransmit::new(
            DxTimeDeadline::new(DwTime::from_ticks(deadline as i64)),
            OnAirTimestamp::new(DwTime::from_ticks(on_air as i64)),
        )
    }

    fn recover_exchange(&mut self, action: RecoveryAction) -> Result<()> {
        self.record_timeout();
        self.state = ExchangeState::Idle;
        self.deadline = None;
        self.arm_receive(action)
    }

    fn record_timeout(&mut self) {
        let Some(pair) = self.active_pair.take() else {
            return;
        };
        if pair.initiator == self.config.address
            && let Some(index) = pair_index(self.roster, pair)
        {
            self.pair_states[index].record_failure(self.monotonic_us());
            self.stats.contention_backoffs += 1;
        }
    }

    fn record_deadline_phase(&mut self) {
        match self.state {
            ExchangeState::SendingPoll { .. }
            | ExchangeState::SendingResponse { .. }
            | ExchangeState::SendingFinal { .. }
            | ExchangeState::SendingReport { .. } => {
                self.stats.tx_completion_timeouts += 1;
            }
            ExchangeState::AwaitResponse { .. }
            | ExchangeState::AwaitFinal { .. }
            | ExchangeState::AwaitReport { .. } => {
                self.stats.rx_completion_timeouts += 1;
            }
            ExchangeState::Idle => {}
        }
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
        let (mission_event_time_us, mission_time_error_us) = self
            .map_event_time(range_event_time_dtu)
            .map_or((None, u32::MAX), |(time, error)| (Some(time), error));
        let measurement = RangeMeasurement {
            receiver: self.config.address,
            source,
            distance_metres: distance,
            sequence: self.sequence,
            range_event_time_dtu,
            mission_event_time_us,
            mission_generation: mission_event_time_us.map(|_| self.mission_generation),
            mission_time_error_us,
        };
        self.sequence += 1;
        self.stats.ranges += 1;
        self.stats.exchange_completions += 1;
        on_range(measurement);
    }

    fn map_event_time(&mut self, event_dtu: u64) -> Option<(u64, u32)> {
        let host_before_us = unix_time_us()?;
        let radio_now_dtu = self
            .hardware
            .radio
            .read_timestamps()
            .ok()
            .map(|timestamps| ticks(timestamps.system))?;
        let host_after_us = unix_time_us()?;
        let bracket_us = host_after_us.checked_sub(host_before_us)?;
        let host_now_us = host_before_us.saturating_add(bracket_us / 2);
        let age_us = wrapping_delta(radio_now_dtu, event_dtu).div_ceil(DTU_PER_US);
        let event_time_us = host_now_us.checked_sub(age_us)?;
        let error_us = u32::try_from(bracket_us.div_ceil(2)).unwrap_or(u32::MAX);
        (error_us <= 750).then_some((event_time_us, error_us))
    }
}

fn ticks(time: DwTime) -> u64 {
    (time.get_timestamp() as u64) & TIMESTAMP_MASK
}

fn unix_time_us() -> Option<u64> {
    let elapsed = SystemTime::now().duration_since(UNIX_EPOCH).ok()?;
    u64::try_from(elapsed.as_micros()).ok()
}

fn millimetres(distance: f64) -> u32 {
    (distance * 1_000.0 + 0.5).clamp(0.0, u32::MAX as f64) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranging_config_accepts_multiple_addressed_initiator_peers() {
        let node0 = NodeAddress::new(0).unwrap();
        let node1 = NodeAddress::new(1).unwrap();
        let node2 = NodeAddress::new(2).unwrap();
        let timing = Duration::from_millis(5);

        assert!(RangingConfig::new(node0, vec![], 2_000, timing).is_err());
        assert!(RangingConfig::new(node0, vec![node0], 2_000, timing).is_err());
        assert!(RangingConfig::new(node0, vec![node1, node1], 2_000, timing).is_err());
        assert!(RangingConfig::new(node0, vec![node1, node2], 2_000, timing).is_ok());
        assert!(
            RangingConfig::new(
                node0,
                vec![
                    node1,
                    node2,
                    NodeAddress::new(3).unwrap(),
                    NodeAddress::new(0x8000).unwrap(),
                    NodeAddress::new(0x8001).unwrap(),
                ],
                2_000,
                timing,
            )
            .is_ok()
        );
        assert!(
            RangingConfig::new(
                node0,
                vec![
                    node1,
                    node2,
                    NodeAddress::new(3).unwrap(),
                    NodeAddress::new(0x8000).unwrap(),
                    NodeAddress::new(0x8001).unwrap(),
                    NodeAddress::new(0x8002).unwrap(),
                ],
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
        let timing = Duration::from_millis(5);

        assert!(RangingConfig::new(node0, vec![node1], 0, timing).is_err());
        assert!(RangingConfig::new(node0, vec![node1], 2_000, Duration::ZERO).is_err());
        assert!(RangingConfig::new(node0, vec![node1], 2_000, Duration::from_millis(11),).is_err());
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
