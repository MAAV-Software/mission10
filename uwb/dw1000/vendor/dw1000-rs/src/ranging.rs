//! Caller-driven tag and anchor ranging state machine.

use heapless::Vec;

use crate::config::{RxOptions, TxOptions};
use crate::device::{
    DelayedTransmit, DeviceIdentity, Eui64, Peer, PeerSnapshot, RxFrame, ShortAddress, Timestamps,
};
use crate::error::{Error, ProtocolError};
use crate::protocol::{
    decode_poll_targets, decode_range_timings, encode_discovery_blink, encode_poll,
    encode_poll_ack, encode_range, encode_range_failed, encode_range_report, encode_ranging_init,
    encode_schedule_sync, parse_frame, BlinkFrame, Frame, FrameHeader, FrameKind, PollTarget,
    RangeReportPayload, RangeTiming, RangingInitFrame, ScheduleSyncFrame,
};
use crate::time::DwTime;

const MAX_FRAME_LEN: usize = 127;

/// Role of the ranging state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub enum Role {
    /// Discovery initiator and range calculator.
    Tag,
    /// Discovery responder.
    Anchor,
}

/// User-facing ranging event.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub enum RangingEvent {
    /// A blink frame was observed.
    BlinkReceived(PeerSnapshot),
    /// A peer was added to the table.
    NewPeer(PeerSnapshot),
    /// A peer was pruned due to inactivity.
    PeerInactive(ShortAddress),
    /// A new range measurement is available.
    RangeUpdated(PeerSnapshot),
    /// A ranging-init message was received.
    RangingInitReceived(ShortAddress),
    /// A schedule-sync message was received.
    ScheduleSyncReceived(ShortAddress),
}

/// Deterministic ranging schedule for multi-anchor and multi-tag systems.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct RangingSchedule {
    /// Anchor response slot used during discovery.
    pub anchor_slot: u8,
    /// Spacing between delayed anchor discovery responses, in microseconds.
    pub discovery_slot_spacing_us: u32,
    /// Tag TDMA slot used for initiating ranging.
    pub tag_slot: u8,
    /// Number of tag TDMA slots.
    pub tag_slot_count: u8,
    /// Duration of each tag TDMA slot, in milliseconds.
    pub tag_slot_ms: u32,
    /// Timeout for poll-ack and range-report collection, in milliseconds.
    pub session_timeout_ms: u32,
    /// Minimum period between tag ranging attempts, in milliseconds.
    pub range_period_ms: u32,
}

impl RangingSchedule {
    /// Default schedule for one tag and up to four anchors.
    pub const fn new() -> Self {
        Self {
            anchor_slot: 0,
            discovery_slot_spacing_us: 12_000,
            tag_slot: 0,
            tag_slot_count: 1,
            tag_slot_ms: 250,
            session_timeout_ms: 160,
            range_period_ms: 80,
        }
    }
}

impl Default for RangingSchedule {
    fn default() -> Self {
        Self::new()
    }
}

/// Ranging configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct RangingConfig {
    /// Local identity used for frame addressing.
    pub identity: DeviceIdentity,
    /// Reply delay used by the protocol, in microseconds.
    pub reply_delay_us: u32,
    /// Inactivity timeout, in milliseconds.
    pub reset_period_ms: u32,
    /// Periodic timer tick, in milliseconds.
    pub timer_period_ms: u32,
    /// Optional exponential moving average factor.
    pub range_filter: Option<u16>,
    /// Multi-node deterministic schedule.
    pub schedule: RangingSchedule,
    /// Whether this anchor broadcasts schedule-sync frames.
    pub anchor_is_coordinator: bool,
}

impl RangingConfig {
    /// Default tag/anchor configuration.
    pub const fn new(identity: DeviceIdentity) -> Self {
        Self {
            identity,
            reply_delay_us: 7000,
            reset_period_ms: 200,
            timer_period_ms: 80,
            range_filter: None,
            schedule: RangingSchedule::new(),
            anchor_is_coordinator: false,
        }
    }
}

/// Radio operations required by the ranging node.
pub trait RangingRadio<SpiE, PinE> {
    /// Starts the receiver with the supplied options.
    fn start_receive(&mut self, options: RxOptions) -> Result<(), Error<SpiE, PinE>>;
    /// Transmits a raw frame.
    fn transmit(&mut self, frame: &[u8], options: TxOptions) -> Result<(), Error<SpiE, PinE>>;
    /// Reads an RX frame into the supplied buffer.
    fn read_frame<'a>(&mut self, buffer: &'a mut [u8]) -> Result<RxFrame<'a>, Error<SpiE, PinE>>;
    /// Reads the latest timestamps.
    fn read_timestamps(&mut self) -> Result<Timestamps, Error<SpiE, PinE>>;
    /// Computes a delayed transmit schedule relative to now.
    fn schedule_delayed_transmit(
        &mut self,
        delay: DwTime,
    ) -> Result<DelayedTransmit, Error<SpiE, PinE>>;
    /// Transmits at an already-computed delayed-transmit schedule.
    fn transmit_at(
        &mut self,
        frame: &[u8],
        schedule: DelayedTransmit,
        wait_for_response: bool,
    ) -> Result<(), Error<SpiE, PinE>>;
}

/// Async radio operations required by the ranging node.
#[allow(async_fn_in_trait)]
pub trait AsyncRangingRadio<SpiE, PinE> {
    /// Starts the receiver with the supplied options.
    async fn start_receive(&mut self, options: RxOptions) -> Result<(), Error<SpiE, PinE>>;
    /// Transmits a raw frame.
    async fn transmit(&mut self, frame: &[u8], options: TxOptions)
        -> Result<(), Error<SpiE, PinE>>;
    /// Reads an RX frame into the supplied buffer.
    async fn read_frame<'a>(
        &mut self,
        buffer: &'a mut [u8],
    ) -> Result<RxFrame<'a>, Error<SpiE, PinE>>;
    /// Reads the latest timestamps.
    async fn read_timestamps(&mut self) -> Result<Timestamps, Error<SpiE, PinE>>;
    /// Computes a delayed transmit schedule relative to now.
    async fn schedule_delayed_transmit(
        &mut self,
        delay: DwTime,
    ) -> Result<DelayedTransmit, Error<SpiE, PinE>>;
    /// Transmits at an already-computed delayed-transmit schedule.
    async fn transmit_at(
        &mut self,
        frame: &[u8],
        schedule: DelayedTransmit,
        wait_for_response: bool,
    ) -> Result<(), Error<SpiE, PinE>>;
}

impl<SPI, IRQ, RST, SpiE, PinE> RangingRadio<SpiE, PinE> for crate::dw1000::Dw1000<SPI, IRQ, RST>
where
    SPI: embedded_hal::spi::SpiDevice<Error = SpiE>,
    IRQ: embedded_hal::digital::InputPin<Error = PinE>,
    RST: embedded_hal::digital::OutputPin<Error = PinE>,
{
    fn start_receive(&mut self, options: RxOptions) -> Result<(), Error<SpiE, PinE>> {
        crate::dw1000::Dw1000::start_receive(self, options)
    }

    fn transmit(&mut self, frame: &[u8], options: TxOptions) -> Result<(), Error<SpiE, PinE>> {
        crate::dw1000::Dw1000::transmit(self, frame, options)
    }

    fn read_frame<'a>(&mut self, buffer: &'a mut [u8]) -> Result<RxFrame<'a>, Error<SpiE, PinE>> {
        crate::dw1000::Dw1000::read_frame(self, buffer)
    }

    fn read_timestamps(&mut self) -> Result<Timestamps, Error<SpiE, PinE>> {
        crate::dw1000::Dw1000::read_timestamps(self)
    }

    fn schedule_delayed_transmit(
        &mut self,
        delay: DwTime,
    ) -> Result<DelayedTransmit, Error<SpiE, PinE>> {
        crate::dw1000::Dw1000::schedule_delayed_transmit(self, delay)
    }

    fn transmit_at(
        &mut self,
        frame: &[u8],
        schedule: DelayedTransmit,
        wait_for_response: bool,
    ) -> Result<(), Error<SpiE, PinE>> {
        crate::dw1000::Dw1000::transmit_at(self, frame, schedule, wait_for_response)
    }
}

impl<SPI, IRQ, RST, SpiE, PinE> AsyncRangingRadio<SpiE, PinE>
    for crate::async_dw1000::AsyncDw1000<SPI, IRQ, RST>
where
    SPI: embedded_hal_async::spi::SpiDevice<Error = SpiE>,
    IRQ: embedded_hal::digital::InputPin<Error = PinE>
        + embedded_hal_async::digital::Wait<Error = PinE>,
    RST: embedded_hal::digital::OutputPin<Error = PinE>,
{
    async fn start_receive(&mut self, options: RxOptions) -> Result<(), Error<SpiE, PinE>> {
        crate::async_dw1000::AsyncDw1000::start_receive(self, options).await
    }

    async fn transmit(
        &mut self,
        frame: &[u8],
        options: TxOptions,
    ) -> Result<(), Error<SpiE, PinE>> {
        crate::async_dw1000::AsyncDw1000::transmit(self, frame, options).await
    }

    async fn read_frame<'a>(
        &mut self,
        buffer: &'a mut [u8],
    ) -> Result<RxFrame<'a>, Error<SpiE, PinE>> {
        crate::async_dw1000::AsyncDw1000::read_frame(self, buffer).await
    }

    async fn read_timestamps(&mut self) -> Result<Timestamps, Error<SpiE, PinE>> {
        crate::async_dw1000::AsyncDw1000::read_timestamps(self).await
    }

    async fn schedule_delayed_transmit(
        &mut self,
        delay: DwTime,
    ) -> Result<DelayedTransmit, Error<SpiE, PinE>> {
        crate::async_dw1000::AsyncDw1000::schedule_delayed_transmit(self, delay).await
    }

    async fn transmit_at(
        &mut self,
        frame: &[u8],
        schedule: DelayedTransmit,
        wait_for_response: bool,
    ) -> Result<(), Error<SpiE, PinE>> {
        crate::async_dw1000::AsyncDw1000::transmit_at(self, frame, schedule, wait_for_response)
            .await
    }
}

/// Tag/anchor ranging state machine.
#[derive(Debug)]
pub struct RangingNode<const N: usize> {
    role: Role,
    config: RangingConfig,
    peers: Vec<Peer, N>,
    sequence: u8,
    expected: FrameKind,
    last_activity_ms: u32,
    blink_counter: u8,
    last_tick_ms: u32,
    last_tx_kind: Option<FrameKind>,
    last_tx_destination: ShortAddress,
    poll_acknowledged: Vec<ShortAddress, N>,
    range_report_received: Vec<ShortAddress, N>,
    session_deadline_ms: Option<u32>,
    last_range_start_ms: u32,
    schedule_epoch_ms: Option<u32>,
}

impl<const N: usize> RangingNode<N> {
    /// Creates a ranging node.
    pub fn new(role: Role, config: RangingConfig) -> Self {
        Self {
            role,
            config,
            peers: Vec::new(),
            sequence: 0,
            expected: match role {
                Role::Tag => FrameKind::PollAck,
                Role::Anchor => FrameKind::Poll,
            },
            last_activity_ms: 0,
            blink_counter: 0,
            last_tick_ms: 0,
            last_tx_kind: None,
            last_tx_destination: ShortAddress::BROADCAST,
            poll_acknowledged: Vec::new(),
            range_report_received: Vec::new(),
            session_deadline_ms: None,
            last_range_start_ms: 0,
            schedule_epoch_ms: None,
        }
    }

    /// Starts the node and arms permanent receive mode.
    pub fn start<R, SpiE, PinE>(
        &mut self,
        radio: &mut R,
        now_ms: u32,
    ) -> Result<(), Error<SpiE, PinE>>
    where
        R: RangingRadio<SpiE, PinE>,
    {
        self.last_activity_ms = now_ms;
        self.last_tick_ms = now_ms;
        radio.start_receive(RxOptions {
            delayed_time: None,
            permanent: true,
        })
    }

    /// Resets protocol state and re-arms permanent receive without pruning peers.
    pub fn recover_link<R, SpiE, PinE>(
        &mut self,
        radio: &mut R,
        now_ms: u32,
    ) -> Result<(), Error<SpiE, PinE>>
    where
        R: RangingRadio<SpiE, PinE>,
    {
        self.reset_protocol_state();
        self.last_activity_ms = now_ms;
        self.last_tick_ms = now_ms;
        radio.start_receive(RxOptions {
            delayed_time: None,
            permanent: true,
        })
    }

    /// Handles a completed transmission.
    pub fn on_tx_done<R, SpiE, PinE>(
        &mut self,
        radio: &mut R,
    ) -> Result<Option<RangingEvent>, Error<SpiE, PinE>>
    where
        R: RangingRadio<SpiE, PinE>,
    {
        let Some(kind) = self.last_tx_kind else {
            return Ok(None);
        };
        let timestamps = radio.read_timestamps()?;
        match (self.role, kind) {
            (Role::Anchor, FrameKind::PollAck) => {
                if let Some(peer) = self.peer_mut(self.last_tx_destination) {
                    peer.poll_ack_sent = timestamps.tx;
                }
            }
            (Role::Tag, FrameKind::Poll) => {
                if self.last_tx_destination.is_broadcast() {
                    for peer in self.peers.iter_mut() {
                        peer.poll_sent = timestamps.tx;
                    }
                } else if let Some(peer) = self.peer_mut(self.last_tx_destination) {
                    peer.poll_sent = timestamps.tx;
                }
            }
            (Role::Tag, FrameKind::Range) => {
                if self.last_tx_destination.is_broadcast() {
                    if self.poll_acknowledged.is_empty() {
                        for peer in self.peers.iter_mut() {
                            peer.range_sent = timestamps.tx;
                        }
                    } else {
                        for index in 0..self.poll_acknowledged.len() {
                            let short = self.poll_acknowledged[index];
                            if let Some(peer) = self.peer_mut(short) {
                                peer.range_sent = timestamps.tx;
                            }
                        }
                    }
                } else if let Some(peer) = self.peer_mut(self.last_tx_destination) {
                    peer.range_sent = timestamps.tx;
                }
            }
            _ => {}
        }
        Ok(None)
    }

    /// Handles an incoming frame.
    pub fn on_rx<R, SpiE, PinE>(
        &mut self,
        radio: &mut R,
        now_ms: u32,
        buffer: &mut [u8],
    ) -> Result<Option<RangingEvent>, Error<SpiE, PinE>>
    where
        R: RangingRadio<SpiE, PinE>,
    {
        let frame = radio.read_frame(buffer)?;
        self.last_activity_ms = now_ms;
        match parse_frame(frame.bytes)? {
            Frame::Blink(blink) if self.role == Role::Anchor => {
                let Some(snapshot) = self.accept_blink(blink, now_ms)? else {
                    return Ok(None);
                };
                self.reset_protocol_state();
                self.send_ranging_init(radio, blink.source_eui, blink.source_short)?;
                Ok(Some(RangingEvent::BlinkReceived(snapshot)))
            }
            Frame::ScheduleSync(sync) => {
                let Some(short_address) = self.observe_schedule_sync(sync) else {
                    return Ok(None);
                };
                if self.role == Role::Tag {
                    self.accept_schedule_sync(sync, now_ms)?;
                }
                Ok(Some(RangingEvent::ScheduleSyncReceived(short_address)))
            }
            Frame::RangingInit(init) if self.role == Role::Tag => {
                let Some(short_address) = self.accept_ranging_init(init, now_ms)? else {
                    return Ok(None);
                };
                self.reset_protocol_state();
                Ok(Some(RangingEvent::RangingInitReceived(short_address)))
            }
            Frame::Poll { header, payload } if self.role == Role::Anchor => {
                let Some(peer_short) = self.accept_header(header, true, now_ms)? else {
                    return Ok(None);
                };
                if self.expected != FrameKind::Poll {
                    self.reset_protocol_state();
                    return Ok(None);
                }
                let mut targets = [PollTarget {
                    short_address: ShortAddress::new(0),
                    reply_delay_us: 0,
                }; N];
                let count = decode_poll_targets(payload, &mut targets)?;
                let local = self.config.identity.short_address;
                let Some(target) = targets[..count]
                    .iter()
                    .find(|target| target.short_address == local)
                    .copied()
                else {
                    return Ok(None);
                };
                let Some(peer) = self.peer_mut(peer_short) else {
                    return Ok(None);
                };
                peer.poll_received = frame.timestamp;
                peer.last_activity_ms = now_ms;
                peer.reply_delay_us = target.reply_delay_us;
                let destination = peer.short_address;
                self.expected = FrameKind::Range;
                self.send_poll_ack(radio, destination, target.reply_delay_us)?;
                Ok(None)
            }
            Frame::PollAck { header } if self.role == Role::Tag => {
                let Some(peer_short) = self.accept_header(header, false, now_ms)? else {
                    return Ok(None);
                };
                if self.expected != FrameKind::PollAck {
                    self.reset_protocol_state();
                    return Ok(None);
                }
                let short_address = {
                    let Some(peer) = self.peer_mut(peer_short) else {
                        return Ok(None);
                    };
                    peer.poll_ack_received = frame.timestamp;
                    peer.last_activity_ms = now_ms;
                    peer.short_address
                };
                if !self.poll_acknowledged.contains(&short_address) {
                    self.poll_acknowledged
                        .push(short_address)
                        .map_err(|_| ProtocolError::PeerTableFull)?;
                }
                if self.poll_acknowledged.len() == self.peers.len() {
                    self.begin_range_report_phase(now_ms);
                    self.send_range(radio, None)?;
                }
                Ok(None)
            }
            Frame::Range { header, payload } if self.role == Role::Anchor => {
                let Some(peer_short) = self.accept_header(header, true, now_ms)? else {
                    return Ok(None);
                };
                if self.expected != FrameKind::Range {
                    self.reset_protocol_state();
                    return Ok(None);
                }
                let mut timings = [RangeTiming {
                    short_address: ShortAddress::new(0),
                    poll_sent: DwTime::zero(),
                    poll_ack_received: DwTime::zero(),
                    range_sent: DwTime::zero(),
                }; N];
                let count = decode_range_timings(payload, &mut timings)?;
                let local = self.config.identity.short_address;
                let Some(timing) = timings[..count]
                    .iter()
                    .find(|timing| timing.short_address == local)
                    .copied()
                else {
                    return Ok(None);
                };
                self.expected = FrameKind::Poll;
                let range_filter = self.config.range_filter;
                let (destination, reply_delay_us, report, snapshot) = {
                    let Some(peer) = self.peer_mut(peer_short) else {
                        return Ok(None);
                    };
                    peer.range_received = frame.timestamp;
                    peer.poll_sent = timing.poll_sent;
                    peer.poll_ack_received = timing.poll_ack_received;
                    peer.range_sent = timing.range_sent;
                    peer.metrics = frame.metrics;
                    peer.last_activity_ms = now_ms;
                    let tof = DwTime::asymmetric_tof(
                        peer.poll_sent,
                        peer.poll_received,
                        peer.poll_ack_sent,
                        peer.poll_ack_received,
                        peer.range_sent,
                        peer.range_received,
                    );
                    let distance = filtered_range(range_filter, peer.range_m, tof.as_meters());
                    peer.range_m = distance;
                    (
                        peer.short_address,
                        peer.reply_delay_us,
                        RangeReportPayload {
                            poll_received: peer.poll_received,
                            poll_ack_sent: peer.poll_ack_sent,
                            range_received: peer.range_received,
                            receive_power_dbm: frame.metrics.receive_power_dbm,
                        },
                        PeerSnapshot::from(&*peer),
                    )
                };
                self.send_range_report(radio, destination, report, reply_delay_us)?;
                Ok(Some(RangingEvent::RangeUpdated(snapshot)))
            }
            Frame::RangeReport { header, payload } if self.role == Role::Tag => {
                let Some(peer_short) = self.accept_header(header, false, now_ms)? else {
                    return Ok(None);
                };
                if self.expected != FrameKind::RangeReport {
                    self.reset_protocol_state();
                    return Ok(None);
                }
                let range_filter = self.config.range_filter;
                let (short_address, snapshot) = {
                    let Some(peer) = self.peer_mut(peer_short) else {
                        return Ok(None);
                    };
                    let tof = DwTime::asymmetric_tof(
                        peer.poll_sent,
                        payload.poll_received,
                        payload.poll_ack_sent,
                        peer.poll_ack_received,
                        peer.range_sent,
                        payload.range_received,
                    );
                    peer.range_m = filtered_range(range_filter, peer.range_m, tof.as_meters());
                    peer.metrics.receive_power_dbm = payload.receive_power_dbm;
                    peer.metrics.first_path_power_dbm = frame.metrics.first_path_power_dbm;
                    peer.metrics.quality = frame.metrics.quality;
                    peer.last_activity_ms = now_ms;
                    (peer.short_address, PeerSnapshot::from(&*peer))
                };
                if !self.range_report_received.contains(&short_address) {
                    self.range_report_received
                        .push(short_address)
                        .map_err(|_| ProtocolError::PeerTableFull)?;
                }
                if self.range_report_received.len() >= self.poll_acknowledged.len() {
                    self.reset_protocol_state();
                }
                Ok(Some(RangingEvent::RangeUpdated(snapshot)))
            }
            Frame::RangeFailed { header } if self.role == Role::Tag => {
                let Some(peer_short) = self.accept_header(header, false, now_ms)? else {
                    return Ok(None);
                };
                self.reset_protocol_state();
                if let Some(peer) = self.peer_mut(peer_short) {
                    peer.last_activity_ms = now_ms;
                }
                Ok(None)
            }
            _ => Ok(None),
        }
    }

    /// Periodic maintenance and discovery tick.
    pub fn tick<R, SpiE, PinE>(
        &mut self,
        radio: &mut R,
        now_ms: u32,
    ) -> Result<Option<RangingEvent>, Error<SpiE, PinE>>
    where
        R: RangingRadio<SpiE, PinE>,
    {
        if self.finish_timed_out_session(radio, now_ms)? {
            return Ok(None);
        }

        if let Some(position) = self.peers.iter().position(|peer| {
            now_ms.saturating_sub(peer.last_activity_ms) > self.config.reset_period_ms
        }) {
            let short = self.peers.swap_remove(position).short_address;
            self.reset_protocol_state();
            return Ok(Some(RangingEvent::PeerInactive(short)));
        }

        if now_ms.saturating_sub(self.last_tick_ms) < self.config.timer_period_ms {
            return Ok(None);
        }
        self.last_tick_ms = now_ms;

        if self.session_active() {
            return Ok(None);
        }

        if self.role == Role::Anchor {
            if self.config.anchor_is_coordinator
                && now_ms.wrapping_sub(self.last_range_start_ms)
                    >= self.config.schedule.range_period_ms
            {
                self.last_range_start_ms = now_ms;
                self.send_schedule_sync(radio, now_ms)?;
            }
        } else if self.tag_slot_is_open(now_ms)
            && now_ms.wrapping_sub(self.last_range_start_ms) >= self.config.schedule.range_period_ms
        {
            if !self.peers.is_empty() && self.blink_counter != 0 {
                self.begin_poll_ack_phase(now_ms);
                self.send_poll(radio, None)?;
            } else {
                self.send_blink(radio)?;
            }
            self.last_range_start_ms = now_ms;
            self.blink_counter = if self.blink_counter >= 20 {
                0
            } else {
                self.blink_counter + 1
            };
        }
        Ok(None)
    }

    /// Returns the current peer snapshots.
    pub fn peers(&self) -> impl Iterator<Item = PeerSnapshot> + '_ {
        self.peers.iter().map(PeerSnapshot::from)
    }

    /// Returns the role this node was configured with.
    pub const fn role(&self) -> Role {
        self.role
    }

    /// Returns a lightweight snapshot of the most recent transmit state.
    pub fn tx_debug_snapshot(&self) -> (u8, Option<FrameKind>) {
        (self.sequence, self.last_tx_kind)
    }

    fn send_blink<R, SpiE, PinE>(&mut self, radio: &mut R) -> Result<(), Error<SpiE, PinE>>
    where
        R: RangingRadio<SpiE, PinE>,
    {
        let mut frame = [0u8; MAX_FRAME_LEN];
        let len = encode_discovery_blink(
            self.next_sequence(),
            self.config.identity.eui,
            self.config.identity.short_address,
            &mut frame,
        )?;
        self.last_tx_kind = Some(FrameKind::Blink);
        self.last_tx_destination = ShortAddress::BROADCAST;
        radio.transmit(&frame[..len], TxOptions::default())
    }

    fn send_ranging_init<R, SpiE, PinE>(
        &mut self,
        radio: &mut R,
        destination_eui: crate::device::Eui64,
        destination_short: ShortAddress,
    ) -> Result<(), Error<SpiE, PinE>>
    where
        R: RangingRadio<SpiE, PinE>,
    {
        let mut frame = [0u8; MAX_FRAME_LEN];
        let len = encode_ranging_init(
            self.next_sequence(),
            self.config.identity.short_address,
            destination_eui,
            &mut frame,
        )?;
        self.last_tx_kind = Some(FrameKind::RangingInit);
        self.last_tx_destination = destination_short;
        radio.transmit(
            &frame[..len],
            TxOptions {
                delayed_time: self.anchor_discovery_delay(),
                wait_for_response: false,
            },
        )
    }

    fn send_schedule_sync<R, SpiE, PinE>(
        &mut self,
        radio: &mut R,
        now_ms: u32,
    ) -> Result<(), Error<SpiE, PinE>>
    where
        R: RangingRadio<SpiE, PinE>,
    {
        let mut frame = [0u8; MAX_FRAME_LEN];
        let len = encode_schedule_sync(
            self.next_sequence(),
            self.config.identity.short_address,
            ShortAddress::BROADCAST,
            now_ms,
            self.config.schedule.tag_slot_count,
            self.config.schedule.tag_slot_ms,
            &mut frame,
        )?;
        self.last_tx_kind = Some(FrameKind::ScheduleSync);
        self.last_tx_destination = ShortAddress::BROADCAST;
        radio.transmit(&frame[..len], TxOptions::default())
    }

    fn send_poll<R, SpiE, PinE>(
        &mut self,
        radio: &mut R,
        destination: Option<ShortAddress>,
    ) -> Result<(), Error<SpiE, PinE>>
    where
        R: RangingRadio<SpiE, PinE>,
    {
        let mut frame = [0u8; MAX_FRAME_LEN];
        let mut targets = [PollTarget {
            short_address: ShortAddress::new(0),
            reply_delay_us: 0,
        }; N];
        let destination = destination.unwrap_or(ShortAddress::BROADCAST);
        let count = if destination.is_broadcast() {
            let base_reply_delay_us = self.config.reply_delay_us;
            for (index, peer) in self.peers.iter_mut().enumerate() {
                peer.reply_delay_us = scheduled_reply_delay_us(base_reply_delay_us, index)?;
                targets[index] = PollTarget {
                    short_address: peer.short_address,
                    reply_delay_us: peer.reply_delay_us,
                };
            }
            self.peers.len()
        } else {
            let reply_delay = self.config.reply_delay_us;
            let peer = self
                .peer_mut(destination)
                .ok_or(ProtocolError::UnknownPeer)?;
            peer.reply_delay_us = reply_delay;
            targets[0] = PollTarget {
                short_address: peer.short_address,
                reply_delay_us: peer.reply_delay_us,
            };
            1
        };
        let len = encode_poll(
            self.next_sequence(),
            self.config.identity.short_address,
            destination,
            &targets[..count],
            &mut frame,
        )?;
        self.last_tx_kind = Some(FrameKind::Poll);
        self.last_tx_destination = destination;
        radio.transmit(&frame[..len], TxOptions::default())
    }

    fn send_poll_ack<R, SpiE, PinE>(
        &mut self,
        radio: &mut R,
        destination: ShortAddress,
        reply_delay_us: u32,
    ) -> Result<(), Error<SpiE, PinE>>
    where
        R: RangingRadio<SpiE, PinE>,
    {
        let mut frame = [0u8; MAX_FRAME_LEN];
        let len = encode_poll_ack(
            self.next_sequence(),
            self.config.identity.short_address,
            destination,
            &mut frame,
        )?;
        self.last_tx_kind = Some(FrameKind::PollAck);
        self.last_tx_destination = destination;
        radio.transmit(
            &frame[..len],
            TxOptions {
                delayed_time: Some(DwTime::from_micros(reply_delay_us as f32)),
                wait_for_response: false,
            },
        )
    }

    fn send_range<R, SpiE, PinE>(
        &mut self,
        radio: &mut R,
        destination: Option<ShortAddress>,
    ) -> Result<(), Error<SpiE, PinE>>
    where
        R: RangingRadio<SpiE, PinE>,
    {
        let mut frame = [0u8; MAX_FRAME_LEN];
        let mut timings = [RangeTiming {
            short_address: ShortAddress::new(0),
            poll_sent: DwTime::zero(),
            poll_ack_received: DwTime::zero(),
            range_sent: DwTime::zero(),
        }; N];
        let destination = destination.unwrap_or(ShortAddress::BROADCAST);
        let delay = DwTime::from_micros(self.config.reply_delay_us as f32);
        let schedule = radio.schedule_delayed_transmit(delay)?;
        let range_sent = schedule.timestamp().value();
        let count = if destination.is_broadcast() {
            if self.poll_acknowledged.is_empty() {
                for (index, peer) in self.peers.iter().enumerate() {
                    timings[index] = RangeTiming {
                        short_address: peer.short_address,
                        poll_sent: peer.poll_sent,
                        poll_ack_received: peer.poll_ack_received,
                        range_sent,
                    };
                }
                self.peers.len()
            } else {
                let mut count = 0;
                for index in 0..self.poll_acknowledged.len() {
                    let short = self.poll_acknowledged[index];
                    if let Some(peer) = self.peer(short) {
                        timings[count] = RangeTiming {
                            short_address: peer.short_address,
                            poll_sent: peer.poll_sent,
                            poll_ack_received: peer.poll_ack_received,
                            range_sent,
                        };
                        count += 1;
                    }
                }
                count
            }
        } else {
            let peer = self.peer(destination).ok_or(ProtocolError::UnknownPeer)?;
            timings[0] = RangeTiming {
                short_address: peer.short_address,
                poll_sent: peer.poll_sent,
                poll_ack_received: peer.poll_ack_received,
                range_sent,
            };
            1
        };
        let len = encode_range(
            self.next_sequence(),
            self.config.identity.short_address,
            destination,
            &timings[..count],
            &mut frame,
        )?;
        self.last_tx_kind = Some(FrameKind::Range);
        self.last_tx_destination = destination;
        radio.transmit_at(&frame[..len], schedule, false)
    }

    fn send_range_report<R, SpiE, PinE>(
        &mut self,
        radio: &mut R,
        destination: ShortAddress,
        payload: RangeReportPayload,
        reply_delay_us: u32,
    ) -> Result<(), Error<SpiE, PinE>>
    where
        R: RangingRadio<SpiE, PinE>,
    {
        let mut frame = [0u8; MAX_FRAME_LEN];
        let len = encode_range_report(
            self.next_sequence(),
            self.config.identity.short_address,
            destination,
            payload,
            &mut frame,
        )?;
        self.last_tx_kind = Some(FrameKind::RangeReport);
        self.last_tx_destination = destination;
        radio.transmit(
            &frame[..len],
            TxOptions {
                delayed_time: Some(DwTime::from_micros(reply_delay_us as f32)),
                wait_for_response: false,
            },
        )
    }

    #[allow(dead_code)]
    fn send_range_failed<R, SpiE, PinE>(
        &mut self,
        radio: &mut R,
        destination: ShortAddress,
    ) -> Result<(), Error<SpiE, PinE>>
    where
        R: RangingRadio<SpiE, PinE>,
    {
        let mut frame = [0u8; MAX_FRAME_LEN];
        let len = encode_range_failed(
            self.next_sequence(),
            self.config.identity.short_address,
            destination,
            &mut frame,
        )?;
        self.last_tx_kind = Some(FrameKind::RangeFailed);
        self.last_tx_destination = destination;
        radio.transmit(&frame[..len], TxOptions::default())
    }

    fn ensure_peer(
        &mut self,
        eui: Option<crate::device::Eui64>,
        short_address: ShortAddress,
        now_ms: u32,
    ) -> Result<&mut Peer, ProtocolError> {
        if let Some(index) = self
            .peers
            .iter()
            .position(|peer| peer.short_address == short_address)
        {
            let peer = &mut self.peers[index];
            peer.last_activity_ms = now_ms;
            if peer.eui.is_none() {
                peer.eui = eui;
            }
            return Ok(peer);
        }
        self.peers
            .push(Peer::new(eui, short_address))
            .map_err(|_| ProtocolError::PeerTableFull)?;
        let last = self.peers.last_mut().ok_or(ProtocolError::PeerTableFull)?;
        last.last_activity_ms = now_ms;
        Ok(last)
    }

    fn peer(&self, short_address: ShortAddress) -> Option<&Peer> {
        self.peers
            .iter()
            .find(|peer| peer.short_address == short_address)
    }

    fn peer_mut(&mut self, short_address: ShortAddress) -> Option<&mut Peer> {
        self.peers
            .iter_mut()
            .find(|peer| peer.short_address == short_address)
    }

    fn next_sequence(&mut self) -> u8 {
        let sequence = self.sequence;
        self.sequence = self.sequence.wrapping_add(1);
        sequence
    }

    fn accept_blink(
        &mut self,
        blink: BlinkFrame,
        now_ms: u32,
    ) -> Result<Option<PeerSnapshot>, ProtocolError> {
        let peer = self.ensure_peer(Some(blink.source_eui), blink.source_short, now_ms)?;
        if !observe_peer_sequence(peer, blink.sequence) {
            return Ok(None);
        }
        Ok(Some(PeerSnapshot::from(&*peer)))
    }

    fn accept_ranging_init(
        &mut self,
        init: RangingInitFrame,
        now_ms: u32,
    ) -> Result<Option<ShortAddress>, ProtocolError> {
        if !self.matches_eui_destination(init.destination_eui) {
            return Ok(None);
        }
        let peer = self.ensure_peer(None, init.source_short, now_ms)?;
        if !observe_peer_sequence(peer, init.sequence) {
            return Ok(None);
        }
        Ok(Some(peer.short_address))
    }

    fn observe_schedule_sync(&self, sync: ScheduleSyncFrame) -> Option<ShortAddress> {
        self.matches_short_destination(sync.header.destination, true)
            .then_some(sync.header.source)
    }

    fn accept_schedule_sync(
        &mut self,
        sync: ScheduleSyncFrame,
        now_ms: u32,
    ) -> Result<Option<ShortAddress>, ProtocolError> {
        let Some(source) = self.observe_schedule_sync(sync) else {
            return Ok(None);
        };
        let peer = self.ensure_peer(None, source, now_ms)?;
        if !observe_peer_sequence(peer, sync.header.sequence) {
            return Ok(None);
        }
        let short_address = peer.short_address;
        let frame_ms = u32::from(sync.tag_slot_count).saturating_mul(sync.tag_slot_ms);
        self.schedule_epoch_ms = Some(if frame_ms == 0 {
            now_ms
        } else {
            now_ms.wrapping_sub(sync.epoch_ms % frame_ms)
        });
        Ok(Some(short_address))
    }

    fn accept_header(
        &mut self,
        header: FrameHeader,
        allow_broadcast: bool,
        now_ms: u32,
    ) -> Result<Option<ShortAddress>, ProtocolError> {
        if !self.matches_short_destination(header.destination, allow_broadcast) {
            return Ok(None);
        }
        let Some(peer) = self.peer_mut(header.source) else {
            return Ok(None);
        };
        if !observe_peer_sequence(peer, header.sequence) {
            return Ok(None);
        }
        peer.last_activity_ms = now_ms;
        Ok(Some(peer.short_address))
    }

    fn matches_short_destination(&self, destination: ShortAddress, allow_broadcast: bool) -> bool {
        destination == self.config.identity.short_address
            || (allow_broadcast && destination.is_broadcast())
    }

    fn matches_eui_destination(&self, destination: Eui64) -> bool {
        destination == self.config.identity.eui
    }

    fn reset_protocol_state(&mut self) {
        self.expected = match self.role {
            Role::Tag => FrameKind::PollAck,
            Role::Anchor => FrameKind::Poll,
        };
        self.last_tx_kind = None;
        self.last_tx_destination = ShortAddress::BROADCAST;
        self.poll_acknowledged.clear();
        self.range_report_received.clear();
        self.session_deadline_ms = None;
    }

    fn begin_poll_ack_phase(&mut self, now_ms: u32) {
        self.expected = FrameKind::PollAck;
        self.poll_acknowledged.clear();
        self.range_report_received.clear();
        self.session_deadline_ms =
            Some(now_ms.wrapping_add(self.config.schedule.session_timeout_ms));
    }

    fn begin_range_report_phase(&mut self, now_ms: u32) {
        self.expected = FrameKind::RangeReport;
        self.range_report_received.clear();
        self.session_deadline_ms =
            Some(now_ms.wrapping_add(self.config.schedule.session_timeout_ms));
    }

    fn finish_timed_out_session<R, SpiE, PinE>(
        &mut self,
        radio: &mut R,
        now_ms: u32,
    ) -> Result<bool, Error<SpiE, PinE>>
    where
        R: RangingRadio<SpiE, PinE>,
    {
        if !self.session_expired(now_ms) {
            return Ok(false);
        }
        match (self.role, self.expected) {
            (Role::Tag, FrameKind::PollAck) if !self.poll_acknowledged.is_empty() => {
                self.begin_range_report_phase(now_ms);
                self.send_range(radio, None)?;
                Ok(true)
            }
            (Role::Tag, FrameKind::PollAck | FrameKind::RangeReport) => {
                self.reset_protocol_state();
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    fn session_active(&self) -> bool {
        matches!(self.expected, FrameKind::PollAck | FrameKind::RangeReport)
            && self.session_deadline_ms.is_some()
    }

    fn session_expired(&self, now_ms: u32) -> bool {
        self.session_deadline_ms
            .map(|deadline| now_ms.wrapping_sub(deadline) < 0x8000_0000)
            .unwrap_or(false)
    }

    fn tag_slot_is_open(&self, now_ms: u32) -> bool {
        let count = u32::from(self.config.schedule.tag_slot_count.max(1));
        if count <= 1 {
            return true;
        }
        let slot_ms = self.config.schedule.tag_slot_ms;
        if slot_ms == 0 {
            return true;
        }
        let tag_slot = u32::from(self.config.schedule.tag_slot).min(count - 1);
        let epoch = self.schedule_epoch_ms.unwrap_or(0);
        let current_slot = (now_ms.wrapping_sub(epoch) / slot_ms) % count;
        current_slot == tag_slot
    }

    fn anchor_discovery_delay(&self) -> Option<DwTime> {
        let delay_us = u32::from(self.config.schedule.anchor_slot)
            .saturating_mul(self.config.schedule.discovery_slot_spacing_us);
        (delay_us != 0).then(|| DwTime::from_micros(delay_us as f32))
    }
}

impl<const N: usize> RangingNode<N> {
    /// Starts the node and arms permanent receive mode on an async radio.
    pub async fn start_async<R, SpiE, PinE>(
        &mut self,
        radio: &mut R,
        now_ms: u32,
    ) -> Result<(), Error<SpiE, PinE>>
    where
        R: AsyncRangingRadio<SpiE, PinE>,
    {
        self.last_activity_ms = now_ms;
        self.last_tick_ms = now_ms;
        radio
            .start_receive(RxOptions {
                delayed_time: None,
                permanent: true,
            })
            .await
    }

    /// Resets protocol state and re-arms permanent receive on an async radio.
    pub async fn recover_link_async<R, SpiE, PinE>(
        &mut self,
        radio: &mut R,
        now_ms: u32,
    ) -> Result<(), Error<SpiE, PinE>>
    where
        R: AsyncRangingRadio<SpiE, PinE>,
    {
        self.reset_protocol_state();
        self.last_activity_ms = now_ms;
        self.last_tick_ms = now_ms;
        radio
            .start_receive(RxOptions {
                delayed_time: None,
                permanent: true,
            })
            .await
    }

    /// Handles a completed transmission for an async radio.
    pub async fn on_tx_done_async<R, SpiE, PinE>(
        &mut self,
        radio: &mut R,
    ) -> Result<Option<RangingEvent>, Error<SpiE, PinE>>
    where
        R: AsyncRangingRadio<SpiE, PinE>,
    {
        let Some(kind) = self.last_tx_kind else {
            return Ok(None);
        };
        let timestamps = radio.read_timestamps().await?;
        match (self.role, kind) {
            (Role::Anchor, FrameKind::PollAck) => {
                if let Some(peer) = self.peer_mut(self.last_tx_destination) {
                    peer.poll_ack_sent = timestamps.tx;
                }
            }
            (Role::Tag, FrameKind::Poll) => {
                if self.last_tx_destination.is_broadcast() {
                    for peer in self.peers.iter_mut() {
                        peer.poll_sent = timestamps.tx;
                    }
                } else if let Some(peer) = self.peer_mut(self.last_tx_destination) {
                    peer.poll_sent = timestamps.tx;
                }
            }
            (Role::Tag, FrameKind::Range) => {
                if self.last_tx_destination.is_broadcast() {
                    if self.poll_acknowledged.is_empty() {
                        for peer in self.peers.iter_mut() {
                            peer.range_sent = timestamps.tx;
                        }
                    } else {
                        for index in 0..self.poll_acknowledged.len() {
                            let short = self.poll_acknowledged[index];
                            if let Some(peer) = self.peer_mut(short) {
                                peer.range_sent = timestamps.tx;
                            }
                        }
                    }
                } else if let Some(peer) = self.peer_mut(self.last_tx_destination) {
                    peer.range_sent = timestamps.tx;
                }
            }
            _ => {}
        }
        Ok(None)
    }

    /// Handles an incoming frame for an async radio.
    pub async fn on_rx_async<R, SpiE, PinE>(
        &mut self,
        radio: &mut R,
        now_ms: u32,
        buffer: &mut [u8],
    ) -> Result<Option<RangingEvent>, Error<SpiE, PinE>>
    where
        R: AsyncRangingRadio<SpiE, PinE>,
    {
        let frame = radio.read_frame(buffer).await?;
        self.last_activity_ms = now_ms;
        match parse_frame(frame.bytes)? {
            Frame::Blink(blink) if self.role == Role::Anchor => {
                let Some(snapshot) = self.accept_blink(blink, now_ms)? else {
                    return Ok(None);
                };
                self.reset_protocol_state();
                self.send_ranging_init_async(radio, blink.source_eui, blink.source_short)
                    .await?;
                Ok(Some(RangingEvent::BlinkReceived(snapshot)))
            }
            Frame::ScheduleSync(sync) => {
                let Some(short_address) = self.observe_schedule_sync(sync) else {
                    return Ok(None);
                };
                if self.role == Role::Tag {
                    self.accept_schedule_sync(sync, now_ms)?;
                }
                Ok(Some(RangingEvent::ScheduleSyncReceived(short_address)))
            }
            Frame::RangingInit(init) if self.role == Role::Tag => {
                let Some(short_address) = self.accept_ranging_init(init, now_ms)? else {
                    return Ok(None);
                };
                self.reset_protocol_state();
                Ok(Some(RangingEvent::RangingInitReceived(short_address)))
            }
            Frame::Poll { header, payload } if self.role == Role::Anchor => {
                let Some(peer_short) = self.accept_header(header, true, now_ms)? else {
                    return Ok(None);
                };
                if self.expected != FrameKind::Poll {
                    self.reset_protocol_state();
                    return Ok(None);
                }
                let mut targets = [PollTarget {
                    short_address: ShortAddress::new(0),
                    reply_delay_us: 0,
                }; N];
                let count = decode_poll_targets(payload, &mut targets)?;
                let local = self.config.identity.short_address;
                let Some(target) = targets[..count]
                    .iter()
                    .find(|target| target.short_address == local)
                    .copied()
                else {
                    return Ok(None);
                };
                let Some(peer) = self.peer_mut(peer_short) else {
                    return Ok(None);
                };
                peer.poll_received = frame.timestamp;
                peer.last_activity_ms = now_ms;
                peer.reply_delay_us = target.reply_delay_us;
                let destination = peer.short_address;
                self.expected = FrameKind::Range;
                self.send_poll_ack_async(radio, destination, target.reply_delay_us)
                    .await?;
                Ok(None)
            }
            Frame::PollAck { header } if self.role == Role::Tag => {
                let Some(peer_short) = self.accept_header(header, false, now_ms)? else {
                    return Ok(None);
                };
                if self.expected != FrameKind::PollAck {
                    self.reset_protocol_state();
                    return Ok(None);
                }
                let short_address = {
                    let Some(peer) = self.peer_mut(peer_short) else {
                        return Ok(None);
                    };
                    peer.poll_ack_received = frame.timestamp;
                    peer.last_activity_ms = now_ms;
                    peer.short_address
                };
                if !self.poll_acknowledged.contains(&short_address) {
                    self.poll_acknowledged
                        .push(short_address)
                        .map_err(|_| ProtocolError::PeerTableFull)?;
                }
                if self.poll_acknowledged.len() == self.peers.len() {
                    self.begin_range_report_phase(now_ms);
                    self.send_range_async(radio, None).await?;
                }
                Ok(None)
            }
            Frame::Range { header, payload } if self.role == Role::Anchor => {
                let Some(peer_short) = self.accept_header(header, true, now_ms)? else {
                    return Ok(None);
                };
                if self.expected != FrameKind::Range {
                    self.reset_protocol_state();
                    return Ok(None);
                }
                let mut timings = [RangeTiming {
                    short_address: ShortAddress::new(0),
                    poll_sent: DwTime::zero(),
                    poll_ack_received: DwTime::zero(),
                    range_sent: DwTime::zero(),
                }; N];
                let count = decode_range_timings(payload, &mut timings)?;
                let local = self.config.identity.short_address;
                let Some(timing) = timings[..count]
                    .iter()
                    .find(|timing| timing.short_address == local)
                    .copied()
                else {
                    return Ok(None);
                };
                self.expected = FrameKind::Poll;
                let range_filter = self.config.range_filter;
                let (destination, reply_delay_us, report, snapshot) = {
                    let Some(peer) = self.peer_mut(peer_short) else {
                        return Ok(None);
                    };
                    peer.range_received = frame.timestamp;
                    peer.poll_sent = timing.poll_sent;
                    peer.poll_ack_received = timing.poll_ack_received;
                    peer.range_sent = timing.range_sent;
                    peer.metrics = frame.metrics;
                    peer.last_activity_ms = now_ms;
                    let tof = DwTime::asymmetric_tof(
                        peer.poll_sent,
                        peer.poll_received,
                        peer.poll_ack_sent,
                        peer.poll_ack_received,
                        peer.range_sent,
                        peer.range_received,
                    );
                    let distance = filtered_range(range_filter, peer.range_m, tof.as_meters());
                    peer.range_m = distance;
                    (
                        peer.short_address,
                        peer.reply_delay_us,
                        RangeReportPayload {
                            poll_received: peer.poll_received,
                            poll_ack_sent: peer.poll_ack_sent,
                            range_received: peer.range_received,
                            receive_power_dbm: frame.metrics.receive_power_dbm,
                        },
                        PeerSnapshot::from(&*peer),
                    )
                };
                self.send_range_report_async(radio, destination, report, reply_delay_us)
                    .await?;
                Ok(Some(RangingEvent::RangeUpdated(snapshot)))
            }
            Frame::RangeReport { header, payload } if self.role == Role::Tag => {
                let Some(peer_short) = self.accept_header(header, false, now_ms)? else {
                    return Ok(None);
                };
                if self.expected != FrameKind::RangeReport {
                    self.reset_protocol_state();
                    return Ok(None);
                }
                let range_filter = self.config.range_filter;
                let (short_address, snapshot) = {
                    let Some(peer) = self.peer_mut(peer_short) else {
                        return Ok(None);
                    };
                    let tof = DwTime::asymmetric_tof(
                        peer.poll_sent,
                        payload.poll_received,
                        payload.poll_ack_sent,
                        peer.poll_ack_received,
                        peer.range_sent,
                        payload.range_received,
                    );
                    peer.range_m = filtered_range(range_filter, peer.range_m, tof.as_meters());
                    peer.metrics.receive_power_dbm = payload.receive_power_dbm;
                    peer.metrics.first_path_power_dbm = frame.metrics.first_path_power_dbm;
                    peer.metrics.quality = frame.metrics.quality;
                    peer.last_activity_ms = now_ms;
                    (peer.short_address, PeerSnapshot::from(&*peer))
                };
                if !self.range_report_received.contains(&short_address) {
                    self.range_report_received
                        .push(short_address)
                        .map_err(|_| ProtocolError::PeerTableFull)?;
                }
                if self.range_report_received.len() >= self.poll_acknowledged.len() {
                    self.reset_protocol_state();
                }
                Ok(Some(RangingEvent::RangeUpdated(snapshot)))
            }
            Frame::RangeFailed { header } if self.role == Role::Tag => {
                let Some(peer_short) = self.accept_header(header, false, now_ms)? else {
                    return Ok(None);
                };
                self.reset_protocol_state();
                if let Some(peer) = self.peer_mut(peer_short) {
                    peer.last_activity_ms = now_ms;
                }
                Ok(None)
            }
            _ => Ok(None),
        }
    }

    /// Periodic maintenance and discovery tick for an async radio.
    pub async fn tick_async<R, SpiE, PinE>(
        &mut self,
        radio: &mut R,
        now_ms: u32,
    ) -> Result<Option<RangingEvent>, Error<SpiE, PinE>>
    where
        R: AsyncRangingRadio<SpiE, PinE>,
    {
        if self.finish_timed_out_session_async(radio, now_ms).await? {
            return Ok(None);
        }

        if let Some(position) = self.peers.iter().position(|peer| {
            now_ms.saturating_sub(peer.last_activity_ms) > self.config.reset_period_ms
        }) {
            let short = self.peers.swap_remove(position).short_address;
            self.reset_protocol_state();
            return Ok(Some(RangingEvent::PeerInactive(short)));
        }

        if now_ms.saturating_sub(self.last_tick_ms) < self.config.timer_period_ms {
            return Ok(None);
        }
        self.last_tick_ms = now_ms;

        if self.session_active() {
            return Ok(None);
        }

        if self.role == Role::Anchor {
            if self.config.anchor_is_coordinator
                && now_ms.wrapping_sub(self.last_range_start_ms)
                    >= self.config.schedule.range_period_ms
            {
                self.last_range_start_ms = now_ms;
                self.send_schedule_sync_async(radio, now_ms).await?;
            }
        } else if self.tag_slot_is_open(now_ms)
            && now_ms.wrapping_sub(self.last_range_start_ms) >= self.config.schedule.range_period_ms
        {
            if !self.peers.is_empty() && self.blink_counter != 0 {
                self.begin_poll_ack_phase(now_ms);
                self.send_poll_async(radio, None).await?;
            } else {
                self.send_blink_async(radio).await?;
            }
            self.last_range_start_ms = now_ms;
            self.blink_counter = if self.blink_counter >= 20 {
                0
            } else {
                self.blink_counter + 1
            };
        }
        Ok(None)
    }

    async fn finish_timed_out_session_async<R, SpiE, PinE>(
        &mut self,
        radio: &mut R,
        now_ms: u32,
    ) -> Result<bool, Error<SpiE, PinE>>
    where
        R: AsyncRangingRadio<SpiE, PinE>,
    {
        if !self.session_expired(now_ms) {
            return Ok(false);
        }
        match (self.role, self.expected) {
            (Role::Tag, FrameKind::PollAck) if !self.poll_acknowledged.is_empty() => {
                self.begin_range_report_phase(now_ms);
                self.send_range_async(radio, None).await?;
                Ok(true)
            }
            (Role::Tag, FrameKind::PollAck | FrameKind::RangeReport) => {
                self.reset_protocol_state();
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    async fn send_blink_async<R, SpiE, PinE>(
        &mut self,
        radio: &mut R,
    ) -> Result<(), Error<SpiE, PinE>>
    where
        R: AsyncRangingRadio<SpiE, PinE>,
    {
        let mut frame = [0u8; MAX_FRAME_LEN];
        let len = encode_discovery_blink(
            self.next_sequence(),
            self.config.identity.eui,
            self.config.identity.short_address,
            &mut frame,
        )?;
        self.last_tx_kind = Some(FrameKind::Blink);
        self.last_tx_destination = ShortAddress::BROADCAST;
        radio.transmit(&frame[..len], TxOptions::default()).await
    }

    async fn send_ranging_init_async<R, SpiE, PinE>(
        &mut self,
        radio: &mut R,
        destination_eui: crate::device::Eui64,
        destination_short: ShortAddress,
    ) -> Result<(), Error<SpiE, PinE>>
    where
        R: AsyncRangingRadio<SpiE, PinE>,
    {
        let mut frame = [0u8; MAX_FRAME_LEN];
        let len = encode_ranging_init(
            self.next_sequence(),
            self.config.identity.short_address,
            destination_eui,
            &mut frame,
        )?;
        self.last_tx_kind = Some(FrameKind::RangingInit);
        self.last_tx_destination = destination_short;
        radio
            .transmit(
                &frame[..len],
                TxOptions {
                    delayed_time: self.anchor_discovery_delay(),
                    wait_for_response: false,
                },
            )
            .await
    }

    async fn send_schedule_sync_async<R, SpiE, PinE>(
        &mut self,
        radio: &mut R,
        now_ms: u32,
    ) -> Result<(), Error<SpiE, PinE>>
    where
        R: AsyncRangingRadio<SpiE, PinE>,
    {
        let mut frame = [0u8; MAX_FRAME_LEN];
        let len = encode_schedule_sync(
            self.next_sequence(),
            self.config.identity.short_address,
            ShortAddress::BROADCAST,
            now_ms,
            self.config.schedule.tag_slot_count,
            self.config.schedule.tag_slot_ms,
            &mut frame,
        )?;
        self.last_tx_kind = Some(FrameKind::ScheduleSync);
        self.last_tx_destination = ShortAddress::BROADCAST;
        radio.transmit(&frame[..len], TxOptions::default()).await
    }

    async fn send_poll_async<R, SpiE, PinE>(
        &mut self,
        radio: &mut R,
        destination: Option<ShortAddress>,
    ) -> Result<(), Error<SpiE, PinE>>
    where
        R: AsyncRangingRadio<SpiE, PinE>,
    {
        let mut frame = [0u8; MAX_FRAME_LEN];
        let mut targets = [PollTarget {
            short_address: ShortAddress::new(0),
            reply_delay_us: 0,
        }; N];
        let destination = destination.unwrap_or(ShortAddress::BROADCAST);
        let count = if destination.is_broadcast() {
            let base_reply_delay_us = self.config.reply_delay_us;
            for (index, peer) in self.peers.iter_mut().enumerate() {
                peer.reply_delay_us = scheduled_reply_delay_us(base_reply_delay_us, index)?;
                targets[index] = PollTarget {
                    short_address: peer.short_address,
                    reply_delay_us: peer.reply_delay_us,
                };
            }
            self.peers.len()
        } else {
            let reply_delay = self.config.reply_delay_us;
            let peer = self
                .peer_mut(destination)
                .ok_or(ProtocolError::UnknownPeer)?;
            peer.reply_delay_us = reply_delay;
            targets[0] = PollTarget {
                short_address: peer.short_address,
                reply_delay_us: peer.reply_delay_us,
            };
            1
        };
        let len = encode_poll(
            self.next_sequence(),
            self.config.identity.short_address,
            destination,
            &targets[..count],
            &mut frame,
        )?;
        self.last_tx_kind = Some(FrameKind::Poll);
        self.last_tx_destination = destination;
        radio.transmit(&frame[..len], TxOptions::default()).await
    }

    async fn send_poll_ack_async<R, SpiE, PinE>(
        &mut self,
        radio: &mut R,
        destination: ShortAddress,
        reply_delay_us: u32,
    ) -> Result<(), Error<SpiE, PinE>>
    where
        R: AsyncRangingRadio<SpiE, PinE>,
    {
        let mut frame = [0u8; MAX_FRAME_LEN];
        let len = encode_poll_ack(
            self.next_sequence(),
            self.config.identity.short_address,
            destination,
            &mut frame,
        )?;
        self.last_tx_kind = Some(FrameKind::PollAck);
        self.last_tx_destination = destination;
        radio
            .transmit(
                &frame[..len],
                TxOptions {
                    delayed_time: Some(DwTime::from_micros(reply_delay_us as f32)),
                    wait_for_response: false,
                },
            )
            .await
    }

    async fn send_range_async<R, SpiE, PinE>(
        &mut self,
        radio: &mut R,
        destination: Option<ShortAddress>,
    ) -> Result<(), Error<SpiE, PinE>>
    where
        R: AsyncRangingRadio<SpiE, PinE>,
    {
        let mut frame = [0u8; MAX_FRAME_LEN];
        let mut timings = [RangeTiming {
            short_address: ShortAddress::new(0),
            poll_sent: DwTime::zero(),
            poll_ack_received: DwTime::zero(),
            range_sent: DwTime::zero(),
        }; N];
        let destination = destination.unwrap_or(ShortAddress::BROADCAST);
        let delay = DwTime::from_micros(self.config.reply_delay_us as f32);
        let schedule = radio.schedule_delayed_transmit(delay).await?;
        let range_sent = schedule.timestamp().value();
        let count = if destination.is_broadcast() {
            if self.poll_acknowledged.is_empty() {
                for (index, peer) in self.peers.iter().enumerate() {
                    timings[index] = RangeTiming {
                        short_address: peer.short_address,
                        poll_sent: peer.poll_sent,
                        poll_ack_received: peer.poll_ack_received,
                        range_sent,
                    };
                }
                self.peers.len()
            } else {
                let mut count = 0;
                for index in 0..self.poll_acknowledged.len() {
                    let short = self.poll_acknowledged[index];
                    if let Some(peer) = self.peer(short) {
                        timings[count] = RangeTiming {
                            short_address: peer.short_address,
                            poll_sent: peer.poll_sent,
                            poll_ack_received: peer.poll_ack_received,
                            range_sent,
                        };
                        count += 1;
                    }
                }
                count
            }
        } else {
            let peer = self.peer(destination).ok_or(ProtocolError::UnknownPeer)?;
            timings[0] = RangeTiming {
                short_address: peer.short_address,
                poll_sent: peer.poll_sent,
                poll_ack_received: peer.poll_ack_received,
                range_sent,
            };
            1
        };
        let len = encode_range(
            self.next_sequence(),
            self.config.identity.short_address,
            destination,
            &timings[..count],
            &mut frame,
        )?;
        self.last_tx_kind = Some(FrameKind::Range);
        self.last_tx_destination = destination;
        radio.transmit_at(&frame[..len], schedule, false).await
    }

    async fn send_range_report_async<R, SpiE, PinE>(
        &mut self,
        radio: &mut R,
        destination: ShortAddress,
        payload: RangeReportPayload,
        reply_delay_us: u32,
    ) -> Result<(), Error<SpiE, PinE>>
    where
        R: AsyncRangingRadio<SpiE, PinE>,
    {
        let mut frame = [0u8; MAX_FRAME_LEN];
        let len = encode_range_report(
            self.next_sequence(),
            self.config.identity.short_address,
            destination,
            payload,
            &mut frame,
        )?;
        self.last_tx_kind = Some(FrameKind::RangeReport);
        self.last_tx_destination = destination;
        radio
            .transmit(
                &frame[..len],
                TxOptions {
                    delayed_time: Some(DwTime::from_micros(reply_delay_us as f32)),
                    wait_for_response: false,
                },
            )
            .await
    }

    #[allow(dead_code)]
    async fn send_range_failed_async<R, SpiE, PinE>(
        &mut self,
        radio: &mut R,
        destination: ShortAddress,
    ) -> Result<(), Error<SpiE, PinE>>
    where
        R: AsyncRangingRadio<SpiE, PinE>,
    {
        let mut frame = [0u8; MAX_FRAME_LEN];
        let len = encode_range_failed(
            self.next_sequence(),
            self.config.identity.short_address,
            destination,
            &mut frame,
        )?;
        self.last_tx_kind = Some(FrameKind::RangeFailed);
        self.last_tx_destination = destination;
        radio.transmit(&frame[..len], TxOptions::default()).await
    }
}

fn filtered_range(range_filter: Option<u16>, previous: f32, distance: f32) -> f32 {
    if let Some(filter) = range_filter {
        if previous != 0.0 {
            let k = 2.0 / (filter as f32 + 1.0);
            return distance * k + previous * (1.0 - k);
        }
    }
    distance
}

fn scheduled_reply_delay_us(base_reply_delay_us: u32, index: usize) -> Result<u32, ProtocolError> {
    let slot = index
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(ProtocolError::ReplyDelayOverflow)?;
    let multiplier = u32::try_from(slot).map_err(|_| ProtocolError::ReplyDelayOverflow)?;
    base_reply_delay_us
        .checked_mul(multiplier)
        .ok_or(ProtocolError::ReplyDelayOverflow)
}

fn observe_peer_sequence(peer: &mut Peer, sequence: u8) -> bool {
    let is_fresh = match peer.last_sequence {
        None => true,
        Some(last_sequence) => {
            let delta = sequence.wrapping_sub(last_sequence);
            delta != 0 && delta < 128
        }
    };
    if is_fresh {
        peer.last_sequence = Some(sequence);
    }
    is_fresh
}
