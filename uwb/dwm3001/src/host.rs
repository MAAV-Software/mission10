use core::cell::RefCell;
use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use defmt::warn;
use embassy_futures::join::join;
use embassy_futures::select::{Either3, select3};
use embassy_sync::blocking_mutex::Mutex;
use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::channel::Channel;
use embassy_sync::signal::Signal;
use embassy_time::{Duration, Instant};
use embassy_usb::class::cdc_acm::{CdcAcmClass, Receiver, Sender};
use embassy_usb::driver::Driver;
use mission10_uwb_protocol::host::{
    Diagnostic, HOST_FRAME_MAX_SIZE, HOST_RAW_MAX_SIZE, HealthCounters, HostToRadio,
    LatestExchanges, RadioConfiguration, RadioToHost, RadioToHostEnvelope, decode_host_to_radio,
    encode_radio_to_host,
};
use mission10_uwb_protocol::{EgoState, FleetMode, StateValidity};

const USB_PACKET_SIZE: usize = 64;

static MEASUREMENTS: Mutex<CriticalSectionRawMutex, RefCell<LatestExchanges>> =
    Mutex::new(RefCell::new(LatestExchanges::new()));
static MEASUREMENT_SIGNAL: Signal<CriticalSectionRawMutex, ()> = Signal::new();
static CONTROL: Channel<CriticalSectionRawMutex, RadioToHost, 8> = Channel::new();
static DIAGNOSTICS: Channel<CriticalSectionRawMutex, RadioToHost, 8> = Channel::new();
static CONFIGURATIONS: Channel<CriticalSectionRawMutex, RadioConfiguration, 2> = Channel::new();
static FLEET_MODES: Channel<CriticalSectionRawMutex, FleetMode, 2> = Channel::new();

#[derive(Clone, Copy)]
pub struct ClockReply {
    pub request_id: u16,
    pub mission_rx_us: u64,
    pub mission_tx_us: u64,
    pub mission_generation: u32,
    pub source_error_us: u32,
    pub local_rx_us: u64,
}

#[derive(Clone, Copy)]
pub struct ClockProbeSent {
    pub request_id: u16,
    pub local_tx_us: u64,
}

static CLOCK_REPLIES: Channel<CriticalSectionRawMutex, ClockReply, 4> = Channel::new();
static CLOCK_PROBE_SENT: Channel<CriticalSectionRawMutex, ClockProbeSent, 4> = Channel::new();

#[derive(Clone, Copy)]
struct TimedEgoState {
    state: EgoState,
    received_at: Instant,
}

static EGO_STATE: Mutex<CriticalSectionRawMutex, RefCell<Option<TimedEgoState>>> =
    Mutex::new(RefCell::new(None));
static RADIO_COUNTERS: Mutex<CriticalSectionRawMutex, RefCell<HealthCounters>> =
    Mutex::new(RefCell::new(HealthCounters::ZERO));

static HOST_DECODE_ERRORS: AtomicU32 = AtomicU32::new(0);
static HOST_COMMAND_DROPS: AtomicU32 = AtomicU32::new(0);
static CONTROL_DROPS: AtomicU32 = AtomicU32::new(0);
static MEASUREMENT_DROPS: AtomicU32 = AtomicU32::new(0);
static DIAGNOSTIC_DROPS: AtomicU32 = AtomicU32::new(0);
static HOST_STALE_TRANSITIONS: AtomicU32 = AtomicU32::new(0);
static EGO_WAS_STALE: AtomicBool = AtomicBool::new(true);
static HOST_CONNECTED: AtomicBool = AtomicBool::new(false);
const EGO_STALE_AFTER: Duration = Duration::from_millis(100);

pub fn publish(event: RadioToHost) {
    match event {
        RadioToHost::CompletedExchange { exchange } => {
            let replaced = MEASUREMENTS.lock(|slots| slots.borrow_mut().push(exchange));
            if replaced {
                MEASUREMENT_DROPS.fetch_add(1, Ordering::Relaxed);
            }
            MEASUREMENT_SIGNAL.signal(());
        }
        RadioToHost::RadioId { .. }
        | RadioToHost::Otp { .. }
        | RadioToHost::Ready { .. }
        | RadioToHost::Configured { .. }
        | RadioToHost::ClockProbe { .. }
        | RadioToHost::ClockStatus { .. }
        | RadioToHost::Health { .. }
        | RadioToHost::FleetModeReceived { .. } => {
            if CONTROL.try_send(event).is_err() {
                CONTROL_DROPS.fetch_add(1, Ordering::Relaxed);
            }
        }
        RadioToHost::Error { .. } => {
            if DIAGNOSTICS.try_send(event).is_err() {
                DIAGNOSTIC_DROPS.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

fn try_measurement() -> Option<RadioToHost> {
    MEASUREMENTS.lock(|slots| {
        slots
            .borrow_mut()
            .pop_oldest()
            .map(|exchange| RadioToHost::CompletedExchange { exchange })
    })
}

pub fn latest_ego_state() -> EgoState {
    let now = Instant::now();
    let (mut state, stale) = EGO_STATE.lock(|slot| match *slot.borrow() {
        Some(timed) => (
            timed.state,
            now.saturating_duration_since(timed.received_at) >= EGO_STALE_AFTER,
        ),
        None => (EgoState::default(), true),
    });
    if stale {
        state.validity.0 |= StateValidity::HOST_STALE;
    }
    let was_stale = EGO_WAS_STALE.swap(stale, Ordering::Relaxed);
    if stale && !was_stale {
        HOST_STALE_TRANSITIONS.fetch_add(1, Ordering::Relaxed);
    }
    state
}

pub fn try_fleet_mode() -> Option<FleetMode> {
    FLEET_MODES.try_receive().ok()
}

pub fn try_configuration() -> Option<RadioConfiguration> {
    CONFIGURATIONS.try_receive().ok()
}

pub fn try_clock_reply() -> Option<ClockReply> {
    CLOCK_REPLIES.try_receive().ok()
}

pub fn try_clock_probe_sent() -> Option<ClockProbeSent> {
    CLOCK_PROBE_SENT.try_receive().ok()
}

pub fn host_connected() -> bool {
    HOST_CONNECTED.load(Ordering::Relaxed)
}

pub fn clear_measurements() {
    MEASUREMENTS.lock(|slots| slots.borrow_mut().clear());
}

pub fn transport_counters() -> HealthCounters {
    HealthCounters {
        host_decode_errors: HOST_DECODE_ERRORS.load(Ordering::Relaxed),
        host_command_drops: HOST_COMMAND_DROPS.load(Ordering::Relaxed),
        control_drops: CONTROL_DROPS.load(Ordering::Relaxed),
        measurement_drops: MEASUREMENT_DROPS.load(Ordering::Relaxed),
        diagnostic_drops: DIAGNOSTIC_DROPS.load(Ordering::Relaxed),
        host_stale_transitions: HOST_STALE_TRANSITIONS.load(Ordering::Relaxed),
        ..HealthCounters::default()
    }
}

pub fn update_radio_counters(counters: HealthCounters) {
    RADIO_COUNTERS.lock(|slot| *slot.borrow_mut() = counters);
}

fn health_counters() -> HealthCounters {
    let mut counters = RADIO_COUNTERS.lock(|slot| *slot.borrow());
    counters.merge_transport(transport_counters());
    counters
}

pub async fn run<'d, D>(cdc: CdcAcmClass<'d, D>)
where
    D: Driver<'d>,
{
    let (sender, receiver) = cdc.split();
    join(tx_task(sender), rx_task(receiver)).await;
}

async fn tx_task<'d, D>(mut sender: Sender<'d, D>)
where
    D: Driver<'d>,
{
    let mut sequence = 0_u32;
    let mut frame = [0_u8; HOST_FRAME_MAX_SIZE];
    let mut schedule = QueueSchedule::default();
    let mut pending = None;
    loop {
        HOST_CONNECTED.store(false, Ordering::Relaxed);
        sender.wait_connection().await;
        HOST_CONNECTED.store(true, Ordering::Relaxed);
        loop {
            let event = match pending {
                Some(event) => event,
                None => next_event(&mut schedule).await,
            };
            let envelope = RadioToHostEnvelope::new(sequence, event);
            let length = match encode_radio_to_host(&envelope, &mut frame) {
                Ok(length) => length,
                Err(_) => {
                    warn!("failed to encode host event");
                    continue;
                }
            };
            let mut offset = 0;
            let mut failed = false;
            while offset < length {
                let end = core::cmp::min(offset + USB_PACKET_SIZE, length);
                if sender.write_packet(&frame[offset..end]).await.is_err() {
                    failed = true;
                    break;
                }
                offset = end;
            }
            // A full-size final bulk packet needs a short packet to terminate
            // the transfer promptly on the host side.
            if !failed && length % USB_PACKET_SIZE == 0 && sender.write_packet(&[]).await.is_err() {
                failed = true;
            }
            if failed {
                HOST_CONNECTED.store(false, Ordering::Relaxed);
                if let RadioToHost::CompletedExchange { exchange } = event {
                    // Return an interrupted measurement to the latest-per-peer
                    // slots. A newer exchange can replace it while disconnected.
                    let discarded = MEASUREMENTS.lock(|slots| slots.borrow_mut().restore(exchange));
                    if discarded {
                        MEASUREMENT_DROPS.fetch_add(1, Ordering::Relaxed);
                    }
                    MEASUREMENT_SIGNAL.signal(());
                    pending = None;
                } else {
                    pending = Some(event);
                }
                break;
            }
            if let RadioToHost::ClockProbe { request_id } = event {
                let _ = CLOCK_PROBE_SENT.try_send(ClockProbeSent {
                    request_id,
                    local_tx_us: Instant::now().as_micros(),
                });
            }
            pending = None;
            sequence = sequence.wrapping_add(1);
        }
    }
}

const MAX_MEASUREMENTS_BEFORE_CONTROL: u8 = 8;
const MAX_EVENTS_BEFORE_DIAGNOSTIC: u8 = 32;

#[derive(Default)]
struct QueueSchedule {
    measurements_since_control: u8,
    events_since_diagnostic: u8,
}

enum EventQueue {
    Measurement,
    Control,
    Diagnostic,
}

impl QueueSchedule {
    fn selected(&mut self, queue: EventQueue) {
        match queue {
            EventQueue::Measurement => {
                self.measurements_since_control = self.measurements_since_control.saturating_add(1);
                self.events_since_diagnostic = self.events_since_diagnostic.saturating_add(1);
            }
            EventQueue::Control => {
                self.measurements_since_control = 0;
                self.events_since_diagnostic = self.events_since_diagnostic.saturating_add(1);
            }
            EventQueue::Diagnostic => self.events_since_diagnostic = 0,
        }
    }
}

async fn next_event(schedule: &mut QueueSchedule) -> RadioToHost {
    if schedule.events_since_diagnostic >= MAX_EVENTS_BEFORE_DIAGNOSTIC
        && let Ok(event) = DIAGNOSTICS.try_receive()
    {
        schedule.selected(EventQueue::Diagnostic);
        return event;
    }
    if schedule.measurements_since_control >= MAX_MEASUREMENTS_BEFORE_CONTROL
        && let Ok(event) = CONTROL.try_receive()
    {
        schedule.selected(EventQueue::Control);
        return event;
    }
    if let Some(event) = try_measurement() {
        schedule.selected(EventQueue::Measurement);
        return event;
    }
    if let Ok(event) = CONTROL.try_receive() {
        schedule.selected(EventQueue::Control);
        return event;
    }
    if let Ok(event) = DIAGNOSTICS.try_receive() {
        schedule.selected(EventQueue::Diagnostic);
        return event;
    }
    let (event, queue) = match select3(
        MEASUREMENT_SIGNAL.wait(),
        CONTROL.receive(),
        DIAGNOSTICS.receive(),
    )
    .await
    {
        Either3::First(()) => loop {
            if let Some(event) = try_measurement() {
                break (event, EventQueue::Measurement);
            }
            MEASUREMENT_SIGNAL.wait().await;
        },
        Either3::Second(event) => (event, EventQueue::Control),
        Either3::Third(event) => (event, EventQueue::Diagnostic),
    };
    schedule.selected(queue);
    event
}

async fn rx_task<'d, D>(mut receiver: Receiver<'d, D>)
where
    D: Driver<'d>,
{
    let mut packet = [0_u8; USB_PACKET_SIZE];
    let mut frame = [0_u8; HOST_FRAME_MAX_SIZE];
    let mut raw = [0_u8; HOST_RAW_MAX_SIZE];
    loop {
        receiver.wait_connection().await;
        let mut used = 0_usize;
        let mut discarding = false;
        let mut synchronizing = true;
        loop {
            let count = match receiver.read_packet(&mut packet).await {
                Ok(count) => count,
                Err(_) => break,
            };
            for &byte in &packet[..count] {
                if synchronizing {
                    if byte == 0 {
                        synchronizing = false;
                        used = 0;
                    }
                    continue;
                }
                if discarding {
                    if byte == 0 {
                        discarding = false;
                        used = 0;
                    }
                    continue;
                }
                // An empty frame is an explicit stream synchronization marker.
                // Hosts send one after opening USB/UART so any stale partial
                // bytes from a prior connection cannot poison the first command.
                if byte == 0 && used == 0 {
                    continue;
                }
                if used == frame.len() {
                    HOST_DECODE_ERRORS.fetch_add(1, Ordering::Relaxed);
                    discarding = true;
                    used = 0;
                    continue;
                }
                frame[used] = byte;
                used += 1;
                if byte != 0 {
                    continue;
                }
                match decode_host_to_radio(&frame[..used], &mut raw) {
                    Ok(envelope) => route_command(envelope.message),
                    Err(_) => {
                        HOST_DECODE_ERRORS.fetch_add(1, Ordering::Relaxed);
                        publish(RadioToHost::Error {
                            diagnostic: Diagnostic::HostFrame,
                        });
                    }
                }
                used = 0;
            }
        }
    }
}

fn route_command(command: HostToRadio) {
    match command {
        HostToRadio::SetEgoState { state } => {
            EGO_STATE.lock(|slot| {
                *slot.borrow_mut() = Some(TimedEgoState {
                    state,
                    received_at: Instant::now(),
                })
            });
            EGO_WAS_STALE.store(false, Ordering::Relaxed);
        }
        HostToRadio::Configure { configuration } => {
            if !configuration.is_valid() {
                publish(RadioToHost::Error {
                    diagnostic: Diagnostic::InvalidConfiguration,
                });
                return;
            }
            if CONFIGURATIONS.try_send(configuration).is_err() {
                HOST_COMMAND_DROPS.fetch_add(1, Ordering::Relaxed);
            }
        }
        HostToRadio::RequestHealth { request_id } => {
            publish(RadioToHost::Health {
                request_id,
                counters: health_counters(),
            });
        }
        HostToRadio::ClockReply {
            request_id,
            mission_rx_us,
            mission_tx_us,
            mission_generation,
            source_error_us,
        } => {
            let reply = ClockReply {
                request_id,
                mission_rx_us,
                mission_tx_us,
                mission_generation,
                source_error_us,
                local_rx_us: Instant::now().as_micros(),
            };
            if CLOCK_REPLIES.try_send(reply).is_err() {
                HOST_COMMAND_DROPS.fetch_add(1, Ordering::Relaxed);
            }
        }
        HostToRadio::BroadcastFleetMode { mode } => {
            if FLEET_MODES.try_send(mode).is_err() {
                HOST_COMMAND_DROPS.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}
