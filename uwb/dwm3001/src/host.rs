use core::cell::RefCell;
use core::sync::atomic::{AtomicU32, Ordering};

use defmt::warn;
use embassy_futures::join::join;
use embassy_futures::select::{Either3, select3};
use embassy_sync::blocking_mutex::Mutex;
use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::channel::Channel;
use embassy_usb::class::cdc_acm::{CdcAcmClass, Receiver, Sender};
use embassy_usb::driver::Driver;
use mission10_uwb_protocol::EgoState;
use mission10_uwb_protocol::host::{
    Diagnostic, HOST_FRAME_MAX_SIZE, HOST_RAW_MAX_SIZE, HealthCounters, HostToRadio, RadioToHost,
    RadioToHostEnvelope, decode_host_to_radio, encode_radio_to_host,
};

const USB_PACKET_SIZE: usize = 64;

static MEASUREMENTS: Channel<CriticalSectionRawMutex, RadioToHost, 16> = Channel::new();
static CONTROL: Channel<CriticalSectionRawMutex, RadioToHost, 8> = Channel::new();
static DIAGNOSTICS: Channel<CriticalSectionRawMutex, RadioToHost, 8> = Channel::new();
static EGO_STATE: Mutex<CriticalSectionRawMutex, RefCell<Option<EgoState>>> =
    Mutex::new(RefCell::new(None));
static RADIO_COUNTERS: Mutex<CriticalSectionRawMutex, RefCell<HealthCounters>> =
    Mutex::new(RefCell::new(HealthCounters {
        irq_wakes: 0,
        spurious_irq_wakes: 0,
        wait_timeouts: 0,
        recoveries: 0,
        missed_deadlines: 0,
        malformed_air_frames: 0,
        unexpected_air_frames: 0,
        host_decode_errors: 0,
        host_command_drops: 0,
        control_drops: 0,
        measurement_drops: 0,
        diagnostic_drops: 0,
    }));

static HOST_DECODE_ERRORS: AtomicU32 = AtomicU32::new(0);
static HOST_COMMAND_DROPS: AtomicU32 = AtomicU32::new(0);
static CONTROL_DROPS: AtomicU32 = AtomicU32::new(0);
static MEASUREMENT_DROPS: AtomicU32 = AtomicU32::new(0);
static DIAGNOSTIC_DROPS: AtomicU32 = AtomicU32::new(0);

pub fn publish(event: RadioToHost) {
    match event {
        RadioToHost::Range { .. } | RadioToHost::PeerState { .. } => {
            publish_latest_measurement(event);
        }
        RadioToHost::RadioId { .. }
        | RadioToHost::Otp { .. }
        | RadioToHost::Ready { .. }
        | RadioToHost::Configured { .. }
        | RadioToHost::Health { .. } => {
            if CONTROL.try_send(event).is_err() {
                CONTROL_DROPS.fetch_add(1, Ordering::Relaxed);
            }
        }
        RadioToHost::Rx { .. } | RadioToHost::Error { .. } => {
            if DIAGNOSTICS.try_send(event).is_err() {
                DIAGNOSTIC_DROPS.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

fn publish_latest_measurement(event: RadioToHost) {
    if MEASUREMENTS.try_send(event).is_ok() {
        return;
    }
    if MEASUREMENTS.try_receive().is_ok() {
        MEASUREMENT_DROPS.fetch_add(1, Ordering::Relaxed);
    }
    if MEASUREMENTS.try_send(event).is_err() {
        MEASUREMENT_DROPS.fetch_add(1, Ordering::Relaxed);
    }
}

#[allow(dead_code)] // Consumed by the native ranging FSM once its peer is available.
pub fn latest_ego_state() -> EgoState {
    EGO_STATE.lock(|state| state.borrow().unwrap_or_default())
}

pub fn transport_counters() -> HealthCounters {
    HealthCounters {
        host_decode_errors: HOST_DECODE_ERRORS.load(Ordering::Relaxed),
        host_command_drops: HOST_COMMAND_DROPS.load(Ordering::Relaxed),
        control_drops: CONTROL_DROPS.load(Ordering::Relaxed),
        measurement_drops: MEASUREMENT_DROPS.load(Ordering::Relaxed),
        diagnostic_drops: DIAGNOSTIC_DROPS.load(Ordering::Relaxed),
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
    loop {
        sender.wait_connection().await;
        loop {
            let event = next_event(&mut schedule).await;
            let envelope = RadioToHostEnvelope::new(sequence, event);
            sequence = sequence.wrapping_add(1);
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
                break;
            }
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
    if let Ok(event) = MEASUREMENTS.try_receive() {
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
        MEASUREMENTS.receive(),
        CONTROL.receive(),
        DIAGNOSTICS.receive(),
    )
    .await
    {
        Either3::First(event) => (event, EventQueue::Measurement),
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
            EGO_STATE.lock(|slot| *slot.borrow_mut() = Some(state));
        }
        HostToRadio::Configure { configuration } => {
            if !configuration.is_valid() {
                publish(RadioToHost::Error {
                    diagnostic: Diagnostic::InvalidConfiguration,
                });
                return;
            }
            publish(RadioToHost::Error {
                diagnostic: Diagnostic::UnsupportedInMode,
            });
        }
        HostToRadio::RequestHealth { request_id } => {
            publish(RadioToHost::Health {
                request_id,
                counters: health_counters(),
            });
        }
    }
}
