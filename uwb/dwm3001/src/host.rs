use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::channel::Channel;
use embassy_usb::class::cdc_acm::CdcAcmClass;
use embassy_usb::driver::Driver;
use mission10_uwb_protocol::{HOST_FRAME_MAX_SIZE, HostEnvelope, RadioToHost, encode_host_frame};

use defmt::warn;

static EVENTS: Channel<CriticalSectionRawMutex, RadioToHost, 16> = Channel::new();

pub fn publish(event: RadioToHost) {
    if EVENTS.try_send(event).is_err() {
        warn!("diagnostic event queue full");
    }
}

pub async fn run<'d, D>(mut cdc: CdcAcmClass<'d, D>)
where
    D: Driver<'d>,
{
    let mut sequence = 0_u32;
    let mut frame = [0_u8; HOST_FRAME_MAX_SIZE];
    loop {
        cdc.wait_connection().await;
        loop {
            let envelope = HostEnvelope::new(sequence, EVENTS.receive().await);
            sequence = sequence.wrapping_add(1);
            let length = match encode_host_frame(&envelope, &mut frame) {
                Ok(length) => length,
                Err(_) => {
                    warn!("failed to encode host event");
                    continue;
                }
            };
            if cdc.write_packet(&frame[..length]).await.is_err() {
                break;
            }
        }
    }
}
