use dw3000_ng::configs::{
    BitRate, PdoaMode, PhrMode, PhrRate, PreambleLength, PulseRepetitionFrequency, SfdSequence,
    StsMode, UwbChannel,
};
use dw3000_ng::time::Instant as RadioInstant;
use dw3000_ng::{
    Config as RadioConfig, DW3000, Error as RadioError, FastCommand, Ready, Sending,
    SingleBufferReceiving,
};
use embassy_futures::select::{Either, select};
use embassy_nrf::gpio::{Input, Pull};
use embassy_nrf::{Peri, peripherals};
use embassy_time::Timer;
use embedded_hal_async::spi::SpiDevice;
use mission10_uwb_protocol::{Diagnostic, RadioToHost};

pub const OWN_ADDRESS: [u8; 2] = if cfg!(feature = "initiator") {
    [0xa0, 0xc0]
} else {
    [0xa1, 0xc1]
};
pub const PEER_ADDRESS: [u8; 2] = if cfg!(feature = "initiator") {
    [0xa1, 0xc1]
} else {
    [0xa0, 0xc0]
};
pub const PEER_INDEX: u8 = if cfg!(feature = "initiator") { 1 } else { 0 };
pub const FALLBACK_ANTENNA_DELAY: u16 = 16_390;

// Both delayed DS-TWR legs use the same bench-tunable turnaround. Two
// milliseconds is intentionally still generous: once it is stable against the
// DW1000 peer we can lower it toward the few-hundred-microsecond hardware limit.
pub const REPLY_DELAY_US: u32 = 2_000;

// Avoid monopolizing the channel in the one-pair diagnostic. Together with the
// two delayed legs this targets roughly 100 ranging exchanges per second.
pub const INITIATOR_INTER_EXCHANGE_GUARD_MS: u64 = 5;

pub struct RadioHardware {
    pub spi: Peri<'static, peripherals::SPI3>,
    pub sck: Peri<'static, peripherals::P0_03>,
    pub miso: Peri<'static, peripherals::P0_29>,
    pub mosi: Peri<'static, peripherals::P0_08>,
    pub cs: Peri<'static, peripherals::P1_06>,
    pub reset: Peri<'static, peripherals::P0_25>,
    /// DWM3001C schematic net `DW_IRQ`, active-high, routed to nRF P1.02.
    pub irq: Peri<'static, peripherals::P1_02>,
}

pub fn radio_config() -> RadioConfig {
    RadioConfig {
        channel: UwbChannel::Channel5,
        sfd_sequence: SfdSequence::IeeeShort,
        pulse_repetition_frequency: PulseRepetitionFrequency::Mhz64,
        preamble_length: PreambleLength::Symbols128,
        bitrate: BitRate::Kbps6800,
        frame_filtering: false,
        ranging_enable: true,
        sts_mode: StsMode::StsModeOff,
        sfd_timeout: 121,
        tx_preamble_code: Some(10),
        rx_preamble_code: Some(10),
        phr_mode: PhrMode::Standard,
        phr_rate: PhrRate::Standard,
        pdoa_mode: PdoaMode::Mode0,
        ..Default::default()
    }
}

#[derive(Clone, Copy, Default)]
pub struct WaitCounters {
    pub irq_wakes: u32,
    pub spurious_irq_wakes: u32,
    pub wait_timeouts: u32,
    pub recoveries: u32,
}

impl WaitCounters {
    pub fn host_message(self) -> RadioToHost {
        RadioToHost::Health {
            irq_wakes: self.irq_wakes,
            spurious_irq_wakes: self.spurious_irq_wakes,
            wait_timeouts: self.wait_timeouts,
            recoveries: self.recoveries,
        }
    }
}

/// Wakes the radio status loop from the DW IRQ line. `wait_for_high` subscribes
/// before checking the pin, so an IRQ arriving between the status read and
/// sleep cannot be lost.
pub struct RadioWait<'d> {
    irq: Input<'d>,
    counters: WaitCounters,
    woke_on_irq: bool,
}

impl<'d> RadioWait<'d> {
    pub fn new(irq: Peri<'d, impl embassy_nrf::gpio::Pin>) -> Self {
        Self {
            irq: Input::new(irq, Pull::Down),
            counters: WaitCounters::default(),
            woke_on_irq: false,
        }
    }

    pub async fn pending(&mut self) {
        if self.woke_on_irq {
            self.counters.spurious_irq_wakes = self.counters.spurious_irq_wakes.wrapping_add(1);
        }

        if self.irq.is_high() {
            self.counters.irq_wakes = self.counters.irq_wakes.wrapping_add(1);
            self.woke_on_irq = true;
            // A high line without a status bit should not become a tight SPI loop.
            Timer::after_micros(50).await;
            return;
        }

        match select(self.irq.wait_for_high(), Timer::after_secs(1)).await {
            Either::First(()) => {
                self.counters.irq_wakes = self.counters.irq_wakes.wrapping_add(1);
                self.woke_on_irq = true;
            }
            Either::Second(()) => {
                self.counters.wait_timeouts = self.counters.wait_timeouts.wrapping_add(1);
                self.woke_on_irq = false;
            }
        }
    }

    pub fn completed(&mut self) {
        self.woke_on_irq = false;
    }

    pub fn recovered(&mut self) {
        self.counters.recoveries = self.counters.recoveries.wrapping_add(1);
        self.completed();
    }

    pub fn counters(&self) -> WaitCounters {
        self.counters
    }
}

pub async fn prepare_tx<SPI>(radio: &mut DW3000<SPI, Ready>) -> Result<(), Diagnostic>
where
    SPI: SpiDevice<u8>,
{
    radio
        .disable_interrupts()
        .await
        .map_err(|_| Diagnostic::Spi)?;
    radio
        .fast_cmd(FastCommand::CMD_CLR_IRQS)
        .await
        .map_err(|_| Diagnostic::Spi)?;
    radio
        .enable_tx_interrupts()
        .await
        .map_err(|_| Diagnostic::Spi)?;
    Ok(())
}

pub async fn prepare_rx<SPI>(radio: &mut DW3000<SPI, Ready>) -> Result<(), Diagnostic>
where
    SPI: SpiDevice<u8>,
{
    radio
        .disable_interrupts()
        .await
        .map_err(|_| Diagnostic::Spi)?;
    radio
        .fast_cmd(FastCommand::CMD_CLR_IRQS)
        .await
        .map_err(|_| Diagnostic::Spi)?;
    radio
        .enable_rx_interrupts()
        .await
        .map_err(|_| Diagnostic::Spi)?;
    Ok(())
}

pub async fn wait_send<SPI>(
    sending: &mut DW3000<SPI, Sending>,
    wait: &mut RadioWait<'_>,
) -> Result<RadioInstant, Diagnostic>
where
    SPI: SpiDevice<u8>,
{
    loop {
        match sending.s_wait().await {
            Ok(timestamp) => {
                wait.completed();
                return Ok(timestamp);
            }
            Err(nb::Error::WouldBlock) => wait.pending().await,
            Err(nb::Error::Other(error)) => {
                wait.completed();
                return Err(diagnostic(&error));
            }
        }
    }
}

pub async fn wait_receive<SPI>(
    receiving: &mut DW3000<SPI, SingleBufferReceiving>,
    buffer: &mut [u8],
    wait: &mut RadioWait<'_>,
) -> Result<(usize, RadioInstant), Diagnostic>
where
    SPI: SpiDevice<u8>,
{
    loop {
        match receiving.r_wait_buf(buffer).await {
            Ok((length, timestamp, _quality)) => {
                wait.completed();
                return Ok((length, timestamp));
            }
            Err(nb::Error::WouldBlock) => wait.pending().await,
            Err(nb::Error::Other(error)) => {
                wait.completed();
                return Err(diagnostic(&error));
            }
        }
    }
}

pub const fn recoverable_receive_error(error: Diagnostic) -> bool {
    matches!(
        error,
        Diagnostic::RxFcs
            | Diagnostic::RxPhy
            | Diagnostic::RxReedSolomon
            | Diagnostic::RxFrameWaitTimeout
            | Diagnostic::RxOverrun
            | Diagnostic::RxPreambleDetectionTimeout
            | Diagnostic::RxSfdTimeout
            | Diagnostic::RxFrameFilteringRejection
    )
}

fn diagnostic<SPI>(error: &RadioError<SPI>) -> Diagnostic
where
    SPI: SpiDevice<u8>,
{
    match error {
        RadioError::Fcs => Diagnostic::RxFcs,
        RadioError::Phy => Diagnostic::RxPhy,
        RadioError::BufferTooSmall { .. } => Diagnostic::RxBufferTooSmall,
        RadioError::ReedSolomon => Diagnostic::RxReedSolomon,
        RadioError::FrameWaitTimeout => Diagnostic::RxFrameWaitTimeout,
        RadioError::Overrun => Diagnostic::RxOverrun,
        RadioError::PreambleDetectionTimeout => Diagnostic::RxPreambleDetectionTimeout,
        RadioError::SfdTimeout => Diagnostic::RxSfdTimeout,
        RadioError::FrameFilteringRejection => Diagnostic::RxFrameFilteringRejection,
        RadioError::Spi(_) => Diagnostic::Spi,
        RadioError::Frame(_) => Diagnostic::FrameDecode,
        RadioError::DelayedSendTooLate => Diagnostic::DelayedSendTooLate,
        RadioError::DelayedSendPowerUpWarning => Diagnostic::DelayedSendPowerUpWarning,
        RadioError::RxNotFinished | RadioError::StillAsleep => Diagnostic::RadioState,
        _ => Diagnostic::Unknown,
    }
}
