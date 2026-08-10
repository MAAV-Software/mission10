use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use dw1000_rs::{
    AddressConfig, AntennaDelay, Channel, DataRate, DeviceIdentity, Dw1000, Eui64, PanId,
    PhyConfig, PreambleCode, PreambleLength, PulseFrequency, RadioConfig, ShortAddress,
};
use embedded_hal::digital::{ErrorKind as PinErrorKind, ErrorType, InputPin, OutputPin};
use embedded_hal::spi::{Operation, SpiDevice};
use gpiocdev::Request;
use gpiocdev::line::{Bias, EdgeDetection, Value};
use linux_embedded_hal::spidev::{SpiModeFlags, SpidevOptions};
use linux_embedded_hal::{Delay, SPIError, SpidevDevice};

use crate::{DEFAULT_ANTENNA_DELAY, NodeAddress, eui_for, short_address_for};

pub const DEFAULT_SPI_PATH: &str = "/dev/spidev0.0";
pub const DEFAULT_GPIO_CHIP: &str = "/dev/gpiochip0";
pub const DEFAULT_IRQ_GPIO: u32 = 24;
pub const DEFAULT_RESET_GPIO: u32 = 25;
pub const DEFAULT_RUN_SPI_HZ: u32 = 20_000_000;
const INIT_SPI_HZ: u32 = 2_000_000;
const EXPECTED_DEVICE_ID: u32 = 0xdeca_0130;
// Zero coarse and mixer gain: 8.5 dB below the channel-5/64-MHz reference setting.
const CLOSE_RANGE_TX_POWER: u32 = 0xC0C0_C0C0;

#[derive(Debug, Clone, Copy, Default)]
pub enum PhyProfile {
    #[default]
    Operational,
    CloseRangeDiagnostic,
}

#[derive(Clone)]
pub(crate) struct SharedSpi(Arc<Mutex<SpidevDevice>>);

impl SharedSpi {
    fn open(path: &Path, speed_hz: u32) -> Result<Self> {
        let device = SpidevDevice::open(path)
            .with_context(|| format!("open SPI device {}", path.display()))?;
        let spi = Self(Arc::new(Mutex::new(device)));
        spi.set_speed(speed_hz)?;
        Ok(spi)
    }

    fn set_speed(&self, speed_hz: u32) -> Result<()> {
        let options = SpidevOptions::new()
            .bits_per_word(8)
            .max_speed_hz(speed_hz)
            .mode(SpiModeFlags::SPI_MODE_0)
            .build();
        self.0
            .lock()
            .map_err(|_| anyhow::anyhow!("SPI mutex poisoned"))?
            .configure(&options)
            .with_context(|| format!("configure SPI for {speed_hz} Hz"))
    }
}

impl embedded_hal::spi::ErrorType for SharedSpi {
    type Error = SPIError;
}

impl SpiDevice for SharedSpi {
    fn transaction(&mut self, operations: &mut [Operation<'_, u8>]) -> Result<(), Self::Error> {
        let mut spi = self
            .0
            .lock()
            .map_err(|_| SPIError::from(io::Error::other("DW1000 SPI mutex poisoned")))?;
        SpiDevice::transaction(&mut *spi, operations)
    }
}

#[derive(Debug)]
pub(crate) struct PinError(gpiocdev::Error);

impl std::fmt::Display for PinError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for PinError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}

impl embedded_hal::digital::Error for PinError {
    fn kind(&self) -> PinErrorKind {
        PinErrorKind::Other
    }
}

impl From<gpiocdev::Error> for PinError {
    fn from(error: gpiocdev::Error) -> Self {
        Self(error)
    }
}

#[derive(Clone)]
pub(crate) struct IrqPin {
    request: Arc<Request>,
}

pub(crate) struct IrqWait {
    request: Arc<Request>,
}

impl IrqPin {
    fn request(chip: &Path, offset: u32) -> Result<(Self, IrqWait)> {
        let request = Arc::new(
            Request::builder()
                .on_chip(chip)
                .with_consumer("dw1000-radio-irq")
                .with_line(offset)
                .as_input()
                .with_bias(Bias::Disabled)
                .with_edge_detection(EdgeDetection::RisingEdge)
                .request()
                .with_context(|| format!("request IRQ GPIO {offset} from {}", chip.display()))?,
        );
        Ok((
            Self {
                request: request.clone(),
            },
            IrqWait { request },
        ))
    }
}

impl ErrorType for IrqPin {
    type Error = PinError;
}

impl InputPin for IrqPin {
    fn is_high(&mut self) -> Result<bool, Self::Error> {
        self.request
            .lone_value()
            .map(|value| value == Value::Active)
            .map_err(Into::into)
    }

    fn is_low(&mut self) -> Result<bool, Self::Error> {
        self.is_high().map(|high| !high)
    }
}

impl IrqWait {
    pub(crate) fn wait(&self, timeout: Duration) -> Result<bool> {
        if self.request.lone_value().context("read DW1000 IRQ level")? == Value::Active {
            return Ok(true);
        }
        if !self
            .request
            .wait_edge_event(timeout)
            .context("wait for DW1000 IRQ edge")?
        {
            return Ok(false);
        }
        self.request
            .read_edge_event()
            .context("read DW1000 IRQ edge")?;
        while self
            .request
            .has_edge_event()
            .context("check pending DW1000 IRQ edges")?
        {
            self.request
                .read_edge_event()
                .context("drain DW1000 IRQ edge")?;
        }
        Ok(true)
    }
}

pub(crate) trait ResetLine {
    fn drive_low(&mut self) -> Result<(), PinError>;
    fn release_pull_up(&mut self) -> Result<(), PinError>;
}

pub(crate) struct LinuxResetLine {
    chip: PathBuf,
    offset: u32,
    request: Option<Request>,
}

impl LinuxResetLine {
    fn new(chip: &Path, offset: u32) -> Self {
        Self {
            chip: chip.to_owned(),
            offset,
            request: None,
        }
    }
}

impl ResetLine for LinuxResetLine {
    fn drive_low(&mut self) -> Result<(), PinError> {
        self.request.take();
        self.request = Some(
            Request::builder()
                .on_chip(&self.chip)
                .with_consumer("dw1000-radio-reset")
                .with_line(self.offset)
                .as_output(Value::Inactive)
                .request()?,
        );
        Ok(())
    }

    fn release_pull_up(&mut self) -> Result<(), PinError> {
        self.request.take();
        self.request = Some(
            Request::builder()
                .on_chip(&self.chip)
                .with_consumer("dw1000-radio-reset")
                .with_line(self.offset)
                .as_input()
                .with_bias(Bias::PullUp)
                .request()?,
        );
        Ok(())
    }
}

pub(crate) struct OpenDrainReset<L = LinuxResetLine> {
    line: L,
}

impl OpenDrainReset<LinuxResetLine> {
    fn request(chip: &Path, offset: u32) -> Result<Self> {
        let mut line = LinuxResetLine::new(chip, offset);
        line.release_pull_up().map_err(anyhow::Error::from)?;
        Ok(Self { line })
    }
}

impl<L: ResetLine> ErrorType for OpenDrainReset<L> {
    type Error = PinError;
}

impl<L: ResetLine> OutputPin for OpenDrainReset<L> {
    fn set_low(&mut self) -> Result<(), Self::Error> {
        self.line.drive_low()
    }

    fn set_high(&mut self) -> Result<(), Self::Error> {
        // RSTn is open drain: logical high is implemented as high impedance.
        self.line.release_pull_up()
    }
}

pub(crate) type LinuxRadio = Dw1000<SharedSpi, IrqPin, OpenDrainReset>;

pub struct RadioHardware {
    pub(crate) radio: LinuxRadio,
    pub(crate) irq: IrqWait,
    device_id: u32,
    identity: DeviceIdentity,
}

impl RadioHardware {
    pub const fn device_id(&self) -> u32 {
        self.device_id
    }

    pub const fn identity(&self) -> DeviceIdentity {
        self.identity
    }
}

#[derive(Debug, Clone)]
pub struct HardwareConfig {
    pub spi_path: PathBuf,
    pub gpio_chip: PathBuf,
    pub irq_gpio: u32,
    pub reset_gpio: u32,
    pub run_spi_hz: u32,
}

impl Default for HardwareConfig {
    fn default() -> Self {
        Self {
            spi_path: DEFAULT_SPI_PATH.into(),
            gpio_chip: DEFAULT_GPIO_CHIP.into(),
            irq_gpio: DEFAULT_IRQ_GPIO,
            reset_gpio: DEFAULT_RESET_GPIO,
            run_spi_hz: DEFAULT_RUN_SPI_HZ,
        }
    }
}

pub fn open_and_initialize(
    config: &HardwareConfig,
    address: NodeAddress,
    profile: PhyProfile,
) -> Result<RadioHardware> {
    if config.run_spi_hz == 0 {
        bail!("runtime SPI frequency must be positive");
    }
    let spi = SharedSpi::open(&config.spi_path, INIT_SPI_HZ)?;
    let (irq_pin, irq) = IrqPin::request(&config.gpio_chip, config.irq_gpio)?;
    let reset = OpenDrainReset::request(&config.gpio_chip, config.reset_gpio)?;
    let mut radio = Dw1000::new(spi.clone(), irq_pin, reset);
    let mut delay = Delay;
    let radio_config = radio_config(address);
    radio
        .init(&mut delay, &radio_config)
        .map_err(|error| anyhow::anyhow!("DW1000 initialization failed: {error:?}"))?;
    if matches!(profile, PhyProfile::CloseRangeDiagnostic) {
        radio
            .set_transmit_power(CLOSE_RANGE_TX_POWER)
            .map_err(|error| anyhow::anyhow!("set close-range DW1000 TX power: {error:?}"))?;
    }
    spi.set_speed(config.run_spi_hz)?;
    let device_id = radio
        .read_device_id()
        .map_err(|error| anyhow::anyhow!("read DW1000 device ID: {error:?}"))?;
    if device_id != EXPECTED_DEVICE_ID {
        bail!("unexpected DW1000 device ID 0x{device_id:08x}; expected 0x{EXPECTED_DEVICE_ID:08x}")
    }
    Ok(RadioHardware {
        radio,
        irq,
        device_id,
        identity: radio_config.address.identity,
    })
}

fn radio_config(address: NodeAddress) -> RadioConfig {
    let identity = DeviceIdentity::new(
        PanId::new(mission10_uwb_protocol::air::PAN_ID),
        ShortAddress::new(short_address_for(address)),
        Eui64::new(eui_for(address)),
    );
    let (data_rate, preamble_length) = (DataRate::Mbps6800, PreambleLength::Symbols128);
    RadioConfig {
        address: AddressConfig { identity },
        phy: PhyConfig {
            data_rate,
            pulse_frequency: PulseFrequency::Mhz64,
            preamble_length,
            channel: Channel::Channel5,
            preamble_code: Some(PreambleCode::Code10),
            smart_power: false,
        },
        antenna_delay: AntennaDelay::new(DEFAULT_ANTENNA_DELAY),
        receiver_auto_reenable: false,
        interrupt_polarity_high: true,
        frame_check: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct MockResetLine {
        actions: Vec<&'static str>,
    }

    impl ResetLine for MockResetLine {
        fn drive_low(&mut self) -> Result<(), PinError> {
            self.actions.push("low");
            Ok(())
        }

        fn release_pull_up(&mut self) -> Result<(), PinError> {
            self.actions.push("release");
            Ok(())
        }
    }

    #[test]
    fn reset_high_releases_the_open_drain_line() {
        let mut pin = OpenDrainReset {
            line: MockResetLine::default(),
        };
        pin.set_low().unwrap();
        pin.set_high().unwrap();
        assert_eq!(pin.line.actions, ["low", "release"]);
    }

    #[test]
    fn radio_config_matches_the_dwm3001_native_phy() {
        let config = radio_config(NodeAddress::new(0).unwrap());
        assert_eq!(config.phy.channel, Channel::Channel5);
        assert_eq!(config.phy.data_rate, DataRate::Mbps6800);
        assert_eq!(config.phy.pulse_frequency, PulseFrequency::Mhz64);
        assert_eq!(config.phy.preamble_length, PreambleLength::Symbols128);
        assert_eq!(config.phy.preamble_code, Some(PreambleCode::Code10));
        assert!(!config.receiver_auto_reenable);
    }
}
