//! DW1000 configuration types
//!
//! This module provides type-safe configuration structures and enums for
//! configuring the DW1000 UWB transceiver.

#[cfg(feature = "defmt")]
use defmt::Format;

use crate::device::{AntennaDelay, DeviceIdentity};
use crate::time::DwTime;

/// Data transmission/reception bit rate
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "defmt", derive(Format))]
pub enum DataRate {
    /// 110 kbps data rate
    Kbps110 = 0x00,
    /// 850 kbps data rate
    Kbps850 = 0x01,
    /// 6.8 Mbps data rate
    Mbps6800 = 0x02,
}

/// Transmission pulse repetition frequency (PRF)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "defmt", derive(Format))]
pub enum PulseFrequency {
    /// 16 MHz PRF (more power efficient)
    Mhz16 = 0x01,
    /// 64 MHz PRF (better performance, more power)
    Mhz64 = 0x02,
}

/// Preamble length configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "defmt", derive(Format))]
pub enum PreambleLength {
    /// 64 symbols
    Symbols64 = 0x01,
    /// 128 symbols
    Symbols128 = 0x05,
    /// 256 symbols
    Symbols256 = 0x09,
    /// 512 symbols
    Symbols512 = 0x0D,
    /// 1024 symbols
    Symbols1024 = 0x02,
    /// 1536 symbols
    Symbols1536 = 0x06,
    /// 2048 symbols
    Symbols2048 = 0x0A,
    /// 4096 symbols
    Symbols4096 = 0x03,
}

/// Preamble Acquisition Chunk (PAC) size
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "defmt", derive(Format))]
pub enum PacSize {
    /// 8 symbols
    Symbols8 = 8,
    /// 16 symbols
    Symbols16 = 16,
    /// 32 symbols
    Symbols32 = 32,
    /// 64 symbols
    Symbols64 = 64,
}

/// RF channel selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "defmt", derive(Format))]
pub enum Channel {
    /// Channel 1 (3.5 GHz)
    Channel1 = 1,
    /// Channel 2 (4.0 GHz)
    Channel2 = 2,
    /// Channel 3 (4.5 GHz)
    Channel3 = 3,
    /// Channel 4 (4.0 GHz)
    Channel4 = 4,
    /// Channel 5 (6.5 GHz)
    Channel5 = 5,
    /// Channel 7 (6.5 GHz)
    Channel7 = 7,
}

/// Preamble codes for different PRF settings
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "defmt", derive(Format))]
pub enum PreambleCode {
    /// Preamble code 1 (16 MHz PRF)
    Code1 = 1,
    /// Preamble code 2 (16 MHz PRF)
    Code2 = 2,
    /// Preamble code 3 (16 MHz PRF)
    Code3 = 3,
    /// Preamble code 4 (16 MHz PRF)
    Code4 = 4,
    /// Preamble code 5 (16 MHz PRF)
    Code5 = 5,
    /// Preamble code 6 (16 MHz PRF)
    Code6 = 6,
    /// Preamble code 7 (16 MHz PRF)
    Code7 = 7,
    /// Preamble code 8 (16 MHz PRF)
    Code8 = 8,
    /// Preamble code 9 (64 MHz PRF)
    Code9 = 9,
    /// Preamble code 10 (64 MHz PRF)
    Code10 = 10,
    /// Preamble code 11 (64 MHz PRF)
    Code11 = 11,
    /// Preamble code 12 (64 MHz PRF)
    Code12 = 12,
    /// Preamble code 17 (64 MHz PRF)
    Code17 = 17,
    /// Preamble code 18 (64 MHz PRF)
    Code18 = 18,
    /// Preamble code 19 (64 MHz PRF)
    Code19 = 19,
    /// Preamble code 20 (64 MHz PRF)
    Code20 = 20,
}

impl PreambleCode {
    /// Returns the raw preamble-code value expected by the DW1000.
    pub const fn raw(self) -> u8 {
        self as u8
    }
}

/// Frame length mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "defmt", derive(Format))]
pub enum FrameLength {
    /// Normal frame length (up to 127 bytes)
    Normal = 0x00,
    /// Extended frame length (up to 1023 bytes)
    Extended = 0x03,
}

/// Pre-defined operation modes combining data rate, PRF, and preamble length
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "defmt", derive(Format))]
pub enum OperatingMode {
    /// Long data range, low power (110 kbps, 16 MHz PRF, 2048 preamble)
    LongDataRangeLowPower,
    /// Short data fast, low power (6.8 Mbps, 16 MHz PRF, 128 preamble)
    ShortDataFastLowPower,
    /// Long data fast, low power (6.8 Mbps, 16 MHz PRF, 1024 preamble)
    LongDataFastLowPower,
    /// Short data fast, accuracy (6.8 Mbps, 64 MHz PRF, 128 preamble)
    ShortDataFastAccuracy,
    /// Long data fast, accuracy (6.8 Mbps, 64 MHz PRF, 1024 preamble)
    LongDataFastAccuracy,
    /// Long data range, accuracy (110 kbps, 64 MHz PRF, 2048 preamble)
    LongDataRangeAccuracy,
}

impl OperatingMode {
    /// Returns the data rate, pulse frequency, and preamble length for this mode
    pub const fn config(&self) -> (DataRate, PulseFrequency, PreambleLength) {
        match self {
            OperatingMode::LongDataRangeLowPower => (
                DataRate::Kbps110,
                PulseFrequency::Mhz16,
                PreambleLength::Symbols2048,
            ),
            OperatingMode::ShortDataFastLowPower => (
                DataRate::Mbps6800,
                PulseFrequency::Mhz16,
                PreambleLength::Symbols128,
            ),
            OperatingMode::LongDataFastLowPower => (
                DataRate::Mbps6800,
                PulseFrequency::Mhz16,
                PreambleLength::Symbols1024,
            ),
            OperatingMode::ShortDataFastAccuracy => (
                DataRate::Mbps6800,
                PulseFrequency::Mhz64,
                PreambleLength::Symbols128,
            ),
            OperatingMode::LongDataFastAccuracy => (
                DataRate::Mbps6800,
                PulseFrequency::Mhz64,
                PreambleLength::Symbols1024,
            ),
            OperatingMode::LongDataRangeAccuracy => (
                DataRate::Kbps110,
                PulseFrequency::Mhz64,
                PreambleLength::Symbols2048,
            ),
        }
    }
}

/// Clock selection for the DW1000
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "defmt", derive(Format))]
pub enum ClockMode {
    /// Automatic clock selection
    Auto = 0x00,
    /// External crystal oscillator
    Xti = 0x01,
    /// PLL clock
    Pll = 0x02,
}
impl ClockMode {
    /// Converts a u8 to ClockMode, defaulting to Auto for invalid values
    pub fn from_u8(value: u8) -> Self {
        match value {
            0x01 => ClockMode::Xti,
            0x02 => ClockMode::Pll,
            _ => ClockMode::Auto,
        }
    }

    /// Converts ClockMode to u8
    pub fn to_u8(&self) -> u8 {
        *self as u8
    }
}

/// Range bias correction tables for different configurations
pub struct RangeBias;

impl RangeBias {
    /// Range bias zero offset for 500 MHz band, 16 MHz PRF
    pub const BIAS_500_16_ZERO: usize = 10;
    /// Range bias zero offset for 500 MHz band, 64 MHz PRF
    pub const BIAS_500_64_ZERO: usize = 8;
    /// Range bias zero offset for 900 MHz band, 16 MHz PRF
    pub const BIAS_900_16_ZERO: usize = 7;
    /// Range bias zero offset for 900 MHz band, 64 MHz PRF
    pub const BIAS_900_64_ZERO: usize = 7;

    /// Range bias table for 500 MHz band, 16 MHz PRF (in mm, -61 to -95 dBm)
    pub const BIAS_500_16: [u8; 18] = [
        198, 187, 179, 163, 143, 127, 109, 84, 59, 31, 0, 36, 65, 84, 97, 106, 110, 112,
    ];

    /// Range bias table for 500 MHz band, 64 MHz PRF (in mm, -61 to -95 dBm)
    pub const BIAS_500_64: [u8; 18] = [
        110, 105, 100, 93, 82, 69, 51, 27, 0, 21, 35, 42, 49, 62, 71, 76, 81, 86,
    ];

    /// Range bias table for 900 MHz band, 16 MHz PRF (in 2mm units, -61 to -95 dBm)
    pub const BIAS_900_16: [u8; 18] = [
        137, 122, 105, 88, 69, 47, 25, 0, 21, 48, 79, 105, 127, 147, 160, 169, 178, 197,
    ];

    /// Range bias table for 900 MHz band, 64 MHz PRF (in 2mm units, -61 to -95 dBm)
    pub const BIAS_900_64: [u8; 18] = [
        147, 133, 117, 99, 75, 50, 29, 0, 24, 45, 63, 76, 87, 98, 116, 122, 132, 142,
    ];
}

/// Configuration builder for the DW1000
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "defmt", derive(Format))]
pub struct DW1000Configuration {
    /// Data transmission rate
    pub data_rate: DataRate,
    /// Pulse repetition frequency
    pub pulse_frequency: PulseFrequency,
    /// Preamble length
    pub preamble_length: PreambleLength,
    /// RF channel
    pub channel: Channel,
    /// Preamble code
    pub preamble_code: PreambleCode,
    /// PAC size
    pub pac_size: PacSize,
    /// Frame length mode
    pub frame_length: FrameLength,
    /// Smart power control
    pub smart_power: bool,
    /// Suppress frame check sequence
    pub suppress_frame_check: bool,
    /// Receiver auto re-enable
    pub receiver_auto_reenable: bool,
    /// Interrupt polarity (true = active high)
    pub interrupt_polarity: bool,
}

impl Default for DW1000Configuration {
    fn default() -> Self {
        // Default is LongDataRangeLowPower mode
        let (data_rate, pulse_frequency, preamble_length) =
            OperatingMode::LongDataRangeLowPower.config();

        Self {
            data_rate,
            pulse_frequency,
            preamble_length,
            channel: Channel::Channel5,
            preamble_code: PreambleCode::Code4,
            pac_size: PacSize::Symbols8,
            frame_length: FrameLength::Normal,
            smart_power: false,
            receiver_auto_reenable: true,
            suppress_frame_check: false,
            interrupt_polarity: true,
        }
    }
}

impl DW1000Configuration {
    /// Creates a new configuration with default settings
    pub const fn new() -> Self {
        Self {
            data_rate: DataRate::Kbps110,
            pulse_frequency: PulseFrequency::Mhz16,
            preamble_length: PreambleLength::Symbols2048,
            channel: Channel::Channel5,
            preamble_code: PreambleCode::Code4,
            pac_size: PacSize::Symbols8,
            frame_length: FrameLength::Normal,
            smart_power: false,
            receiver_auto_reenable: true,
            suppress_frame_check: false,
            interrupt_polarity: true,
        }
    }

    /// Creates a configuration from an operating mode
    pub fn from_mode(mode: OperatingMode) -> Self {
        let (data_rate, pulse_frequency, preamble_length) = mode.config();
        Self {
            data_rate,
            pulse_frequency,
            preamble_length,
            ..Default::default()
        }
    }

    /// Sets the data rate
    pub const fn with_data_rate(mut self, rate: DataRate) -> Self {
        self.data_rate = rate;
        self
    }

    /// Sets the pulse frequency
    pub const fn with_pulse_frequency(mut self, freq: PulseFrequency) -> Self {
        self.pulse_frequency = freq;
        self
    }

    /// Sets the preamble length
    pub const fn with_preamble_length(mut self, length: PreambleLength) -> Self {
        self.preamble_length = length;
        self
    }

    /// Sets the channel
    pub const fn with_channel(mut self, channel: Channel) -> Self {
        self.channel = channel;
        self
    }

    /// Sets the preamble code
    pub const fn with_preamble_code(mut self, code: PreambleCode) -> Self {
        self.preamble_code = code;
        self
    }

    /// Sets smart power control
    pub const fn with_smart_power(mut self, enabled: bool) -> Self {
        self.smart_power = enabled;
        self
    }

    /// Sets receiver auto re-enable
    pub const fn with_receiver_auto_reenable(mut self, enabled: bool) -> Self {
        self.receiver_auto_reenable = enabled;
        self
    }
}

/// Driver configuration validation errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "defmt", derive(Format))]
pub enum ConfigError {
    /// The requested preamble length cannot be used with the selected PHY setup.
    UnsupportedPreambleLength,
    /// The selected preamble code cannot be used with the selected pulse frequency.
    InvalidPreambleCode,
}

/// Addressing configuration written into the DW1000.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "defmt", derive(Format))]
pub struct AddressConfig {
    /// Local device identity.
    pub identity: DeviceIdentity,
}

/// User-provided PHY configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "defmt", derive(Format))]
pub struct PhyConfig {
    /// Data rate.
    pub data_rate: DataRate,
    /// Pulse repetition frequency.
    pub pulse_frequency: PulseFrequency,
    /// Preamble length.
    pub preamble_length: PreambleLength,
    /// RF channel.
    pub channel: Channel,
    /// Optional explicit preamble code.
    pub preamble_code: Option<PreambleCode>,
    /// Smart power control.
    pub smart_power: bool,
}

/// Driver-ready PHY configuration with all derived fields resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "defmt", derive(Format))]
pub struct ValidatedPhyConfig {
    /// Data rate.
    pub data_rate: DataRate,
    /// Pulse repetition frequency.
    pub pulse_frequency: PulseFrequency,
    /// Preamble length.
    pub preamble_length: PreambleLength,
    /// RF channel.
    pub channel: Channel,
    /// Resolved preamble code.
    pub preamble_code: PreambleCode,
    /// Resolved PAC size.
    pub pac_size: PacSize,
    /// Smart power control.
    pub smart_power: bool,
}

impl PhyConfig {
    fn validated(self) -> Result<ValidatedPhyConfig, ConfigError> {
        let preamble_code = self
            .preamble_code
            .unwrap_or(default_preamble_code(self.pulse_frequency));
        if !preamble_code_matches_pulse_frequency(preamble_code, self.pulse_frequency) {
            return Err(ConfigError::InvalidPreambleCode);
        }

        Ok(ValidatedPhyConfig {
            data_rate: self.data_rate,
            pulse_frequency: self.pulse_frequency,
            preamble_length: self.preamble_length,
            channel: self.channel,
            preamble_code,
            pac_size: pac_size_for_preamble(self.preamble_length),
            smart_power: self.smart_power,
        })
    }
}

/// Complete radio configuration consumed by the driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "defmt", derive(Format))]
pub struct RadioConfig {
    /// Local address configuration.
    pub address: AddressConfig,
    /// PHY configuration.
    pub phy: PhyConfig,
    /// Symmetric antenna delay.
    pub antenna_delay: AntennaDelay,
    /// Re-enable RX automatically after receive completion.
    pub receiver_auto_reenable: bool,
    /// Interrupt polarity (`true` = active high).
    pub interrupt_polarity_high: bool,
    /// Include/check the IEEE 802.15.4 frame check sequence.
    pub frame_check: bool,
}

impl RadioConfig {
    /// Builds a radio configuration from a logical identity and operating mode.
    pub fn from_mode(identity: DeviceIdentity, mode: OperatingMode) -> Self {
        let (data_rate, pulse_frequency, preamble_length) = mode.config();
        Self {
            address: AddressConfig { identity },
            phy: PhyConfig {
                data_rate,
                pulse_frequency,
                preamble_length,
                channel: Channel::Channel5,
                preamble_code: None,
                smart_power: false,
            },
            antenna_delay: AntennaDelay::LEGACY_DEFAULT,
            receiver_auto_reenable: true,
            interrupt_polarity_high: true,
            frame_check: true,
        }
    }

    /// Validates and resolves derived PHY fields.
    pub fn validated_phy(&self) -> Result<ValidatedPhyConfig, ConfigError> {
        self.phy.validated()
    }
}

/// Receive options for a single DW1000 receive session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "defmt", derive(Format))]
pub struct RxOptions {
    /// Optional delayed start time relative to the current system timestamp.
    pub delayed_time: Option<DwTime>,
    /// Keep the receiver permanently armed.
    pub permanent: bool,
}

/// Transmit options for a single DW1000 frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "defmt", derive(Format))]
pub struct TxOptions {
    /// Optional delayed transmit time relative to the current system timestamp.
    pub delayed_time: Option<DwTime>,
    /// Set the WAIT4RESP bit after transmit.
    pub wait_for_response: bool,
}

const fn default_preamble_code(pulse_frequency: PulseFrequency) -> PreambleCode {
    match pulse_frequency {
        PulseFrequency::Mhz16 => PreambleCode::Code4,
        PulseFrequency::Mhz64 => PreambleCode::Code10,
    }
}

const fn preamble_code_matches_pulse_frequency(
    preamble_code: PreambleCode,
    pulse_frequency: PulseFrequency,
) -> bool {
    matches!(
        (pulse_frequency, preamble_code),
        (
            PulseFrequency::Mhz16,
            PreambleCode::Code1
                | PreambleCode::Code2
                | PreambleCode::Code3
                | PreambleCode::Code4
                | PreambleCode::Code5
                | PreambleCode::Code6
                | PreambleCode::Code7
                | PreambleCode::Code8
        ) | (
            PulseFrequency::Mhz64,
            PreambleCode::Code9
                | PreambleCode::Code10
                | PreambleCode::Code11
                | PreambleCode::Code12
                | PreambleCode::Code17
                | PreambleCode::Code18
                | PreambleCode::Code19
                | PreambleCode::Code20
        )
    )
}

const fn pac_size_for_preamble(preamble_length: PreambleLength) -> PacSize {
    match preamble_length {
        PreambleLength::Symbols64 | PreambleLength::Symbols128 => PacSize::Symbols8,
        PreambleLength::Symbols256 | PreambleLength::Symbols512 => PacSize::Symbols16,
        PreambleLength::Symbols1024 => PacSize::Symbols32,
        PreambleLength::Symbols1536 | PreambleLength::Symbols2048 | PreambleLength::Symbols4096 => {
            PacSize::Symbols64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Channel, ConfigError, DataRate, PhyConfig, PreambleCode, PreambleLength, PulseFrequency,
    };

    fn phy_config(
        pulse_frequency: PulseFrequency,
        preamble_code: Option<PreambleCode>,
    ) -> PhyConfig {
        PhyConfig {
            data_rate: DataRate::Kbps110,
            pulse_frequency,
            preamble_length: PreambleLength::Symbols2048,
            channel: Channel::Channel5,
            preamble_code,
            smart_power: false,
        }
    }

    #[test]
    fn explicit_preamble_code_must_match_pulse_frequency() {
        assert_eq!(
            phy_config(PulseFrequency::Mhz16, Some(PreambleCode::Code4))
                .validated()
                .unwrap()
                .preamble_code,
            PreambleCode::Code4
        );
        assert_eq!(
            phy_config(PulseFrequency::Mhz64, Some(PreambleCode::Code10))
                .validated()
                .unwrap()
                .preamble_code,
            PreambleCode::Code10
        );
        assert_eq!(
            phy_config(PulseFrequency::Mhz16, Some(PreambleCode::Code10)).validated(),
            Err(ConfigError::InvalidPreambleCode)
        );
        assert_eq!(
            phy_config(PulseFrequency::Mhz64, Some(PreambleCode::Code4)).validated(),
            Err(ConfigError::InvalidPreambleCode)
        );
    }

    #[test]
    fn default_preamble_codes_still_resolve_for_each_pulse_frequency() {
        assert_eq!(
            phy_config(PulseFrequency::Mhz16, None)
                .validated()
                .unwrap()
                .preamble_code,
            PreambleCode::Code4
        );
        assert_eq!(
            phy_config(PulseFrequency::Mhz64, None)
                .validated()
                .unwrap()
                .preamble_code,
            PreambleCode::Code10
        );
    }
}
