//! DW1000 register and bit field constants
//!
//! This module contains all register addresses, sub-addresses, bit positions,
//! and length constants for the Decawave DW1000 UWB transceiver.
//!
//! Based on the DW1000 User Manual and the Arduino DW1000 library by Thomas Trojer.

/// Time stamp byte length (40-bit timestamp = 5 bytes)
pub const LEN_STAMP: usize = 5;

/// Device operation modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub enum DeviceMode {
    /// Idle mode - device is not transmitting or receiving
    Idle = 0x00,
    /// Receive mode - device is ready to receive
    Rx = 0x01,
    /// Transmit mode - device is ready to transmit
    Tx = 0x02,
}

/// Used for SPI ready without actual writes
pub const JUNK: u8 = 0x00;

/// No sub-address for register write
pub const NO_SUB: u8 = 0xFF;

// ============================================================================
// Register Addresses
// ============================================================================

/// Device ID register
pub const DEV_ID: u8 = 0x00;
/// Length of DEV_ID register
pub const LEN_DEV_ID: usize = 4;

/// Extended Unique Identifier register
pub const EUI: u8 = 0x01;
/// Length of EUI register
pub const LEN_EUI: usize = 8;

/// PAN identifier and short address register
pub const PANADR: u8 = 0x03;
/// Length of PANADR register
pub const LEN_PANADR: usize = 4;

/// Device configuration register
pub const SYS_CFG: u8 = 0x04;
/// Length of SYS_CFG register
pub const LEN_SYS_CFG: usize = 4;

// SYS_CFG bit positions
/// Frame Filtering Enable
pub const FFEN_BIT: u8 = 0;
/// Frame Filtering Behave as Coordinator
pub const FFBC_BIT: u8 = 1;
/// Frame Filtering Allow Beacon
pub const FFAB_BIT: u8 = 2;
/// Frame Filtering Allow Data
pub const FFAD_BIT: u8 = 3;
/// Frame Filtering Allow Acknowledgment
pub const FFAA_BIT: u8 = 4;
/// Frame Filtering Allow MAC command
pub const FFAM_BIT: u8 = 5;
/// Frame Filtering Allow Reserved
pub const FFAR_BIT: u8 = 6;
/// Disable Double RX Buffer
pub const DIS_DRXB_BIT: u8 = 12;
/// Disable STXP warning
pub const DIS_STXP_BIT: u8 = 18;
/// Host interrupt polarity
pub const HIRQ_POL_BIT: u8 = 9;
/// Receiver Auto-Re-enable
pub const RXAUTR_BIT: u8 = 29;
/// PHR Mode sub-register position
pub const PHR_MODE_SUB: u8 = 16;
/// Length of PHR Mode sub-register
pub const LEN_PHR_MODE_SUB: usize = 2;
/// Receiver Mode 110 kbps
pub const RXM110K_BIT: u8 = 22;

/// Device control register
pub const SYS_CTRL: u8 = 0x0D;
/// Length of SYS_CTRL register
pub const LEN_SYS_CTRL: usize = 4;

// SYS_CTRL bit positions
/// Suppress Frame Check Sequence Transmission
pub const SFCST_BIT: u8 = 0;
/// Transmit Start
pub const TXSTRT_BIT: u8 = 1;
/// Transmit Delayed Send
pub const TXDLYS_BIT: u8 = 2;
/// Transceiver Off
pub const TRXOFF_BIT: u8 = 6;
/// Wait for Response
pub const WAIT4RESP_BIT: u8 = 7;
/// Receiver Enable
pub const RXENAB_BIT: u8 = 8;
/// Receiver Delayed Enable
pub const RXDLYS_BIT: u8 = 9;

/// System event status register
pub const SYS_STATUS: u8 = 0x0F;
/// Length of SYS_STATUS register
pub const LEN_SYS_STATUS: usize = 5;

// SYS_STATUS bit positions
/// Clock PLL Lock
pub const CPLOCK_BIT: u8 = 1;
/// Automatic Acknowledge Trigger
pub const AAT_BIT: u8 = 3;
/// Transmit Frame Begins
pub const TXFRB_BIT: u8 = 4;
/// Transmit Preamble Sent
pub const TXPRS_BIT: u8 = 5;
/// Transmit PHY Header Sent
pub const TXPHS_BIT: u8 = 6;
/// Transmit Frame Sent
pub const TXFRS_BIT: u8 = 7;
/// LDE Processing Done
pub const LDEDONE_BIT: u8 = 10;
/// Receiver PHY Header Error
pub const RXPHE_BIT: u8 = 12;
/// Receiver Data Frame Ready
pub const RXDFR_BIT: u8 = 13;
/// Receiver Frame Control Good
pub const RXFCG_BIT: u8 = 14;
/// Receiver Frame Control Error
pub const RXFCE_BIT: u8 = 15;
/// Receiver Reed Solomon Frame Sync Loss
pub const RXRFSL_BIT: u8 = 16;
/// Receiver Frame Wait Timeout
pub const RXRFTO_BIT: u8 = 17;
/// LDE Error
pub const LDEERR_BIT: u8 = 18;
/// Receiver Preamble Timeout
pub const RXPTO_BIT: u8 = 21;
/// RF PLL Losing Lock
pub const RFPLL_LL_BIT: u8 = 24;
/// Clock PLL Losing Lock
pub const CLKPLL_LL_BIT: u8 = 25;
/// Receiver SFD Timeout
pub const RXSFDTO_BIT: u8 = 26;

/// System event mask register (uses SYS_STATUS bit definitions below 32)
pub const SYS_MASK: u8 = 0x0E;
/// Length of SYS_MASK register
pub const LEN_SYS_MASK: usize = 4;

/// System time counter
pub const SYS_TIME: u8 = 0x06;
/// Length of SYS_TIME register
pub const LEN_SYS_TIME: usize = LEN_STAMP;

/// RX timestamp register
pub const RX_TIME: u8 = 0x15;
/// Length of RX_TIME register
pub const LEN_RX_TIME: usize = 14;
/// RX timestamp sub-register
pub const RX_STAMP_SUB: u8 = 0x00;
/// First Path Amplitude point 1 sub-register
pub const FP_AMPL1_SUB: u8 = 0x07;
/// Length of RX timestamp
pub const LEN_RX_STAMP: usize = LEN_STAMP;
/// Length of FP_AMPL1
pub const LEN_FP_AMPL1: usize = 2;

/// RX frame quality register
pub const RX_FQUAL: u8 = 0x12;
/// Length of RX_FQUAL register
pub const LEN_RX_FQUAL: usize = 8;
/// Standard Deviation of Noise sub-register
pub const STD_NOISE_SUB: u8 = 0x00;
/// First Path Amplitude point 2 sub-register
pub const FP_AMPL2_SUB: u8 = 0x02;
/// First Path Amplitude point 3 sub-register
pub const FP_AMPL3_SUB: u8 = 0x04;
/// Channel Impulse Response Power sub-register
pub const CIR_PWR_SUB: u8 = 0x06;
/// Length of STD_NOISE
pub const LEN_STD_NOISE: usize = 2;
/// Length of FP_AMPL2
pub const LEN_FP_AMPL2: usize = 2;
/// Length of FP_AMPL3
pub const LEN_FP_AMPL3: usize = 2;
/// Length of CIR_PWR
pub const LEN_CIR_PWR: usize = 2;

/// TX timestamp register
pub const TX_TIME: u8 = 0x17;
/// Length of TX_TIME register
pub const LEN_TX_TIME: usize = 10;
/// TX timestamp sub-register
pub const TX_STAMP_SUB: u8 = 0;
/// Length of TX timestamp
pub const LEN_TX_STAMP: usize = LEN_STAMP;

/// Timing register (for delayed RX/TX)
pub const DX_TIME: u8 = 0x0A;
/// Length of DX_TIME register
pub const LEN_DX_TIME: usize = LEN_STAMP;

/// Transmit data buffer
pub const TX_BUFFER: u8 = 0x09;
/// Length of TX_BUFFER
pub const LEN_TX_BUFFER: usize = 1024;
/// Length of standard UWB frames
pub const LEN_UWB_FRAMES: usize = 127;
/// Length of extended UWB frames
pub const LEN_EXT_UWB_FRAMES: usize = 1023;

/// RX frame info register
pub const RX_FINFO: u8 = 0x10;
/// Length of RX_FINFO register
pub const LEN_RX_FINFO: usize = 4;

/// Receive data buffer
pub const RX_BUFFER: u8 = 0x11;
/// Length of RX_BUFFER
pub const LEN_RX_BUFFER: usize = 1024;

/// Transmit control register
pub const TX_FCTRL: u8 = 0x08;
/// Length of TX_FCTRL register
pub const LEN_TX_FCTRL: usize = 5;

/// Channel control register
pub const CHAN_CTRL: u8 = 0x1F;
/// Length of CHAN_CTRL register
pub const LEN_CHAN_CTRL: usize = 4;
/// Decawave proprietary SFD enable
pub const DWSFD_BIT: u8 = 17;
/// Non-standard SFD in TX
pub const TNSSFD_BIT: u8 = 20;
/// Non-standard SFD in RX
pub const RNSSFD_BIT: u8 = 21;

/// User-defined SFD register
pub const USR_SFD: u8 = 0x21;
/// Length of USR_SFD register
pub const LEN_USR_SFD: usize = 41;
/// SFD length sub-register
pub const SFD_LENGTH_SUB: u8 = 0x00;
/// Length of SFD_LENGTH sub-register
pub const LEN_SFD_LENGTH: usize = 1;

// ============================================================================
// OTP (One Time Programmable) Memory Interface
// ============================================================================

/// OTP control register (for LDE micro code loading only)
pub const OTP_IF: u8 = 0x2D;
/// OTP address sub-register
pub const OTP_ADDR_SUB: u8 = 0x04;
/// OTP control sub-register
pub const OTP_CTRL_SUB: u8 = 0x06;
/// OTP read data sub-register
pub const OTP_RDAT_SUB: u8 = 0x0A;
/// Length of OTP_ADDR
pub const LEN_OTP_ADDR: usize = 2;
/// Length of OTP_CTRL
pub const LEN_OTP_CTRL: usize = 2;
/// Length of OTP_RDAT
pub const LEN_OTP_RDAT: usize = 4;

// ============================================================================
// AGC (Automatic Gain Control) Configuration
// ============================================================================

/// AGC tune register
pub const AGC_TUNE: u8 = 0x23;
/// AGC_TUNE1 sub-register
pub const AGC_TUNE1_SUB: u8 = 0x04;
/// AGC_TUNE2 sub-register
pub const AGC_TUNE2_SUB: u8 = 0x0C;
/// AGC_TUNE3 sub-register
pub const AGC_TUNE3_SUB: u8 = 0x12;
/// Length of AGC_TUNE1
pub const LEN_AGC_TUNE1: usize = 2;
/// Length of AGC_TUNE2
pub const LEN_AGC_TUNE2: usize = 4;
/// Length of AGC_TUNE3
pub const LEN_AGC_TUNE3: usize = 2;

// ============================================================================
// DRX (Digital Receiver) Configuration
// ============================================================================

/// DRX tune register
pub const DRX_TUNE: u8 = 0x27;
/// DRX_TUNE0b sub-register
pub const DRX_TUNE0B_SUB: u8 = 0x02;
/// DRX_TUNE1a sub-register
pub const DRX_TUNE1A_SUB: u8 = 0x04;
/// DRX_TUNE1b sub-register
pub const DRX_TUNE1B_SUB: u8 = 0x06;
/// DRX_TUNE2 sub-register
pub const DRX_TUNE2_SUB: u8 = 0x08;
/// DRX_TUNE4H sub-register
pub const DRX_TUNE4H_SUB: u8 = 0x26;
/// Length of DRX_TUNE0b
pub const LEN_DRX_TUNE0B: usize = 2;
/// Length of DRX_TUNE1a
pub const LEN_DRX_TUNE1A: usize = 2;
/// Length of DRX_TUNE1b
pub const LEN_DRX_TUNE1B: usize = 2;
/// Length of DRX_TUNE2
pub const LEN_DRX_TUNE2: usize = 4;
/// Length of DRX_TUNE4H
pub const LEN_DRX_TUNE4H: usize = 2;

// ============================================================================
// LDE (Leading Edge Detection) Interface
// ============================================================================

/// LDE interface register
pub const LDE_IF: u8 = 0x2E;
/// LDE_CFG1 sub-register
pub const LDE_CFG1_SUB: u16 = 0x0806;
/// LDE_RXANTD sub-register
pub const LDE_RXANTD_SUB: u16 = 0x1804;
/// LDE_CFG2 sub-register
pub const LDE_CFG2_SUB: u16 = 0x1806;
/// LDE_REPC sub-register
pub const LDE_REPC_SUB: u16 = 0x2804;
/// Length of LDE_CFG1
pub const LEN_LDE_CFG1: usize = 1;
/// Length of LDE_CFG2
pub const LEN_LDE_CFG2: usize = 2;
/// Length of LDE_REPC
pub const LEN_LDE_REPC: usize = 2;
/// Length of LDE_RXANTD
pub const LEN_LDE_RXANTD: usize = 2;

// ============================================================================
// TX Power Configuration
// ============================================================================

/// TX power register
pub const TX_POWER: u8 = 0x1E;
/// Length of TX_POWER register
pub const LEN_TX_POWER: usize = 4;

// ============================================================================
// RF Configuration
// ============================================================================

/// RF configuration register
pub const RF_CONF: u8 = 0x28;
/// RF_RXCTRLH sub-register
pub const RF_RXCTRLH_SUB: u8 = 0x0B;
/// RF_TXCTRL sub-register
pub const RF_TXCTRL_SUB: u8 = 0x0C;
/// Length of RF_RXCTRLH
pub const LEN_RF_RXCTRLH: usize = 1;
/// Length of RF_TXCTRL
pub const LEN_RF_TXCTRL: usize = 4;

// ============================================================================
// TX Calibration
// ============================================================================

/// TX calibration register
pub const TX_CAL: u8 = 0x2A;
/// Transmitter Calibration - Pulse Generator Delay sub-register
pub const TC_PGDELAY_SUB: u8 = 0x0B;
/// Length of TC_PGDELAY
pub const LEN_TC_PGDELAY: usize = 1;
/// Transmitter Calibration - SAR Control
pub const TC_SARC: u8 = 0x00;
/// Transmitter Calibration - SAR Latch
pub const TC_SARL: u8 = 0x03;

// ============================================================================
// Frequency Synthesizer Control
// ============================================================================

/// Frequency Synthesizer control register
pub const FS_CTRL: u8 = 0x2B;
/// FS_PLLCFG sub-register
pub const FS_PLLCFG_SUB: u8 = 0x07;
/// FS_PLLTUNE sub-register
pub const FS_PLLTUNE_SUB: u8 = 0x0B;
/// FS_XTALT sub-register (Crystal Trim)
pub const FS_XTALT_SUB: u8 = 0x0E;
/// Length of FS_PLLCFG
pub const LEN_FS_PLLCFG: usize = 4;
/// Length of FS_PLLTUNE
pub const LEN_FS_PLLTUNE: usize = 1;
/// Length of FS_XTALT
pub const LEN_FS_XTALT: usize = 1;

// ============================================================================
// Always-On (AON) Memory Interface
// ============================================================================

/// Always-On register
pub const AON: u8 = 0x2C;
/// AON Wakeup Configuration sub-register
pub const AON_WCFG_SUB: u8 = 0x00;
/// Length of AON_WCFG
pub const LEN_AON_WCFG: usize = 2;
/// On-Wake Load Configurations from AON
pub const ONW_LDC_BIT: u8 = 6;
/// On-Wake Load the LDO tune value
pub const ONW_LDD0_BIT: u8 = 12;
/// AON Control sub-register
pub const AON_CTRL_SUB: u8 = 0x02;
/// Length of AON_CTRL
pub const LEN_AON_CTRL: usize = 1;
/// Restore bit
pub const RESTORE_BIT: u8 = 0;
/// Save bit
pub const SAVE_BIT: u8 = 1;
/// Upload Configuration bit
pub const UPL_CFG_BIT: u8 = 2;
/// AON Configuration 0 sub-register
pub const AON_CFG0_SUB: u8 = 0x06;
/// Length of AON_CFG0
pub const LEN_AON_CFG0: usize = 4;
/// Sleep Enable bit
pub const SLEEP_EN_BIT: u8 = 0;
/// Wake using WAKEUP pin
pub const WAKE_PIN_BIT: u8 = 1;
/// Wake using SPI access
pub const WAKE_SPI_BIT: u8 = 2;
/// Wake when sleep counter elapses
pub const WAKE_CNT_BIT: u8 = 3;

// ============================================================================
// Power Management and System Control (PMSC)
// ============================================================================

/// PMSC register
pub const PMSC: u8 = 0x36;
/// PMSC_CTRL0 sub-register
pub const PMSC_CTRL0_SUB: u8 = 0x00;
/// PMSC_CTRL1 sub-register
pub const PMSC_CTRL1_SUB: u8 = 0x04;
/// PMSC_LEDC sub-register (LED Control)
pub const PMSC_LEDC_SUB: u8 = 0x28;
/// Length of PMSC_CTRL0
pub const LEN_PMSC_CTRL0: usize = 4;
/// Length of PMSC_CTRL1
pub const LEN_PMSC_CTRL1: usize = 4;
/// Length of PMSC_LEDC
pub const LEN_PMSC_LEDC: usize = 4;
/// GPIO De-bounce Clock Enable
pub const GPDCE_BIT: u8 = 18;
/// Kilohertz Clock Enable
pub const KHZCLKEN_BIT: u8 = 23;
/// Blink Enable
pub const BLNKEN: u8 = 8;
/// After Transmit Sleep bit
pub const ATXSLP_BIT: u8 = 11;
/// After Receive Sleep bit
pub const ARXSLP_BIT: u8 = 12;

// ============================================================================
// TX Antenna Delay
// ============================================================================

/// TX antenna delay register
pub const TX_ANTD: u8 = 0x18;
/// Length of TX_ANTD register
pub const LEN_TX_ANTD: usize = 2;

// ============================================================================
// GPIO Control
// ============================================================================

/// GPIO control register
pub const GPIO_CTRL: u8 = 0x26;
/// GPIO mode sub-register
pub const GPIO_MODE_SUB: u8 = 0x00;
/// Length of GPIO_MODE
pub const LEN_GPIO_MODE: usize = 4;

// GPIO pin message mode positions
/// Message mode GPIO 0
pub const MSGP0: u8 = 6;
/// Message mode GPIO 1
pub const MSGP1: u8 = 8;
/// Message mode GPIO 2
pub const MSGP2: u8 = 10;
/// Message mode GPIO 3
pub const MSGP3: u8 = 12;
/// Message mode GPIO 4
pub const MSGP4: u8 = 14;
/// Message mode GPIO 5
pub const MSGP5: u8 = 16;
/// Message mode GPIO 6
pub const MSGP6: u8 = 18;
/// Message mode GPIO 7
pub const MSGP7: u8 = 20;
/// Message mode GPIO 8
pub const MSGP8: u8 = 22;

/// GPIO mode value
pub const GPIO_MODE: u8 = 0;
/// LED mode value
pub const LED_MODE: u8 = 1;

/// First IEEE 802.15.4 frame-control byte used by standard ranging frames.
pub const FC_1: u8 = 0x41;
/// First frame-control byte used by DW1000 blink frames.
pub const FC_1_BLINK: u8 = 0xC5;
/// Second IEEE 802.15.4 frame-control byte for long addressing.
pub const FC_2: u8 = 0x8C;
/// Second IEEE 802.15.4 frame-control byte for short addressing.
pub const FC_2_SHORT: u8 = 0x88;

/// Little-endian low byte of the default PAN ID.
pub const PAN_ID_1: u8 = 0xCA;
/// Little-endian high byte of the default PAN ID.
pub const PAN_ID_2: u8 = 0xDE;

/// Serialized MAC header length when using short addresses.
pub const SHORT_MAC_LEN: u8 = 9;
/// Serialized MAC header length when using extended addresses.
pub const LONG_MAC_LEN: u8 = 15;
