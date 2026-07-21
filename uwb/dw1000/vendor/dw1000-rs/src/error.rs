//! Shared error types for the DW1000 driver and ranging state machine.

#[cfg(feature = "defmt")]
use defmt::Format;

use crate::config::ConfigError;

/// Receive-side driver errors derived from `SYS_STATUS`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "defmt", derive(Format))]
pub enum RxError {
    /// No completed RX frame is available to read.
    FrameNotReady,
    /// Leading-edge detection failed.
    LeadingEdgeDetection,
    /// Frame check sequence failed.
    FrameCheck,
    /// PHY header error.
    Header,
    /// Reed-Solomon decoder error.
    ReedSolomon,
    /// Receive timeout.
    Timeout,
}

/// Protocol-level decoding and state-machine errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "defmt", derive(Format))]
pub enum ProtocolError {
    /// The frame contents were malformed.
    InvalidFrame,
    /// The frame kind byte was not recognized.
    UnknownFrameKind,
    /// The output buffer was too small to hold the decoded data.
    BufferTooSmall,
    /// The peer table is full.
    PeerTableFull,
    /// The requested peer is unknown.
    UnknownPeer,
    /// The configured reply delay cannot be represented on the wire.
    ReplyDelayOverflow,
}

/// Top-level crate error.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "defmt", derive(Format))]
pub enum Error<SpiE, PinE> {
    /// SPI transaction error.
    Spi(SpiE),
    /// GPIO error.
    Pin(PinE),
    /// Receive-side radio error.
    Receive(RxError),
    /// Protocol decode or state-machine error.
    Protocol(ProtocolError),
    /// Invalid configuration supplied to the driver.
    InvalidConfig(ConfigError),
    /// The driver operation requires a completed radio configuration.
    NotConfigured,
    /// Caller-provided buffer was too small.
    BufferTooSmall {
        /// Actual buffer length.
        len: usize,
        /// Required length.
        needed: usize,
    },
    /// Frame exceeded the supported size.
    FrameTooLong {
        /// Actual frame length.
        len: usize,
        /// Maximum supported length.
        max: usize,
    },
}

impl<SpiE, PinE> From<ProtocolError> for Error<SpiE, PinE> {
    fn from(value: ProtocolError) -> Self {
        Self::Protocol(value)
    }
}

impl<SpiE, PinE> From<ConfigError> for Error<SpiE, PinE> {
    fn from(value: ConfigError) -> Self {
        Self::InvalidConfig(value)
    }
}
