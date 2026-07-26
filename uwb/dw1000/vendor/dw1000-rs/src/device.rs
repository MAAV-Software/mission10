//! Typed DW1000-facing data structures shared by the driver and ranging code.

use core::ops::{BitOr, BitOrAssign};

#[cfg(feature = "defmt")]
use defmt::Format;

use crate::time::DwTime;

/// Raw 40-bit system deadline written to the DW1000 `DX_TIME` register.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "defmt", derive(Format))]
pub struct DxTimeDeadline(DwTime);

impl DxTimeDeadline {
    /// Creates a raw deadline for a radio implementation or test double.
    pub const fn new(value: DwTime) -> Self {
        Self(value)
    }

    /// Returns the raw DW1000 deadline.
    pub const fn value(self) -> DwTime {
        self.0
    }
}

/// Antenna-adjusted 40-bit RMARKER timestamp for a transmitted frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "defmt", derive(Format))]
pub struct OnAirTimestamp(DwTime);

impl OnAirTimestamp {
    /// Creates an on-air timestamp for a radio implementation or test double.
    pub const fn new(value: DwTime) -> Self {
        Self(value)
    }

    /// Returns the antenna-adjusted DW1000 timestamp.
    pub const fn value(self) -> DwTime {
        self.0
    }
}

/// One delayed-transmit schedule in both DW1000 time domains.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "defmt", derive(Format))]
pub struct DelayedTransmit {
    deadline: DxTimeDeadline,
    timestamp: OnAirTimestamp,
}

impl DelayedTransmit {
    /// Creates a schedule from a raw deadline and its corresponding RMARKER timestamp.
    pub const fn new(deadline: DxTimeDeadline, timestamp: OnAirTimestamp) -> Self {
        Self {
            deadline,
            timestamp,
        }
    }

    /// Returns the raw deadline programmed into `DX_TIME`.
    pub const fn deadline(self) -> DxTimeDeadline {
        self.deadline
    }

    /// Returns the RMARKER timestamp to place in a ranging frame.
    pub const fn timestamp(self) -> OnAirTimestamp {
        self.timestamp
    }
}

/// 16-bit PAN identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "defmt", derive(Format))]
pub struct PanId(u16);

impl PanId {
    /// Creates a PAN ID from the raw 16-bit value.
    pub const fn new(raw: u16) -> Self {
        Self(raw)
    }

    /// Returns the raw 16-bit value.
    pub const fn raw(self) -> u16 {
        self.0
    }

    /// Returns the value encoded in little-endian byte order.
    pub const fn to_le_bytes(self) -> [u8; 2] {
        self.0.to_le_bytes()
    }
}

/// 16-bit short address.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "defmt", derive(Format))]
pub struct ShortAddress(u16);

impl ShortAddress {
    /// Broadcast short address.
    pub const BROADCAST: Self = Self(0xFFFF);

    /// Creates a short address from the raw 16-bit value.
    pub const fn new(raw: u16) -> Self {
        Self(raw)
    }

    /// Returns the raw 16-bit value.
    pub const fn raw(self) -> u16 {
        self.0
    }

    /// Returns `true` if this is the broadcast address.
    pub const fn is_broadcast(self) -> bool {
        self.0 == Self::BROADCAST.0
    }

    /// Returns the value encoded in little-endian byte order.
    pub const fn to_le_bytes(self) -> [u8; 2] {
        self.0.to_le_bytes()
    }
}

/// 64-bit extended unique identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "defmt", derive(Format))]
pub struct Eui64([u8; 8]);

impl Eui64 {
    /// Creates an EUI-64 from raw bytes.
    pub const fn new(bytes: [u8; 8]) -> Self {
        Self(bytes)
    }

    /// Returns the raw bytes in natural order.
    pub const fn raw(self) -> [u8; 8] {
        self.0
    }

    /// Returns the value encoded for the DW1000 EUI register.
    pub fn to_register_bytes(self) -> [u8; 8] {
        let bytes = self.0;
        [
            bytes[7], bytes[6], bytes[5], bytes[4], bytes[3], bytes[2], bytes[1], bytes[0],
        ]
    }
}

/// Logical identity configured into the DW1000.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "defmt", derive(Format))]
pub struct DeviceIdentity {
    /// PAN identifier.
    pub pan_id: PanId,
    /// 16-bit short address.
    pub short_address: ShortAddress,
    /// 64-bit extended unique identifier.
    pub eui: Eui64,
}

impl DeviceIdentity {
    /// Creates a new device identity.
    pub const fn new(pan_id: PanId, short_address: ShortAddress, eui: Eui64) -> Self {
        Self {
            pan_id,
            short_address,
            eui,
        }
    }
}

/// Symmetric DW1000 antenna delay calibration, in device ticks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "defmt", derive(Format))]
pub struct AntennaDelay(u16);

impl AntennaDelay {
    /// Legacy default used by the examples before board-specific calibration.
    pub const LEGACY_DEFAULT: Self = Self(16_456);

    /// Creates a new antenna delay.
    pub const fn new(raw: u16) -> Self {
        Self(raw)
    }

    /// Returns the raw DW1000 tick value.
    pub const fn raw(self) -> u16 {
        self.0
    }

    /// Encodes the delay as a DW1000 timestamp byte array.
    pub fn to_time_bytes(self) -> [u8; 5] {
        DwTime::from_ticks(self.0 as i64).to_bytes()
    }
}

/// Signal-quality measurements derived from the last received frame.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "defmt", derive(Format))]
pub struct SignalMetrics {
    /// Estimated receive power in dBm.
    pub receive_power_dbm: f32,
    /// Estimated first-path power in dBm.
    pub first_path_power_dbm: f32,
    /// Driver-defined receive quality metric.
    pub quality: f32,
}

/// Collection of recent DW1000 timestamps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "defmt", derive(Format))]
pub struct Timestamps {
    /// Last transmit timestamp.
    pub tx: DwTime,
    /// Last receive timestamp.
    pub rx: DwTime,
    /// Current system timestamp.
    pub system: DwTime,
}

/// Typed `SYS_STATUS` bit mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "defmt", derive(Format))]
pub struct SysStatus(pub u64);

impl SysStatus {
    /// Empty status mask.
    pub const EMPTY: Self = Self(0);

    /// Returns `true` when all bits in `other` are set in `self`.
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl BitOr for SysStatus {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for SysStatus {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// Raw radio registers useful when diagnosing receiver state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "defmt", derive(Format))]
pub struct RadioDebugState {
    /// System event status.
    pub sys_status: SysStatus,
    /// System state machine state, stored in the low 40 bits.
    pub sys_state: u64,
    /// System control register.
    pub sys_ctrl: u32,
    /// System configuration register.
    pub sys_cfg: u32,
    /// Interrupt mask register.
    pub sys_mask: u32,
    /// Power-management and system-clocking control register.
    pub pmsc_ctrl0: u32,
}

/// Borrowed view of a received frame.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "defmt", derive(Format))]
pub struct RxFrame<'a> {
    /// Received frame bytes, excluding the FCS when frame checking is enabled.
    pub bytes: &'a [u8],
    /// Receive timestamp.
    pub timestamp: DwTime,
    /// Signal metrics derived from the frame.
    pub metrics: SignalMetrics,
    /// Raw status bits observed while processing the frame.
    pub status: SysStatus,
}

/// Internal peer state tracked by the ranging state machine.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "defmt", derive(Format))]
pub struct Peer {
    /// Optional peer EUI, available once discovered via blink.
    pub eui: Option<Eui64>,
    /// Peer short address.
    pub short_address: ShortAddress,
    /// Last activity timestamp in host milliseconds.
    pub last_activity_ms: u32,
    /// Most recent inbound sequence number accepted from this peer.
    pub last_sequence: Option<u8>,
    /// Reply delay selected for this peer.
    pub reply_delay_us: u32,
    /// Ranging timestamps.
    pub poll_sent: DwTime,
    /// Ranging timestamps.
    pub poll_received: DwTime,
    /// Ranging timestamps.
    pub poll_ack_sent: DwTime,
    /// Ranging timestamps.
    pub poll_ack_received: DwTime,
    /// Ranging timestamps.
    pub range_sent: DwTime,
    /// Ranging timestamps.
    pub range_received: DwTime,
    /// Most recent filtered range in meters.
    pub range_m: f32,
    /// Most recent receive metrics.
    pub metrics: SignalMetrics,
}

impl Peer {
    /// Creates a fresh peer record.
    pub const fn new(eui: Option<Eui64>, short_address: ShortAddress) -> Self {
        Self {
            eui,
            short_address,
            last_activity_ms: 0,
            last_sequence: None,
            reply_delay_us: 0,
            poll_sent: DwTime::zero(),
            poll_received: DwTime::zero(),
            poll_ack_sent: DwTime::zero(),
            poll_ack_received: DwTime::zero(),
            range_sent: DwTime::zero(),
            range_received: DwTime::zero(),
            range_m: 0.0,
            metrics: SignalMetrics {
                receive_power_dbm: 0.0,
                first_path_power_dbm: 0.0,
                quality: 0.0,
            },
        }
    }
}

/// User-facing immutable snapshot of a peer.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "defmt", derive(Format))]
pub struct PeerSnapshot {
    /// Peer short address.
    pub short_address: ShortAddress,
    /// Current range estimate in meters.
    pub range_m: f32,
    /// Most recent receive power estimate in dBm.
    pub receive_power_dbm: f32,
    /// Most recent first-path power estimate in dBm.
    pub first_path_power_dbm: f32,
    /// Most recent receive quality metric.
    pub quality: f32,
}

impl From<&Peer> for PeerSnapshot {
    fn from(peer: &Peer) -> Self {
        Self {
            short_address: peer.short_address,
            range_m: peer.range_m,
            receive_power_dbm: peer.metrics.receive_power_dbm,
            first_path_power_dbm: peer.metrics.first_path_power_dbm,
            quality: peer.metrics.quality,
        }
    }
}

impl From<Peer> for PeerSnapshot {
    fn from(peer: Peer) -> Self {
        Self::from(&peer)
    }
}
