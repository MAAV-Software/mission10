//! DW1000 timestamp handling module
//!
//! This module provides the `DW1000Time` type for handling timestamps from the
//! DW1000 UWB transceiver. The DW1000 uses 40-bit timestamps where each bit
//! represents approximately 15.65 picoseconds.

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

#[cfg(feature = "defmt")]
use defmt::Format;

/// Time resolution in microseconds of time-based registers/values.
/// Each bit in a timestamp counts for a period of approx. 15.65ps
pub const TIME_RES: f32 = 0.000_015_650_041;

/// Inverse of TIME_RES for faster multiplication instead of division
pub const TIME_RES_INV: f32 = 63897.6;

/// Speed of radio waves [m/s] * timestamp resolution [~15.65ps] of DW1000
pub const DISTANCE_OF_RADIO: f32 = 0.004_691_764;

/// Distance represented by one DW1000 timestamp tick, in meters.
pub const DISTANCE_PER_TICK_M: f32 = DISTANCE_OF_RADIO;

/// Inverse of DISTANCE_OF_RADIO for faster multiplication instead of division
pub const DISTANCE_OF_RADIO_INV: f32 = 213.139_45;

/// Timestamp byte length - 40 bit -> 5 bytes
pub const LENGTH_TIMESTAMP: usize = 5;

/// Timer/counter overflow (40 bits) -> overflow approx. every 17.2 seconds
pub const TIME_OVERFLOW: i64 = 0x10000000000; // 1099511627776

/// Maximum valid timestamp value (40-bit max)
pub const TIME_MAX: i64 = 0xffffffffff;

/// Time factors (relative to microseconds) for setting delayed transceive
/// Time factor for seconds (1 second = 1e6 microseconds)
pub const SECONDS: f32 = 1e6;

/// Time factor for milliseconds (1 millisecond = 1e3 microseconds)
pub const MILLISECONDS: f32 = 1e3;

/// Time factor for microseconds (1 microsecond = 1 microsecond)
pub const MICROSECONDS: f32 = 1.0;

/// Time factor for nanoseconds (1 nanosecond = 1e-3 microseconds)
pub const NANOSECONDS: f32 = 1e-3;

/// Represents a timestamp from the DW1000 UWB transceiver.
///
/// The DW1000 uses 40-bit timestamps where each increment represents
/// approximately 15.65 picoseconds. This structure provides methods
/// to convert between raw timestamps and real-world time units.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "defmt", derive(Format))]
pub struct DW1000Time {
    /// Internal timestamp value (40-bit, but stored as i64 for calculations)
    timestamp: i64,
}

impl DW1000Time {
    /// Creates a new `DW1000Time` with timestamp 0
    #[inline]
    pub const fn new() -> Self {
        Self { timestamp: 0 }
    }

    /// Creates a zero-valued timestamp.
    #[inline]
    pub const fn zero() -> Self {
        Self::new()
    }

    /// Creates a new `DW1000Time` from a raw timestamp value
    ///
    /// # Arguments
    /// * `timestamp` - Raw timestamp where 1 unit ≈ 15.65ps
    #[inline]
    pub fn from_timestamp(timestamp: i64) -> Self {
        let mut time = Self::new();
        time.set_timestamp(timestamp);
        time
    }

    /// Creates a new `DW1000Time` from raw DW1000 ticks.
    #[inline]
    pub const fn from_ticks(ticks: i64) -> Self {
        Self { timestamp: ticks }
    }

    /// Creates a new `DW1000Time` from a byte array (little-endian)
    ///
    /// # Arguments
    /// * `data` - 5-byte array containing the timestamp
    pub fn from_bytes(data: &[u8; LENGTH_TIMESTAMP]) -> Self {
        let mut time = Self::new();
        time.set_timestamp_from_bytes(data);
        time
    }

    /// Creates a new `DW1000Time` from microseconds
    ///
    /// # Arguments
    /// * `time_us` - Time in microseconds
    #[inline]
    pub fn from_microseconds(time_us: f32) -> Self {
        let mut time = Self::new();
        time.set_time(time_us);
        time
    }

    /// Creates a new `DW1000Time` from microseconds.
    #[inline]
    pub fn from_micros(time_us: f32) -> Self {
        Self::from_microseconds(time_us)
    }

    /// Creates a new `DW1000Time` from a time value and factor
    ///
    /// # Arguments
    /// * `value` - Time value
    /// * `factor_us` - Multiplication factor to convert to microseconds
    #[inline]
    pub fn from_time_with_factor(value: i32, factor_us: f32) -> Self {
        let mut time = Self::new();
        time.set_time_with_factor(value, factor_us);
        time
    }

    /// Sets the timestamp to a raw value
    ///
    /// # Arguments
    /// * `value` - Raw timestamp where 1 unit ≈ 15.65ps
    #[inline]
    pub fn set_timestamp(&mut self, value: i64) {
        self.timestamp = value;
    }

    /// Sets the timestamp from a byte array (little-endian)
    ///
    /// # Arguments
    /// * `data` - 5-byte array containing the timestamp
    pub fn set_timestamp_from_bytes(&mut self, data: &[u8; LENGTH_TIMESTAMP]) {
        self.timestamp = 0;
        for (i, &byte) in data.iter().enumerate() {
            self.timestamp |= (byte as i64) << (i * 8);
        }
    }

    /// Sets the time in microseconds
    ///
    /// # Arguments
    /// * `time_us` - Time in microseconds
    #[inline]
    pub fn set_time(&mut self, time_us: f32) {
        self.timestamp = (time_us * TIME_RES_INV) as i64;
    }

    /// Sets the time from a value and factor
    ///
    /// # Arguments
    /// * `value` - Time value
    /// * `factor_us` - Multiplication factor to convert to microseconds
    #[inline]
    pub fn set_time_with_factor(&mut self, value: i32, factor_us: f32) {
        self.set_time(value as f32 * factor_us);
    }

    /// Gets the raw timestamp value
    #[inline]
    pub const fn get_timestamp(&self) -> i64 {
        self.timestamp
    }

    /// Gets the timestamp as a byte array (little-endian)
    ///
    /// # Arguments
    /// * `data` - 5-byte array where the timestamp will be written
    pub fn get_timestamp_bytes(&self) -> [u8; LENGTH_TIMESTAMP] {
        let mut data = [0u8; LENGTH_TIMESTAMP];
        for (index, byte) in data.iter_mut().enumerate() {
            *byte = ((self.timestamp >> (index * 8)) & 0xFF) as u8;
        }
        data
    }

    /// Returns the timestamp encoded as the DW1000 40-bit little-endian wire format.
    #[inline]
    pub fn to_bytes(&self) -> [u8; LENGTH_TIMESTAMP] {
        self.get_timestamp_bytes()
    }

    /// Returns the time in microseconds
    #[inline]
    pub fn as_microseconds(&self) -> f32 {
        (self.timestamp % TIME_OVERFLOW) as f32 * TIME_RES
    }

    /// Returns the time as distance in meters (d = c * t)
    ///
    /// This is useful for time-of-flight calculations
    #[inline]
    pub fn as_meters(&self) -> f32 {
        (self.timestamp % TIME_OVERFLOW) as f32 * DISTANCE_OF_RADIO
    }

    /// Computes asymmetric two-way-ranging time of flight from the six protocol timestamps.
    pub fn asymmetric_tof(
        poll_sent: Self,
        poll_received: Self,
        poll_ack_sent: Self,
        poll_ack_received: Self,
        range_sent: Self,
        range_received: Self,
    ) -> Self {
        let round1 = (poll_ack_received - poll_sent).wrapped().timestamp as i128;
        let reply1 = (poll_ack_sent - poll_received).wrapped().timestamp as i128;
        let round2 = (range_received - poll_ack_sent).wrapped().timestamp as i128;
        let reply2 = (range_sent - poll_ack_received).wrapped().timestamp as i128;
        let denominator = round1 + round2 + reply1 + reply2;
        if denominator == 0 {
            return Self::zero();
        }
        Self::from_ticks(((round1 * round2 - reply1 * reply2) / denominator) as i64)
    }

    /// Wraps negative timestamps due to overflow
    ///
    /// Converts negative values that occur due to overflow of one node to the
    /// correct positive value.
    ///
    /// # Example
    /// Maximum timestamp is 1000. Node N1 sends 999 as timestamp. N2 receives
    /// and sends delayed and increased timestamp back. Delay is 10, so timestamp
    /// would be 1009, but due to overflow 009 is sent back.
    /// Now calculate TOF: 009 - 999 = -990 -> incorrect time, so wrap()
    /// Wrap calculation: -990 + 1000 = 10 -> correct time
    #[inline]
    pub fn wrap(&mut self) -> &mut Self {
        if self.timestamp < 0 {
            self.timestamp += TIME_OVERFLOW;
        }
        self
    }

    /// Returns a new wrapped timestamp
    ///
    /// Same as `wrap()` but returns a new instance instead of modifying self
    #[inline]
    pub fn wrapped(mut self) -> Self {
        self.wrap();
        self
    }

    /// Checks if the timestamp is valid for use with the DW1000 device
    ///
    /// Returns `true` if the timestamp is within the valid range [0, TIME_MAX],
    /// `false` if negative or overflow (maybe after calculation)
    #[inline]
    pub const fn is_valid_timestamp(&self) -> bool {
        self.timestamp >= 0 && self.timestamp <= TIME_MAX
    }
}

impl Default for DW1000Time {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Short alias used across the driver and ranging layers.
pub type DwTime = DW1000Time;

// Implement Add operations
impl Add for DW1000Time {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            timestamp: self.timestamp + other.timestamp,
        }
    }
}

impl AddAssign for DW1000Time {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.timestamp += other.timestamp;
    }
}

// Implement Sub operations
impl Sub for DW1000Time {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            timestamp: self.timestamp - other.timestamp,
        }
    }
}

impl SubAssign for DW1000Time {
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        self.timestamp -= other.timestamp;
    }
}

// Implement Mul operations with f32
impl Mul<f32> for DW1000Time {
    type Output = Self;

    #[inline]
    fn mul(self, factor: f32) -> Self {
        Self {
            timestamp: (self.timestamp as f32 * factor) as i64,
        }
    }
}

impl MulAssign<f32> for DW1000Time {
    #[inline]
    fn mul_assign(&mut self, factor: f32) {
        self.timestamp = (self.timestamp as f32 * factor) as i64;
    }
}

// Implement Mul operations with DW1000Time
impl Mul<DW1000Time> for DW1000Time {
    type Output = Self;

    #[inline]
    fn mul(self, factor: DW1000Time) -> Self {
        Self {
            timestamp: self.timestamp * factor.timestamp,
        }
    }
}

impl MulAssign<DW1000Time> for DW1000Time {
    #[inline]
    fn mul_assign(&mut self, factor: DW1000Time) {
        self.timestamp *= factor.timestamp;
    }
}

// Implement Div operations with f32
impl Div<f32> for DW1000Time {
    type Output = Self;

    #[inline]
    fn div(self, factor: f32) -> Self {
        Self {
            timestamp: (self.timestamp as f32 / factor) as i64,
        }
    }
}

impl DivAssign<f32> for DW1000Time {
    #[inline]
    fn div_assign(&mut self, factor: f32) {
        self.timestamp = (self.timestamp as f32 / factor) as i64;
    }
}

// Implement Div operations with DW1000Time
impl Div<DW1000Time> for DW1000Time {
    type Output = Self;

    #[inline]
    fn div(self, factor: DW1000Time) -> Self {
        Self {
            timestamp: self.timestamp / factor.timestamp,
        }
    }
}

impl DivAssign<DW1000Time> for DW1000Time {
    #[inline]
    fn div_assign(&mut self, factor: DW1000Time) {
        self.timestamp /= factor.timestamp;
    }
}
