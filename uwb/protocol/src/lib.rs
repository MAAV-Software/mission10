#![cfg_attr(not(test), no_std)]

//! Mission 10 UWB wire types and ranging arithmetic.
//!
//! The native addressed air link and CM5 host stream are separate protocols.

use core::fmt;

use hubpack::SerializedSize;
use serde::de::{Unexpected, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub mod air;
pub mod clock;
pub mod fleet;
pub mod host;
pub mod ranging;
pub mod scheduler;
pub mod state;

pub use fleet::{FleetMode, FleetNetwork};
pub use ranging::{
    DTU_METRES, DTU_PER_US, MAX_SCHEDULED_TX_ERROR_DTU, REPORT_TURNAROUND_US, ResponderTimestamps,
    TIMESTAMP_MASK, delayed_tx_time, distance_metres, scheduled_tx_matches, wrapping_delta,
};
pub use state::{AvoidanceMode, EgoState, StateValidity};

/// Checked Mission 10 IEEE 802.15.4 short address.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd, SerializedSize)]
pub struct NodeAddress(u16);

impl NodeAddress {
    /// Creates an aircraft (`0..3`) or development (`0x8000..0x80ff`) address.
    pub const fn new(value: u16) -> Option<Self> {
        if value <= 3 || (value >= 0x8000 && value <= 0x80ff) {
            Some(Self(value))
        } else {
            None
        }
    }

    /// Returns the 16-bit IEEE short address.
    pub const fn get(self) -> u16 {
        self.0
    }
}

impl fmt::Display for NodeAddress {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "0x{:04x}", self.0)
    }
}

impl Serialize for NodeAddress {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u16(self.0)
    }
}

struct NodeAddressVisitor;

impl Visitor<'_> for NodeAddressVisitor {
    type Value = NodeAddress;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("an aircraft address 0..3 or development address 0x8000..0x80ff")
    }

    fn visit_u16<E>(self, value: u16) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        NodeAddress::new(value)
            .ok_or_else(|| E::invalid_value(Unexpected::Unsigned(value.into()), &self))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        let raw = u16::try_from(value)
            .map_err(|_| E::invalid_value(Unexpected::Unsigned(value), &self))?;
        self.visit_u16(raw)
    }
}

impl<'de> Deserialize<'de> for NodeAddress {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_u16(NodeAddressVisitor)
    }
}

/// MAC destination.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Destination {
    /// One validated Mission 10 node.
    Node(NodeAddress),
    /// IEEE short broadcast address.
    Broadcast,
}

impl Destination {
    /// Returns the IEEE short-address representation.
    pub const fn get(self) -> u16 {
        match self {
            Self::Node(address) => address.get(),
            Self::Broadcast => air::BROADCAST_ADDRESS,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::NodeAddress;

    #[test]
    fn node_address_namespace_is_exhaustive() {
        for value in 0_u32..=u32::from(u16::MAX) {
            let value = value as u16;
            let expected = value <= 3 || (0x8000..=0x80ff).contains(&value);
            assert_eq!(NodeAddress::new(value).is_some(), expected);
            let raw = value.to_le_bytes();
            assert_eq!(hubpack::deserialize::<NodeAddress>(&raw).is_ok(), expected);
        }
    }
}
