#![cfg_attr(not(test), no_std)]

//! Mission 10 UWB wire types and ranging arithmetic.
//!
//! The native air link, the temporary DW1000-compatible air link, and the
//! CM5 host stream are deliberately separate protocols.

pub mod air;
pub mod dw1000_bench;
pub mod host;
pub mod ranging;
pub mod state;

pub use ranging::{
    DTU_METRES, DTU_PER_US, ResponderTimestamps, TIMESTAMP_MASK, delayed_tx_time, distance_metres,
    wrapping_delta,
};
pub use state::{AvoidanceMode, EgoState, StateValidity};
