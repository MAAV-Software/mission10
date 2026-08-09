#![no_std]

//! Embedded Rust driver for the Decawave DW1000 UWB transceiver
//!
//! This crate provides a no_std driver for the DW1000 Ultra-Wideband (UWB)
//! transceiver chip, suitable for use in embedded systems.

pub mod async_dw1000;
pub mod config;
pub mod constants;
mod device;
mod driver_core;
pub mod dw1000;
pub mod error;
pub mod registers;
pub mod time;

pub use async_dw1000::AsyncDw1000;
pub use config::*;
pub use device::{
    AntennaDelay, DelayedTransmit, DeviceIdentity, DxTimeDeadline, Eui64, OnAirTimestamp, PanId,
    PeerSnapshot, RadioDebugState, RxFrame, ShortAddress, SignalMetrics, SysStatus, Timestamps,
};
pub use dw1000::Dw1000;
pub use error::{Error, RxError};
pub use time::{DW1000Time, DwTime};
