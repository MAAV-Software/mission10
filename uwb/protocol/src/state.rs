use hubpack::SerializedSize;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, SerializedSize)]
#[repr(u8)]
pub enum AvoidanceMode {
    Nominal,
    Phase,
    Reflex,
    Deconflict,
    Yielding,
    Saturated,
    Climb,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize, SerializedSize)]
pub struct StateValidity(pub u16);

impl StateValidity {
    pub const PHASE: u16 = 1 << 0;
    pub const YAW: u16 = 1 << 1;
    pub const POSITION: u16 = 1 << 2;
    pub const VELOCITY: u16 = 1 << 3;
    pub const HOST_STALE: u16 = 1 << 15;

    pub const fn contains(self, flag: u16) -> bool {
        self.0 & flag != 0
    }
}

/// Compact state shared once by each participant in a ranging exchange.
/// Coordinate values are ENU; angles are signed milliradians.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, SerializedSize)]
pub struct EgoState {
    pub sample_time_us: u64,
    pub sequence: u32,
    pub phase_mrad: i16,
    pub phase_rate_mrad_s: i16,
    pub yaw_mrad: i16,
    pub position_enu_mm: [i32; 3],
    pub velocity_enu_mm_s: [i16; 3],
    pub mode: AvoidanceMode,
    pub validity: StateValidity,
}

impl Default for EgoState {
    fn default() -> Self {
        Self {
            sample_time_us: 0,
            sequence: 0,
            phase_mrad: 0,
            phase_rate_mrad_s: 0,
            yaw_mrad: 0,
            position_enu_mm: [0; 3],
            velocity_enu_mm_s: [0; 3],
            mode: AvoidanceMode::Nominal,
            validity: StateValidity::default(),
        }
    }
}
