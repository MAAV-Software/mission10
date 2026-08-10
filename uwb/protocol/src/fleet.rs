use hubpack::SerializedSize;
use serde::{Deserialize, Serialize};

/// Fleet-wide Wi-Fi operating mode selected over UWB.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, SerializedSize)]
pub enum FleetNetwork {
    Field,
    Internet,
}

/// Complete idempotent fleet master and network state.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, SerializedSize)]
pub struct FleetMode {
    pub master: u8,
    pub network: FleetNetwork,
}

impl FleetMode {
    pub const fn new(master: u8, network: FleetNetwork) -> Option<Self> {
        if master <= 3 {
            Some(Self { master, network })
        } else {
            None
        }
    }

    pub const fn is_valid(self) -> bool {
        self.master <= 3
    }
}
