pub mod board;
pub mod ranging;

pub const BENCH_PAN_ID: u16 = 0x4d10;
pub const DEFAULT_ANTENNA_DELAY: u16 = 16_390;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct NodeIndex(u8);

impl NodeIndex {
    pub const MAX: u8 = 15;

    pub const fn new(value: u8) -> Option<Self> {
        if value <= Self::MAX {
            Some(Self(value))
        } else {
            None
        }
    }

    pub const fn get(self) -> u8 {
        self.0
    }
}

impl std::fmt::Display for NodeIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

pub const fn address_for(index: NodeIndex) -> [u8; 2] {
    [0xa0 + index.get(), 0xc0 + index.get()]
}

pub const fn short_address_for(index: NodeIndex) -> u16 {
    u16::from_le_bytes(address_for(index))
}

pub const fn eui_for(index: NodeIndex) -> [u8; 8] {
    [0x7d, 0x00, 0x22, 0xea, 0x82, 0x60, 0x3b, 0x90 + index.get()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bench_addresses_match_the_existing_nodes() {
        let node0 = NodeIndex::new(0).unwrap();
        let node1 = NodeIndex::new(1).unwrap();
        assert_eq!(address_for(node0), [0xa0, 0xc0]);
        assert_eq!(address_for(node1), [0xa1, 0xc1]);
        assert_eq!(short_address_for(node0), 0xc0a0);
        assert_ne!(eui_for(node0), eui_for(node1));
        assert_eq!(NodeIndex::new(16), None);
    }
}
