pub mod board;
pub mod host_service;
pub mod oracle;
pub mod ranging;

pub use mission10_uwb_protocol::NodeAddress;

pub const DEFAULT_ANTENNA_DELAY: u16 = 16_390;

pub const fn short_address_for(address: NodeAddress) -> u16 {
    address.get()
}

pub const fn eui_for(address: NodeAddress) -> [u8; 8] {
    let [high, low] = address.get().to_be_bytes();
    [0x7d, 0x00, 0x22, 0xea, 0x82, 0x60, high, low]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_addresses_map_to_short_addresses_and_unique_euis() {
        let node0 = NodeAddress::new(0).unwrap();
        let node1 = NodeAddress::new(1).unwrap();
        let development = NodeAddress::new(0x8000).unwrap();
        assert_eq!(short_address_for(node0), 0);
        assert_eq!(short_address_for(node1), 1);
        assert_ne!(eui_for(node0), eui_for(node1));
        assert_eq!(&eui_for(development)[6..], &[0x80, 0x00]);
    }
}
