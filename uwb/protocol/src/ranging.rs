pub const TIMESTAMP_MASK: u64 = (1_u64 << 40) - 1;
pub const DTU_PER_US: u64 = 63_897;
pub const DTU_METRES: f64 = 0.004_691_763_978_615_9;

pub const fn wrapping_delta(later: u64, earlier: u64) -> u64 {
    later.wrapping_sub(earlier) & TIMESTAMP_MASK
}

/// Round a delayed-transmit timestamp to the resolution required by DX_TIME.
pub const fn delayed_tx_time(base: u64, delay_us: u32) -> u64 {
    let delayed = base.wrapping_add(delay_us as u64 * DTU_PER_US) & TIMESTAMP_MASK;
    delayed & !0x1ff
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResponderTimestamps {
    pub poll_rx: u64,
    pub poll_ack_tx: u64,
    pub range_rx: u64,
}

/// Asymmetric double-sided TWR. Both native peers evaluate this function over
/// the same six hardware timestamps.
pub fn distance_metres(initiator: [u64; 3], responder: ResponderTimestamps) -> Option<f64> {
    let [poll_tx, poll_ack_rx, range_tx] = initiator;
    let round1 = wrapping_delta(poll_ack_rx, poll_tx) as i128;
    let reply1 = wrapping_delta(responder.poll_ack_tx, responder.poll_rx) as i128;
    let round2 = wrapping_delta(responder.range_rx, responder.poll_ack_tx) as i128;
    let reply2 = wrapping_delta(range_tx, poll_ack_rx) as i128;
    let denominator = round1 + round2 + reply1 + reply2;
    if denominator == 0 {
        return None;
    }
    let numerator = round1 * round2 - reply1 * reply2;
    let tof_dtu = numerator as f64 / denominator as f64;
    (tof_dtu >= 0.0).then_some(tof_dtu * DTU_METRES)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamp_delta_wraps_at_40_bits() {
        assert_eq!(wrapping_delta(25, TIMESTAMP_MASK - 24), 50);
    }

    #[test]
    fn delayed_transmit_is_512_dtu_aligned() {
        let scheduled = delayed_tx_time(TIMESTAMP_MASK - 1_000, 2_000);
        assert_eq!(scheduled & 0x1ff, 0);
        assert!(scheduled <= TIMESTAMP_MASK);
    }

    #[test]
    fn synthetic_ds_twr_recovers_known_time_of_flight() {
        let tof = 2_130_u64;
        let poll_tx = TIMESTAMP_MASK - 10_000;
        let poll_rx = (poll_tx + tof) & TIMESTAMP_MASK;
        let poll_ack_tx = (poll_rx + 447_283_000) & TIMESTAMP_MASK;
        let poll_ack_rx = (poll_ack_tx + tof) & TIMESTAMP_MASK;
        let range_tx = (poll_ack_rx + 447_279_000) & TIMESTAMP_MASK;
        let range_rx = (range_tx + tof) & TIMESTAMP_MASK;
        let distance = distance_metres(
            [poll_tx, poll_ack_rx, range_tx],
            ResponderTimestamps {
                poll_rx,
                poll_ack_tx,
                range_rx,
            },
        )
        .unwrap();
        assert!((distance - 9.993).abs() < 0.01, "{distance}");
    }
}
