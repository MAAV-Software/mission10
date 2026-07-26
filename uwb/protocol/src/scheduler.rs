use hubpack::SerializedSize;
use serde::{Deserialize, Serialize};

use crate::NodeAddress;

/// Four competition aircraft plus two movable development radios.
pub const MAX_NODES: usize = 6;
pub const MAX_PAIRS: usize = 15;
pub const FAR_PERIOD_US: u64 = 100_000;
pub const CLOSE_PERIOD_US: u64 = 10_000;
pub const CLOSE_EXPIRY_US: u64 = 300_000;
pub const MAX_BACKOFF_US: u64 = 400_000;
pub const CLOSE_ENTER_MM: u32 = 3_000;
pub const CLOSE_EXIT_MM: u32 = 3_500;

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize, SerializedSize)]
pub struct FlightRoster {
    count: u8,
    nodes: [NodeAddress; MAX_NODES],
}

impl FlightRoster {
    pub fn new(local: NodeAddress, peers: &[NodeAddress]) -> Option<Self> {
        if peers.is_empty() || peers.len() >= MAX_NODES || peers.contains(&local) {
            return None;
        }
        let mut roster = Self {
            count: (peers.len() + 1) as u8,
            nodes: [NodeAddress::default(); MAX_NODES],
        };
        roster.nodes[0] = local;
        for (index, peer) in peers.iter().copied().enumerate() {
            roster.nodes[index + 1] = peer;
        }
        let count = roster.count as usize;
        roster.nodes[..count].sort_unstable();
        roster.is_valid().then_some(roster)
    }

    pub fn is_valid(self) -> bool {
        let count = self.count as usize;
        (2..=MAX_NODES).contains(&count)
            && self.nodes[..count].windows(2).all(|pair| pair[0] < pair[1])
    }

    pub fn nodes(&self) -> &[NodeAddress] {
        &self.nodes[..self.count as usize]
    }

    pub fn contains(self, node: NodeAddress) -> bool {
        self.nodes().contains(&node)
    }

    pub fn pairs(self) -> PairIter {
        PairIter {
            roster: self,
            lower: 0,
            upper: 1,
        }
    }

    pub fn pair_count(self) -> usize {
        let count = self.count as usize;
        count * (count - 1) / 2
    }
}

pub struct PairIter {
    roster: FlightRoster,
    lower: usize,
    upper: usize,
}

impl Iterator for PairIter {
    type Item = Pair;

    fn next(&mut self) -> Option<Self::Item> {
        let count = self.roster.count as usize;
        if self.lower >= count || self.upper >= count {
            return None;
        }
        let pair = Pair::new(self.roster.nodes[self.lower], self.roster.nodes[self.upper])?;
        self.upper += 1;
        if self.upper >= count {
            self.lower += 1;
            self.upper = self.lower + 1;
        }
        Some(pair)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, SerializedSize)]
pub struct Pair {
    pub initiator: NodeAddress,
    pub responder: NodeAddress,
}

impl Pair {
    pub fn new(first: NodeAddress, second: NodeAddress) -> Option<Self> {
        if first == second {
            return None;
        }
        let (initiator, responder) = if first < second {
            (first, second)
        } else {
            (second, first)
        };
        Some(Self {
            initiator,
            responder,
        })
    }

    pub fn contains(self, node: NodeAddress) -> bool {
        self.initiator == node || self.responder == node
    }

    pub fn role(self, node: NodeAddress) -> PairRole {
        if node == self.initiator {
            PairRole::Initiator {
                peer: self.responder,
            }
        } else if node == self.responder {
            PairRole::Responder {
                peer: self.initiator,
            }
        } else {
            PairRole::Observer
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PairRole {
    Initiator { peer: NodeAddress },
    Responder { peer: NodeAddress },
    Observer,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize, SerializedSize)]
pub struct ExchangeId(u16);

impl ExchangeId {
    pub const fn new(raw: u16) -> Self {
        Self(raw)
    }

    pub const fn get(self) -> u16 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PairMacState {
    next_due_us: u64,
    last_attempt_us: u64,
    close_valid_until_us: u64,
    random_state: u32,
    consecutive_failures: u8,
    initialized: bool,
    attempted: bool,
}

impl PairMacState {
    pub const fn new(seed: u32) -> Self {
        Self {
            next_due_us: 0,
            last_attempt_us: 0,
            close_valid_until_us: 0,
            random_state: seed,
            consecutive_failures: 0,
            initialized: false,
            attempted: false,
        }
    }

    pub fn initialize(&mut self, now_us: u64, pair: Pair) {
        if self.initialized {
            return;
        }
        self.random_state ^= u32::from(pair.initiator.get()) << 16;
        self.random_state ^= u32::from(pair.responder.get());
        let phase = u64::from(self.random()) % FAR_PERIOD_US;
        self.next_due_us = now_us.saturating_add(phase);
        self.initialized = true;
    }

    pub const fn next_due_us(self) -> u64 {
        self.next_due_us
    }

    pub const fn is_due(self, now_us: u64) -> bool {
        self.initialized && now_us >= self.next_due_us
    }

    pub const fn is_close(self, now_us: u64) -> bool {
        self.close_valid_until_us != 0 && now_us <= self.close_valid_until_us
    }

    pub fn begin_attempt(&mut self, now_us: u64) {
        self.last_attempt_us = now_us;
        self.attempted = true;
    }

    pub fn expire(&mut self, now_us: u64) -> bool {
        if self.close_valid_until_us != 0 && now_us > self.close_valid_until_us {
            self.close_valid_until_us = 0;
            true
        } else {
            false
        }
    }

    pub fn record_success(&mut self, now_us: u64, close: bool) {
        self.consecutive_failures = 0;
        if close {
            self.close_valid_until_us = now_us.saturating_add(CLOSE_EXPIRY_US);
        } else {
            self.close_valid_until_us = 0;
        }
        let schedule_anchor = if self.attempted {
            self.last_attempt_us
        } else {
            now_us
        };
        self.next_due_us = schedule_anchor
            .saturating_add(self.period_us(now_us))
            .max(now_us);
    }

    pub fn record_failure(&mut self, now_us: u64) {
        self.consecutive_failures = self.consecutive_failures.saturating_add(1);
        let period = self.period_us(now_us);
        let exponent = self.consecutive_failures.min(3);
        let upper = period.saturating_mul(1_u64 << exponent).min(MAX_BACKOFF_US);
        let width = upper.saturating_sub(period);
        let extra = if width == 0 {
            0
        } else {
            u64::from(self.random()) % (width + 1)
        };
        self.next_due_us = now_us.saturating_add(period).saturating_add(extra);
    }

    fn period_us(self, now_us: u64) -> u64 {
        if self.is_close(now_us) {
            CLOSE_PERIOD_US
        } else {
            FAR_PERIOD_US
        }
    }

    fn random(&mut self) -> u32 {
        let mut value = self.random_state;
        if value == 0 {
            value = 0x6d2b_79f5;
        }
        value ^= value << 13;
        value ^= value >> 17;
        value ^= value << 5;
        self.random_state = value;
        value
    }
}

pub fn due_pair(
    roster: FlightRoster,
    local: NodeAddress,
    now_us: u64,
    pair_states: &[PairMacState; MAX_PAIRS],
) -> Option<Pair> {
    roster
        .pairs()
        .enumerate()
        .filter(|(_, pair)| pair.initiator == local)
        .filter(|(index, _)| pair_states[*index].is_due(now_us))
        .min_by_key(|(index, pair)| {
            (
                pair_states[*index].next_due_us(),
                pair.initiator,
                pair.responder,
            )
        })
        .map(|(_, pair)| pair)
}

pub fn next_due_us(
    roster: FlightRoster,
    local: NodeAddress,
    pair_states: &[PairMacState; MAX_PAIRS],
) -> Option<u64> {
    roster
        .pairs()
        .enumerate()
        .filter(|(_, pair)| pair.initiator == local)
        .map(|(index, _)| pair_states[index].next_due_us())
        .min()
}

pub fn pair_index(roster: FlightRoster, pair: Pair) -> Option<usize> {
    roster.pairs().position(|candidate| candidate == pair)
}

pub fn classify_range(previously_close: bool, millimetres: u32) -> bool {
    if previously_close {
        millimetres < CLOSE_EXIT_MM
    } else {
        millimetres < CLOSE_ENTER_MM
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(value: u16) -> NodeAddress {
        NodeAddress::new(value).unwrap()
    }

    #[test]
    fn pair_role_and_close_hysteresis_are_stable() {
        let pair = Pair::new(node(3), node(1)).unwrap();
        assert_eq!(pair.initiator, node(1));
        assert_eq!(pair.role(node(1)), PairRole::Initiator { peer: node(3) });
        assert!(classify_range(false, 2_999));
        assert!(classify_range(true, 3_499));
        assert!(!classify_range(true, 3_500));
    }

    #[test]
    fn successful_close_pair_uses_ten_millisecond_period() {
        let pair = Pair::new(node(0), node(1)).unwrap();
        let mut state = PairMacState::new(7);
        state.initialize(1_000, pair);
        state.begin_attempt(20_000);
        state.record_success(26_000, true);
        assert_eq!(state.next_due_us(), 30_000);
        assert!(state.is_close(30_000));
        assert!(state.expire(326_001));
    }

    #[test]
    fn failure_backoff_is_bounded() {
        let pair = Pair::new(node(0), node(1)).unwrap();
        let mut state = PairMacState::new(9);
        state.initialize(0, pair);
        state.record_failure(1_000_000);
        assert!((1_100_000..=1_200_000).contains(&state.next_due_us()));
    }
}
