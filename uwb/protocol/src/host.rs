//! Bidirectional host protocol carried as CRC-protected, COBS-delimited Hubpack.

use crc::{CRC_32_ISO_HDLC, Crc};
use hubpack::SerializedSize;
use serde::{Deserialize, Serialize};

use crate::scheduler::{ExchangeId, FlightRoster};
use crate::{EgoState, NodeAddress};

pub const HOST_PROTOCOL_VERSION: u8 = 8;
/// The other five radios in the complete development inventory.
pub const MAX_PEERS: usize = 5;

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize, SerializedSize)]
pub struct RadioConfiguration {
    pub node_address: NodeAddress,
    pub peer_count: u8,
    pub peers: [NodeAddress; MAX_PEERS],
}

impl RadioConfiguration {
    pub fn peers(&self) -> Option<&[NodeAddress]> {
        (self.peer_count as usize <= MAX_PEERS).then(|| &self.peers[..self.peer_count as usize])
    }

    pub fn is_valid(&self) -> bool {
        self.roster().is_some()
    }

    pub fn roster(&self) -> Option<FlightRoster> {
        FlightRoster::new(self.node_address, self.peers()?)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, SerializedSize)]
pub enum HostToRadio {
    Configure {
        configuration: RadioConfiguration,
    },
    SetEgoState {
        state: EgoState,
    },
    RequestHealth {
        request_id: u32,
    },
    ClockReply {
        request_id: u16,
        mission_rx_us: u64,
        mission_tx_us: u64,
        mission_generation: u32,
        source_error_us: u32,
    },
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, SerializedSize)]
pub struct HostToRadioEnvelope {
    pub protocol_version: u8,
    pub sequence: u32,
    pub message: HostToRadio,
}

impl HostToRadioEnvelope {
    pub const fn new(sequence: u32, message: HostToRadio) -> Self {
        Self {
            protocol_version: HOST_PROTOCOL_VERSION,
            sequence,
            message,
        }
    }
}

/// Stable, non-overlapping diagnostics reported by the radio and host tasks.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, SerializedSize)]
#[repr(u8)]
pub enum Diagnostic {
    RxFcs,
    RxPhy,
    RxBufferTooSmall,
    RxReedSolomon,
    RxFrameWaitTimeout,
    RxOverrun,
    RxPreambleDetectionTimeout,
    RxSfdTimeout,
    RxFrameFilteringRejection,
    Spi,
    FrameDecode,
    InvalidDistance,
    DelayedSendTooLate,
    DelayedSendPowerUpWarning,
    RadioState,
    HostFrame,
    InvalidConfiguration,
    MalformedAir,
    UnexpectedAir,
    InvalidTimestamp,
    Unknown,
    RadioReset,
    WatchdogReset,
}

impl Diagnostic {
    /// Hubpack encodes fieldless enum variants by declaration-order index.
    /// This inverse is also used for the nRF retained reset-reason byte.
    pub const fn from_wire_index(value: u8) -> Option<Self> {
        Some(match value {
            0 => Self::RxFcs,
            1 => Self::RxPhy,
            2 => Self::RxBufferTooSmall,
            3 => Self::RxReedSolomon,
            4 => Self::RxFrameWaitTimeout,
            5 => Self::RxOverrun,
            6 => Self::RxPreambleDetectionTimeout,
            7 => Self::RxSfdTimeout,
            8 => Self::RxFrameFilteringRejection,
            9 => Self::Spi,
            10 => Self::FrameDecode,
            11 => Self::InvalidDistance,
            12 => Self::DelayedSendTooLate,
            13 => Self::DelayedSendPowerUpWarning,
            14 => Self::RadioState,
            15 => Self::HostFrame,
            16 => Self::InvalidConfiguration,
            17 => Self::MalformedAir,
            18 => Self::UnexpectedAir,
            19 => Self::InvalidTimestamp,
            20 => Self::Unknown,
            21 => Self::RadioReset,
            22 => Self::WatchdogReset,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize, SerializedSize)]
pub struct HealthCounters {
    pub irq_wakes: u32,
    pub spurious_irq_wakes: u32,
    pub wait_timeouts: u32,
    pub recoveries: u32,
    pub malformed_air_frames: u32,
    pub unexpected_air_frames: u32,
    pub clock_samples_accepted: u32,
    pub clock_samples_rejected: u32,
    pub time_mapping_unavailable: u32,
    pub clock_generation_changes: u32,
    pub polls_sent: u32,
    pub polls_received: u32,
    pub responses_sent: u32,
    pub responses_received: u32,
    pub finals_sent: u32,
    pub finals_received: u32,
    pub reports_sent: u32,
    pub reports_received: u32,
    pub exchange_completions: u32,
    pub exchange_timeouts: u32,
    pub contention_backoffs: u32,
    pub late_delayed_transmits: u32,
    pub host_stale_transitions: u32,
    pub peer_stale_transitions: u32,
    pub radio_reinitializations: u32,
    pub host_decode_errors: u32,
    pub host_command_drops: u32,
    pub control_drops: u32,
    pub measurement_drops: u32,
    pub diagnostic_drops: u32,
}

impl HealthCounters {
    pub const ZERO: Self = Self {
        irq_wakes: 0,
        spurious_irq_wakes: 0,
        wait_timeouts: 0,
        recoveries: 0,
        malformed_air_frames: 0,
        unexpected_air_frames: 0,
        clock_samples_accepted: 0,
        clock_samples_rejected: 0,
        time_mapping_unavailable: 0,
        clock_generation_changes: 0,
        polls_sent: 0,
        polls_received: 0,
        responses_sent: 0,
        responses_received: 0,
        finals_sent: 0,
        finals_received: 0,
        reports_sent: 0,
        reports_received: 0,
        exchange_completions: 0,
        exchange_timeouts: 0,
        contention_backoffs: 0,
        late_delayed_transmits: 0,
        host_stale_transitions: 0,
        peer_stale_transitions: 0,
        radio_reinitializations: 0,
        host_decode_errors: 0,
        host_command_drops: 0,
        control_drops: 0,
        measurement_drops: 0,
        diagnostic_drops: 0,
    };

    /// Merge the counters owned by the host transport without disturbing
    /// radio-owned counters in `self`.
    pub fn merge_transport(&mut self, transport: Self) {
        self.host_decode_errors = transport.host_decode_errors;
        self.host_command_drops = transport.host_command_drops;
        self.control_drops = transport.control_drops;
        self.measurement_drops = transport.measurement_drops;
        self.diagnostic_drops = transport.diagnostic_drops;
        self.host_stale_transitions = transport.host_stale_transitions;
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, SerializedSize)]
pub enum MissionEventTime {
    Unavailable,
    Mapped {
        mission_time_us: u64,
        generation: u32,
        error_us: u32,
    },
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, SerializedSize)]
pub struct CompletedExchange {
    pub peer: NodeAddress,
    pub exchange_id: ExchangeId,
    pub range_event_time_dtu: u64,
    pub mission_event_time: MissionEventTime,
    pub millimetres: u32,
    pub rssi_cdbm: i16,
    pub quality_flags: u16,
    pub state: EgoState,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, SerializedSize)]
pub enum RadioToHost {
    RadioId {
        ridtag: u16,
        model: u8,
        version: u8,
        revision: u8,
    },
    Otp {
        tx_power: u32,
        antenna: u32,
        xtal: u32,
        revision: u32,
    },
    Ready {
        rx_delay: u16,
        tx_delay: u16,
    },
    ClockProbe {
        request_id: u16,
    },
    ClockStatus {
        generation: u32,
        error_us: u32,
        age_us: u32,
        mapped: bool,
    },
    Configured {
        configuration: RadioConfiguration,
    },
    CompletedExchange {
        exchange: CompletedExchange,
    },
    Error {
        diagnostic: Diagnostic,
    },
    Health {
        request_id: u32,
        counters: HealthCounters,
    },
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LatestExchanges {
    entries: [Option<(u32, CompletedExchange)>; MAX_PEERS],
    generation: u32,
}

impl LatestExchanges {
    pub const fn new() -> Self {
        Self {
            entries: [None; MAX_PEERS],
            generation: 0,
        }
    }

    /// Inserts the newest exchange for a peer. Returns true when an older
    /// unsent exchange was replaced.
    pub fn push(&mut self, exchange: CompletedExchange) -> bool {
        self.generation = self.generation.wrapping_add(1);
        if let Some(entry) = self.entries.iter_mut().find(|entry| {
            entry
                .as_ref()
                .is_some_and(|(_, current)| current.peer == exchange.peer)
        }) {
            *entry = Some((self.generation, exchange));
            return true;
        }
        if let Some(entry) = self.entries.iter_mut().find(|entry| entry.is_none()) {
            *entry = Some((self.generation, exchange));
            return false;
        }
        let oldest = self
            .entries
            .iter()
            .enumerate()
            .min_by_key(|(_, entry)| entry.map(|(generation, _)| generation))
            .map(|(index, _)| index)
            .unwrap_or(0);
        self.entries[oldest] = Some((self.generation, exchange));
        true
    }

    pub fn pop_oldest(&mut self) -> Option<CompletedExchange> {
        let oldest = self
            .entries
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| entry.map(|(generation, _)| (index, generation)))
            .min_by_key(|(_, generation)| *generation)
            .map(|(index, _)| index)?;
        self.entries[oldest].take().map(|(_, exchange)| exchange)
    }

    /// Restores an interrupted transmission unless a newer exchange for the
    /// same peer is already pending. Returns true when the restored value was
    /// obsolete and was discarded.
    pub fn restore(&mut self, exchange: CompletedExchange) -> bool {
        if self.entries.iter().any(|entry| {
            entry
                .as_ref()
                .is_some_and(|(_, current)| current.peer == exchange.peer)
        }) {
            return true;
        }
        self.push(exchange)
    }

    pub fn clear(&mut self) {
        self.entries = [None; MAX_PEERS];
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, SerializedSize)]
pub struct RadioToHostEnvelope {
    pub protocol_version: u8,
    pub sequence: u32,
    pub message: RadioToHost,
}

impl RadioToHostEnvelope {
    pub const fn new(sequence: u32, message: RadioToHost) -> Self {
        Self {
            protocol_version: HOST_PROTOCOL_VERSION,
            sequence,
            message,
        }
    }
}

const fn max(a: usize, b: usize) -> usize {
    if a > b { a } else { b }
}

pub const HOST_ENVELOPE_MAX_SIZE: usize = max(
    <HostToRadioEnvelope as SerializedSize>::MAX_SIZE,
    <RadioToHostEnvelope as SerializedSize>::MAX_SIZE,
);
pub const HOST_RAW_MAX_SIZE: usize = HOST_ENVELOPE_MAX_SIZE + 4;
pub const HOST_FRAME_MAX_SIZE: usize = corncobs::max_encoded_len(HOST_RAW_MAX_SIZE);
const _: () = assert!(HOST_FRAME_MAX_SIZE <= 192);

const HOST_CRC: Crc<u32> = Crc::<u32>::new(&CRC_32_ISO_HDLC);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostFrameError {
    Cobs,
    TooShort,
    Crc,
    Hubpack,
    TrailingData,
    Version(u8),
}

fn encode<T: Serialize>(
    envelope: &T,
    output: &mut [u8; HOST_FRAME_MAX_SIZE],
) -> Result<usize, HostFrameError> {
    let mut raw = [0_u8; HOST_RAW_MAX_SIZE];
    let payload_len = hubpack::serialize(&mut raw[..HOST_ENVELOPE_MAX_SIZE], envelope)
        .map_err(|_| HostFrameError::Hubpack)?;
    let crc = HOST_CRC.checksum(&raw[..payload_len]).to_le_bytes();
    raw[payload_len..payload_len + 4].copy_from_slice(&crc);
    Ok(corncobs::encode_buf(&raw[..payload_len + 4], output))
}

fn decode_payload<'a>(
    frame: &[u8],
    raw: &'a mut [u8; HOST_RAW_MAX_SIZE],
) -> Result<&'a [u8], HostFrameError> {
    let raw_len = corncobs::decode_buf(frame, raw).map_err(|_| HostFrameError::Cobs)?;
    let payload_len = raw_len.checked_sub(4).ok_or(HostFrameError::TooShort)?;
    let expected = u32::from_le_bytes(
        raw[payload_len..raw_len]
            .try_into()
            .expect("four-byte CRC suffix"),
    );
    if HOST_CRC.checksum(&raw[..payload_len]) != expected {
        return Err(HostFrameError::Crc);
    }
    Ok(&raw[..payload_len])
}

pub fn encode_host_to_radio(
    envelope: &HostToRadioEnvelope,
    output: &mut [u8; HOST_FRAME_MAX_SIZE],
) -> Result<usize, HostFrameError> {
    encode(envelope, output)
}

pub fn decode_host_to_radio(
    frame: &[u8],
    raw: &mut [u8; HOST_RAW_MAX_SIZE],
) -> Result<HostToRadioEnvelope, HostFrameError> {
    let payload = decode_payload(frame, raw)?;
    let (envelope, rest) = hubpack::deserialize::<HostToRadioEnvelope>(payload)
        .map_err(|_| HostFrameError::Hubpack)?;
    validate_decoded(envelope.protocol_version, rest)?;
    Ok(envelope)
}

pub fn encode_radio_to_host(
    envelope: &RadioToHostEnvelope,
    output: &mut [u8; HOST_FRAME_MAX_SIZE],
) -> Result<usize, HostFrameError> {
    encode(envelope, output)
}

pub fn decode_radio_to_host(
    frame: &[u8],
    raw: &mut [u8; HOST_RAW_MAX_SIZE],
) -> Result<RadioToHostEnvelope, HostFrameError> {
    let payload = decode_payload(frame, raw)?;
    let (envelope, rest) = hubpack::deserialize::<RadioToHostEnvelope>(payload)
        .map_err(|_| HostFrameError::Hubpack)?;
    validate_decoded(envelope.protocol_version, rest)?;
    Ok(envelope)
}

fn validate_decoded(version: u8, rest: &[u8]) -> Result<(), HostFrameError> {
    if !rest.is_empty() {
        return Err(HostFrameError::TrailingData);
    }
    if version != HOST_PROTOCOL_VERSION {
        return Err(HostFrameError::Version(version));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(value: u16) -> NodeAddress {
        NodeAddress::new(value).unwrap()
    }

    fn exchange_id(value: u16) -> ExchangeId {
        ExchangeId::new(value)
    }

    #[test]
    fn configuration_rejects_reserved_duplicate_and_self_addresses() {
        assert!(
            RadioConfiguration {
                node_address: node(2),
                peer_count: 3,
                peers: [node(0), node(1), node(3), node(0), node(0)],
            }
            .is_valid()
        );
        assert!(
            !RadioConfiguration {
                node_address: node(2),
                peer_count: 2,
                peers: [node(1), node(1), node(0), node(0), node(0)],
            }
            .is_valid()
        );
        assert!(
            !RadioConfiguration {
                node_address: node(2),
                peer_count: 1,
                peers: [node(2), node(0), node(0), node(0), node(0)],
            }
            .is_valid()
        );
        assert!(
            !RadioConfiguration {
                node_address: node(2),
                peer_count: 6,
                peers: [node(0); 5],
            }
            .is_valid()
        );
    }

    #[test]
    fn diagnostic_wire_indices_are_explicitly_pinned() {
        let variants = [
            Diagnostic::RxFcs,
            Diagnostic::RxPhy,
            Diagnostic::RxBufferTooSmall,
            Diagnostic::RxReedSolomon,
            Diagnostic::RxFrameWaitTimeout,
            Diagnostic::RxOverrun,
            Diagnostic::RxPreambleDetectionTimeout,
            Diagnostic::RxSfdTimeout,
            Diagnostic::RxFrameFilteringRejection,
            Diagnostic::Spi,
            Diagnostic::FrameDecode,
            Diagnostic::InvalidDistance,
            Diagnostic::DelayedSendTooLate,
            Diagnostic::DelayedSendPowerUpWarning,
            Diagnostic::RadioState,
            Diagnostic::HostFrame,
            Diagnostic::InvalidConfiguration,
            Diagnostic::MalformedAir,
            Diagnostic::UnexpectedAir,
            Diagnostic::InvalidTimestamp,
            Diagnostic::Unknown,
            Diagnostic::RadioReset,
            Diagnostic::WatchdogReset,
        ];
        for (index, diagnostic) in variants.into_iter().enumerate() {
            assert_eq!(diagnostic as usize, index);
            assert_eq!(Diagnostic::from_wire_index(index as u8), Some(diagnostic));
        }
        assert_eq!(Diagnostic::from_wire_index(variants.len() as u8), None);
    }

    #[test]
    fn corrupted_radio_frame_is_rejected() {
        let envelope = RadioToHostEnvelope::new(
            7,
            RadioToHost::CompletedExchange {
                exchange: CompletedExchange {
                    peer: node(3),
                    exchange_id: exchange_id(0x1234),
                    range_event_time_dtu: 0x12_3456_789a,
                    mission_event_time: MissionEventTime::Unavailable,
                    millimetres: 2_345,
                    rssi_cdbm: -7_225,
                    quality_flags: 3,
                    state: golden_state(),
                },
            },
        );
        let mut frame = [0; HOST_FRAME_MAX_SIZE];
        let len = encode_radio_to_host(&envelope, &mut frame).unwrap();
        let mut raw = [0; HOST_RAW_MAX_SIZE];

        frame[2] ^= 0x40;
        assert_eq!(
            decode_radio_to_host(&frame[..len], &mut raw),
            Err(HostFrameError::Crc)
        );
    }

    #[test]
    fn latest_exchange_slots_replace_per_peer_and_preserve_other_peers() {
        fn exchange(peer: u16, id: u16) -> CompletedExchange {
            CompletedExchange {
                peer: node(peer),
                exchange_id: exchange_id(id),
                range_event_time_dtu: 0,
                mission_event_time: MissionEventTime::Unavailable,
                millimetres: 0,
                rssi_cdbm: 0,
                quality_flags: 0,
                state: EgoState::default(),
            }
        }

        let mut slots = LatestExchanges::default();
        assert!(!slots.push(exchange(1, 9)));
        assert!(!slots.push(exchange(3, 0x14)));
        assert!(slots.push(exchange(1, 0x18)));
        assert_eq!(slots.pop_oldest().unwrap().exchange_id.get(), 0x14);
        assert_eq!(slots.pop_oldest().unwrap().exchange_id.get(), 0x18);
        assert!(slots.pop_oldest().is_none());

        assert!(!slots.push(exchange(1, 0x22)));
        assert!(slots.restore(exchange(1, 0x18)));
        assert_eq!(slots.pop_oldest().unwrap().exchange_id.get(), 0x22);
    }

    /// Ordered radio→host golden frames. Sequence numbers are part of the
    /// committed wire contract and are assigned by position here.
    fn radio_fixture_frames() -> Vec<(&'static str, Vec<u8>)> {
        let configuration = golden_configuration();
        let cases = [
            (
                "radio.radio_id",
                RadioToHost::RadioId {
                    ridtag: 0xdeca,
                    model: 3,
                    version: 0,
                    revision: 2,
                },
            ),
            (
                "radio.otp",
                RadioToHost::Otp {
                    tx_power: 0x6161_6161,
                    antenna: 0x3ff0_3ff0,
                    xtal: 0x00be_0019,
                    revision: 0x0001_0201,
                },
            ),
            (
                "radio.ready",
                RadioToHost::Ready {
                    rx_delay: 0x3ff0,
                    tx_delay: 0x3ff1,
                },
            ),
            (
                "radio.clock_probe",
                RadioToHost::ClockProbe { request_id: 0x5678 },
            ),
            (
                "radio.clock_status",
                RadioToHost::ClockStatus {
                    generation: 7,
                    error_us: 125,
                    age_us: 12_345,
                    mapped: true,
                },
            ),
            (
                "radio.configured",
                RadioToHost::Configured { configuration },
            ),
            (
                "radio.completed_exchange",
                RadioToHost::CompletedExchange {
                    exchange: CompletedExchange {
                        peer: node(3),
                        exchange_id: exchange_id(0x1234),
                        range_event_time_dtu: 0x01_0203_0405,
                        mission_event_time: MissionEventTime::Mapped {
                            mission_time_us: 0x0102_0304_0506_0708,
                            generation: 7,
                            error_us: 125,
                        },
                        millimetres: 2_345,
                        rssi_cdbm: -7_225,
                        quality_flags: 3,
                        state: golden_state(),
                    },
                },
            ),
            (
                "radio.error",
                RadioToHost::Error {
                    diagnostic: Diagnostic::RadioReset,
                },
            ),
            (
                "radio.health",
                RadioToHost::Health {
                    request_id: 0x89ab_cdef,
                    counters: HealthCounters {
                        irq_wakes: 1,
                        spurious_irq_wakes: 2,
                        wait_timeouts: 3,
                        recoveries: 4,
                        malformed_air_frames: 5,
                        unexpected_air_frames: 6,
                        clock_samples_accepted: 7,
                        clock_samples_rejected: 8,
                        time_mapping_unavailable: 9,
                        clock_generation_changes: 10,
                        polls_sent: 11,
                        polls_received: 12,
                        responses_sent: 13,
                        responses_received: 14,
                        finals_sent: 15,
                        finals_received: 16,
                        reports_sent: 17,
                        reports_received: 18,
                        exchange_completions: 19,
                        exchange_timeouts: 20,
                        contention_backoffs: 21,
                        late_delayed_transmits: 22,
                        host_stale_transitions: 23,
                        peer_stale_transitions: 24,
                        radio_reinitializations: 25,
                        host_decode_errors: 26,
                        host_command_drops: 27,
                        control_drops: 28,
                        measurement_drops: 29,
                        diagnostic_drops: 30,
                    },
                },
            ),
        ];

        let mut frame = [0; HOST_FRAME_MAX_SIZE];
        cases
            .into_iter()
            .enumerate()
            .map(|(index, (name, message))| {
                let envelope = RadioToHostEnvelope::new(0x1020_3040 + index as u32, message);
                let len = encode_radio_to_host(&envelope, &mut frame).unwrap();
                (name, frame[..len].to_vec())
            })
            .collect()
    }

    /// Ordered host→radio golden frames.
    fn host_fixture_frames() -> Vec<(&'static str, Vec<u8>)> {
        let cases = [
            (
                "host.configure",
                HostToRadioEnvelope::new(
                    0x0102_0304,
                    HostToRadio::Configure {
                        configuration: golden_configuration(),
                    },
                ),
            ),
            (
                "host.set_ego_state",
                HostToRadioEnvelope::new(
                    0x0102_0305,
                    HostToRadio::SetEgoState {
                        state: golden_state(),
                    },
                ),
            ),
            (
                "host.request_health",
                HostToRadioEnvelope::new(
                    0x0102_0306,
                    HostToRadio::RequestHealth {
                        request_id: 0x89ab_cdef,
                    },
                ),
            ),
            (
                "host.clock_reply",
                HostToRadioEnvelope::new(
                    0x0102_0307,
                    HostToRadio::ClockReply {
                        request_id: 0x5678,
                        mission_rx_us: 0x0102_0304_0506_0708,
                        mission_tx_us: 0x1112_1314_1516_1718,
                        mission_generation: 7,
                        source_error_us: 125,
                    },
                ),
            ),
        ];
        let mut frame = [0; HOST_FRAME_MAX_SIZE];
        cases
            .into_iter()
            .map(|(name, envelope)| {
                let len = encode_host_to_radio(&envelope, &mut frame).unwrap();
                (name, frame[..len].to_vec())
            })
            .collect()
    }

    const FIXTURE_HEADER: &str = "\
# Committed wire contract shared by the Rust encoder and Python codec tests.
# Format: direction.variant=COBS(Hubpack(envelope) || CRC-32/ISO-HDLC) || 00
# These are test payloads, not node configuration. Changing a frame normally
# requires a host protocol version bump rather than regenerating this file.
# Generated by the Rust encoders. Regenerate (never hand-edit) with:
#   UPDATE_GOLDEN=1 cargo test --target x86_64-unknown-linux-gnu \
#     -p mission10-uwb-protocol committed_fixture
";

    fn render_committed_fixture() -> String {
        use std::fmt::Write;
        let mut text = String::from(FIXTURE_HEADER);
        for (name, bytes) in radio_fixture_frames()
            .into_iter()
            .chain(host_fixture_frames())
        {
            text.push_str(name);
            text.push('=');
            for byte in bytes {
                write!(text, "{byte:02x}").unwrap();
            }
            text.push('\n');
        }
        text
    }

    /// Gate: the checked-in fixture must equal what the encoders produce, so the
    /// Rust and Python codecs cannot silently drift. The Python suite reads the
    /// same file and asserts its decoded meaning, keeping the two independent.
    #[test]
    fn committed_fixture_matches_the_encoders() {
        let rendered = render_committed_fixture();
        if std::env::var_os("PRINT_HOST_GOLDEN").is_some() {
            println!("{rendered}");
            return;
        }
        if std::env::var_os("UPDATE_GOLDEN").is_some() {
            let path = concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/testdata/host_protocol_v8.frames"
            );
            std::fs::write(path, &rendered).expect("write golden fixture");
            return;
        }
        assert_eq!(
            rendered,
            include_str!("../testdata/host_protocol_v8.frames"),
            "golden fixture is stale; regenerate with \
             UPDATE_GOLDEN=1 cargo test --target x86_64-unknown-linux-gnu \
             -p mission10-uwb-protocol committed_fixture",
        );
    }

    fn golden_configuration() -> RadioConfiguration {
        RadioConfiguration {
            node_address: node(2),
            peer_count: 3,
            peers: [node(0), node(3), node(0x8000), node(0), node(0)],
        }
    }

    fn golden_state() -> EgoState {
        EgoState {
            sample_time_us: 0x0102_0304_0506_0708,
            sequence: 0x1112_1314,
            frame_epoch: 0x1516,
            phase_mrad: -1,
            phase_rate_mrad_s: 2,
            yaw_mrad: -3,
            position_enu_mm: [4, -5, 6],
            velocity_enu_mm_s: [-7, 8, -9],
            mode: crate::AvoidanceMode::Phase,
            validity: crate::StateValidity(0x800d),
        }
    }
}
