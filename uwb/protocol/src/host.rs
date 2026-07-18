//! Bidirectional host protocol carried as CRC-protected, COBS-delimited Hubpack.

use crc::{CRC_32_ISO_HDLC, Crc};
use hubpack::SerializedSize;
use serde::{Deserialize, Serialize};

use crate::EgoState;

pub const HOST_PROTOCOL_VERSION: u8 = 3;
pub const MAX_PEERS: usize = 3;

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize, SerializedSize)]
pub struct RadioConfiguration {
    pub node_address: u16,
    pub peer_count: u8,
    pub peers: [u16; MAX_PEERS],
}

impl RadioConfiguration {
    pub fn peers(&self) -> Option<&[u16]> {
        (self.peer_count as usize <= MAX_PEERS).then(|| &self.peers[..self.peer_count as usize])
    }

    pub fn is_valid(&self) -> bool {
        let Some(peers) = self.peers() else {
            return false;
        };
        if !is_unicast_address(self.node_address) {
            return false;
        }
        for (index, peer) in peers.iter().enumerate() {
            if !is_unicast_address(*peer)
                || *peer == self.node_address
                || peers[..index].contains(peer)
            {
                return false;
            }
        }
        true
    }
}

pub const fn is_unicast_address(address: u16) -> bool {
    address < 0xfffe
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, SerializedSize)]
pub enum HostToRadio {
    Configure { configuration: RadioConfiguration },
    SetEgoState { state: EgoState },
    RequestHealth { request_id: u32 },
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

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, SerializedSize)]
#[repr(u8)]
pub enum OperatingMode {
    Dw1000BenchResponder,
    Dw1000BenchInitiator,
    Native,
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
    ShortPoll,
    ShortPollAck,
    ShortRange,
    ShortRangeReport,
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
    UnsupportedInMode,
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
            11 => Self::ShortPoll,
            12 => Self::ShortPollAck,
            13 => Self::ShortRange,
            14 => Self::ShortRangeReport,
            15 => Self::InvalidDistance,
            16 => Self::DelayedSendTooLate,
            17 => Self::DelayedSendPowerUpWarning,
            18 => Self::RadioState,
            19 => Self::HostFrame,
            20 => Self::InvalidConfiguration,
            21 => Self::MalformedAir,
            22 => Self::UnexpectedAir,
            23 => Self::InvalidTimestamp,
            24 => Self::Unknown,
            25 => Self::UnsupportedInMode,
            26 => Self::RadioReset,
            27 => Self::WatchdogReset,
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
    pub missed_deadlines: u32,
    pub malformed_air_frames: u32,
    pub unexpected_air_frames: u32,
    pub host_decode_errors: u32,
    pub host_command_drops: u32,
    pub control_drops: u32,
    pub measurement_drops: u32,
    pub diagnostic_drops: u32,
}

impl HealthCounters {
    /// Merge the counters owned by the host transport without disturbing
    /// radio-owned counters in `self`.
    pub fn merge_transport(&mut self, transport: Self) {
        self.host_decode_errors = transport.host_decode_errors;
        self.host_command_drops = transport.host_command_drops;
        self.control_drops = transport.control_drops;
        self.measurement_drops = transport.measurement_drops;
        self.diagnostic_drops = transport.diagnostic_drops;
    }
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
        mode: OperatingMode,
        rx_delay: u16,
        tx_delay: u16,
        node_address: u16,
    },
    Configured {
        configuration: RadioConfiguration,
    },
    Rx {
        kind: u8,
    },
    Range {
        peer: u16,
        exchange_id: u16,
        range_event_time_dtu: u64,
        millimetres: u32,
        rssi_cdbm: i16,
        quality_flags: u16,
    },
    PeerState {
        peer: u16,
        exchange_id: u16,
        state: EgoState,
    },
    Error {
        diagnostic: Diagnostic,
    },
    Health {
        request_id: u32,
        counters: HealthCounters,
    },
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
const _: () = assert!(HOST_FRAME_MAX_SIZE <= 64);

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

    #[test]
    fn configuration_rejects_reserved_duplicate_and_self_addresses() {
        assert!(
            RadioConfiguration {
                node_address: 2,
                peer_count: 3,
                peers: [0, 1, 3],
            }
            .is_valid()
        );
        assert!(
            !RadioConfiguration {
                node_address: 2,
                peer_count: 2,
                peers: [1, 1, 0],
            }
            .is_valid()
        );
        assert!(
            !RadioConfiguration {
                node_address: 2,
                peer_count: 1,
                peers: [2, 0, 0],
            }
            .is_valid()
        );
        assert!(
            !RadioConfiguration {
                node_address: 0xffff,
                peer_count: 0,
                peers: [0; 3],
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
            Diagnostic::ShortPoll,
            Diagnostic::ShortPollAck,
            Diagnostic::ShortRange,
            Diagnostic::ShortRangeReport,
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
            Diagnostic::UnsupportedInMode,
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
    fn host_to_radio_round_trip() {
        let envelope = HostToRadioEnvelope::new(
            42,
            HostToRadio::Configure {
                configuration: RadioConfiguration {
                    node_address: 2,
                    peer_count: 2,
                    peers: [0, 4, 0],
                },
            },
        );
        let mut frame = [0; HOST_FRAME_MAX_SIZE];
        let len = encode_host_to_radio(&envelope, &mut frame).unwrap();
        assert_eq!(frame[len - 1], 0);
        let mut raw = [0; HOST_RAW_MAX_SIZE];
        assert_eq!(decode_host_to_radio(&frame[..len], &mut raw), Ok(envelope));
    }

    #[test]
    fn radio_to_host_round_trip_and_crc_rejection() {
        let envelope = RadioToHostEnvelope::new(
            7,
            RadioToHost::Range {
                peer: 4,
                exchange_id: 0x1234,
                range_event_time_dtu: 0x12_3456_789a,
                millimetres: 2_345,
                rssi_cdbm: -7_225,
                quality_flags: 3,
            },
        );
        let mut frame = [0; HOST_FRAME_MAX_SIZE];
        let len = encode_radio_to_host(&envelope, &mut frame).unwrap();
        let mut raw = [0; HOST_RAW_MAX_SIZE];
        assert_eq!(decode_radio_to_host(&frame[..len], &mut raw), Ok(envelope));

        frame[2] ^= 0x40;
        assert_eq!(
            decode_radio_to_host(&frame[..len], &mut raw),
            Err(HostFrameError::Crc)
        );
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
                    mode: OperatingMode::Dw1000BenchInitiator,
                    rx_delay: 0x3ff0,
                    tx_delay: 0x3ff1,
                    node_address: 2,
                },
            ),
            (
                "radio.configured",
                RadioToHost::Configured { configuration },
            ),
            ("radio.rx", RadioToHost::Rx { kind: 3 }),
            (
                "radio.range",
                RadioToHost::Range {
                    peer: 4,
                    exchange_id: 0x1234,
                    range_event_time_dtu: 0x01_0203_0405,
                    millimetres: 2_345,
                    rssi_cdbm: -7_225,
                    quality_flags: 3,
                },
            ),
            (
                "radio.peer_state",
                RadioToHost::PeerState {
                    peer: 4,
                    exchange_id: 0x1234,
                    state: golden_state(),
                },
            ),
            (
                "radio.error",
                RadioToHost::Error {
                    diagnostic: Diagnostic::UnsupportedInMode,
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
                        missed_deadlines: 5,
                        malformed_air_frames: 6,
                        unexpected_air_frames: 7,
                        host_decode_errors: 8,
                        host_command_drops: 9,
                        control_drops: 10,
                        measurement_drops: 11,
                        diagnostic_drops: 12,
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
#   UPDATE_GOLDEN=1 cargo test -p mission10-uwb-protocol committed_fixture
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
        if std::env::var_os("UPDATE_GOLDEN").is_some() {
            let path = concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/testdata/host_protocol_v3.frames"
            );
            std::fs::write(path, &rendered).expect("write golden fixture");
            return;
        }
        assert_eq!(
            rendered,
            include_str!("../testdata/host_protocol_v3.frames"),
            "golden fixture is stale; regenerate with \
             UPDATE_GOLDEN=1 cargo test -p mission10-uwb-protocol committed_fixture",
        );
    }

    fn golden_configuration() -> RadioConfiguration {
        RadioConfiguration {
            node_address: 2,
            peer_count: 3,
            peers: [0, 4, 0x8000],
        }
    }

    fn golden_state() -> EgoState {
        EgoState {
            sample_time_us: 0x0102_0304_0506_0708,
            sequence: 0x1112_1314,
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
