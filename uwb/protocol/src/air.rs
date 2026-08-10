//! Native addressed UWB air protocol.
//!
//! Inputs to [`decode`] exclude the two-byte PHY FCS reported by the radio.

use hubpack::SerializedSize;
use serde::{Deserialize, Serialize};
use smoltcp::wire::{
    Ieee802154Address as Address, Ieee802154Frame as MacFrame, Ieee802154FrameType as FrameType,
    Ieee802154FrameVersion as FrameVersion, Ieee802154Pan as Pan, Ieee802154Repr as MacRepr,
};

use crate::scheduler::ExchangeId;
use crate::{Destination, EgoState, FleetMode, NodeAddress, TIMESTAMP_MASK};

pub const AIR_PROTOCOL_VERSION: u8 = 6;
pub const PAN_ID: u16 = 0x4d10;
pub const BROADCAST_ADDRESS: u16 = 0xffff;
pub const AIR_FRAME_MAX_NO_FCS: usize = 125;
pub const REPORT_STATUS_CLOSE: u16 = 1 << 0;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, SerializedSize)]
pub enum AirMessage {
    Poll {
        state: EgoState,
    },
    Response {
        poll_rx: u64,
        response_tx: u64,
        state: EgoState,
    },
    Final {
        poll_tx: u64,
        response_rx: u64,
        final_tx: u64,
    },
    Report {
        final_rx: u64,
        status: u16,
    },
    FleetMode {
        mode: FleetMode,
    },
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, SerializedSize)]
pub struct AirEnvelope {
    pub protocol_version: u8,
    pub exchange_id: ExchangeId,
    pub message: AirMessage,
}

impl AirEnvelope {
    pub const fn new(exchange_id: ExchangeId, message: AirMessage) -> Self {
        Self {
            protocol_version: AIR_PROTOCOL_VERSION,
            exchange_id,
            message,
        }
    }
}

pub const AIR_ENVELOPE_MAX_SIZE: usize = <AirEnvelope as SerializedSize>::MAX_SIZE;
const MAC_HEADER_SIZE: usize = 9;
const _: () = assert!(MAC_HEADER_SIZE + AIR_ENVELOPE_MAX_SIZE <= AIR_FRAME_MAX_NO_FCS);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AirEncodeError {
    BroadcastRanging,
    FleetModeUnicast,
    InvalidFleetMode,
    InvalidTimestamp,
    Hubpack,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AirDecodeError {
    Mac,
    FrameType,
    Security,
    FrameVersion,
    Pan,
    Addressing,
    Destination,
    BroadcastRanging,
    FleetModeUnicast,
    InvalidFleetMode,
    Hubpack,
    TrailingData,
    ProtocolVersion(u8),
    InvalidTimestamp,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EncodedAirFrame {
    bytes: [u8; AIR_FRAME_MAX_NO_FCS],
    length: usize,
}

impl EncodedAirFrame {
    pub fn bytes(&self) -> &[u8] {
        &self.bytes[..self.length]
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodedAirFrame {
    pub source: NodeAddress,
    pub destination: Destination,
    pub mac_sequence: u8,
    pub envelope: AirEnvelope,
}

pub fn encode(
    source: NodeAddress,
    destination: Destination,
    mac_sequence: u8,
    envelope: &AirEnvelope,
) -> Result<EncodedAirFrame, AirEncodeError> {
    match (destination, envelope.message) {
        (Destination::Broadcast, AirMessage::FleetMode { .. }) => {}
        (Destination::Broadcast, _) => return Err(AirEncodeError::BroadcastRanging),
        (Destination::Node(_), AirMessage::FleetMode { .. }) => {
            return Err(AirEncodeError::FleetModeUnicast);
        }
        (Destination::Node(_), _) => {}
    }
    if let AirMessage::FleetMode { mode } = envelope.message
        && !mode.is_valid()
    {
        return Err(AirEncodeError::InvalidFleetMode);
    }
    if !timestamps_valid(&envelope.message) {
        return Err(AirEncodeError::InvalidTimestamp);
    }

    let repr = MacRepr {
        frame_type: FrameType::Data,
        security_enabled: false,
        frame_pending: false,
        ack_request: false,
        sequence_number: Some(mac_sequence),
        pan_id_compression: true,
        frame_version: FrameVersion::Ieee802154_2006,
        dst_pan_id: Some(Pan(PAN_ID)),
        dst_addr: Some(short_address(destination.get())),
        src_pan_id: None,
        src_addr: Some(short_address(source.get())),
    };
    debug_assert_eq!(repr.buffer_len(), MAC_HEADER_SIZE);

    let mut result = EncodedAirFrame {
        bytes: [0; AIR_FRAME_MAX_NO_FCS],
        length: 0,
    };
    let payload_len = hubpack::serialize(
        &mut result.bytes[MAC_HEADER_SIZE..MAC_HEADER_SIZE + AIR_ENVELOPE_MAX_SIZE],
        envelope,
    )
    .map_err(|_| AirEncodeError::Hubpack)?;
    result.length = MAC_HEADER_SIZE + payload_len;
    let mut frame = MacFrame::new_unchecked(&mut result.bytes[..result.length]);
    repr.emit(&mut frame);
    Ok(result)
}

pub fn decode(bytes: &[u8], local_address: NodeAddress) -> Result<DecodedAirFrame, AirDecodeError> {
    let frame = MacFrame::new_checked(bytes).map_err(|_| AirDecodeError::Mac)?;
    let repr = MacRepr::parse(&frame).map_err(|_| AirDecodeError::Mac)?;
    if repr.frame_type != FrameType::Data {
        return Err(AirDecodeError::FrameType);
    }
    if repr.security_enabled {
        return Err(AirDecodeError::Security);
    }
    if repr.frame_version != FrameVersion::Ieee802154_2006 {
        return Err(AirDecodeError::FrameVersion);
    }
    if repr.dst_pan_id != Some(Pan(PAN_ID)) {
        return Err(AirDecodeError::Pan);
    }
    let source = decode_short(repr.src_addr)
        .and_then(NodeAddress::new)
        .ok_or(AirDecodeError::Addressing)?;
    let destination_raw = decode_short(repr.dst_addr).ok_or(AirDecodeError::Addressing)?;
    if destination_raw != local_address.get() && destination_raw != BROADCAST_ADDRESS {
        return Err(AirDecodeError::Destination);
    }
    let destination = if destination_raw == BROADCAST_ADDRESS {
        Destination::Broadcast
    } else {
        Destination::Node(NodeAddress::new(destination_raw).ok_or(AirDecodeError::Addressing)?)
    };
    let mac_sequence = repr.sequence_number.ok_or(AirDecodeError::Addressing)?;
    let payload = frame.payload().ok_or(AirDecodeError::FrameType)?;
    let (envelope, rest) =
        hubpack::deserialize::<AirEnvelope>(payload).map_err(|_| AirDecodeError::Hubpack)?;
    if !rest.is_empty() {
        return Err(AirDecodeError::TrailingData);
    }
    if envelope.protocol_version != AIR_PROTOCOL_VERSION {
        return Err(AirDecodeError::ProtocolVersion(envelope.protocol_version));
    }
    match (destination, envelope.message) {
        (Destination::Broadcast, AirMessage::FleetMode { .. }) => {}
        (Destination::Broadcast, _) => return Err(AirDecodeError::BroadcastRanging),
        (Destination::Node(_), AirMessage::FleetMode { .. }) => {
            return Err(AirDecodeError::FleetModeUnicast);
        }
        (Destination::Node(_), _) => {}
    }
    if let AirMessage::FleetMode { mode } = envelope.message
        && !mode.is_valid()
    {
        return Err(AirDecodeError::InvalidFleetMode);
    }
    if !timestamps_valid(&envelope.message) {
        return Err(AirDecodeError::InvalidTimestamp);
    }
    Ok(DecodedAirFrame {
        source,
        destination,
        mac_sequence,
        envelope,
    })
}

const fn short_address(address: u16) -> Address {
    Address::Short(address.to_be_bytes())
}

fn decode_short(address: Option<Address>) -> Option<u16> {
    match address {
        Some(Address::Short(bytes)) => Some(u16::from_be_bytes(bytes)),
        _ => None,
    }
}

const fn timestamp_valid(timestamp: u64) -> bool {
    timestamp <= TIMESTAMP_MASK
}

const fn timestamps_valid(message: &AirMessage) -> bool {
    match message {
        AirMessage::Poll { .. } => true,
        AirMessage::Response {
            poll_rx,
            response_tx,
            ..
        } => timestamp_valid(*poll_rx) && timestamp_valid(*response_tx),
        AirMessage::Final {
            poll_tx,
            response_rx,
            final_tx,
        } => {
            timestamp_valid(*poll_tx) && timestamp_valid(*response_rx) && timestamp_valid(*final_tx)
        }
        AirMessage::Report { final_rx, .. } => timestamp_valid(*final_rx),
        AirMessage::FleetMode { .. } => true,
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Write as _;

    use super::*;
    use crate::scheduler::ExchangeId;

    fn node(value: u16) -> NodeAddress {
        NodeAddress::new(value).unwrap()
    }

    fn exchange_id() -> ExchangeId {
        ExchangeId::new(0x5678)
    }

    fn state() -> EgoState {
        EgoState {
            sample_time_us: 123_456,
            sequence: 72,
            phase_mrad: -120,
            phase_rate_mrad_s: 33,
            yaw_mrad: 1_570,
            position_enu_mm: [1_000, -2_000, 3_000],
            velocity_enu_mm_s: [40, -50, 60],
            ..EgoState::default()
        }
    }

    fn golden_cases() -> [(&'static str, AirMessage); 5] {
        [
            ("poll", AirMessage::Poll { state: state() }),
            (
                "response",
                AirMessage::Response {
                    poll_rx: 0x01_0203_0405,
                    response_tx: 0x11_1213_1415,
                    state: state(),
                },
            ),
            (
                "final",
                AirMessage::Final {
                    poll_tx: 0x01_0203_0405,
                    response_rx: 0x11_1213_1415,
                    final_tx: 0x21_2223_2425,
                },
            ),
            (
                "report",
                AirMessage::Report {
                    final_rx: 0x31_3233_3435,
                    status: 0x1234,
                },
            ),
            (
                "fleet_mode",
                AirMessage::FleetMode {
                    mode: FleetMode::new(2, crate::FleetNetwork::Field).unwrap(),
                },
            ),
        ]
    }

    fn golden_frames() -> String {
        let mut rendered = String::new();
        for (sequence, (name, message)) in golden_cases().into_iter().enumerate() {
            let destination = if matches!(message, AirMessage::FleetMode { .. }) {
                Destination::Broadcast
            } else {
                Destination::Node(node(3))
            };
            let frame = encode(
                node(2),
                destination,
                sequence as u8,
                &AirEnvelope::new(exchange_id(), message),
            )
            .unwrap();
            write!(rendered, "{name}=").unwrap();
            for byte in frame.bytes() {
                write!(rendered, "{byte:02x}").unwrap();
            }
            rendered.push('\n');
        }
        rendered
    }

    #[test]
    fn every_air_message_matches_committed_golden_bytes() {
        let rendered = golden_frames();
        if std::env::var_os("PRINT_AIR_GOLDEN").is_some() {
            println!("{rendered}");
            return;
        }
        if std::env::var_os("UPDATE_AIR_GOLDEN").is_some() {
            let path = concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/testdata/air_protocol_v6.frames"
            );
            std::fs::write(path, &rendered).expect("write air fixture");
            return;
        }
        assert_eq!(rendered, include_str!("../testdata/air_protocol_v6.frames"));
    }

    #[test]
    fn committed_air_golden_bytes_decode_independently() {
        for (sequence, ((expected_name, expected_message), line)) in golden_cases()
            .into_iter()
            .zip(include_str!("../testdata/air_protocol_v6.frames").lines())
            .enumerate()
        {
            let (name, encoded) = line.split_once('=').unwrap();
            assert_eq!(name, expected_name);
            let bytes = encoded
                .as_bytes()
                .chunks_exact(2)
                .map(|pair| {
                    let digits = core::str::from_utf8(pair).unwrap();
                    u8::from_str_radix(digits, 16).unwrap()
                })
                .collect::<Vec<_>>();
            let decoded = decode(&bytes, node(3)).unwrap();
            assert_eq!(decoded.source, node(2));
            let expected_destination = if matches!(expected_message, AirMessage::FleetMode { .. }) {
                Destination::Broadcast
            } else {
                Destination::Node(node(3))
            };
            assert_eq!(decoded.destination, expected_destination);
            assert_eq!(decoded.mac_sequence, sequence as u8);
            assert_eq!(decoded.envelope.exchange_id, exchange_id());
            assert_eq!(decoded.envelope.message, expected_message);
        }
    }

    #[test]
    fn mac_header_is_short_addressed_compressed_2006_data() {
        let frame = encode(
            node(0x8000),
            Destination::Node(node(0x8001)),
            0x9a,
            &AirEnvelope::new(exchange_id(), AirMessage::Poll { state: state() }),
        )
        .unwrap();
        assert_eq!(
            &frame.bytes()[..9],
            &[0x41, 0x98, 0x9a, 0x10, 0x4d, 0x01, 0x80, 0x00, 0x80]
        );
    }

    #[test]
    fn ranging_is_unicast_and_timestamp_checked() {
        let envelope = AirEnvelope::new(
            exchange_id(),
            AirMessage::Report {
                final_rx: 1,
                status: 0,
            },
        );
        assert_eq!(
            encode(node(1), Destination::Broadcast, 3, &envelope),
            Err(AirEncodeError::BroadcastRanging)
        );
        assert_eq!(
            encode(
                node(1),
                Destination::Node(node(2)),
                3,
                &AirEnvelope::new(
                    exchange_id(),
                    AirMessage::Report {
                        final_rx: TIMESTAMP_MASK + 1,
                        status: 0,
                    },
                ),
            ),
            Err(AirEncodeError::InvalidTimestamp)
        );
    }

    #[test]
    fn fleet_mode_is_validated_and_broadcast_only() {
        let mode = FleetMode::new(1, crate::FleetNetwork::Internet).unwrap();
        let envelope = AirEnvelope::new(exchange_id(), AirMessage::FleetMode { mode });
        let frame = encode(node(0x8000), Destination::Broadcast, 7, &envelope).unwrap();
        let decoded = decode(frame.bytes(), node(2)).unwrap();
        assert_eq!(decoded.destination, Destination::Broadcast);
        assert_eq!(decoded.envelope.message, AirMessage::FleetMode { mode });
        assert_eq!(
            encode(node(0x8000), Destination::Node(node(2)), 7, &envelope),
            Err(AirEncodeError::FleetModeUnicast)
        );
    }

    #[test]
    fn rejects_other_destination_pan_version_and_trailing_payload() {
        let envelope = AirEnvelope::new(
            exchange_id(),
            AirMessage::Report {
                final_rx: 1,
                status: REPORT_STATUS_CLOSE,
            },
        );
        let frame = encode(node(1), Destination::Node(node(2)), 3, &envelope).unwrap();
        assert_eq!(
            decode(frame.bytes(), node(3)),
            Err(AirDecodeError::Destination)
        );

        let mut wrong_pan = frame;
        wrong_pan.bytes[4] ^= 1;
        assert_eq!(decode(wrong_pan.bytes(), node(2)), Err(AirDecodeError::Pan));

        let mut wrong_version = frame;
        wrong_version.bytes[MAC_HEADER_SIZE] = AIR_PROTOCOL_VERSION + 1;
        assert_eq!(
            decode(wrong_version.bytes(), node(2)),
            Err(AirDecodeError::ProtocolVersion(AIR_PROTOCOL_VERSION + 1))
        );

        let mut trailing = frame;
        trailing.bytes[trailing.length] = 0xaa;
        trailing.length += 1;
        assert_eq!(
            decode(trailing.bytes(), node(2)),
            Err(AirDecodeError::TrailingData)
        );
    }
}
