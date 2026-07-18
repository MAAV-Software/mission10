//! Native addressed UWB air protocol.
//!
//! Inputs to [`decode`] exclude the two-byte PHY FCS reported by the DW3000.

use hubpack::SerializedSize;
use serde::{Deserialize, Serialize};
use smoltcp::wire::{
    Ieee802154Address as Address, Ieee802154Frame as MacFrame, Ieee802154FrameType as FrameType,
    Ieee802154FrameVersion as FrameVersion, Ieee802154Pan as Pan, Ieee802154Repr as MacRepr,
};

use crate::{EgoState, TIMESTAMP_MASK};

pub const AIR_PROTOCOL_VERSION: u8 = 1;
pub const PAN_ID: u16 = 0x4d10;
pub const BROADCAST_ADDRESS: u16 = 0xffff;
pub const AIR_FRAME_MAX_NO_FCS: usize = 125;

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
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, SerializedSize)]
pub struct AirEnvelope {
    pub protocol_version: u8,
    pub exchange_id: u16,
    pub message: AirMessage,
}

impl AirEnvelope {
    pub const fn new(exchange_id: u16, message: AirMessage) -> Self {
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
    InvalidSource,
    InvalidDestination,
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
    pub source: u16,
    pub destination: u16,
    pub mac_sequence: u8,
    pub envelope: AirEnvelope,
}

pub fn encode(
    source: u16,
    destination: u16,
    mac_sequence: u8,
    envelope: &AirEnvelope,
) -> Result<EncodedAirFrame, AirEncodeError> {
    if !is_unicast(source) {
        return Err(AirEncodeError::InvalidSource);
    }
    if destination != BROADCAST_ADDRESS && !is_unicast(destination) {
        return Err(AirEncodeError::InvalidDestination);
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
        dst_addr: Some(short_address(destination)),
        src_pan_id: None,
        src_addr: Some(short_address(source)),
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

pub fn decode(bytes: &[u8], local_address: u16) -> Result<DecodedAirFrame, AirDecodeError> {
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
    let source = decode_short(repr.src_addr).ok_or(AirDecodeError::Addressing)?;
    let destination = decode_short(repr.dst_addr).ok_or(AirDecodeError::Addressing)?;
    if !is_unicast(source) {
        return Err(AirDecodeError::Addressing);
    }
    if destination != local_address && destination != BROADCAST_ADDRESS {
        return Err(AirDecodeError::Destination);
    }
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

const fn is_unicast(address: u16) -> bool {
    address < 0xfffe
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn round_trip(message: AirMessage) {
        let envelope = AirEnvelope::new(0x1234, message);
        let encoded = encode(2, 4, 9, &envelope).unwrap();
        let decoded = decode(encoded.bytes(), 4).unwrap();
        assert_eq!(
            decoded,
            DecodedAirFrame {
                source: 2,
                destination: 4,
                mac_sequence: 9,
                envelope,
            }
        );
    }

    #[test]
    fn every_message_round_trips() {
        round_trip(AirMessage::Poll { state: state() });
        round_trip(AirMessage::Response {
            poll_rx: 11,
            response_tx: 22,
            state: state(),
        });
        round_trip(AirMessage::Final {
            poll_tx: 11,
            response_rx: 22,
            final_tx: 33,
        });
        round_trip(AirMessage::Report {
            final_rx: 44,
            status: 0,
        });
    }

    #[test]
    fn mac_header_is_short_addressed_compressed_2006_data() {
        let frame = encode(
            0x1234,
            0x5678,
            0x9a,
            &AirEnvelope::new(7, AirMessage::Poll { state: state() }),
        )
        .unwrap();
        assert_eq!(
            &frame.bytes()[..9],
            &[0x41, 0x98, 0x9a, 0x10, 0x4d, 0x78, 0x56, 0x34, 0x12]
        );
    }

    #[test]
    fn rejects_other_destination_and_out_of_range_timestamp() {
        let envelope = AirEnvelope::new(
            0,
            AirMessage::Report {
                final_rx: 1,
                status: 0,
            },
        );
        let frame = encode(1, 2, 3, &envelope).unwrap();
        assert_eq!(decode(frame.bytes(), 4), Err(AirDecodeError::Destination));
        assert_eq!(
            encode(
                1,
                2,
                3,
                &AirEnvelope::new(
                    0,
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
    fn rejects_wrong_pan_version_and_trailing_payload() {
        let envelope = AirEnvelope::new(
            0,
            AirMessage::Report {
                final_rx: 1,
                status: 0,
            },
        );
        let frame = encode(1, 2, 3, &envelope).unwrap();

        let mut wrong_pan = frame;
        wrong_pan.bytes[4] ^= 1;
        assert_eq!(decode(wrong_pan.bytes(), 2), Err(AirDecodeError::Pan));

        let mut wrong_version = frame;
        wrong_version.bytes[MAC_HEADER_SIZE] = AIR_PROTOCOL_VERSION + 1;
        assert_eq!(
            decode(wrong_version.bytes(), 2),
            Err(AirDecodeError::ProtocolVersion(AIR_PROTOCOL_VERSION + 1))
        );

        let mut trailing = frame;
        trailing.bytes[trailing.length] = 0xaa;
        trailing.length += 1;
        assert_eq!(
            decode(trailing.bytes(), 2),
            Err(AirDecodeError::TrailingData)
        );
    }
}
