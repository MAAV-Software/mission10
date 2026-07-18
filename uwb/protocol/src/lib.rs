#![cfg_attr(not(test), no_std)]

//! Mission 10 UWB wire types.
//!
//! The host link uses typed Hubpack messages with COBS framing and CRC-32. The
//! separate legacy air codec exactly mirrors `ros/uwb/uwb_core/ranger.py` so a
//! DWM3001CDK can range with the surviving DW1000 during bring-up.

use crc::{CRC_32_ISO_HDLC, Crc};
use hubpack::SerializedSize;
use serde::{Deserialize, Serialize};

pub const HOST_PROTOCOL_VERSION: u8 = 2;

/// Stable, non-overlapping diagnostics reported by the radio task.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, SerializedSize)]
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
    Unknown,
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
        role: u8,
        rx_delay: u16,
        tx_delay: u16,
    },
    Rx {
        kind: u8,
    },
    Range {
        peer: u8,
        millimetres: u32,
    },
    Error {
        diagnostic: Diagnostic,
    },
    Health {
        irq_wakes: u32,
        spurious_irq_wakes: u32,
        wait_timeouts: u32,
        recoveries: u32,
    },
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, SerializedSize)]
pub struct HostEnvelope {
    pub protocol_version: u8,
    pub sequence: u32,
    pub message: RadioToHost,
}

impl HostEnvelope {
    pub const fn new(sequence: u32, message: RadioToHost) -> Self {
        Self {
            protocol_version: HOST_PROTOCOL_VERSION,
            sequence,
            message,
        }
    }
}

pub const HOST_ENVELOPE_MAX_SIZE: usize = <HostEnvelope as SerializedSize>::MAX_SIZE;
pub const HOST_RAW_MAX_SIZE: usize = HOST_ENVELOPE_MAX_SIZE + 4;
pub const HOST_FRAME_MAX_SIZE: usize = corncobs::max_encoded_len(HOST_RAW_MAX_SIZE);

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

pub fn encode_host_frame(
    envelope: &HostEnvelope,
    output: &mut [u8; HOST_FRAME_MAX_SIZE],
) -> Result<usize, HostFrameError> {
    let mut raw = [0_u8; HOST_RAW_MAX_SIZE];
    let payload_len = hubpack::serialize(&mut raw[..HOST_ENVELOPE_MAX_SIZE], envelope)
        .map_err(|_| HostFrameError::Hubpack)?;
    let crc = HOST_CRC.checksum(&raw[..payload_len]).to_le_bytes();
    raw[payload_len..payload_len + crc.len()].copy_from_slice(&crc);
    Ok(corncobs::encode_buf(
        &raw[..payload_len + crc.len()],
        output,
    ))
}

pub fn decode_host_frame(
    frame: &[u8],
    raw: &mut [u8; HOST_RAW_MAX_SIZE],
) -> Result<HostEnvelope, HostFrameError> {
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
    let (envelope, rest) = hubpack::deserialize::<HostEnvelope>(&raw[..payload_len])
        .map_err(|_| HostFrameError::Hubpack)?;
    if !rest.is_empty() {
        return Err(HostFrameError::TrailingData);
    }
    if envelope.protocol_version != HOST_PROTOCOL_VERSION {
        return Err(HostFrameError::Version(envelope.protocol_version));
    }
    Ok(envelope)
}

pub const FRAME_LEN: usize = 18;
pub const TIMESTAMP_MASK: u64 = (1_u64 << 40) - 1;
pub const REPLY_DELAY_US: u32 = 7_000;
pub const DTU_PER_US: u64 = 63_897;
pub const DTU_METRES: f64 = 0.004_691_763_978_615_9;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum MessageKind {
    Poll = 0,
    PollAck = 1,
    Range = 2,
    RangeReport = 3,
}

impl TryFrom<u8> for MessageKind {
    type Error = DecodeError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Poll),
            1 => Ok(Self::PollAck),
            2 => Ok(Self::Range),
            3 => Ok(Self::RangeReport),
            other => Err(DecodeError::UnknownKind(other)),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodeError {
    WrongLength(usize),
    UnknownKind(u8),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Frame {
    bytes: [u8; FRAME_LEN],
}

impl Frame {
    pub fn new(kind: MessageKind, source: [u8; 2]) -> Self {
        let mut bytes = [0; FRAME_LEN];
        bytes[0] = kind as u8;
        bytes[1..3].copy_from_slice(&source);
        Self { bytes }
    }

    pub fn poll(source: [u8; 2]) -> Self {
        Self::new(MessageKind::Poll, source)
    }

    pub fn poll_ack(source: [u8; 2]) -> Self {
        Self::new(MessageKind::PollAck, source)
    }

    pub fn range(source: [u8; 2], poll_tx: u64, poll_ack_rx: u64, range_tx: u64) -> Self {
        let mut frame = Self::new(MessageKind::Range, source);
        frame.write_timestamp(3, poll_tx);
        frame.write_timestamp(8, poll_ack_rx);
        frame.write_timestamp(13, range_tx);
        frame
    }

    pub fn range_report(source: [u8; 2], distance_mm: u32) -> Self {
        let mut frame = Self::new(MessageKind::RangeReport, source);
        frame.bytes[3..7].copy_from_slice(&distance_mm.to_le_bytes());
        frame
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, DecodeError> {
        if bytes.len() != FRAME_LEN {
            return Err(DecodeError::WrongLength(bytes.len()));
        }
        MessageKind::try_from(bytes[0])?;
        let mut frame = [0; FRAME_LEN];
        frame.copy_from_slice(bytes);
        Ok(Self { bytes: frame })
    }

    pub fn kind(&self) -> MessageKind {
        MessageKind::try_from(self.bytes[0]).expect("constructed Frame always has a valid kind")
    }

    pub fn source(&self) -> [u8; 2] {
        [self.bytes[1], self.bytes[2]]
    }

    pub fn timestamps(&self) -> Option<[u64; 3]> {
        (self.kind() == MessageKind::Range).then(|| {
            [
                self.read_timestamp(3),
                self.read_timestamp(8),
                self.read_timestamp(13),
            ]
        })
    }

    pub fn distance_mm(&self) -> Option<u32> {
        (self.kind() == MessageKind::RangeReport)
            .then(|| u32::from_le_bytes(self.bytes[3..7].try_into().expect("fixed-width field")))
    }

    pub fn as_bytes(&self) -> &[u8; FRAME_LEN] {
        &self.bytes
    }

    fn write_timestamp(&mut self, offset: usize, timestamp: u64) {
        let bytes = (timestamp & TIMESTAMP_MASK).to_le_bytes();
        self.bytes[offset..offset + 5].copy_from_slice(&bytes[..5]);
    }

    fn read_timestamp(&self, offset: usize) -> u64 {
        let mut bytes = [0; 8];
        bytes[..5].copy_from_slice(&self.bytes[offset..offset + 5]);
        u64::from_le_bytes(bytes)
    }
}

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

/// Asymmetric double-sided TWR, with the same ordering as the Python ranger.
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
    (tof_dtu.is_finite() && tof_dtu >= 0.0).then_some(tof_dtu * DTU_METRES)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_range_frame_matches_python_layout() {
        let frame = Frame::range(
            [0xa0, 0xc0],
            0x01_02_03_04_05,
            0x11_12_13_14_15,
            0x21_22_23_24_25,
        );
        assert_eq!(
            frame.as_bytes(),
            &[
                2, 0xa0, 0xc0, 5, 4, 3, 2, 1, 0x15, 0x14, 0x13, 0x12, 0x11, 0x25, 0x24, 0x23, 0x22,
                0x21,
            ]
        );
        assert_eq!(
            Frame::decode(frame.as_bytes()).unwrap().timestamps(),
            frame.timestamps()
        );
    }

    #[test]
    fn golden_range_report_is_little_endian_millimetres() {
        let frame = Frame::range_report([0xa1, 0xc1], 0x1234_5678);
        assert_eq!(
            &frame.as_bytes()[..8],
            &[3, 0xa1, 0xc1, 0x78, 0x56, 0x34, 0x12, 0]
        );
        assert_eq!(frame.distance_mm(), Some(0x1234_5678));
    }

    #[test]
    fn timestamp_delta_wraps_at_40_bits() {
        assert_eq!(wrapping_delta(25, TIMESTAMP_MASK - 24), 50);
    }

    #[test]
    fn delayed_transmit_is_512_dtu_aligned() {
        let scheduled = delayed_tx_time(TIMESTAMP_MASK - 1_000, REPLY_DELAY_US);
        assert_eq!(scheduled & 0x1ff, 0);
        assert!(scheduled <= TIMESTAMP_MASK);
    }

    #[test]
    fn synthetic_ds_twr_recovers_known_time_of_flight() {
        let tof = 2_130_u64; // almost exactly 10 metres
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

    #[test]
    fn decode_rejects_bad_length_and_kind() {
        assert_eq!(Frame::decode(&[0; 4]), Err(DecodeError::WrongLength(4)));
        let mut raw = [0; FRAME_LEN];
        raw[0] = 99;
        assert_eq!(Frame::decode(&raw), Err(DecodeError::UnknownKind(99)));
    }

    #[test]
    fn host_frame_round_trips_hubpack_crc_and_cobs() {
        let envelope = HostEnvelope::new(
            0x1234_5678,
            RadioToHost::Otp {
                tx_power: 0x6161_6161,
                antenna: 0x3ff0_3ff0,
                xtal: 0x00be_0019,
                revision: 0x0001_0201,
            },
        );
        let mut encoded = [0_u8; HOST_FRAME_MAX_SIZE];
        let length = encode_host_frame(&envelope, &mut encoded).unwrap();
        assert_eq!(
            &encoded[..length],
            &[
                0x10, 0x02, 0x78, 0x56, 0x34, 0x12, 0x01, 0x61, 0x61, 0x61, 0x61, 0xf0, 0x3f, 0xf0,
                0x3f, 0x19, 0x02, 0xbe, 0x04, 0x01, 0x02, 0x01, 0x05, 0x93, 0xc6, 0xb3, 0xfb, 0x00,
            ]
        );
        assert_eq!(encoded[length - 1], 0);
        assert!(!encoded[..length - 1].contains(&0));

        let mut raw = [0_u8; HOST_RAW_MAX_SIZE];
        assert_eq!(
            decode_host_frame(&encoded[..length], &mut raw),
            Ok(envelope)
        );
    }

    #[test]
    fn host_frame_rejects_corruption_and_truncation() {
        let envelope = HostEnvelope::new(
            7,
            RadioToHost::Range {
                peer: 1,
                millimetres: 12_345,
            },
        );
        let mut encoded = [0_u8; HOST_FRAME_MAX_SIZE];
        let length = encode_host_frame(&envelope, &mut encoded).unwrap();
        encoded[2] ^= 0x40;
        let mut raw = [0_u8; HOST_RAW_MAX_SIZE];
        assert_eq!(
            decode_host_frame(&encoded[..length], &mut raw),
            Err(HostFrameError::Crc)
        );
        assert_eq!(
            decode_host_frame(&encoded[..length - 1], &mut raw),
            Err(HostFrameError::Cobs)
        );
    }
}
