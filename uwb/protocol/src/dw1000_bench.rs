//! Temporary DW1000 interoperability-bench codec matching
//! `ros/uwb/uwb_core/ranger.py`.

use crate::TIMESTAMP_MASK;

pub const FRAME_LEN: usize = 18;

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
        MessageKind::try_from(self.bytes[0]).expect("constructed frame kind")
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
            .then(|| u32::from_le_bytes(self.bytes[3..7].try_into().expect("fixed field")))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_range_matches_python_layout() {
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
        assert_eq!(Frame::decode(frame.as_bytes()).unwrap(), frame);
    }

    #[test]
    fn golden_report_is_little_endian_millimetres() {
        let frame = Frame::range_report([0xa1, 0xc1], 0x1234_5678);
        assert_eq!(
            &frame.as_bytes()[..8],
            &[3, 0xa1, 0xc1, 0x78, 0x56, 0x34, 0x12, 0]
        );
        assert_eq!(frame.distance_mm(), Some(0x1234_5678));
    }

    #[test]
    fn rejects_bad_length_and_kind() {
        assert_eq!(Frame::decode(&[0; 4]), Err(DecodeError::WrongLength(4)));
        let mut raw = [0; FRAME_LEN];
        raw[0] = 99;
        assert_eq!(Frame::decode(&raw), Err(DecodeError::UnknownKind(99)));
    }
}
