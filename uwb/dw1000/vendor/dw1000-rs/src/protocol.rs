//! Compact wire protocol used by the ranging state machine tests and examples.

use crate::device::{Eui64, ShortAddress};
use crate::error::ProtocolError;
use crate::time::DwTime;

const KIND_BLINK: u8 = 0x10;
const KIND_RANGING_INIT: u8 = 0x11;
const KIND_SCHEDULE_SYNC: u8 = 0x12;
const KIND_POLL: u8 = 0x20;
const KIND_POLL_ACK: u8 = 0x21;
const KIND_RANGE: u8 = 0x22;
const KIND_RANGE_REPORT: u8 = 0x23;
const KIND_RANGE_FAILED: u8 = 0x24;

const COMMON_HEADER_LEN: usize = 6;
const BLINK_LEN: usize = 12;
const RANGING_INIT_LEN: usize = 12;
const SCHEDULE_SYNC_PAYLOAD_LEN: usize = 9;
const POLL_TARGET_LEN: usize = 6;
const RANGE_TIMING_LEN: usize = 17;
const RANGE_REPORT_LEN: usize = 19;

/// High-level frame kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub enum FrameKind {
    /// Discovery blink from a tag.
    Blink,
    /// Anchor response that seeds the ranging session.
    RangingInit,
    /// Coordinator frame that aligns tag TDMA slots.
    ScheduleSync,
    /// Tag poll frame.
    Poll,
    /// Anchor delayed poll acknowledgement.
    PollAck,
    /// Tag range frame carrying timing snapshots.
    Range,
    /// Anchor range report carrying the final timing tuple.
    RangeReport,
    /// Anchor range failure notification.
    RangeFailed,
}

/// Common frame header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct FrameHeader {
    /// Sequence number.
    pub sequence: u8,
    /// Source short address.
    pub source: ShortAddress,
    /// Destination short address.
    pub destination: ShortAddress,
}

/// Parsed blink frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct BlinkFrame {
    /// Sequence number.
    pub sequence: u8,
    /// Source short address.
    pub source_short: ShortAddress,
    /// Source EUI-64.
    pub source_eui: Eui64,
}

/// Parsed ranging-init frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct RangingInitFrame {
    /// Sequence number.
    pub sequence: u8,
    /// Source short address.
    pub source_short: ShortAddress,
    /// Destination EUI-64.
    pub destination_eui: Eui64,
}

/// Parsed TDMA schedule-sync frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct ScheduleSyncFrame {
    /// Common header.
    pub header: FrameHeader,
    /// Monotonic coordinator epoch in milliseconds.
    pub epoch_ms: u32,
    /// Total tag slots in the TDMA frame.
    pub tag_slot_count: u8,
    /// Duration of each tag slot, in milliseconds.
    pub tag_slot_ms: u32,
}

/// Poll target entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct PollTarget {
    /// Short address of the intended anchor.
    pub short_address: ShortAddress,
    /// Reply delay in microseconds.
    pub reply_delay_us: u32,
}

/// Range timing entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct RangeTiming {
    /// Short address of the intended anchor.
    pub short_address: ShortAddress,
    /// Poll transmit timestamp from the tag.
    pub poll_sent: DwTime,
    /// Poll-ack receive timestamp from the tag.
    pub poll_ack_received: DwTime,
    /// Range transmit timestamp from the tag.
    pub range_sent: DwTime,
}

/// Payload carried by a range-report frame.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct RangeReportPayload {
    /// Poll receive timestamp from the anchor.
    pub poll_received: DwTime,
    /// Poll-ack transmit timestamp from the anchor.
    pub poll_ack_sent: DwTime,
    /// Range receive timestamp from the anchor.
    pub range_received: DwTime,
    /// Receive power measured at the anchor.
    pub receive_power_dbm: f32,
}

/// Parsed protocol frame.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub enum Frame<'a> {
    /// Blink frame.
    Blink(BlinkFrame),
    /// Ranging-init frame.
    RangingInit(RangingInitFrame),
    /// Schedule-sync frame.
    ScheduleSync(ScheduleSyncFrame),
    /// Poll frame.
    Poll {
        /// Common header.
        header: FrameHeader,
        /// Encoded poll target payload.
        payload: &'a [u8],
    },
    /// Poll-ack frame.
    PollAck {
        /// Common header.
        header: FrameHeader,
    },
    /// Range frame.
    Range {
        /// Common header.
        header: FrameHeader,
        /// Encoded range timing payload.
        payload: &'a [u8],
    },
    /// Range-report frame.
    RangeReport {
        /// Common header.
        header: FrameHeader,
        /// Parsed payload.
        payload: RangeReportPayload,
    },
    /// Range-failed frame.
    RangeFailed {
        /// Common header.
        header: FrameHeader,
    },
}

/// Detects the high-level frame kind from raw bytes.
pub fn detect_frame_kind(bytes: &[u8]) -> Result<FrameKind, ProtocolError> {
    if bytes.is_empty() {
        return Err(ProtocolError::InvalidFrame);
    }
    match bytes[0] {
        KIND_BLINK => Ok(FrameKind::Blink),
        KIND_RANGING_INIT => Ok(FrameKind::RangingInit),
        KIND_SCHEDULE_SYNC => Ok(FrameKind::ScheduleSync),
        KIND_POLL => Ok(FrameKind::Poll),
        KIND_POLL_ACK => Ok(FrameKind::PollAck),
        KIND_RANGE => Ok(FrameKind::Range),
        KIND_RANGE_REPORT => Ok(FrameKind::RangeReport),
        KIND_RANGE_FAILED => Ok(FrameKind::RangeFailed),
        _ => Err(ProtocolError::UnknownFrameKind),
    }
}

/// Parses a raw frame.
pub fn parse_frame(bytes: &[u8]) -> Result<Frame<'_>, ProtocolError> {
    match detect_frame_kind(bytes)? {
        FrameKind::Blink => {
            if bytes.len() != BLINK_LEN {
                return Err(ProtocolError::InvalidFrame);
            }
            Ok(Frame::Blink(BlinkFrame {
                sequence: bytes[1],
                source_short: short_address_from(&bytes[2..4]),
                source_eui: eui_from(&bytes[4..12]),
            }))
        }
        FrameKind::RangingInit => {
            if bytes.len() != RANGING_INIT_LEN {
                return Err(ProtocolError::InvalidFrame);
            }
            Ok(Frame::RangingInit(RangingInitFrame {
                sequence: bytes[1],
                source_short: short_address_from(&bytes[2..4]),
                destination_eui: eui_from(&bytes[4..12]),
            }))
        }
        FrameKind::ScheduleSync => {
            let header = parse_common_header(bytes)?;
            let payload = parse_schedule_sync_payload(&bytes[COMMON_HEADER_LEN..])?;
            Ok(Frame::ScheduleSync(ScheduleSyncFrame {
                header,
                epoch_ms: payload.0,
                tag_slot_count: payload.1,
                tag_slot_ms: payload.2,
            }))
        }
        FrameKind::Poll => {
            let header = parse_common_header(bytes)?;
            Ok(Frame::Poll {
                header,
                payload: &bytes[COMMON_HEADER_LEN..],
            })
        }
        FrameKind::PollAck => Ok(Frame::PollAck {
            header: parse_common_header(bytes)?,
        }),
        FrameKind::Range => {
            let header = parse_common_header(bytes)?;
            Ok(Frame::Range {
                header,
                payload: &bytes[COMMON_HEADER_LEN..],
            })
        }
        FrameKind::RangeReport => {
            let header = parse_common_header(bytes)?;
            let payload: RangeReportPayload =
                parse_range_report_payload(&bytes[COMMON_HEADER_LEN..])?;
            Ok(Frame::RangeReport { header, payload })
        }
        FrameKind::RangeFailed => Ok(Frame::RangeFailed {
            header: parse_common_header(bytes)?,
        }),
    }
}

/// Decodes poll targets from a borrowed payload slice.
pub fn decode_poll_targets(
    payload: &[u8],
    output: &mut [PollTarget],
) -> Result<usize, ProtocolError> {
    if payload.is_empty() {
        return Err(ProtocolError::InvalidFrame);
    }
    let count = payload[0] as usize;
    let needed = 1 + count * POLL_TARGET_LEN;
    if payload.len() != needed || output.len() < count {
        return Err(ProtocolError::BufferTooSmall);
    }
    for (index, target) in output.iter_mut().enumerate().take(count) {
        let base = 1 + index * POLL_TARGET_LEN;
        *target = PollTarget {
            short_address: short_address_from(&payload[base..base + 2]),
            reply_delay_us: u32::from_le_bytes([
                payload[base + 2],
                payload[base + 3],
                payload[base + 4],
                payload[base + 5],
            ]),
        };
    }
    Ok(count)
}

/// Decodes range timings from a borrowed payload slice.
pub fn decode_range_timings(
    payload: &[u8],
    output: &mut [RangeTiming],
) -> Result<usize, ProtocolError> {
    if payload.is_empty() {
        return Err(ProtocolError::InvalidFrame);
    }
    let count = payload[0] as usize;
    let needed = 1 + count * RANGE_TIMING_LEN;
    if payload.len() != needed || output.len() < count {
        return Err(ProtocolError::BufferTooSmall);
    }
    for (index, timing) in output.iter_mut().enumerate().take(count) {
        let base = 1 + index * RANGE_TIMING_LEN;
        *timing = RangeTiming {
            short_address: short_address_from(&payload[base..base + 2]),
            poll_sent: time_from(&payload[base + 2..base + 7]),
            poll_ack_received: time_from(&payload[base + 7..base + 12]),
            range_sent: time_from(&payload[base + 12..base + 17]),
        };
    }
    Ok(count)
}

/// Encodes a discovery blink.
pub fn encode_discovery_blink(
    sequence: u8,
    source_eui: Eui64,
    source_short: ShortAddress,
    buffer: &mut [u8],
) -> Result<usize, ProtocolError> {
    ensure_capacity(buffer, BLINK_LEN)?;
    buffer[0] = KIND_BLINK;
    buffer[1] = sequence;
    buffer[2..4].copy_from_slice(&source_short.to_le_bytes());
    buffer[4..12].copy_from_slice(&source_eui.raw());
    Ok(BLINK_LEN)
}

/// Encodes a ranging-init frame.
pub fn encode_ranging_init(
    sequence: u8,
    source: ShortAddress,
    destination_eui: Eui64,
    buffer: &mut [u8],
) -> Result<usize, ProtocolError> {
    ensure_capacity(buffer, RANGING_INIT_LEN)?;
    buffer[0] = KIND_RANGING_INIT;
    buffer[1] = sequence;
    buffer[2..4].copy_from_slice(&source.to_le_bytes());
    buffer[4..12].copy_from_slice(&destination_eui.raw());
    Ok(RANGING_INIT_LEN)
}

/// Encodes a schedule-sync frame.
pub fn encode_schedule_sync(
    sequence: u8,
    source: ShortAddress,
    destination: ShortAddress,
    epoch_ms: u32,
    tag_slot_count: u8,
    tag_slot_ms: u32,
    buffer: &mut [u8],
) -> Result<usize, ProtocolError> {
    let len = COMMON_HEADER_LEN + SCHEDULE_SYNC_PAYLOAD_LEN;
    ensure_capacity(buffer, len)?;
    write_common_header(KIND_SCHEDULE_SYNC, sequence, source, destination, buffer);
    let payload_start = COMMON_HEADER_LEN;
    buffer[payload_start..payload_start + 4].copy_from_slice(&epoch_ms.to_le_bytes());
    buffer[payload_start + 4] = tag_slot_count;
    buffer[payload_start + 5..payload_start + 9].copy_from_slice(&tag_slot_ms.to_le_bytes());
    Ok(len)
}

/// Encodes a poll frame.
pub fn encode_poll(
    sequence: u8,
    source: ShortAddress,
    destination: ShortAddress,
    targets: &[PollTarget],
    buffer: &mut [u8],
) -> Result<usize, ProtocolError> {
    let len = COMMON_HEADER_LEN + 1 + targets.len() * POLL_TARGET_LEN;
    ensure_capacity(buffer, len)?;
    write_common_header(KIND_POLL, sequence, source, destination, buffer);
    buffer[COMMON_HEADER_LEN] = targets.len() as u8;
    for (index, target) in targets.iter().enumerate() {
        let base = COMMON_HEADER_LEN + 1 + index * POLL_TARGET_LEN;
        buffer[base..base + 2].copy_from_slice(&target.short_address.to_le_bytes());
        buffer[base + 2..base + 6].copy_from_slice(&target.reply_delay_us.to_le_bytes());
    }
    Ok(len)
}

/// Encodes a poll-ack frame.
pub fn encode_poll_ack(
    sequence: u8,
    source: ShortAddress,
    destination: ShortAddress,
    buffer: &mut [u8],
) -> Result<usize, ProtocolError> {
    ensure_capacity(buffer, COMMON_HEADER_LEN)?;
    write_common_header(KIND_POLL_ACK, sequence, source, destination, buffer);
    Ok(COMMON_HEADER_LEN)
}

/// Encodes a range frame.
pub fn encode_range(
    sequence: u8,
    source: ShortAddress,
    destination: ShortAddress,
    timings: &[RangeTiming],
    buffer: &mut [u8],
) -> Result<usize, ProtocolError> {
    let len = COMMON_HEADER_LEN + 1 + timings.len() * RANGE_TIMING_LEN;
    ensure_capacity(buffer, len)?;
    write_common_header(KIND_RANGE, sequence, source, destination, buffer);
    buffer[COMMON_HEADER_LEN] = timings.len() as u8;
    for (index, timing) in timings.iter().enumerate() {
        let base = COMMON_HEADER_LEN + 1 + index * RANGE_TIMING_LEN;
        buffer[base..base + 2].copy_from_slice(&timing.short_address.to_le_bytes());
        buffer[base + 2..base + 7].copy_from_slice(&timing.poll_sent.to_bytes());
        buffer[base + 7..base + 12].copy_from_slice(&timing.poll_ack_received.to_bytes());
        buffer[base + 12..base + 17].copy_from_slice(&timing.range_sent.to_bytes());
    }
    Ok(len)
}

/// Encodes a range-report frame.
pub fn encode_range_report(
    sequence: u8,
    source: ShortAddress,
    destination: ShortAddress,
    payload: RangeReportPayload,
    buffer: &mut [u8],
) -> Result<usize, ProtocolError> {
    let len = COMMON_HEADER_LEN + RANGE_REPORT_LEN;
    ensure_capacity(buffer, len)?;
    write_common_header(KIND_RANGE_REPORT, sequence, source, destination, buffer);
    let payload_start = COMMON_HEADER_LEN;
    buffer[payload_start..payload_start + 5].copy_from_slice(&payload.poll_received.to_bytes());
    buffer[payload_start + 5..payload_start + 10]
        .copy_from_slice(&payload.poll_ack_sent.to_bytes());
    buffer[payload_start + 10..payload_start + 15]
        .copy_from_slice(&payload.range_received.to_bytes());
    buffer[payload_start + 15..payload_start + 19]
        .copy_from_slice(&payload.receive_power_dbm.to_le_bytes());
    Ok(len)
}

/// Encodes a range-failed frame.
pub fn encode_range_failed(
    sequence: u8,
    source: ShortAddress,
    destination: ShortAddress,
    buffer: &mut [u8],
) -> Result<usize, ProtocolError> {
    ensure_capacity(buffer, COMMON_HEADER_LEN)?;
    write_common_header(KIND_RANGE_FAILED, sequence, source, destination, buffer);
    Ok(COMMON_HEADER_LEN)
}

fn ensure_capacity(buffer: &[u8], needed: usize) -> Result<(), ProtocolError> {
    if buffer.len() < needed {
        Err(ProtocolError::BufferTooSmall)
    } else {
        Ok(())
    }
}

fn parse_common_header(bytes: &[u8]) -> Result<FrameHeader, ProtocolError> {
    if bytes.len() < COMMON_HEADER_LEN {
        return Err(ProtocolError::InvalidFrame);
    }
    Ok(FrameHeader {
        sequence: bytes[1],
        source: short_address_from(&bytes[2..4]),
        destination: short_address_from(&bytes[4..6]),
    })
}

fn parse_range_report_payload(bytes: &[u8]) -> Result<RangeReportPayload, ProtocolError> {
    if bytes.len() != RANGE_REPORT_LEN {
        return Err(ProtocolError::InvalidFrame);
    }
    Ok(RangeReportPayload {
        poll_received: time_from(&bytes[0..5]),
        poll_ack_sent: time_from(&bytes[5..10]),
        range_received: time_from(&bytes[10..15]),
        receive_power_dbm: f32::from_le_bytes([bytes[15], bytes[16], bytes[17], bytes[18]]),
    })
}

fn parse_schedule_sync_payload(bytes: &[u8]) -> Result<(u32, u8, u32), ProtocolError> {
    if bytes.len() != SCHEDULE_SYNC_PAYLOAD_LEN {
        return Err(ProtocolError::InvalidFrame);
    }
    Ok((
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        bytes[4],
        u32::from_le_bytes([bytes[5], bytes[6], bytes[7], bytes[8]]),
    ))
}

fn write_common_header(
    kind: u8,
    sequence: u8,
    source: ShortAddress,
    destination: ShortAddress,
    buffer: &mut [u8],
) {
    buffer[0] = kind;
    buffer[1] = sequence;
    buffer[2..4].copy_from_slice(&source.to_le_bytes());
    buffer[4..6].copy_from_slice(&destination.to_le_bytes());
}

fn short_address_from(bytes: &[u8]) -> ShortAddress {
    ShortAddress::new(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn eui_from(bytes: &[u8]) -> Eui64 {
    Eui64::new([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ])
}

fn time_from(bytes: &[u8]) -> DwTime {
    DwTime::from_bytes(&[bytes[0], bytes[1], bytes[2], bytes[3], bytes[4]])
}
