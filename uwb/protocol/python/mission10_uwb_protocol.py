"""Small independent decoder for Mission 10's nRF-to-host UWB stream."""

from __future__ import annotations

import argparse
import dataclasses
import struct
import tty
import zlib
from collections.abc import Iterator
from typing import BinaryIO

PROTOCOL_VERSION = 2
MAX_FRAME_SIZE = 29


@dataclasses.dataclass(frozen=True)
class Envelope:
    protocol_version: int
    sequence: int
    kind: str
    fields: tuple[int, ...]


class FrameError(ValueError):
    pass


def _cobs_decode(frame: bytes) -> bytes:
    if not frame or frame[-1] != 0:
        raise FrameError("truncated COBS frame")
    output = bytearray()
    index = 0
    end = len(frame) - 1
    while index < end:
        code = frame[index]
        if code == 0:
            raise FrameError("unexpected COBS delimiter")
        index += 1
        run_end = index + code - 1
        if run_end > end:
            raise FrameError("truncated COBS run")
        output.extend(frame[index:run_end])
        index = run_end
        if code != 0xFF and index < end:
            output.append(0)
    return bytes(output)


def decode_frame(frame: bytes) -> Envelope:
    raw = _cobs_decode(frame)
    if len(raw) < 10:
        raise FrameError("frame too short")
    payload, crc_bytes = raw[:-4], raw[-4:]
    expected_crc = struct.unpack("<I", crc_bytes)[0]
    if zlib.crc32(payload) != expected_crc:
        raise FrameError("CRC mismatch")

    version, sequence, variant = struct.unpack_from("<BIB", payload)
    if version != PROTOCOL_VERSION:
        raise FrameError(f"unsupported protocol version {version}")
    formats = {
        0: ("radio_id", "<HBBB"),
        1: ("otp", "<IIII"),
        2: ("ready", "<BHH"),
        3: ("rx", "<B"),
        4: ("range", "<BI"),
        5: ("error", "<B"),
        6: ("health", "<IIII"),
    }
    try:
        kind, field_format = formats[variant]
    except KeyError as error:
        raise FrameError(f"unknown message variant {variant}") from error
    if len(payload) != 6 + struct.calcsize(field_format):
        raise FrameError("message length does not match its variant")
    return Envelope(version, sequence, kind, struct.unpack_from(field_format, payload, 6))


def frames(stream: BinaryIO) -> Iterator[bytes]:
    frame = bytearray()
    while chunk := stream.read(64):
        for byte in chunk:
            if len(frame) == MAX_FRAME_SIZE:
                frame.clear()
            frame.append(byte)
            if byte == 0:
                yield bytes(frame)
                frame.clear()


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump typed DWM3001 J20 events")
    parser.add_argument("device", help="CDC device, normally /dev/serial/by-id/usb-MAAV_...")
    args = parser.parse_args()
    with open(args.device, "rb", buffering=0) as stream:
        tty.setraw(stream.fileno())
        for frame in frames(stream):
            try:
                print(decode_frame(frame), flush=True)
            except FrameError as error:
                print(f"discarding invalid frame {frame.hex()}: {error}", flush=True)


if __name__ == "__main__":
    main()
