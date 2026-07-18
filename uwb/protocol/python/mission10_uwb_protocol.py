"""Independent standard-library codec for Mission 10's UWB host stream."""

from __future__ import annotations

import argparse
import dataclasses
import struct
import time
import tty
import zlib
from collections.abc import Iterator
from typing import BinaryIO

PROTOCOL_VERSION = 3
MAX_PEERS = 3
MAX_FRAME_SIZE = 64
EGO_STATE_FORMAT = "<QIhhh3i3hBH"
EGO_STATE_SIZE = struct.calcsize(EGO_STATE_FORMAT)
CONFIGURATION_FORMAT = "<HB3H"
CONFIGURATION_SIZE = struct.calcsize(CONFIGURATION_FORMAT)
DIAGNOSTIC_NAMES = (
    "rx_fcs",
    "rx_phy",
    "rx_buffer_too_small",
    "rx_reed_solomon",
    "rx_frame_wait_timeout",
    "rx_overrun",
    "rx_preamble_detection_timeout",
    "rx_sfd_timeout",
    "rx_frame_filtering_rejection",
    "spi",
    "frame_decode",
    "short_poll",
    "short_poll_ack",
    "short_range",
    "short_range_report",
    "invalid_distance",
    "delayed_send_too_late",
    "delayed_send_power_up_warning",
    "radio_state",
    "host_frame",
    "invalid_configuration",
    "malformed_air",
    "unexpected_air",
    "invalid_timestamp",
    "unknown",
    "unsupported_in_mode",
    "radio_reset",
    "watchdog_reset",
)


@dataclasses.dataclass(frozen=True)
class EgoState:
    sample_time_us: int = 0
    sequence: int = 0
    phase_mrad: int = 0
    phase_rate_mrad_s: int = 0
    yaw_mrad: int = 0
    position_enu_mm: tuple[int, int, int] = (0, 0, 0)
    velocity_enu_mm_s: tuple[int, int, int] = (0, 0, 0)
    mode: int = 0
    validity: int = 0


@dataclasses.dataclass(frozen=True)
class RadioConfiguration:
    node_address: int
    peers: tuple[int, ...]

    def __post_init__(self) -> None:
        if not 0 <= self.node_address < 0xFFFE:
            raise ValueError("node address is reserved")
        if len(self.peers) > MAX_PEERS:
            raise ValueError(f"at most {MAX_PEERS} peers are supported")
        if any(not 0 <= peer < 0xFFFE for peer in self.peers):
            raise ValueError("peer address is reserved")
        if self.node_address in self.peers or len(set(self.peers)) != len(self.peers):
            raise ValueError("peers must be unique and distinct from the node")


@dataclasses.dataclass(frozen=True)
class Diagnostic:
    code: int
    name: str


@dataclasses.dataclass(frozen=True)
class Envelope:
    protocol_version: int
    sequence: int
    kind: str
    fields: tuple[object, ...]


class FrameError(ValueError):
    pass


def _cobs_encode(raw: bytes) -> bytes:
    output = bytearray(b"\0")
    code_index = 0
    code = 1
    for byte in raw:
        if byte == 0:
            output[code_index] = code
            code_index = len(output)
            output.append(0)
            code = 1
        else:
            output.append(byte)
            code += 1
            if code == 0xFF:
                output[code_index] = code
                code_index = len(output)
                output.append(0)
                code = 1
    output[code_index] = code
    output.append(0)
    return bytes(output)


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


def _decode_ego_state(payload: bytes) -> EgoState:
    if len(payload) != EGO_STATE_SIZE:
        raise FrameError("ego state has the wrong length")
    values = struct.unpack(EGO_STATE_FORMAT, payload)
    return EgoState(
        sample_time_us=values[0],
        sequence=values[1],
        phase_mrad=values[2],
        phase_rate_mrad_s=values[3],
        yaw_mrad=values[4],
        position_enu_mm=values[5:8],
        velocity_enu_mm_s=values[8:11],
        mode=values[11],
        validity=values[12],
    )


def _encode_ego_state(state: EgoState) -> bytes:
    return struct.pack(
        EGO_STATE_FORMAT,
        state.sample_time_us,
        state.sequence,
        state.phase_mrad,
        state.phase_rate_mrad_s,
        state.yaw_mrad,
        *state.position_enu_mm,
        *state.velocity_enu_mm_s,
        state.mode,
        state.validity,
    )


def _decode_configuration(payload: bytes) -> RadioConfiguration:
    if len(payload) != CONFIGURATION_SIZE:
        raise FrameError("configuration has the wrong length")
    node_address, peer_count, *peer_slots = struct.unpack(CONFIGURATION_FORMAT, payload)
    if peer_count > MAX_PEERS:
        raise FrameError("configuration peer count is out of range")
    try:
        return RadioConfiguration(node_address, tuple(peer_slots[:peer_count]))
    except ValueError as error:
        raise FrameError(str(error)) from error


def _encode_configuration(configuration: RadioConfiguration) -> bytes:
    slots = (*configuration.peers, *((0,) * (MAX_PEERS - len(configuration.peers))))
    return struct.pack(
        CONFIGURATION_FORMAT,
        configuration.node_address,
        len(configuration.peers),
        *slots,
    )


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
    body = payload[6:]
    fixed_formats = {
        0: ("radio_id", "<HBBB"),
        1: ("otp", "<IIII"),
        2: ("ready", "<BHHH"),
        4: ("rx", "<B"),
        5: ("range", "<HHQIhH"),
        8: ("health", "<13I"),
    }
    if variant == 3:
        return Envelope(version, sequence, "configured", (_decode_configuration(body),))
    if variant == 6:
        if len(body) != 4 + EGO_STATE_SIZE:
            raise FrameError("peer state has the wrong length")
        peer, exchange_id = struct.unpack_from("<HH", body)
        return Envelope(
            version,
            sequence,
            "peer_state",
            (peer, exchange_id, _decode_ego_state(body[4:])),
        )
    if variant == 7:
        if len(body) != 1:
            raise FrameError("diagnostic has the wrong length")
        code = body[0]
        name = DIAGNOSTIC_NAMES[code] if code < len(DIAGNOSTIC_NAMES) else "unrecognized"
        return Envelope(version, sequence, "error", (Diagnostic(code, name),))
    try:
        kind, field_format = fixed_formats[variant]
    except KeyError as error:
        raise FrameError(f"unknown radio message variant {variant}") from error
    if len(body) != struct.calcsize(field_format):
        raise FrameError("message length does not match its variant")
    return Envelope(version, sequence, kind, struct.unpack(field_format, body))


def _encode_host_command(sequence: int, variant: int, body: bytes) -> bytes:
    payload = struct.pack("<BIB", PROTOCOL_VERSION, sequence, variant) + body
    raw = payload + struct.pack("<I", zlib.crc32(payload))
    frame = _cobs_encode(raw)
    if len(frame) > MAX_FRAME_SIZE:
        raise FrameError("encoded host command exceeds the transport bound")
    return frame


def encode_configuration(sequence: int, configuration: RadioConfiguration) -> bytes:
    return _encode_host_command(sequence, 0, _encode_configuration(configuration))


def encode_ego_state(sequence: int, state: EgoState) -> bytes:
    return _encode_host_command(sequence, 1, _encode_ego_state(state))


def encode_health_request(sequence: int, request_id: int) -> bytes:
    return _encode_host_command(sequence, 2, struct.pack("<I", request_id))


def frames(stream: BinaryIO) -> Iterator[bytes]:
    frame = bytearray()
    discarding = False
    while chunk := stream.read(64):
        for byte in chunk:
            if discarding:
                if byte == 0:
                    discarding = False
                continue
            if len(frame) == MAX_FRAME_SIZE:
                frame.clear()
                discarding = True
                continue
            frame.append(byte)
            if byte == 0:
                yield bytes(frame)
                frame.clear()


def _address(value: str) -> int:
    address = int(value, 0)
    if not 0 <= address <= 0xFFFF:
        raise argparse.ArgumentTypeError("address must fit in 16 bits")
    return address


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump and configure typed DWM3001 J20 events")
    parser.add_argument("device", help="CDC device, normally /dev/serial/by-id/usb-MAAV_...")
    parser.add_argument(
        "--configure-node", type=_address, help="set this node's short address (native mode only)"
    )
    parser.add_argument(
        "--peer", type=_address, action="append", default=[], help="peer short address (up to three)"
    )
    parser.add_argument(
        "--request-health", action="store_true", help="request an immediate health snapshot"
    )
    parser.add_argument(
        "--summary-interval",
        type=float,
        default=0.0,
        help="print range/error counts and range rate every N seconds (default: print every event)",
    )
    args = parser.parse_args()
    if args.peer and args.configure_node is None:
        parser.error("--peer requires --configure-node")

    started = time.monotonic()
    summary_started = started
    ranges = errors = invalid = 0
    try:
        with open(args.device, "r+b", buffering=0) as stream:
            tty.setraw(stream.fileno())
            command_sequence = 0
            if args.configure_node is not None or args.request_health:
                stream.write(b"\0")
            if args.configure_node is not None:
                try:
                    configuration = RadioConfiguration(args.configure_node, tuple(args.peer))
                except ValueError as error:
                    parser.error(str(error))
                stream.write(encode_configuration(command_sequence, configuration))
                command_sequence += 1
            if args.request_health:
                stream.write(encode_health_request(command_sequence, int(time.time()) & 0xFFFFFFFF))

            for frame in frames(stream):
                try:
                    envelope = decode_frame(frame)
                except FrameError as error:
                    invalid += 1
                    if args.summary_interval <= 0:
                        print(f"discarding invalid frame {frame.hex()}: {error}", flush=True)
                    continue

                if envelope.kind == "range":
                    ranges += 1
                elif envelope.kind == "error":
                    errors += 1

                if args.summary_interval <= 0:
                    print(envelope, flush=True)
                    continue

                now = time.monotonic()
                if now - summary_started >= args.summary_interval:
                    elapsed = now - started
                    print(
                        f"ranges={ranges} rate={ranges / elapsed:.1f} Hz "
                        f"errors={errors} invalid={invalid} last={envelope}",
                        flush=True,
                    )
                    summary_started = now
    except KeyboardInterrupt:
        pass

    elapsed = max(time.monotonic() - started, 1e-6)
    print(
        f"STATS ranges={ranges} rate={ranges / elapsed:.1f} Hz "
        f"errors={errors} invalid={invalid}",
        flush=True,
    )


if __name__ == "__main__":
    main()
