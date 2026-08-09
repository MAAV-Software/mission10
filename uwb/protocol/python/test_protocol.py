import io
from pathlib import Path

import pytest

from mission10_uwb_protocol import (
    EgoState,
    Diagnostic,
    Envelope,
    FrameError,
    MissionEventTime,
    RadioConfiguration,
    decode_frame,
    encode_clock_reply,
    encode_configuration,
    encode_ego_state,
    encode_health_request,
    frames,
    is_node_address,
)


GOLDEN_CONFIGURATION = RadioConfiguration(2, (0, 3, 0x8000))
GOLDEN_STATE = EgoState(
    sample_time_us=0x0102030405060708,
    sequence=0x11121314,
    frame_epoch=0x1516,
    phase_mrad=-1,
    phase_rate_mrad_s=2,
    yaw_mrad=-3,
    position_enu_mm=(4, -5, 6),
    velocity_enu_mm_s=(-7, 8, -9),
    mode=1,
    validity=0x800D,
)

GOLDEN_FIXTURE = Path(__file__).resolve().parents[1] / "testdata/host_protocol_v8.frames"
GOLDEN_FRAMES = dict(
    line.split("=", 1)
    for line in GOLDEN_FIXTURE.read_text().splitlines()
    if line and not line.startswith("#")
)
GOLDEN_RADIO_FRAMES = {
    name.removeprefix("radio."): frame
    for name, frame in GOLDEN_FRAMES.items()
    if name.startswith("radio.")
}
GOLDEN_HOST_FRAMES = {
    name.removeprefix("host."): frame
    for name, frame in GOLDEN_FRAMES.items()
    if name.startswith("host.")
}


def test_decodes_every_rust_radio_variant():
    expected = [
        Envelope(8, 0x10203040, "radio_id", (0xDECA, 3, 0, 2)),
        Envelope(8, 0x10203041, "otp", (0x61616161, 0x3FF03FF0, 0x00BE0019, 0x00010201)),
        Envelope(8, 0x10203042, "ready", (0x3FF0, 0x3FF1)),
        Envelope(8, 0x10203043, "clock_probe", (0x5678,)),
        Envelope(8, 0x10203044, "clock_status", (7, 125, 12_345, 1)),
        Envelope(8, 0x10203045, "configured", (GOLDEN_CONFIGURATION,)),
        Envelope(
            8,
            0x10203046,
            "completed_exchange",
            (
                    3,
                0x1234,
                0x0102030405,
                MissionEventTime(0x0102030405060708, 7, 125),
                2345,
                -7225,
                3,
                GOLDEN_STATE,
            ),
        ),
        Envelope(8, 0x10203047, "error", (Diagnostic(21, "radio_reset"),)),
        Envelope(8, 0x10203048, "health", (0x89ABCDEF, *range(1, 31))),
    ]
    assert [decode_frame(bytes.fromhex(frame)) for frame in GOLDEN_RADIO_FRAMES.values()] == expected


def test_host_commands_have_valid_cobs_crc_and_bounds():
    frames = (
        encode_configuration(0x01020304, GOLDEN_CONFIGURATION),
        encode_ego_state(0x01020305, GOLDEN_STATE),
        encode_health_request(0x01020306, 0x89ABCDEF),
        encode_clock_reply(
            0x01020307,
            0x5678,
            0x0102030405060708,
            0x1112131415161718,
            7,
            125,
        ),
    )
    for frame in frames:
        assert frame.endswith(b"\0")
        assert len(frame) <= 192
    assert frames == tuple(bytes.fromhex(frame) for frame in GOLDEN_HOST_FRAMES.values())


def test_rejects_corrupt_and_truncated_frames():
    golden_otp = bytes.fromhex(GOLDEN_RADIO_FRAMES["otp"])
    corrupt = bytearray(golden_otp)
    corrupt[4] ^= 0x40
    with pytest.raises(FrameError, match="CRC"):
        decode_frame(bytes(corrupt))
    with pytest.raises(FrameError, match="truncated"):
        decode_frame(golden_otp[:-1])


def test_stream_discards_the_remainder_of_an_oversized_frame():
    valid = bytes.fromhex(GOLDEN_RADIO_FRAMES["otp"])
    stream = io.BytesIO(b"x" * 193 + b"suffix\0" + valid)
    assert list(frames(stream)) == [valid]


def test_configuration_validation():
    with pytest.raises(ValueError, match="unique"):
        RadioConfiguration(2, (1, 1))
    with pytest.raises(ValueError, match="distinct"):
        RadioConfiguration(2, (2,))
    with pytest.raises(ValueError, match="required"):
        RadioConfiguration(2, ())


def test_node_address_namespace_is_exhaustive():
    for address in range(0x10000):
        expected = address <= 3 or 0x8000 <= address <= 0x80FF
        assert is_node_address(address) is expected
