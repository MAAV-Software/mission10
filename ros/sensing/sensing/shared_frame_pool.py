"""Mission-owned shared-memory frames for local and external consumers.

The camera owner is the only writer.  Local sensing consumers borrow immutable
slot leases; the recorder is deliberately not a lease holder and validates a
generation seqlock around its copy.  A dead or slow recorder therefore cannot
hold a camera buffer, exhaust the pool, or delay sensing.
"""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import mmap
import os
import struct
from pathlib import Path
import threading
import time
from typing import Callable


MAGIC = b"M10CM2\0\0"
VERSION = 1
DEFAULT_SLOTS = 8
HEADER_BYTES = 128
META_BYTES = 64
HEADER = struct.Struct("<8sIIIIIIQQQ")
META = struct.Struct("<QQQQIfI20x")

# Header tuple indices and byte offsets for independently updated fields.
PUBLISHED_SEQUENCE_OFFSET = struct.calcsize("<8sIIIIII")
RECORDER_HEARTBEAT_OFFSET = PUBLISHED_SEQUENCE_OFFSET + 16


class Segment:
    """One explicit mmap file."""

    def __init__(self, path: Path, fd: int, size: int, owner: bool) -> None:
        self.path = path
        self.fd = fd
        self.size = size
        self.owner = owner
        self.mapping = mmap.mmap(fd, size, access=mmap.ACCESS_WRITE)
        self.buf = memoryview(self.mapping)

    @classmethod
    def claim_owner(cls, path: Path, size: int) -> "Segment":
        """Open the control file and hold its writer lock until close()."""
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("shared frame pool already has a live owner") from exc
            os.ftruncate(fd, size)
            return cls(path, fd, size, owner=True)
        except Exception:
            os.close(fd)
            raise

    @classmethod
    def prepare_owned(cls, path: Path, size: int) -> "Segment":
        """Open one owner-managed data file while the control lock is held."""
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            os.ftruncate(fd, size)
            return cls(path, fd, size, owner=True)
        except Exception:
            os.close(fd)
            raise

    @classmethod
    def attach(cls, path: Path, *, require_live_owner: bool = False) -> "Segment":
        fd = os.open(path, os.O_RDWR)
        try:
            if require_live_owner:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                    raise RuntimeError("shared frame pool has no live owner")
            return cls(path, fd, os.fstat(fd).st_size, owner=False)
        except Exception:
            os.close(fd)
            raise

    def close(self) -> None:
        self.buf.release()
        self.mapping.close()
        os.close(self.fd)


def segment_paths(name: str) -> tuple[Path, Path]:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
    if not name or any(character not in allowed for character in name):
        raise ValueError("shared frame pool name must be alphanumeric or underscore")
    root = Path("/dev/shm")
    return root / f"{name}_control", root / f"{name}_frames"


@dataclass(frozen=True)
class FrameMetadata:
    sequence: int
    sensor_boottime_ns: int
    realtime_ns: int
    exposure_us: int
    analogue_gain: float
    payload_bytes: int
    slot: int
    generation: int


@dataclass(frozen=True)
class BorrowedImage:
    """The sensor_msgs/Image fields used by in-process sensing frontends."""

    header: object
    height: int
    width: int
    encoding: str
    is_bigendian: int
    step: int
    data: memoryview


class FrameLease:
    """An immutable view whose slot cannot be reused until release()."""

    def __init__(
        self,
        pool: "SharedFramePool",
        metadata: FrameMetadata,
        release: Callable[[], None],
    ) -> None:
        self.pool = pool
        self.metadata = metadata
        self._release = release
        self._released = False
        self._views: list[memoryview] = []

    @property
    def ts_ns(self) -> int:
        return self.metadata.realtime_ns

    def image(
        self,
        frame_id: str = "imx219_nadir",
        encoding: str = "yuyv",
    ) -> BorrowedImage:
        from builtin_interfaces.msg import Time
        from std_msgs.msg import Header

        stamp = Time(
            sec=int(self.ts_ns // 1_000_000_000),
            nanosec=int(self.ts_ns % 1_000_000_000),
        )
        header = Header(stamp=stamp, frame_id=frame_id)
        view = self.pool.slot_view(self.metadata.slot)[: self.metadata.payload_bytes]
        self._views.append(view)
        return BorrowedImage(
            header=header,
            height=self.pool.height,
            width=self.pool.width,
            encoding=encoding,
            is_bigendian=0,
            step=self.pool.stride,
            data=view,
        )

    def release(self) -> None:
        if not self._released:
            self._released = True
            for view in self._views:
                view.release()
            self._views.clear()
            self._release()

    def __enter__(self) -> "FrameLease":
        return self

    def __exit__(self, *_exc) -> None:
        self.release()


class WritableSlot:
    """One producer reservation. Commit publishes; abort leaves it unused."""

    def __init__(self, pool: "SharedFramePool", slot: int) -> None:
        self.pool = pool
        self.slot = slot
        self.buffer = pool.slot_view(slot)
        self._done = False

    def commit(
        self,
        *,
        sensor_boottime_ns: int,
        realtime_ns: int,
        exposure_us: int,
        analogue_gain: float,
        payload_bytes: int | None = None,
    ) -> FrameMetadata:
        if self._done:
            raise RuntimeError("shared frame slot already completed")
        self._done = True
        try:
            return self.pool._commit(
                self.slot,
                sensor_boottime_ns=sensor_boottime_ns,
                realtime_ns=realtime_ns,
                exposure_us=exposure_us,
                analogue_gain=analogue_gain,
                payload_bytes=payload_bytes or self.pool.frame_bytes,
            )
        finally:
            self.buffer.release()

    def abort(self) -> None:
        self._done = True
        self.buffer.release()


class SharedFramePool:
    """Eight-slot, single-producer frame pool backed by POSIX shared memory."""

    def __init__(
        self,
        control: Segment,
        frames: Segment,
        *,
        owner: bool,
    ) -> None:
        self.control = control
        self.frames = frames
        self.owner = owner
        values = HEADER.unpack_from(control.buf)
        if values[0] != MAGIC or values[1] != VERSION:
            raise RuntimeError("shared camera pool has an incompatible header")
        (
            _magic,
            _version,
            self.slot_count,
            self.width,
            self.height,
            self.stride,
            self.frame_bytes,
            _published_sequence,
            _published_slot,
            _heartbeat,
        ) = values
        expected_control = HEADER_BYTES + self.slot_count * META_BYTES
        if control.size < expected_control or frames.size < self.slot_count * self.frame_bytes:
            raise RuntimeError("shared camera pool segments are smaller than their header")
        self._lock = threading.Lock()
        self._leases = [0] * self.slot_count
        self._next_slot = 0
        self._sequence = int(values[7])
        self.pool_full_drops = 0

    @classmethod
    def create(
        cls,
        name: str,
        width: int,
        height: int,
        *,
        slots: int = DEFAULT_SLOTS,
        bytes_per_pixel: int = 2,
    ) -> "SharedFramePool":
        if slots < 3:
            raise ValueError("shared frame pool needs at least three slots")
        stride = width * bytes_per_pixel
        frame_bytes = stride * height
        control_path, frames_path = segment_paths(name)
        control = Segment.claim_owner(control_path, HEADER_BYTES + slots * META_BYTES)
        previous_sequence = 0
        if control.size >= HEADER.size:
            previous_header = HEADER.unpack_from(control.buf)
            if previous_header[0] == MAGIC and previous_header[1] == VERSION:
                previous_sequence = int(previous_header[7])
        try:
            frames = Segment.prepare_owned(frames_path, slots * frame_bytes)
        except Exception:
            control.close()
            raise
        control.buf[:] = b"\0" * control.size
        HEADER.pack_into(
            control.buf,
            0,
            MAGIC,
            VERSION,
            slots,
            width,
            height,
            stride,
            frame_bytes,
            previous_sequence,
            0,
            0,
        )
        return cls(control, frames, owner=True)

    @classmethod
    def attach(cls, name: str) -> "SharedFramePool":
        control_path, frames_path = segment_paths(name)
        control = Segment.attach(control_path, require_live_owner=True)
        try:
            frames = Segment.attach(frames_path)
        except Exception:
            control.close()
            raise
        return cls(control, frames, owner=False)

    def slot_view(self, slot: int) -> memoryview:
        start = slot * self.frame_bytes
        return self.frames.buf[start : start + self.frame_bytes]

    def _meta_offset(self, slot: int) -> int:
        return HEADER_BYTES + slot * META_BYTES

    def _generation(self, slot: int) -> int:
        return struct.unpack_from("<Q", self.control.buf, self._meta_offset(slot))[0]

    def begin_write(self) -> WritableSlot | None:
        if not self.owner:
            raise RuntimeError("only the camera owner can publish frames")
        with self._lock:
            for offset in range(self.slot_count):
                slot = (self._next_slot + offset) % self.slot_count
                if self._leases[slot] == 0:
                    self._next_slot = (slot + 1) % self.slot_count
                    generation = self._generation(slot)
                    if generation & 1:
                        generation += 1
                    struct.pack_into(
                        "<Q", self.control.buf, self._meta_offset(slot), generation + 1
                    )
                    return WritableSlot(self, slot)
            self.pool_full_drops += 1
            return None

    def _commit(
        self,
        slot: int,
        *,
        sensor_boottime_ns: int,
        realtime_ns: int,
        exposure_us: int,
        analogue_gain: float,
        payload_bytes: int,
    ) -> FrameMetadata:
        if payload_bytes < 1 or payload_bytes > self.frame_bytes:
            raise ValueError("published payload does not fit the shared frame slot")
        offset = self._meta_offset(slot)
        odd_generation = self._generation(slot)
        if not odd_generation & 1:
            raise RuntimeError("shared frame slot was not reserved")
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
            even_generation = odd_generation + 1
            META.pack_into(
                self.control.buf,
                offset,
                odd_generation,
                sequence,
                int(sensor_boottime_ns),
                int(realtime_ns),
                int(exposure_us),
                float(analogue_gain),
                int(payload_bytes),
            )
            # Publish the even generation last. Readers reject odd or changed values.
            struct.pack_into("<Q", self.control.buf, offset, even_generation)
            struct.pack_into(
                "<QQ",
                self.control.buf,
                PUBLISHED_SEQUENCE_OFFSET,
                sequence,
                slot,
            )
        metadata = FrameMetadata(
            sequence=sequence,
            sensor_boottime_ns=int(sensor_boottime_ns),
            realtime_ns=int(realtime_ns),
            exposure_us=int(exposure_us),
            analogue_gain=float(analogue_gain),
            payload_bytes=int(payload_bytes),
            slot=slot,
            generation=even_generation,
        )
        return metadata

    def lease(self, metadata: FrameMetadata) -> FrameLease | None:
        if not self.owner:
            raise RuntimeError("only local sensing consumers take leases")
        with self._lock:
            current = self._read_meta(metadata.slot)
            if current is None or current.sequence != metadata.sequence:
                return None
            self._leases[metadata.slot] += 1

        def release() -> None:
            with self._lock:
                if self._leases[metadata.slot] <= 0:
                    raise RuntimeError("shared frame lease released twice")
                self._leases[metadata.slot] -= 1

        return FrameLease(self, metadata, release)

    def _read_meta(self, slot: int) -> FrameMetadata | None:
        values = META.unpack_from(self.control.buf, self._meta_offset(slot))
        generation = int(values[0])
        if generation == 0 or generation & 1:
            return None
        return FrameMetadata(
            sequence=int(values[1]),
            sensor_boottime_ns=int(values[2]),
            realtime_ns=int(values[3]),
            exposure_us=int(values[4]),
            analogue_gain=float(values[5]),
            payload_bytes=int(values[6]),
            slot=slot,
            generation=generation,
        )

    def latest_after(self, last_sequence: int) -> FrameMetadata | None:
        """Return metadata for the freshest complete frame."""
        candidates = []
        for slot in range(self.slot_count):
            metadata = self._read_meta(slot)
            if metadata is not None and metadata.sequence > last_sequence:
                candidates.append(metadata)
        return max(candidates, key=lambda item: item.sequence, default=None)

    def copy(self, metadata: FrameMetadata) -> bytes | None:
        """Copy one selected frame, rejecting a concurrent overwrite."""
        before = self._generation(metadata.slot)
        if before != metadata.generation or before & 1:
            return None
        payload = bytes(self.slot_view(metadata.slot)[: metadata.payload_bytes])
        after = self._generation(metadata.slot)
        return payload if before == after and not after & 1 else None

    @property
    def published_sequence(self) -> int:
        return struct.unpack_from("<Q", self.control.buf, PUBLISHED_SEQUENCE_OFFSET)[0]

    def set_recorder_heartbeat(self, monotonic_ns: int | None = None) -> None:
        struct.pack_into(
            "<Q",
            self.control.buf,
            RECORDER_HEARTBEAT_OFFSET,
            int(monotonic_ns if monotonic_ns is not None else time.monotonic_ns()),
        )

    def recorder_alive(self, stale_after_s: float = 2.0) -> bool:
        heartbeat = struct.unpack_from(
            "<Q", self.control.buf, RECORDER_HEARTBEAT_OFFSET
        )[0]
        return heartbeat != 0 and time.monotonic_ns() - heartbeat <= int(stale_after_s * 1e9)

    def close(self) -> None:
        if self.owner:
            # Remove both names while the control lock is still held. Existing
            # mappings remain valid; a later owner gets fresh files.
            self.control.path.unlink(missing_ok=True)
            self.frames.path.unlink(missing_ok=True)
        self.frames.close()
        self.control.close()
