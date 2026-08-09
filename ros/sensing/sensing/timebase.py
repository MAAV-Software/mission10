"""Camera and PX4 timestamp checks owned by the sensing contract."""
from __future__ import annotations

from collections import deque
import threading
import time

from px4_msgs.msg import TimesyncStatus


def boottime_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_BOOTTIME)


def realtime_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_REALTIME)


class ClockMapper:
    """Map one camera's CLOCK_BOOTTIME stamps into ROS CLOCK_REALTIME."""

    def __init__(self, step_threshold_ns: int) -> None:
        self.step_threshold_ns = step_threshold_ns
        self.initial_offset_ns = self._sample_offset()
        self.last_offset_ns = self.initial_offset_ns
        self.max_offset_delta_ns = 0
        self.last_realtime_ns = 0
        self.nonmonotonic_drops = 0
        self.clock_step_events = 0
        self.max_clock_step_ns = 0

    @staticmethod
    def _sample_offset() -> int:
        best = None
        for _ in range(7):
            b0 = boottime_ns()
            realtime = realtime_ns()
            b1 = boottime_ns()
            sample = (b1 - b0, realtime - (b0 + b1) // 2)
            if best is None or sample[0] < best[0]:
                best = sample
        return int(best[1])

    def map(self, sensor_ns: int) -> tuple[int, int] | None:
        offset = self._sample_offset()
        step = offset - self.last_offset_ns
        absolute_step = abs(step)
        self.max_clock_step_ns = max(self.max_clock_step_ns, absolute_step)
        if absolute_step > self.step_threshold_ns:
            self.clock_step_events += 1
        self.last_offset_ns = offset
        self.max_offset_delta_ns = max(
            self.max_offset_delta_ns, abs(offset - self.initial_offset_ns)
        )
        mapped = sensor_ns + offset
        if mapped <= self.last_realtime_ns:
            self.nonmonotonic_drops += 1
            return None
        self.last_realtime_ns = mapped
        return mapped, offset


class SyncMonitor:
    """Verify that PX4 samples and camera stamps share ROS realtime."""

    def __init__(self, max_rtt_us: int) -> None:
        self.max_rtt_us = max_rtt_us
        self.cv = threading.Condition()
        self.samples = deque(maxlen=5)
        self.source_protocol = 0
        self.last_sync_mono = 0.0
        self.last_imu_mono = 0.0
        self.last_imu_ns = 0

    def note_timesync(self, msg) -> None:
        with self.cv:
            self.samples.append((int(msg.estimated_offset), int(msg.round_trip_time)))
            self.source_protocol = int(msg.source_protocol)
            self.last_sync_mono = time.monotonic()
            self.cv.notify_all()

    def note_imu(self, timestamp_ns: int) -> None:
        with self.cv:
            self.last_imu_ns = timestamp_ns
            self.last_imu_mono = time.monotonic()
            self.cv.notify_all()

    def status(self) -> tuple[bool, str]:
        now = time.monotonic()
        if len(self.samples) < 3:
            return False, f"waiting for timesync samples ({len(self.samples)}/3)"
        if self.source_protocol != TimesyncStatus.SOURCE_PROTOCOL_DDS:
            return False, "timesync source is not DDS"
        if now - self.last_sync_mono > 2.0:
            return False, "timesync_status is stale"
        offsets = [row[0] for row in self.samples]
        rtts = [row[1] for row in self.samples]
        if any(value == 0 for value in offsets):
            return False, "timesync offset is zero"
        if max(rtts) > self.max_rtt_us:
            return False, f"timesync RTT {max(rtts) / 1000:.1f} ms exceeds limit"
        spread = max(offsets) - min(offsets)
        if spread > 5000:
            return False, f"timesync offset spread {spread / 1000:.3f} ms exceeds limit"
        if not self.last_imu_ns or now - self.last_imu_mono > 1.0:
            return False, "sensor_combined is stale"
        if abs(realtime_ns() - self.last_imu_ns) > 2_000_000_000:
            return False, "IMU timestamp is not in ROS realtime"
        return True, (
            f"DDS time sync ready: RTT max={max(rtts) / 1000:.1f} ms, "
            f"offset spread={spread / 1000:.3f} ms"
        )

    def wait_ready(self, timeout: float, stop: threading.Event | None = None) -> tuple[bool, str]:
        deadline = time.monotonic() + timeout
        with self.cv:
            while stop is None or not stop.is_set():
                ok, detail = self.status()
                if ok:
                    return True, detail
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False, detail
                self.cv.wait(min(0.25, remaining))
        return False, "stop requested"
