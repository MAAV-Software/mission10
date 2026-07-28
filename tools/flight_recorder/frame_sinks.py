"""Extra consumers for frames the recorder already holds.

The recorder owns the CM2. One capture feeds every consumer that needs the
nadir view (rfd-single-camera-sensing 3.1), so a consumer that wants those
frames takes them here rather than subscribing to 20 MB/s of imagery over
DDS.

The bag comes first and is unchanged: it is written synchronously in the
capture thread, before any sink runs. A sink is a strictly optional tap. It
runs on its own thread behind a bounded queue that drops the oldest frame
when it is full, and it swallows its own faults, so no sink can slow the
capture thread or stop the recorder.
"""
from __future__ import annotations

import collections
import json
import math
import sys
import threading
import time
from typing import NamedTuple


class Frame(NamedTuple):
    """One captured image and the bag time it was written under."""

    img: object  # sensor_msgs.msg.Image, already written to the bag
    ts_ns: int


class FanoutSink:
    """Submit a captured frame to independent non-blocking consumers."""

    def __init__(self, *sinks):
        self.sinks = tuple(sink for sink in sinks if sink is not None)

    def submit(self, img, ts_ns):
        for sink in self.sinks:
            sink.submit(img, ts_ns)

    def close(self):
        for sink in self.sinks:
            sink.close()

    def summaries(self):
        return [sink.summary() for sink in self.sinks]


class DetectorSink:
    """Runs the AprilTag detector on nadir frames and publishes the results.

    Detection costs about 90 ms per frame on the CM5 against a 100 ms frame
    interval, so the queue is a safety net for the tail, not the rate limiter.
    A frame that arrives while the previous one is still in the detector waits;
    a frame that arrives when the queue is full evicts the oldest, because a
    stale fix is worth less than a fresh one.
    """

    def __init__(
        self, detector, publish, bag=None, topic=None, depth=2,
        debug_publish=None, debug_topic=None,
    ):
        self.detector = detector
        self.publish = publish
        self.bag = bag
        self.topic = topic
        self.debug_publish = debug_publish
        self.debug_topic = debug_topic
        self._queue = collections.deque(maxlen=depth)
        self._wake = threading.Condition()
        self._stop = False
        self.submitted = 0
        self.processed = 0
        self.dropped = 0
        self.detections = 0
        self.faults = 0
        self.last_fault = None
        self._latency_ms = collections.deque(maxlen=200)
        self._thread = threading.Thread(target=self._run, name="detector-sink",
                                        daemon=True)
        self._thread.start()

    def submit(self, img, ts_ns):
        """Hand over one frame. Never blocks, never raises."""
        frame = Frame(img, ts_ns)
        with self._wake:
            self.submitted += 1
            if len(self._queue) == self._queue.maxlen:
                self.dropped += 1
            self._queue.append(frame)
            self._wake.notify()

    def close(self, timeout=2.0):
        with self._wake:
            self._stop = True
            self._wake.notify_all()
        self._thread.join(timeout)

    def _run(self):
        while True:
            with self._wake:
                while not self._queue and not self._stop:
                    self._wake.wait(0.25)
                if not self._queue:
                    if self._stop:
                        return
                    continue
                frame = self._queue.popleft()
            try:
                self._handle(frame)
            except Exception as exc:  # a sink fault is never a capture fault
                self.faults += 1
                self.last_fault = str(exc)
                if self.faults in (1, 10, 100):
                    print(f"recorder: detector sink fault: {exc}",
                          file=sys.stderr, flush=True)

    def _handle(self, frame):
        t0 = time.perf_counter()
        result = self.detector(frame.img)
        if isinstance(result, tuple):
            msg, debug_msg = result
        else:
            msg, debug_msg = result, None
        self.processed += 1
        self.detections += len(msg.detections)
        self._latency_ms.append((time.perf_counter() - t0) * 1e3)
        if self.publish is not None:
            self.publish(msg)
        if self.bag is not None and self.topic is not None:
            from rclpy.serialization import serialize_message

            self.bag.write(self.topic, serialize_message(msg), frame.ts_ns)
        if debug_msg is not None:
            if self.debug_publish is not None:
                self.debug_publish(debug_msg)
            if self.bag is not None and self.debug_topic is not None:
                from rclpy.serialization import serialize_message

                self.bag.write(
                    self.debug_topic, serialize_message(debug_msg), frame.ts_ns
                )

    def summary(self):
        lat = sorted(self._latency_ms)
        med = lat[len(lat) // 2] if lat else 0.0
        worst = lat[-1] if lat else 0.0
        return (
            f"detector: {self.processed} frames, {self.detections} tags, "
            f"{self.dropped} dropped of {self.submitted}, "
            f"{med:.0f} ms median / {worst:.0f} ms worst, {self.faults} faults"
        )


class FlowSink:
    """Run flight-critical flow without blocking the camera receive thread."""

    def __init__(
        self,
        frontend,
        range_history,
        publish,
        debug_publish,
        bag,
        flow_topic="/fmu/in/sensor_optical_flow",
        debug_topic="/localization/cm2_flow/debug",
        fault_topic="/camera_down/image_fault",
        depth=2,
    ):
        self.frontend = frontend
        self.range_history = range_history
        self.publish = publish
        self.debug_publish = debug_publish
        self.bag = bag
        self.flow_topic = flow_topic
        self.debug_topic = debug_topic
        self.fault_topic = fault_topic
        self._queue = collections.deque(maxlen=depth)
        self._wake = threading.Condition()
        self._stop = False
        self.submitted = 0
        self.processed = 0
        self.dropped = 0
        self.valid = 0
        self.faults = 0
        self.error_count = 0
        self.last_fault = None
        self._latency_ms = collections.deque(maxlen=300)
        self._raw_ring = collections.deque(maxlen=30)
        self._fault_post = 0
        self._last_fault_dump_ns = 0
        self._valid_streak = 0
        self._sequence = 0
        self._thread = threading.Thread(
            target=self._run, name="cm2-flow-sink", daemon=True
        )
        self._thread.start()

    def submit(self, img, ts_ns):
        frame = Frame(img, ts_ns)
        with self._wake:
            self.submitted += 1
            if len(self._queue) == self._queue.maxlen:
                self.dropped += 1
            self._queue.append(frame)
            self._wake.notify()

    def trigger_raw_clip(self):
        with self._wake:
            self._dump_preroll_locked()
            self._fault_post = 30

    def close(self, timeout=3.0):
        with self._wake:
            self._stop = True
            self._wake.notify_all()
        self._thread.join(timeout)

    def _dump_preroll_locked(self):
        for frame in self._raw_ring:
            self.bag.write(
                self.fault_topic,
                self._serialize(frame.img),
                frame.ts_ns,
            )

    @staticmethod
    def _serialize(msg):
        from rclpy.serialization import serialize_message

        return serialize_message(msg)

    def _run(self):
        while True:
            with self._wake:
                while not self._queue and not self._stop:
                    self._wake.wait(0.25)
                if not self._queue:
                    if self._stop:
                        return
                    continue
                frame = self._queue.popleft()
            try:
                self._handle(frame)
            except Exception as exc:
                self.faults += 1
                self.error_count += 1
                self.last_fault = str(exc)
                if self.faults in (1, 10, 100):
                    print(
                        f"recorder: CM2 flow sink fault: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )

    def _handle(self, frame):
        from px4_msgs.msg import SensorOpticalFlow
        from std_msgs.msg import String

        t0 = time.perf_counter()
        result = self.frontend.process(frame.img, frame.ts_ns)
        latency_ms = (time.perf_counter() - t0) * 1e3
        self._latency_ms.append(latency_ms)
        self.processed += 1
        self._sequence += 1
        if result.quality > 0:
            self.valid += 1
            self._valid_streak += 1
        else:
            self.error_count += 1
            self._valid_streak = 0

        msg = SensorOpticalFlow()
        msg.timestamp = time.time_ns() // 1000
        msg.timestamp_sample = result.timestamp_sample_ns // 1000
        msg.device_id = 0
        msg.pixel_flow[:] = [float(value) for value in result.pixel_flow_raw]
        msg.delta_angle[:] = [float(value) for value in result.delta_angle]
        msg.delta_angle_available = bool(np_all_finite(result.delta_angle))
        msg.distance_m = math.nan
        msg.distance_available = False
        msg.integration_timespan_us = max(0, int(result.integration_timespan_us))
        msg.quality = int(result.quality)
        msg.error_count = self.error_count
        msg.max_flow_rate = 0.55
        msg.min_ground_distance = 0.4
        msg.max_ground_distance = 4.0
        msg.mode = SensorOpticalFlow.MODE_BRIGHT
        self.publish(msg)
        self.bag.write(self.flow_topic, self._serialize(msg), frame.ts_ns)

        range_row = self.range_history.nearest(result.timestamp_sample_ns)
        if range_row is None:
            distance_m = math.nan
            distance_age_ms = math.nan
            distance_quality = -1
        else:
            distance_m = range_row[1]
            distance_quality = range_row[2]
            distance_age_ms = abs(
                range_row[0] - result.timestamp_sample_ns
            ) / 1e6
        debug = String()
        debug.data = json.dumps(
            {
                "sequence": self._sequence,
                "timestamp_sample_ns": result.timestamp_sample_ns,
                "status": result.status,
                "quality": result.quality,
                "integration_timespan_us": result.integration_timespan_us,
                "error_count": self.error_count,
                "submitted_frames": self.submitted,
                "dropped_frames": self.dropped,
                "detected_features": result.detected,
                "tracked_features": result.tracked,
                "homography_inliers": result.inliers,
                "tile_coverage": result.coverage,
                "homography_inlier_fraction": result.inlier_fraction,
                "fb_error_median_px": result.fb_median_px,
                "compensated_residual_p95_rad": result.residual_p95_rad,
                "processing_latency_ms": latency_ms,
                "pixel_flow_raw_rad": result.pixel_flow_raw.tolist(),
                "pixel_flow_compensated_rad": (
                    result.pixel_flow_compensated.tolist()
                ),
                "delta_angle_rad": result.delta_angle.tolist(),
                "distance_m": distance_m,
                "distance_age_ms": distance_age_ms,
                "distance_quality": distance_quality,
                "tracks_xyxy": result.tracks_xyxy or [],
                "track_fb_error_px": result.track_fb or [],
            },
            separators=(",", ":"),
            allow_nan=True,
        )
        self.debug_publish(debug)
        self.bag.write(self.debug_topic, self._serialize(debug), frame.ts_ns)

        with self._wake:
            self._raw_ring.append(frame)
            if self._fault_post > 0:
                self.bag.write(
                    self.fault_topic, self._serialize(frame.img), frame.ts_ns
                )
                self._fault_post -= 1

    def summary(self):
        lat = sorted(self._latency_ms)
        median = lat[len(lat) // 2] if lat else 0.0
        p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else 0.0
        return (
            f"flow: {self.valid}/{self.processed} valid, "
            f"{self.dropped} queue drops of {self.submitted}, "
            f"{median:.1f} ms median / {p95:.1f} ms p95, "
            f"{self.faults} worker faults"
        )


def np_all_finite(values):
    # Keep NumPy out of this utility module's import path.
    return all(math.isfinite(float(value)) for value in values)
