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
import sys
import threading
import time
from typing import NamedTuple


class Frame(NamedTuple):
    """One captured image and the bag time it was written under."""

    img: object  # sensor_msgs.msg.Image, already written to the bag
    ts_ns: int


class DetectorSink:
    """Runs the AprilTag detector on nadir frames and publishes the results.

    Detection costs about 90 ms per frame on the CM5 against a 100 ms frame
    interval, so the queue is a safety net for the tail, not the rate limiter.
    A frame that arrives while the previous one is still in the detector waits;
    a frame that arrives when the queue is full evicts the oldest, because a
    stale fix is worth less than a fresh one.
    """

    def __init__(self, detector, publish, bag=None, topic=None, depth=2):
        self.detector = detector
        self.publish = publish
        self.bag = bag
        self.topic = topic
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
        msg = self.detector(frame.img)
        self.processed += 1
        self.detections += len(msg.detections)
        self._latency_ms.append((time.perf_counter() - t0) * 1e3)
        if self.publish is not None:
            self.publish(msg)
        if self.bag is not None and self.topic is not None:
            from rclpy.serialization import serialize_message

            self.bag.write(self.topic, serialize_message(msg), frame.ts_ns)

    def summary(self):
        lat = sorted(self._latency_ms)
        med = lat[len(lat) // 2] if lat else 0.0
        worst = lat[-1] if lat else 0.0
        return (
            f"detector: {self.processed} frames, {self.detections} tags, "
            f"{self.dropped} dropped of {self.submitted}, "
            f"{med:.0f} ms median / {worst:.0f} ms worst, {self.faults} faults"
        )
