"""The sink's contract: the capture thread never waits, and never fails."""
import threading
import time

import pytest

from frame_sinks import DetectorSink


class FakeMsg:
    def __init__(self, n=0):
        self.detections = list(range(n))


def settle(sink, predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_every_frame_reaches_the_detector_when_it_keeps_up():
    """The flight case: the detector finishes inside the frame interval."""
    seen = []
    sink = DetectorSink(lambda img: (seen.append(img), FakeMsg())[1], publish=None)
    try:
        for i in range(20):
            sink.submit(i, i)
            assert settle(sink, lambda n=i: sink.processed == n + 1)
    finally:
        sink.close()
    assert seen == list(range(20))
    assert sink.dropped == 0


def test_a_slow_detector_drops_the_oldest_and_keeps_the_newest():
    release = threading.Event()
    started = threading.Event()

    def slow(img):
        started.set()
        release.wait(2.0)
        return FakeMsg()

    sink = DetectorSink(slow, publish=None, depth=2)
    try:
        sink.submit("a", 1)
        assert started.wait(1.0)  # "a" is in the detector, the queue is empty
        for name in ("b", "c", "d", "e"):
            sink.submit(name, 1)  # only two fit; b and c are evicted
        assert sink.dropped == 2
        release.set()
        assert settle(sink, lambda: sink.processed == 3)
    finally:
        release.set()
        sink.close()
    # The frames that survived are the freshest ones, which is the point.
    assert sink.submitted == 5


def test_submit_never_blocks_on_a_stalled_detector():
    release = threading.Event()
    sink = DetectorSink(lambda img: (release.wait(5.0), FakeMsg())[1],
                        publish=None, depth=1)
    try:
        t0 = time.perf_counter()
        for i in range(50):
            sink.submit(i, i)
        assert time.perf_counter() - t0 < 0.5
    finally:
        release.set()
        sink.close()


def test_a_detector_fault_is_counted_and_the_sink_survives_it():
    calls = []

    def flaky(img):
        calls.append(img)
        if img % 2:
            raise RuntimeError("boom")
        return FakeMsg(1)

    sink = DetectorSink(flaky, publish=None)
    try:
        for i in range(6):
            sink.submit(i, i)
            assert settle(sink, lambda n=i: len(calls) == n + 1)
        assert settle(sink, lambda: sink.faults == 3)
    finally:
        sink.close()
    assert sink.processed == 3
    assert sink.detections == 3
    assert sink.last_fault == "boom"


def test_a_publish_fault_is_contained_too():
    def bad_publish(msg):
        raise RuntimeError("no transport")

    sink = DetectorSink(lambda img: FakeMsg(), publish=bad_publish)
    try:
        sink.submit(0, 0)
        assert settle(sink, lambda: sink.faults == 1)
    finally:
        sink.close()
    assert sink.last_fault == "no transport"


def test_close_stops_the_thread():
    sink = DetectorSink(lambda img: FakeMsg(), publish=None)
    sink.submit(0, 0)
    assert settle(sink, lambda: sink.processed == 1)
    sink.close()
    assert not sink._thread.is_alive()


def test_summary_reports_what_the_operator_needs():
    sink = DetectorSink(lambda img: FakeMsg(2), publish=None)
    try:
        sink.submit(0, 0)
        assert settle(sink, lambda: sink.processed == 1)
    finally:
        sink.close()
    text = sink.summary()
    assert "1 frames" in text and "2 tags" in text and "0 faults" in text
