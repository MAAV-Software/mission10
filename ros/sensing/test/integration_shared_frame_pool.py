#!/usr/bin/env python3
"""Process-level qualification for the CM2 shared frame pool.

This is intentionally not a unit-test suite. It exercises the deployment
boundary: a producer, an attaching recorder process, bounded stalled consumer
leases, and recorder death while production continues.
"""
from __future__ import annotations

import multiprocessing
import os
import threading
import time

from sensing.frame_sinks import DetectorSink
from sensing.shared_frame_pool import SharedFramePool


WIDTH = 32
HEIGHT = 24


class EmptyDetections:
    detections = ()


def recorder_process(name: str, stop, result) -> None:
    pool = SharedFramePool.attach(name)
    last_sequence = 0
    copied = 0
    try:
        while not stop.is_set():
            pool.set_recorder_heartbeat()
            metadata = pool.latest_after(last_sequence)
            if metadata is not None:
                payload = pool.copy(metadata)
                if payload is None:
                    continue
                expected = metadata.sequence & 0xFF
                if not payload or any(value != expected for value in payload):
                    result.put((False, "recorder observed a torn frame"))
                    return
                last_sequence = metadata.sequence
                copied += 1
            stop.wait(0.001)
        result.put((True, copied))
    finally:
        pool.close()


def publish(pool: SharedFramePool):
    slot = pool.begin_write()
    if slot is None:
        raise RuntimeError("producer exhausted the bounded pool")
    next_sequence = pool.published_sequence + 1
    slot.buffer[:] = bytes([next_sequence & 0xFF]) * pool.frame_bytes
    metadata = slot.commit(
        sensor_boottime_ns=time.monotonic_ns(),
        realtime_ns=time.time_ns(),
        exposure_us=500,
        analogue_gain=1.0,
    )
    return metadata


def main() -> int:
    name = f"m10_pool_integration_{os.getpid()}"
    pool = SharedFramePool.create(name, WIDTH, HEIGHT, slots=8)
    context = multiprocessing.get_context("spawn")
    stop = context.Event()
    stop_is_usable = True
    result = context.Queue()
    recorder = context.Process(target=recorder_process, args=(name, stop, result))
    sinks = []
    try:
        recorder.start()
        deadline = time.monotonic() + 2.0
        while not pool.recorder_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pool.recorder_alive(), "recorder heartbeat did not arrive"

        for _ in range(100):
            publish(pool)
            time.sleep(0.002)

        # Stall three real bounded sinks. Each keeps one in-flight and replaces
        # its single pending lease as fresher frames arrive.
        release_consumers = threading.Event()
        started = [threading.Event() for _ in range(3)]
        for consumer in range(3):
            sink = DetectorSink(
                lambda _image, event=started[consumer]: (
                    event.set(), release_consumers.wait(2.0), EmptyDetections()
                )[-1],
                publish=None,
                depth=1,
            )
            sinks.append(sink)
            metadata = publish(pool)
            lease = pool.lease(metadata)
            assert lease is not None
            sink.submit(metadata.sequence, metadata.realtime_ns, lease.release)
        assert all(event.wait(1.0) for event in started)
        for sink in sinks:
            for _ in range(4):
                metadata = publish(pool)
                lease = pool.lease(metadata)
                assert lease is not None
                sink.submit(metadata.sequence, metadata.realtime_ns, lease.release)
        for _ in range(100):
            publish(pool)
        assert pool.pool_full_drops == 0, "six stalled leases blocked capture"
        release_consumers.set()
        for sink in sinks:
            sink.close()

        stop.set()
        recorder.join(3.0)
        assert recorder.exitcode == 0, f"recorder exited {recorder.exitcode}"
        ok, detail = result.get(timeout=1.0)
        assert ok and detail > 0, detail

        # Start another recorder and kill it without a shutdown handshake.
        pool.set_recorder_heartbeat(0)
        stop = context.Event()
        recorder = context.Process(target=recorder_process, args=(name, stop, result))
        recorder.start()
        deadline = time.monotonic() + 2.0
        while not pool.recorder_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pool.recorder_alive()
        recorder.terminate()
        recorder.join(2.0)
        # A process terminated inside Event.wait() may leave that Event's
        # semaphore inconsistent. Do not touch the dead child's stop handle.
        stop_is_usable = False
        for _ in range(100):
            publish(pool)
        assert pool.pool_full_drops == 0, "dead recorder blocked capture"
        print("PASS: shared producer, bounded leases, seqlock copy, recorder isolation")
        return 0
    finally:
        if stop_is_usable:
            stop.set()
        if recorder.is_alive():
            recorder.terminate()
            recorder.join(2.0)
        for sink in sinks:
            sink.close()
        pool.close()


if __name__ == "__main__":
    raise SystemExit(main())
