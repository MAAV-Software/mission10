#!/usr/bin/env python3
"""Process-level qualification for the CM2 shared frame pool.

This is intentionally not a unit-test suite. It exercises the deployment
boundary: a producer, an attaching recorder process, bounded stalled consumer
leases, recorder death while production continues, and owner crash recovery.
"""
from __future__ import annotations

import multiprocessing
import os
import threading
import time

from sensing.frame_sinks import DetectorSink
from sensing.shared_frame_pool import SharedFramePool, segment_paths


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


def owner_process(name: str, ready, result) -> None:
    pool = SharedFramePool.create(name, WIDTH, HEIGHT, slots=8)
    metadata = publish(pool)
    result.put(metadata.sequence)
    ready.set()
    time.sleep(60.0)


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
    crash_name = f"{name}_owner_crash"
    crashed_owner = None
    reclaimed = None
    surviving_reader = None
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

        # A live writer keeps the exclusive control-file lock. SIGKILL releases
        # only the kernel lock, leaving both files behind for the next owner.
        owner_ready = context.Event()
        owner_result = context.Queue()
        crashed_owner = context.Process(
            target=owner_process,
            args=(crash_name, owner_ready, owner_result),
        )
        crashed_owner.start()
        assert owner_ready.wait(2.0), "crash-test owner did not initialize"
        previous_sequence = owner_result.get(timeout=1.0)
        surviving_reader = SharedFramePool.attach(crash_name)
        try:
            SharedFramePool.create(crash_name, WIDTH, HEIGHT, slots=8)
        except RuntimeError as exc:
            assert "live owner" in str(exc)
        else:
            raise AssertionError("second live owner acquired the pool")

        crashed_owner.kill()
        crashed_owner.join(2.0)
        assert crashed_owner.exitcode is not None
        try:
            SharedFramePool.attach(crash_name)
        except RuntimeError as exc:
            assert "no live owner" in str(exc)
        else:
            raise AssertionError("reader attached to an ownerless pool")

        reclaimed = SharedFramePool.create(crash_name, WIDTH, HEIGHT, slots=8)
        assert reclaimed.published_sequence == previous_sequence
        metadata = publish(reclaimed)
        assert metadata.sequence == previous_sequence + 1
        resumed = surviving_reader.latest_after(previous_sequence)
        assert resumed is not None
        assert resumed.sequence == metadata.sequence
        copied = surviving_reader.copy(resumed)
        assert copied is not None
        reclaimed.close()
        reclaimed = None
        surviving_reader.close()
        surviving_reader = None

        print(
            "PASS: shared producer, bounded leases, recorder isolation, "
            "live-owner exclusion, crash reclaim"
        )
        return 0
    finally:
        if stop_is_usable:
            stop.set()
        if recorder.is_alive():
            recorder.terminate()
            recorder.join(2.0)
        if crashed_owner is not None and crashed_owner.is_alive():
            crashed_owner.kill()
            crashed_owner.join(2.0)
        if reclaimed is not None:
            reclaimed.close()
        if surviving_reader is not None:
            surviving_reader.close()
        for sink in sinks:
            sink.close()
        pool.close()
        for pool_name in (name, crash_name):
            for path in segment_paths(pool_name):
                path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
