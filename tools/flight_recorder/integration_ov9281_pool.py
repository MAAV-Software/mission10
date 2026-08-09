#!/usr/bin/env python3
"""Short hardware check: OV9281 -> shared pool -> isolated MCAP writer."""
from __future__ import annotations

import argparse
import multiprocessing
from pathlib import Path
import time

import numpy as np
from picamera2 import MappedArray, Picamera2
from rclpy.serialization import serialize_message
from sensor_msgs.msg import CameraInfo, Image

from recording import Bag, stamp_ns
from sensing.camera_tuning import load_ov9281_daylight_tuning
from sensing.frame_sinks import DetectorSink
from sensing.shared_frame_pool import SharedFramePool
from sensing.timebase import ClockMapper, boottime_ns


def temperature_c() -> float:
    return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000.0


def record_pool(name: str, output: str, stop, result, fps: float) -> None:
    pool = SharedFramePool.attach(name)
    bag = Bag(output)
    bag.topic("/camera/image_raw", "sensor_msgs/msg/Image")
    bag.topic("/camera/camera_info", "sensor_msgs/msg/CameraInfo")
    info = CameraInfo(width=pool.width, height=pool.height)
    last_sequence = 0
    next_record_ns = None
    period_ns = int(1e9 / fps)
    copied = 0
    sequence_drops = 0
    try:
        while not stop.is_set():
            pool.set_recorder_heartbeat()
            metadata = pool.latest_after(last_sequence)
            if metadata is None:
                stop.wait(0.01)
                continue
            if last_sequence and metadata.sequence > last_sequence + 1:
                sequence_drops += metadata.sequence - last_sequence - 1
            last_sequence = metadata.sequence
            if next_record_ns is not None and metadata.realtime_ns < next_record_ns:
                continue
            payload = pool.copy(metadata)
            if payload is None:
                continue
            image = Image()
            image.header.stamp = stamp_ns(metadata.realtime_ns)
            image.header.frame_id = "ov9281_smoke"
            image.height = pool.height
            image.width = pool.width
            image.encoding = "mono8"
            image.is_bigendian = 0
            image.step = pool.stride
            image.data = payload
            bag.write(
                "/camera/image_raw",
                serialize_message(image),
                metadata.realtime_ns,
            )
            info.header = image.header
            bag.write(
                "/camera/camera_info",
                serialize_message(info),
                metadata.realtime_ns,
            )
            copied += 1
            if next_record_ns is None:
                next_record_ns = metadata.realtime_ns + period_ns
            else:
                periods = max(
                    1,
                    (metadata.realtime_ns - next_record_ns) // period_ns + 1,
                )
                next_record_ns += periods * period_ns
    finally:
        bag.close()
        pool.close()
        result.put((copied, sequence_drops))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--record-seconds", type=float, default=10.0)
    ap.add_argument("--record-fps", type=float, default=10.0)
    ap.add_argument("--max-temperature-c", type=float, default=85.0)
    ap.add_argument("--detect", action="store_true")
    args = ap.parse_args()
    if not 0 < args.record_seconds <= args.seconds:
        ap.error("record-seconds must be positive and no greater than seconds")

    name = f"m10_ov9281_smoke_{multiprocessing.current_process().pid}"
    pool = SharedFramePool.create(name, 1280, 800, slots=8, bytes_per_pixel=1)
    context = multiprocessing.get_context("spawn")
    recorder_stop = context.Event()
    result = context.Queue()
    recorder = context.Process(
        target=record_pool,
        args=(name, args.out, recorder_stop, result, args.record_fps),
    )
    camera = None
    detector_sink = None
    frames = 0
    max_gap_ns = 0
    last_sensor_ns = 0
    max_temperature = temperature_c()
    clock = ClockMapper(step_threshold_ns=5_000_000)
    started = 0.0
    try:
        tuning = load_ov9281_daylight_tuning(Picamera2, 1000)
        camera = Picamera2(0, tuning=tuning)
        model = camera.camera_properties.get("Model")
        rotation = camera.camera_properties.get("Rotation")
        if model != "ov9281" or rotation != 180:
            raise RuntimeError(f"expected rotated OV9281, got {model!r}/{rotation!r}")
        frame_duration_us = int(1_000_000 / 30)
        camera.configure(
            camera.create_video_configuration(
                main={"size": (1280, 800), "format": "YUV420"},
                controls={
                    "FrameDurationLimits": (frame_duration_us, frame_duration_us),
                    "AeExposureMode": 1,
                    "ExposureTimeMode": 0,
                    "AnalogueGainMode": 0,
                },
                buffer_count=8,
            )
        )
        if args.detect:
            from mission_engine.rim.tag_detector import TagDetector, detect_image

            detector = TagDetector()
            detector_sink = DetectorSink(
                detector=lambda image: detect_image(detector, image),
                publish=None,
                depth=1,
            )
        recorder.start()
        heartbeat_deadline = time.monotonic() + 2.0
        while not pool.recorder_alive() and time.monotonic() < heartbeat_deadline:
            time.sleep(0.01)
        if not pool.recorder_alive():
            raise RuntimeError("isolated recorder did not attach")
        camera.start()
        started = time.monotonic()
        record_deadline = started + args.record_seconds
        deadline = started + args.seconds
        while time.monotonic() < deadline:
            if time.monotonic() >= record_deadline:
                recorder_stop.set()
            request = camera.capture_request()
            try:
                metadata = request.get_metadata()
                sensor_ns = int(metadata.get("SensorTimestamp", boottime_ns()))
                mapped = clock.map(sensor_ns)
                if mapped is None:
                    continue
                realtime_ns, _offset = mapped
                writable = pool.begin_write()
                if writable is None:
                    continue
                with MappedArray(request, "main") as mapped:
                    destination = np.ndarray(
                        (800, 1280), dtype=np.uint8, buffer=writable.buffer
                    )
                    np.copyto(destination, mapped.array[:800, :1280])
                published = writable.commit(
                    sensor_boottime_ns=sensor_ns,
                    realtime_ns=realtime_ns,
                    exposure_us=int(metadata.get("ExposureTime", 0)),
                    analogue_gain=float(metadata.get("AnalogueGain", 0.0)),
                )
                del destination, writable
            finally:
                request.release()
            if detector_sink is not None:
                lease = pool.lease(published)
                if lease is not None:
                    detector_sink.submit(
                        lease.image(frame_id="ov9281_smoke", encoding="mono8"),
                        lease.ts_ns,
                        lease.release,
                    )
            if last_sensor_ns:
                max_gap_ns = max(max_gap_ns, sensor_ns - last_sensor_ns)
            last_sensor_ns = sensor_ns
            frames += 1
            max_temperature = max(max_temperature, temperature_c())
            if max_temperature >= args.max_temperature_c:
                print(f"thermal stop at {max_temperature:.1f} C")
                break
        recorder_stop.set()
        recorder.join(5.0)
        if recorder.is_alive():
            raise RuntimeError("isolated recorder did not stop cleanly")
        if detector_sink is not None:
            detector_sink.close()
        copied, sequence_drops = result.get(timeout=1.0)
        elapsed = time.monotonic() - started
        detector_summary = (
            f", {detector_sink.summary()}" if detector_sink is not None else ""
        )
        print(
            f"PASS: camera={frames} ({frames / elapsed:.1f} Hz), "
            f"recorded={copied}, sequence_drops={sequence_drops}, "
            f"max_gap={max_gap_ns / 1e6:.1f} ms, max_temp={max_temperature:.1f} C"
            f"{detector_summary}"
        )
        return 0
    finally:
        recorder_stop.set()
        if recorder.is_alive():
            recorder.terminate()
            recorder.join(2.0)
        if camera is not None:
            try:
                camera.stop()
            except Exception:
                pass
            camera.close()
        if detector_sink is not None:
            detector_sink.close()
        pool.close()


if __name__ == "__main__":
    raise SystemExit(main())
