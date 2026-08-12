#!/usr/bin/env python3
"""Publish one image repeatedly through the CM2 shared frame pool."""

import argparse
import time

import cv2

from sensing.shared_frame_pool import SharedFramePool


WIDTH = 1640
HEIGHT = 1232
RATE_HZ = 10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"cannot read image: {args.image}")
    image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    payload = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_YUY2).tobytes()

    pool = SharedFramePool.create("cm2", WIDTH, HEIGHT, slots=8)
    period = 1 / RATE_HZ
    deadline = time.monotonic()
    print(f"publishing {args.image} to cm2 at {RATE_HZ} Hz; Ctrl-C stops")
    try:
        while True:
            slot = pool.begin_write()
            if slot is not None:
                slot.buffer[:] = payload
                slot.commit(
                    sensor_boottime_ns=time.monotonic_ns(),
                    realtime_ns=time.time_ns(),
                    exposure_us=1000,
                    analogue_gain=1.0,
                )
            deadline += period
            time.sleep(max(0, deadline - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        pool.close()


if __name__ == "__main__":
    main()
