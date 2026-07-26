#!/usr/bin/env python3
"""Tiered drain for a split mcap recording.

    HOT (tmpfs/RAM)  --fast-->  MID (eMMC, optional)  --slow-->  DEEP (USB)

capture.py writes a split bag into HOT (one <session>_N.mcap chunk at a time).
This relocates each COMPLETED chunk down the tiers *while recording continues*,
so the maximum recording time is bounded by the sum of all tier capacities
rather than the eMMC's small free space. The USB drains continuously at its slow
pace; eMMC + RAM absorb the (produce - usb) deficit.

Policy
------
While the recording flag exists (capture running):
  * thread A (hot->mid): move every completed HOT chunk to MID whenever MID has
    headroom; if MID is full, leave it in HOT (the RAM spill). The active
    (highest-index) chunk is never touched -- capture is still writing it.
  * thread B (mid->deep): continuously move MID chunks to DEEP at USB speed.
When the flag disappears (capture has closed every file + written metadata),
both threads stop and a single-threaded final flush pushes everything still in
MID and HOT to DEEP, then copies metadata.yaml last so the DEEP bag dir is a
self-consistent, replayable bag.

All moves use `rsync --remove-source-files`: rsync writes a hidden temp file and
atomically renames it on completion, so a reader globbing *.mcap never sees a
partial chunk and an interrupted move never leaves a torn file behind.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import threading
import time


def chunk_idx(path):
    """…/<session>_<N>.mcap -> N  (or -1 if unparseable)."""
    base = os.path.basename(path)
    try:
        return int(base.rsplit("_", 1)[1].split(".")[0])
    except Exception:
        return -1


def mcaps(d):
    return sorted(glob.glob(os.path.join(d, "*.mcap")), key=chunk_idx)


def avail_mb(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / (1024 * 1024)


def rsync_move(src, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    r = subprocess.run(
        ["rsync", "-a", "--remove-source-files", src, dst_dir + "/"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if r.returncode != 0:
        sys.stderr.write(f"tier_drain: rsync {src} -> {dst_dir} FAILED: "
                         f"{r.stderr.decode().strip()}\n")
    return r.returncode == 0


class Counter:
    def __init__(self):
        self.n = 0
        self._l = threading.Lock()

    def inc(self):
        with self._l:
            self.n += 1


def hotfree_loop(a, stop, moved):
    """HOT -> MID (or HOT -> DEEP in 2-tier mode), completed chunks only."""
    while not stop.is_set():
        files = mcaps(a.hot)
        active = max((chunk_idx(f) for f in files), default=-1)
        for f in files:
            if chunk_idx(f) >= active:
                continue  # active chunk is still being written -- never move it
            if a.mid:
                if avail_mb(a.mid) > a.mid_headroom_mb:
                    rsync_move(f, a.mid)
                # else MID full -> leave in HOT (RAM spill), retry next pass
            else:
                # 2-tier: DEEP is the eMMC rootfs. Reserve headroom so the RAM
                # spill starts BEFORE the rootfs is hard-full (a full rootfs
                # takes down more than the recording).
                if avail_mb(a.deep) > a.deep_headroom_mb:
                    if rsync_move(f, a.deep):
                        moved.inc()
                # else DEEP full -> leave in HOT (RAM spill), retry next pass
        stop.wait(a.poll)


def middrain_loop(a, stop, moved):
    """MID -> DEEP, continuously, at USB pace."""
    while not stop.is_set():
        for f in mcaps(a.mid):
            if rsync_move(f, a.deep):
                moved.inc()
        stop.wait(a.poll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hot", required=True)
    ap.add_argument("--mid", default="")
    ap.add_argument("--deep", required=True)
    ap.add_argument("--flag", required=True)
    ap.add_argument("--mid-headroom-mb", type=float, default=600.0)
    ap.add_argument("--deep-headroom-mb", type=float, default=256.0)
    ap.add_argument("--poll", type=float, default=0.5)
    a = ap.parse_args()

    os.makedirs(a.deep, exist_ok=True)
    if a.mid:
        os.makedirs(a.mid, exist_ok=True)

    moved = Counter()
    stop = threading.Event()
    t0 = time.time()

    threads = [threading.Thread(target=hotfree_loop, args=(a, stop, moved),
                                daemon=True)]
    if a.mid:
        threads.append(threading.Thread(target=middrain_loop,
                                        args=(a, stop, moved), daemon=True))
    for t in threads:
        t.start()

    # run until capture clears the flag
    while os.path.exists(a.flag):
        time.sleep(a.poll)
    stop.set()
    for t in threads:
        t.join(timeout=120)

    # ---- final flush (capture stopped; every chunk closed) ----
    # Drain MID first (older indices), then whatever is left in HOT (the former
    # active chunk + any RAM spill), straight to DEEP. Metadata last.
    for f in mcaps(a.mid) if a.mid else []:
        if rsync_move(f, a.deep):
            moved.inc()
    for f in mcaps(a.hot):
        if avail_mb(a.deep) <= a.deep_headroom_mb:
            break  # keep the reserve; leftovers stay in HOT (exit 1 reports them)
        if rsync_move(f, a.deep):
            moved.inc()
    for meta in glob.glob(os.path.join(a.hot, "*.yaml")):
        shutil.copy2(meta, a.deep)

    n = len(glob.glob(os.path.join(a.deep, "*.mcap")))
    has_meta = os.path.exists(os.path.join(a.deep, "metadata.yaml"))
    hot_left = len(mcaps(a.hot))
    mid_left = len(mcaps(a.mid)) if a.mid else 0
    print(f"tier_drain: moved {moved.n} chunks in {time.time()-t0:.0f}s; "
          f"DEEP has {n} chunks, metadata={has_meta}; "
          f"leftover hot={hot_left} mid={mid_left}", file=sys.stderr)
    return 0 if (hot_left == 0 and mid_left == 0 and has_meta) else 1


if __name__ == "__main__":
    sys.exit(main())
