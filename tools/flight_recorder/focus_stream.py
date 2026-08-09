#!/usr/bin/env python3
"""Serve low-rate native-resolution previews from either drone camera."""

import argparse
import io
import logging
import multiprocessing
import signal
import threading
from html import escape
from http import server
from urllib.parse import urlsplit

from sensing.camera_tuning import load_imx219_daylight_tuning


CAMERAS = {
    "ov9281": {
        "index": 0,
        "model": "ov9281",
        "size": (1280, 800),
        "description": "forward global shutter",
    },
    "cm2": {
        "index": 1,
        "model": "imx219",
        "size": (1640, 1232),
        "description": "downward Camera Module 2",
    },
}


class Frames:
    def __init__(self):
        self.frame = None
        self.sequence = 0
        self.condition = threading.Condition()

    def update(self, frame):
        with self.condition:
            self.frame = frame
            self.sequence += 1
            self.condition.notify_all()

    def next(self, sequence=-1):
        with self.condition:
            self.condition.wait_for(
                lambda: self.frame is not None and self.sequence != sequence
            )
            return self.frame, self.sequence


class PipeOutput(io.BufferedIOBase):
    """Forward complete JPEG buffers from a camera worker to the web process."""

    def __init__(self, connection):
        self.connection = connection

    def write(self, buf):
        self.connection.send_bytes(buf)
        return len(buf)


def camera_worker(
    name, camera_index, fps, quality, max_exposure_us, connection, ready
):
    """Own one camera in a separate process and emit complete JPEG frames."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from picamera2 import Picamera2
    from picamera2.encoders import JpegEncoder
    from picamera2.outputs import FileOutput

    spec = CAMERAS[name]
    picam = None
    try:
        tuning = None
        if name == "cm2":
            tuning = load_imx219_daylight_tuning(Picamera2, max_exposure_us)
        picam = Picamera2(camera_index, tuning=tuning)
        model = picam.camera_properties.get("Model")
        rotation = picam.camera_properties.get("Rotation")
        if model != spec["model"]:
            raise RuntimeError(
                f"expected {spec['model']} at camera {camera_index}, "
                f"got model={model!r}"
            )
        if name == "ov9281" and rotation != 180:
            raise RuntimeError(
                "expected OV9281 device-tree rotation=180, "
                f"got rotation={rotation!r}"
            )

        ready.send(("initialized", rotation))
        if not connection.poll(10) or connection.recv_bytes() != b"configure":
            raise RuntimeError("camera configuration was not requested")

        frame_us = round(1_000_000 / fps)
        controls = {"FrameDurationLimits": (frame_us, frame_us)}
        if name == "cm2":
            controls.update({
                "AeExposureMode": 1,       # Short, whose tuning is capped above.
                "ExposureTimeMode": 0,     # Auto.
                "AnalogueGainMode": 0,     # Auto.
            })
        config = picam.create_video_configuration(
            main={"size": spec["size"], "format": "YUV420"},
            controls=controls,
            buffer_count=4,
        )
        picam.configure(config)
        picam.start_recording(
            JpegEncoder(q=quality), FileOutput(PipeOutput(connection))
        )
        ready.send(("ready", rotation))
        ready.close()
        while True:
            if connection.poll(0.25):
                command = connection.recv_bytes()
                if command == b"stop":
                    break
    except Exception as exc:
        try:
            ready.send(("error", str(exc)))
        except (BrokenPipeError, OSError):
            pass
    finally:
        if picam is not None:
            try:
                picam.stop_recording()
            except Exception:
                pass
            picam.close()
        connection.close()


class CameraProcess:
    def __init__(self, context, name, camera_index, fps, quality, exposure_us):
        self.name = name
        self.frames = Frames()
        self.rotation = None
        web_connection, camera_connection = context.Pipe(duplex=True)
        ready_parent, ready_child = context.Pipe(duplex=False)
        self.connection = web_connection
        self.ready = ready_parent
        self.process = context.Process(
            target=camera_worker,
            args=(
                name,
                camera_index,
                fps,
                quality,
                exposure_us,
                camera_connection,
                ready_child,
            ),
            name=f"preview-{name}",
        )
        self.process.start()
        camera_connection.close()
        ready_child.close()
        if not self.ready.poll(10):
            raise RuntimeError(f"{name} camera did not initialize within 10 seconds")
        status, value = self.ready.recv()
        if status != "initialized":
            self.process.join(timeout=2)
            raise RuntimeError(f"{name} camera failed: {value}")
        self.rotation = value
        self.reader = None

    def start(self):
        self.connection.send_bytes(b"configure")
        if not self.ready.poll(10):
            raise RuntimeError(f"{self.name} camera did not start within 10 seconds")
        status, value = self.ready.recv()
        self.ready.close()
        if status != "ready":
            self.process.join(timeout=2)
            raise RuntimeError(f"{self.name} camera failed: {value}")
        self.rotation = value
        self.reader = threading.Thread(
            target=self._read_frames,
            name=f"preview-reader-{self.name}",
            daemon=True,
        )
        self.reader.start()

    def _read_frames(self):
        try:
            while True:
                self.frames.update(self.connection.recv_bytes())
        except (EOFError, OSError):
            pass

    def close(self):
        try:
            self.connection.send_bytes(b"stop")
        except (BrokenPipeError, OSError):
            pass
        self.process.join(timeout=3)
        self.connection.close()
        if self.reader is not None:
            self.reader.join(timeout=1)
        if self.process.is_alive():
            raise RuntimeError(f"{self.name} camera worker did not stop")


def make_page(active, fps, max_exposure_us):
    panels = []
    for name in active:
        spec = CAMERAS[name]
        width, height = spec["size"]
        exposure = (
            f" · automatic daylight exposure ≤{max_exposure_us} µs"
            if name == "cm2" else ""
        )
        panels.append(f"""\
    <section>
      <header>{escape(name.upper())} · {width}×{height} · {fps:g} fps ·
        {escape(spec["description"])}{exposure} ·
        <a href="/{name}/snapshot.jpg">snapshot</a></header>
      <img class="preview" data-camera="{name}" data-native-width="{width}"
           src="/{name}/snapshot.jpg" width="{width}" height="{height}"
           alt="{escape(spec["description"])} live preview">
    </section>""")
    return f"""\
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>drone camera preview</title>
  <style>
    body {{ margin: 0; background: #111; color: #eee; font: 16px sans-serif; }}
    main {{ display: flex; flex-direction: column; }}
    section {{ min-width: 0; }}
    header {{ padding: .65rem 1rem; }}
    img {{ display: block; width: auto; height: auto; max-width: none; }}
    a {{ color: #9cf; }}
  </style>
</head>
<body>
  <main>
{"".join(panels)}
  </main>
  <script>
    for (const preview of document.querySelectorAll(".preview")) {{
      const camera = preview.dataset.camera;
      function sizeAtDevicePixels() {{
        const nativeWidth = Number(preview.dataset.nativeWidth);
        preview.style.width = `${{nativeWidth / window.devicePixelRatio}}px`;
      }}
      sizeAtDevicePixels();
      window.addEventListener("resize", sizeAtDevicePixels);
      function refresh() {{
        const next = new Image();
        next.onload = () => {{
          preview.src = next.src;
          window.setTimeout(refresh, {round(1000 / fps)});
        }};
        next.onerror = () => window.setTimeout(refresh, 1000);
        next.src = `/${{camera}}/snapshot.jpg?t=${{Date.now()}}`;
      }}
      window.setTimeout(refresh, {round(1000 / fps)});
    }}
  </script>
</body>
</html>
"""


class PreviewHandler(server.BaseHTTPRequestHandler):
    cameras = {}
    page = b""
    health = b""

    def do_GET(self):
        path = urlsplit(self.path).path
        if path in ("/", "/index.html"):
            self._send("text/html; charset=utf-8", self.page)
            return
        if path == "/health":
            self._send("text/plain; charset=utf-8", self.health)
            return

        parts = path.strip("/").split("/")
        if len(parts) != 2 or parts[0] not in self.cameras:
            self.send_error(404)
            return
        name, resource = parts
        frames = self.cameras[name].frames
        if resource == "snapshot.jpg":
            frame, _ = frames.next()
            self._send("image/jpeg", frame)
            return
        if resource == "stream.mjpg":
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=FRAME"
            )
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            sequence = -1
            try:
                while True:
                    frame, sequence = frames.next(sequence)
                    self.wfile.write(b"--FRAME\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        f"Content-Length: {len(frame)}\r\n\r\n".encode()
                    )
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        self.send_error(404)

    def _send(self, content_type, body):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        logging.info("%s - %s", self.client_address[0], fmt % args)


class ThreadingHTTPServer(server.ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera", choices=("both", "ov9281", "cm2"), default="both"
    )
    parser.add_argument("--ov-camera", type=int, default=0)
    parser.add_argument("--cm2-camera", type=int, default=1)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--cm2-max-exposure-us", type=int, default=1000)
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if not 1 <= args.quality <= 100:
        parser.error("--quality must be between 1 and 100")
    if args.cm2_max_exposure_us <= 0:
        parser.error("--cm2-max-exposure-us must be positive")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    active = ("ov9281", "cm2") if args.camera == "both" else (args.camera,)
    indices = {"ov9281": args.ov_camera, "cm2": args.cm2_camera}
    context = multiprocessing.get_context("spawn")
    cameras = {}
    httpd = None
    try:
        # Initialize every libcamera manager before either sensor streams. The
        # CM2 needs its own process-local tuning file. Configure the OV9281
        # last so rotation=180 resolves to native sensor H+V flips.
        manager_order = tuple(reversed(active)) if args.camera == "both" else active
        for name in manager_order:
            cameras[name] = CameraProcess(
                context,
                name,
                indices[name],
                args.fps,
                args.quality,
                args.cm2_max_exposure_us,
            )
        for name in manager_order:
            cameras[name].start()
        PreviewHandler.cameras = cameras
        PreviewHandler.page = make_page(
            active, args.fps, args.cm2_max_exposure_us
        ).encode()
        health_lines = []
        for name, camera in cameras.items():
            width, height = CAMERAS[name]["size"]
            cap = (
                f" exposure_us<={args.cm2_max_exposure_us}"
                if name == "cm2" else ""
            )
            health_lines.append(
                f"ok {name} {width}x{height} {args.fps:g}fps "
                f"rotation={camera.rotation}{cap}"
            )
        PreviewHandler.health = ("\n".join(health_lines) + "\n").encode()
        httpd = ThreadingHTTPServer((args.bind, args.port), PreviewHandler)
        logging.info(
            "camera preview listening on http://%s:%d for %s",
            args.bind,
            args.port,
            ", ".join(active),
        )
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if httpd is not None:
            httpd.shutdown()
        close_errors = []
        for camera in reversed(tuple(cameras.values())):
            try:
                camera.close()
            except RuntimeError as exc:
                close_errors.append(str(exc))
        if close_errors:
            raise RuntimeError("; ".join(close_errors))


if __name__ == "__main__":
    main()
