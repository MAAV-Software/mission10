"""AprilTag tap for the downward camera: frames in, detections out.

One detector serves two consumers on the same wire (rfd-mission-execution,
detection ingest section): the mine log, where a decoded tag promotes a
cluster straight to `verified`, and the tag anchor, where re-observed tags
measure flight-layer drift.

Detection uses OpenCV's `DICT_APRILTAG_36h11`, which is already installed on
the flight image; `pupil_apriltags` is not, and the offline analysis chain
that established the ground truth used it, so
`reference/flight_bags/20260725_000141_07-24-2005-survey/analysis/compare_detectors.py`
scores this detector against those results before it is trusted in flight.

`TagDetector` holds no ROS state, so the flight recorder can call it directly
on the frame it already has in hand rather than paying a DDS hop for 20 MB/s
of imagery. `TagDetectorNode` is the same detector behind an `Image`
subscription, for SITL, bag replay, and any rig where the camera is not the
recorder's.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

FAMILY = "tag36h11"
TAG_PREFIX = f"{FAMILY}:"


@dataclass(frozen=True)
class TagDetection:
    """One decoded tag in raw pixel coordinates."""

    tag_id: int
    center: Tuple[float, float]
    corners: Tuple[Tuple[float, float], ...]  # 4 corners, detector order

    @property
    def wire_id(self) -> str:
        return f"{TAG_PREFIX}{self.tag_id}"

    @property
    def mean_side_px(self) -> float:
        c = np.asarray(self.corners, dtype=float)
        return float(np.linalg.norm(c - np.roll(c, 1, axis=0), axis=1).mean())

    @property
    def px_per_module(self) -> float:
        """Decode headroom. tag36h11 spans 8 modules across its black square;
        published decode floors sit near 5 px/module in practice, so this is
        the number that bounds usable survey altitude for a given tag size."""
        return self.mean_side_px / 8.0

    def size_px(self) -> Tuple[float, float]:
        # np.ptp, not ndarray.ptp: the method was removed in NumPy 2.0, which
        # is what the flight image carries.
        c = np.asarray(self.corners, dtype=float)
        return (float(np.ptp(c[:, 0])), float(np.ptp(c[:, 1])))


class TagDetector:
    """Grayscale frame -> tag36h11 detections. No ROS, no camera ownership."""

    def __init__(
        self,
        min_side_px: float = 24.0,
        accept_ids: Optional[Sequence[int]] = None,
    ) -> None:
        import cv2  # imported here so the pure core stays importable without it

        self._cv2 = cv2
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        params = cv2.aruco.DetectorParameters()
        # Corner accuracy drives the ground fix directly: a pixel of corner
        # error at 4 m AGL is about 3 mm of ground error, and the anchor
        # residual is the difference of two such fixes. The AprilTag refiner
        # holds 0.15 px against pupil_apriltags; the subpix refiner costs
        # 0.68 px and finds fewer tags, so it is not an alternative here.
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        # This method does not refine the aruco candidates. It runs OpenCV's
        # own AprilTag pipeline, so the aprilTag* parameters govern the cost
        # and the aruco quad parameters below do not reach it.
        #
        # Quad detection runs on a decimated image, as in the reference
        # AprilTag library. OpenCV defaults the factor to 0.0 (full
        # resolution) where the reference library defaults it to 2.0. On the
        # CM5 that difference is 269 ms against 78 ms per 1640x1232 frame,
        # single-threaded either way, against a 100 ms frame interval.
        #
        # Over the whole flight-1 bag (`detector_decimate.json`) it costs 1.2
        # points of recall, 90.3% -> 89.1%, and 0.009 px of centre agreement,
        # which is 0.03 mm on the ground at 4 m AGL against a measured per-fix
        # noise near 50 mm (TAG_ANCHOR_REPLAY.md).
        params.aprilTagQuadDecimate = 2.0
        # Decode tolerance, from the flight-1 sweep (`detector_sweep.json`):
        # recall 84.0% -> 90.3% at unchanged centre accuracy, longest run of
        # consecutive blind frames 32 -> 11.
        params.minMarkerPerimeterRate = 0.01
        params.polygonalApproxAccuracyRate = 0.05
        params.errorCorrectionRate = 0.8
        self._detector = cv2.aruco.ArucoDetector(dictionary, params)
        self.min_side_px = float(min_side_px)
        self.accept_ids = set(accept_ids) if accept_ids is not None else None

    def detect(self, gray: np.ndarray) -> List[TagDetection]:
        if gray.ndim != 2:
            raise ValueError(f"expected a single-channel frame, got shape {gray.shape}")
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None:
            return []
        out: List[TagDetection] = []
        for quad, tag_id in zip(corners, ids.flatten().tolist()):
            if self.accept_ids is not None and int(tag_id) not in self.accept_ids:
                continue
            pts = np.asarray(quad, dtype=float).reshape(4, 2)
            det = TagDetection(
                tag_id=int(tag_id),
                center=(float(pts[:, 0].mean()), float(pts[:, 1].mean())),
                corners=tuple((float(x), float(y)) for x, y in pts),
            )
            if det.mean_side_px < self.min_side_px:
                continue
            out.append(det)
        return out


def luma(data: bytes, height: int, width: int, step: int, encoding: str) -> np.ndarray:
    """Single-channel view of an `Image` payload the detector can read."""
    buf = np.frombuffer(data, dtype=np.uint8)
    if encoding == "mono8":
        return buf.reshape(height, step)[:, :width]
    if encoding in ("yuv422_yuy2", "yuyv"):
        return buf.reshape(height, step)[:, : width * 2 : 2]
    raise ValueError(f"unsupported encoding {encoding!r}")


def to_detection_array(header, detections: Sequence[TagDetection]):
    """`TagDetection`s under an image header -> the wire message.

    The header must equal the source image header: the engine joins detections
    to flight state on this stamp, never on arrival time.

    Imports `vision_msgs` on call so the detector stays usable off a bare
    Python path, and so a caller that only wants pixels pays nothing.
    """
    from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

    out = Detection2DArray()
    out.header = header
    for det in detections:
        d = Detection2D()
        d.header = header
        d.id = det.wire_id
        d.bbox.center.position.x = det.center[0]
        d.bbox.center.position.y = det.center[1]
        d.bbox.size_x, d.bbox.size_y = det.size_px()
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = FAMILY
        # A decode is binary: it either passed the family's checksum or it is
        # not reported at all. Score carries no extra information.
        hyp.hypothesis.score = 1.0
        d.results.append(hyp)
        out.detections.append(d)
    return out


def detect_image(detector: TagDetector, img):
    """`sensor_msgs/Image` in, `Detection2DArray` out.

    The whole path from a captured frame to the wire, so the recorder's frame
    sink and the standalone node cannot drift apart.
    """
    gray = luma(img.data, img.height, img.width, img.step, img.encoding)
    return to_detection_array(img.header, detector.detect(gray))


def main(args=None):
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from vision_msgs.msg import Detection2DArray

    class TagDetectorNode(Node):
        """`Image` in, `Detection2DArray` out, header preserved verbatim."""

        def __init__(self) -> None:
            super().__init__("tag_detector")
            self.declare_parameter("image_topic", "/camera_down/image_raw")
            self.declare_parameter("detections_topic", "/detections/down")
            self.declare_parameter("min_side_px", 24.0)
            self.declare_parameter("stride", 1)
            self.detector = TagDetector(
                min_side_px=float(self.get_parameter("min_side_px").value)
            )
            self.stride = max(1, int(self.get_parameter("stride").value))
            self.frames = 0
            self.pub = self.create_publisher(
                Detection2DArray, str(self.get_parameter("detections_topic").value), 10
            )
            self.create_subscription(
                Image, str(self.get_parameter("image_topic").value), self._cb, 5
            )
            self.get_logger().info(
                f"tag_detector up: {self.get_parameter('image_topic').value} -> "
                f"{self.get_parameter('detections_topic').value}"
            )

        def _cb(self, msg: Image) -> None:
            self.frames += 1
            if self.frames % self.stride:
                return
            try:
                out = detect_image(self.detector, msg)
            except ValueError as exc:
                self.get_logger().error(str(exc), throttle_duration_sec=5.0)
                return
            self.pub.publish(out)

    rclpy.init(args=args)
    node = TagDetectorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
