"""The geometry a detection carries, and the message it becomes.

`TagDetection`'s properties decide what reaches the anchor and the mine log,
so they are checked against hand-computed quads rather than a detector run:
the detector needs a camera and OpenCV, and these do not.
"""
import math

import pytest

from mission_engine.rim.tag_detector import (
    FAMILY,
    TAG_PREFIX,
    TagDetection,
    to_detection_array,
)


def square(cx, cy, side, rot=0.0):
    """A tag quad, centred and optionally rotated in the image plane."""
    half = side / 2.0
    c, s = math.cos(rot), math.sin(rot)
    return tuple(
        (cx + x * c - y * s, cy + x * s + y * c)
        for x, y in ((-half, -half), (half, -half), (half, half), (-half, half))
    )


def detection(tag_id=6, cx=800.0, cy=600.0, side=48.0, rot=0.0):
    return TagDetection(tag_id=tag_id, center=(cx, cy),
                        corners=square(cx, cy, side, rot))


def test_the_wire_id_names_the_family():
    assert detection(tag_id=7).wire_id == f"{TAG_PREFIX}7" == "tag36h11:7"


def test_mean_side_is_the_quad_edge_length():
    assert detection(side=48.0).mean_side_px == pytest.approx(48.0)


def test_px_per_module_spans_the_tags_eight_modules():
    # The decode floor sits near 5 px/module, so this is the number that
    # bounds usable survey altitude for a given tag size.
    assert detection(side=40.0).px_per_module == pytest.approx(5.0)


def test_size_px_is_the_axis_aligned_extent():
    assert detection(side=48.0).size_px() == pytest.approx((48.0, 48.0))


def test_size_px_grows_with_image_plane_rotation():
    """A rotated square's bounding box is wider than its side.

    This runs `np.ptp` on real arrays. `ndarray.ptp` was removed in NumPy 2.0,
    which is the version the flight image carries, and the failure only
    surfaces here on the path from a detection to the wire.
    """
    w, h = detection(side=48.0, rot=math.radians(45.0)).size_px()
    assert w == pytest.approx(48.0 * math.sqrt(2.0))
    assert h == pytest.approx(48.0 * math.sqrt(2.0))


def _has_vision_msgs():
    try:
        import vision_msgs.msg  # noqa: F401
        import std_msgs.msg  # noqa: F401
    except ImportError:
        return False
    return True


# The message tests need a ROS environment. The geometry above does not, and a
# module-level skip would take it down with them.
needs_ros = pytest.mark.skipif(not _has_vision_msgs(), reason="needs vision_msgs")


@needs_ros
def test_the_message_carries_the_image_header_and_the_tags():
    from std_msgs.msg import Header

    header = Header()
    header.frame_id = "imx219_nadir"
    header.stamp.sec, header.stamp.nanosec = 1784937899, 840000000
    dets = [detection(tag_id=6, cx=700.0, cy=500.0, side=48.0),
            detection(tag_id=7, cx=900.0, cy=700.0, side=52.0)]

    msg = to_detection_array(header, dets)

    assert msg.header.stamp.sec == 1784937899
    assert [d.id for d in msg.detections] == ["tag36h11:6", "tag36h11:7"]
    # Every detection repeats the source stamp: the engine joins detections to
    # flight state on it, never on arrival time.
    assert all(d.header.stamp.nanosec == 840000000 for d in msg.detections)
    first = msg.detections[0]
    assert (first.bbox.center.position.x, first.bbox.center.position.y) == (700.0, 500.0)
    assert first.bbox.size_x == pytest.approx(48.0)
    assert first.results[0].hypothesis.class_id == FAMILY
    assert first.results[0].hypothesis.score == 1.0


@needs_ros
def test_an_empty_pass_still_reports_the_frame():
    from std_msgs.msg import Header

    msg = to_detection_array(Header(), [])
    assert msg.detections == []
