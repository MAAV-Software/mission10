import math
from pathlib import Path

from bringup.sitl_spawn import load_fleet


CONFIG = Path(__file__).parents[1] / "config" / "fleet.yaml"


def _xy(fleet):
    return [tuple(float(value) for value in vehicle["pose"].split(",")[:2])
            for vehicle in fleet["vehicles"]]


def test_fixed_mode_preserves_checked_in_poses():
    fixed = load_fleet(str(CONFIG))
    assert _xy(fixed) == [(1.0, 4.5), (1.0, 1.5), (1.0, -1.5), (1.0, -4.5)]
    assert "_random_spawn" not in fixed


def test_random_spawn_is_seeded_and_preserves_staging_slots():
    first = load_fleet(str(CONFIG), random_spawn=True, spawn_seed=8675309)
    again = load_fleet(str(CONFIG), random_spawn=True, spawn_seed=8675309)
    different = load_fleet(str(CONFIG), random_spawn=True, spawn_seed=8675310)

    assert _xy(first) == _xy(again)
    assert _xy(first) != _xy(different)
    assert first["_random_spawn"] == {"enabled": True, "seed": 8675309}
    stages = [tuple(float(value) for value in vehicle["staging_pose"].split(",")[:2])
              for vehicle in first["vehicles"]]
    assert stages == [(1.0, 4.5), (1.0, 1.5), (1.0, -1.5), (1.0, -4.5)]


def test_random_spawn_respects_walls_origin_and_pair_separation():
    fleet = load_fleet(str(CONFIG), random_spawn=True, spawn_seed=42)
    cfg = fleet["random_spawn"]
    points = _xy(fleet)
    yaw = float(cfg["cage_yaw_rad"])
    usable_e = float(cfg["cage_width_east_west_m"]) / 2.0 - float(cfg["wall_clearance_m"])
    usable_n = float(cfg["cage_length_north_south_m"]) / 2.0 - float(cfg["wall_clearance_m"])

    for east, north in points:
        # inverse world->cage rotation
        local_e = math.cos(yaw) * east + math.sin(yaw) * north
        local_n = -math.sin(yaw) * east + math.cos(yaw) * north
        assert abs(local_e) <= usable_e + 1e-6
        assert abs(local_n) <= usable_n + 1e-6
        assert math.hypot(east, north) >= float(cfg["origin_clearance_m"]) - 1e-6

    for index, point in enumerate(points):
        for peer in points[index + 1:]:
            assert math.dist(point, peer) >= float(cfg["min_separation_m"]) - 1e-6


def test_rough_line_spawn_is_seeded_and_jitters_inside_slot_disks():
    first = load_fleet(str(CONFIG), rough_line_spawn=True, spawn_seed=23)
    again = load_fleet(str(CONFIG), rough_line_spawn=True, spawn_seed=23)
    different = load_fleet(str(CONFIG), rough_line_spawn=True, spawn_seed=24)

    assert _xy(first) == _xy(again)
    assert _xy(first) != _xy(different)
    assert first["_rough_line_spawn"] == {"enabled": True, "seed": 23}
    points = _xy(first)
    nominal = [(1.0, 4.5), (1.0, 1.5), (1.0, -1.5), (1.0, -4.5)]
    assert all(math.dist(point, center) <= 0.75 + 1e-6
               for point, center in zip(points, nominal))
    assert any(abs(point[0] - center[0]) > 0.1
               for point, center in zip(points, nominal))
    assert all(points[index][1] > points[index + 1][1] for index in range(3))
    assert all(math.dist(points[index], points[index + 1]) >= 2.0 - 1e-6
               for index in range(3))
    stages = [tuple(float(value) for value in vehicle["staging_pose"].split(",")[:2])
              for vehicle in first["vehicles"]]
    assert stages == nominal


def test_spawn_modes_are_mutually_exclusive():
    try:
        load_fleet(str(CONFIG), random_spawn=True, rough_line_spawn=True)
    except ValueError as error:
        assert "mutually exclusive" in str(error)
    else:
        raise AssertionError("expected mutually exclusive spawn modes to fail")
