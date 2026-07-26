"""Build the per-instance PX4 SITL shell command for a gz Harmonic fleet.

Instance 0 launches the gz server via the `make px4_sitl` target; later instances
attach to that server with PX4_GZ_STANDALONE=1 and the prebuilt px4 binary. Every
instance gets PX4_UXRCE_DDS_NS=px4_<i> so instance 0 is namespaced too.
"""
from __future__ import annotations

import copy
import math
import os
import random

import yaml


def _pose_values(pose: str) -> list[float]:
    values = [float(value) for value in str(pose).split(",")]
    if len(values) != 6:
        raise ValueError(f"pose must contain east,north,z,roll,pitch,yaw, got {pose!r}")
    return values


def _randomize_spawns(fleet: dict, seed: int) -> dict:
    """Return a fleet copy with seeded, collision-free M-Air spawn poses.

    The configured poses remain the deterministic choreography staging slots and
    are copied to ``staging_pose``. Only the physical Gazebo spawn is randomized.
    Sampling happens in the cage's slightly yawed local frame so wall clearance
    is measured normal to the actual nets, not an axis-aligned approximation.
    """
    out = copy.deepcopy(fleet)
    cfg = out.get("random_spawn", {})
    width = float(cfg.get("cage_width_east_west_m", 24.384))
    length = float(cfg.get("cage_length_north_south_m", 36.576))
    yaw = float(cfg.get("cage_yaw_rad", -0.006981317))
    wall = float(cfg.get("wall_clearance_m", 2.0))
    separation = float(cfg.get("min_separation_m", 3.0))
    origin_clearance = float(cfg.get("origin_clearance_m", 1.0))
    attempts = int(cfg.get("max_attempts_per_vehicle", 10_000))
    half_x = width / 2.0 - wall
    half_y = length / 2.0 - wall
    if min(half_x, half_y) <= 0.0:
        raise ValueError("random_spawn wall clearance leaves no usable cage interior")
    if separation <= 0.0 or origin_clearance < 0.0:
        raise ValueError("random_spawn clearances must be nonnegative and separation positive")

    rng = random.Random(int(seed))
    accepted: list[tuple[float, float]] = []
    for index, vehicle in enumerate(out["vehicles"]):
        fixed_pose = vehicle.get("staging_pose", vehicle.get("pose", "0,0,0,0,0,0"))
        vehicle["staging_pose"] = fixed_pose
        pose = _pose_values(fixed_pose)
        for _ in range(attempts):
            local_e = rng.uniform(-half_x, half_x)
            local_n = rng.uniform(-half_y, half_y)
            east = math.cos(yaw) * local_e - math.sin(yaw) * local_n
            north = math.sin(yaw) * local_e + math.cos(yaw) * local_n
            if math.hypot(east, north) < origin_clearance:
                continue
            if any(math.hypot(east - pe, north - pn) < separation for pe, pn in accepted):
                continue
            accepted.append((east, north))
            pose[0], pose[1] = east, north
            vehicle["pose"] = ",".join(f"{value:.6f}" for value in pose)
            break
        else:
            raise RuntimeError(
                f"could not place vehicle {index} after {attempts} attempts; "
                "relax random_spawn clearances"
            )

    out["_random_spawn"] = {"enabled": True, "seed": int(seed)}
    return out


def load_fleet(config_path: str, *, random_spawn: bool = False,
               spawn_seed: int | None = None) -> dict:
    with open(config_path) as f:
        fleet = yaml.safe_load(f)
    if not random_spawn:
        return fleet
    cfg = fleet.get("random_spawn", {})
    seed = int(cfg.get("seed", 0) if spawn_seed is None else spawn_seed)
    return _randomize_spawns(fleet, seed)


def gz_model_name(model: str, instance_id: int) -> str:
    """gz model name PX4 assigns instance `instance_id`, e.g. x500_0."""
    return f"{model}_{instance_id}"


def px4_build_dir(px4_dir: str) -> str:
    return os.path.join(px4_dir, "build", "px4_sitl_default")


def px4_binary(px4_dir: str) -> str:
    return os.path.join(px4_build_dir(px4_dir), "bin", "px4")


def build_sitl_cmd(*, instance_id, px4_dir, model, pose="", autostart=4001,
                   world="default", home_gps=None, dds_ns=None):
    dds_ns = dds_ns or f"px4_{instance_id}"
    env = [
        f"PX4_SYS_AUTOSTART={autostart}",
        f"PX4_UXRCE_DDS_NS={dds_ns}",
    ]
    if home_gps:
        env += [
            f"PX4_HOME_LAT={float(home_gps['lat']):.10f}",
            f"PX4_HOME_LON={float(home_gps['lon']):.10f}",
            f"PX4_HOME_ALT={float(home_gps.get('alt_m', 0.0)):.1f}",
        ]
    if pose:
        env.append(f'PX4_GZ_MODEL_POSE="{pose}"')
    env_str = " ".join(env)

    # exec the long-running binary so it inherits the shell's PID; that lets a
    # ros2 launch SIGINT reach px4 directly instead of orphaning it behind bash.
    if instance_id == 0:
        return f"cd {px4_dir} && PX4_GZ_WORLD={world} {env_str} exec make px4_sitl gz_{model}"

    build_dir = px4_build_dir(px4_dir)
    rootfs = os.path.join(build_dir, "rootfs", str(instance_id))
    etc = os.path.join(build_dir, "etc")
    return (
        f"mkdir -p {rootfs} && cd {rootfs} && "
        f"{env_str} GZ_IP=127.0.0.1 PX4_GZ_STANDALONE=1 PX4_SIM_MODEL=gz_{model} "
        f"exec {px4_binary(px4_dir)} -i {instance_id} -d {etc}"
    )
