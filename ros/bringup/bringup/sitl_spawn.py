"""Build the per-instance PX4 SITL shell command for a gz Harmonic fleet.

Instance 0 launches the gz server via the `make px4_sitl` target; later instances
attach to that server with PX4_GZ_STANDALONE=1 and the prebuilt px4 binary. Every
instance gets PX4_UXRCE_DDS_NS=px4_<i> so instance 0 is namespaced too.
"""
from __future__ import annotations

import os

import yaml


def load_fleet(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


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
