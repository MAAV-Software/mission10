"""Bring up a PX4 SITL + gz Harmonic fleet: MicroXRCEAgent + N PX4 instances.

px4_dir (or the PX4_DIR env var) must point at the PX4 fork checkout whose
build/px4_sitl_default holds the SITL binary. num_vehicles sets the fleet size.
"""
from __future__ import annotations

import os
import subprocess

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnShutdown
from launch.substitutions import LaunchConfiguration

from bringup.sitl_spawn import build_sitl_cmd, load_fleet, px4_build_dir


def _proc(command):
    return ExecuteProcess(cmd=["bash", "-lc", command], output="screen")


def _reaper(bin_px4):
    """Synchronous on-shutdown reaper. exec lets a SIGINT reach px4 directly, but
    instance 0's `make` forks a gz server px4 can't take with it; a launch SIGINT
    can also miss children when the group is split. Kill stray px4/gz in-process
    on shutdown (an OpaqueFunction runs synchronously; spawning a fresh
    ExecuteProcess here is unreliable while the event loop is tearing down)."""
    def _kill(_context, *_args, **_kwargs):
        for pat in (bin_px4, "gz sim"):
            # returns 1 when nothing matches; that is the normal no-op case
            subprocess.run(["pkill", "-9", "-f", pat], check=False)
        return []
    return _kill


def _gz_gated(command, world, timeout_s=60.0):
    """Gate a standalone instance on the gz world clock instead of a fixed sleep.

    PX4_GZ_STANDALONE retries the spawn every 2 s, so once the clock topic is up
    every gated instance can spawn at once. Fail-fast: error out if gz never
    comes up within timeout_s rather than hanging forever.
    """
    tries = max(1, int(timeout_s / 0.5))
    grep = rf"gz topic -l 2>/dev/null | grep -q '/world/{world}/clock'"
    gate = (
        f"for _ in $(seq 1 {tries}); do {grep} && break; sleep 0.5; done; "
        f"{grep} || {{ echo 'gz world clock not up after {timeout_s:g}s' >&2; exit 1; }}"
    )
    return _proc(f"{gate}; {command}")


def _setup(context, *args, **kwargs):
    num = int(LaunchConfiguration("num_vehicles").perform(context))
    px4_dir = LaunchConfiguration("px4_dir").perform(context).strip()
    config_file = LaunchConfiguration("fleet_config").perform(context).strip()
    random_spawn = LaunchConfiguration("random_spawn").perform(context).lower() in (
        "1", "true", "yes", "on")
    spawn_seed_raw = LaunchConfiguration("spawn_seed").perform(context).strip()
    spawn_seed = int(spawn_seed_raw) if spawn_seed_raw else None

    if not px4_dir:
        raise RuntimeError("px4_dir/PX4_DIR is unset; point it at the PX4 fork checkout.")
    if not config_file:
        config_file = os.path.join(get_package_share_directory("bringup"), "config", "fleet.yaml")

    fleet = load_fleet(config_file, random_spawn=random_spawn, spawn_seed=spawn_seed)

    vehicles = fleet["vehicles"]
    if num > len(vehicles):
        raise RuntimeError(f"num_vehicles={num} exceeds {len(vehicles)} configured vehicles.")

    world = LaunchConfiguration("world").perform(context).strip() or fleet.get("world", "default")
    actions = [_proc(f"exec MicroXRCEAgent udp4 -p {fleet.get('agent_port', 8888)}")]

    for i in range(num):
        v = vehicles[i]
        cmd = build_sitl_cmd(
            instance_id=i,
            px4_dir=px4_dir,
            model=fleet["model"],
            pose=v.get("pose", ""),
            autostart=fleet.get("autostart", 4001),
            world=world,
            home_gps=fleet.get("home_gps"),
            dds_ns=v.get("namespace", f"px4_{i}"),
        )
        # instance 0 launches gz; standalone instances gate on the world clock
        # and then spawn concurrently (each its own uniquely-named model).
        actions.append(_proc(cmd) if i == 0 else _gz_gated(cmd, world))

    # Reap any stray px4/gz on shutdown so reruns start clean instead of fighting
    # orphaned duplicates.
    bin_px4 = os.path.join(px4_build_dir(px4_dir), "bin", "px4")
    actions.append(RegisterEventHandler(
        OnShutdown(on_shutdown=[OpaqueFunction(function=_reaper(bin_px4))])))

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("num_vehicles", default_value="1"),
        DeclareLaunchArgument("px4_dir", default_value=os.environ.get("PX4_DIR", "")),
        DeclareLaunchArgument("fleet_config", default_value=""),
        DeclareLaunchArgument("world", default_value="",
                              description="gz world override (e.g. 'windy'); empty uses fleet.yaml."),
        DeclareLaunchArgument("random_spawn", default_value="false"),
        DeclareLaunchArgument("spawn_seed", default_value=""),
        OpaqueFunction(function=_setup),
    ])
