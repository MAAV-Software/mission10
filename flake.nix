{
  description = "mission10 development environments for workstations";

  inputs = {
    nix-ros-overlay.url = "github:lopsided98/nix-ros-overlay/master";
    nixpkgs.follows = "nix-ros-overlay/nixpkgs"; # Remeber to be in lockstep with the overlay!
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { self, nix-ros-overlay, nixpkgs, rust-overlay }:
    nix-ros-overlay.inputs.flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            nix-ros-overlay.overlays.default
            rust-overlay.overlays.default
          ];
        };

        uwbRust = pkgs.rust-bin.stable."1.92.0".default.override {
          targets = [ "thumbv7em-none-eabihf" ];
          extensions = [ "rust-src" "llvm-tools-preview" ];
        };

        # PX4<->ROS2 uXRCE-DDS bridge. Not in nixpkgs; pinned to the
        # PX4-compatible v2.4.3 (v3.x breaks the client). Built against the
        # overlay's Fast-DDS (fastrtps). See nix/microxrce-agent.nix.
        microxrceAgent = pkgs.callPackage ./nix/microxrce-agent.nix {
          inherit (pkgs.rosPackages.jazzy) fastrtps;
        };
      in {
        devShells = {
          # Pure-core development. Runs anywhere, including macOS; mirrors CI.
          # Enough to hack on ros/*/ cores, run every test suite, and work on
          # models/. Python extras that aren't in nixpkgs (qtm-rt, pyulog)
          # live in per-package venvs via uv.
          default = pkgs.mkShell {
            name = "mission10";
            packages = [
              pkgs.python3
              pkgs.vcstool
              pkgs.uv
            ];
          };

          # DWM3001CDK software is always built on a workstation. The Pis only
          # receive the resulting ELF and run probe-rs/serial tooling.
          uwb = pkgs.mkShell {
            name = "mission10-uwb";
            packages = [
              uwbRust
              pkgs.cargo-binutils
              pkgs.flip-link
              pkgs.probe-rs-tools
            ];
          };
        } // nixpkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
          # ROS rim + sim development (Linux only since ros.cachix.org)
          #
          # nix provides the ENVIRONMENT; colcon builds at the repo root,
          # discovering ros/* (incl. the vendored px4_msgs, synced from the
          # PX4 fork by scripts/sync_px4_msgs.sh). Artifacts land in gitignored
          # build/install/log. Our packages never become nix derivations.
          sim = pkgs.mkShell {
            name = "mission10-sim";
            packages = [
              pkgs.colcon
              pkgs.vcstool
              microxrceAgent # PX4<->ROS2 bridge (MicroXRCEAgent udp4 -p 8888)
              # Colcon builds px4_msgs from source; expose its build tools as
              # real prefixes instead of only through the aggregate ros-env.
              # PX4's top-level Makefile also probes PATH for ninja before
              # delegating into its existing CMake/Ninja build directory.
              pkgs.ninja
              pkgs.rosPackages.jazzy.ament-cmake
              pkgs.rosPackages.jazzy.ament-cmake-core
              pkgs.rosPackages.jazzy.rosidl-default-generators
              pkgs.rosPackages.jazzy.builtin-interfaces
              (with pkgs.rosPackages.jazzy; buildEnv {
                paths = [
                  ros-core # rclpy, ros2cli, ament, ...
                  nav-msgs # gz truth odometry -> sim_truth_ev
                  vision-msgs # detection wire contract
                  ros-gz-sim # Gazebo (gz) integration
                  ros-gz-bridge
                  # px4_msgs is deliberately NOT here: pinned in
                  # externals.repos, colcon-built in the workspace so it
                  # always matches deployed PX4 firmware.
                ];
              })
            ];
            shellHook = ''
              export CMAKE_PREFIX_PATH="''${AMENT_PREFIX_PATH}''${CMAKE_PREFIX_PATH:+:}''${CMAKE_PREFIX_PATH:-}"
              export QT_QPA_PLATFORM=xcb
              # PX4 SITL fork checkout (the fork's flake owns the binary; this
              # shell only points at it).
              if [ -z "''${PX4_DIR:-}" ]; then
                export PX4_DIR="$(cd "$(git rev-parse --show-toplevel 2>/dev/null)/.." 2>/dev/null && pwd)/PX4-Autopilot"
              fi
              [ -x "$PX4_DIR/build/px4_sitl_default/bin/px4" ] || \
                echo "note: no PX4 SITL binary under PX4_DIR=$PX4_DIR (build the fork, or set PX4_DIR)"
            '';
            # TODO(first sim session):
            # - PX4 SITL binary: belongs to the PX4-Autopilot fork's flake,
            #   not this one. This shell only needs to talk to a running SITL.
            # - GUI apps (rviz2, gz gui) on non-NixOS need nixGL or
            #   nix-system-graphics for GL.
          };
        };
      });

  nixConfig = {
    extra-substituters = [ "https://ros.cachix.org" ];
    extra-trusted-public-keys = [ "ros.cachix.org-1:dSyZxI8geDCJrwgvCOHDoAfOm5sV1wCPjBkKL+38Rvo=" ];
  };
}
