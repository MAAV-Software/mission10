# Micro XRCE-DDS Agent (eProsima) — the PX4 <-> ROS 2 uXRCE-DDS bridge.
#
# Pinned to v2.4.3 ON PURPOSE: PX4's uXRCE-DDS *client* rides the v2.x line and
# is INCOMPATIBLE with agent v3.x (eProsima's current release). v2.4.3 is the
# Jazzy-matched tag — a plain `nix build` of upstream HEAD would grab the broken
# v3.x.
#
# Built against SYSTEM deps (superbuild OFF) so nix controls every input instead
# of the agent git-fetching them at build time (which the sandbox forbids):
#   - Fast-DDS  -> rosPackages.jazzy.fastrtps  (Fast-DDS 2.14.6; "fastrtps" is
#                  the legacy ROS package name for Fast-DDS)
#   - Fast-CDR  -> nixpkgs fastcdr 2.3.5  (agent's USE_SYSTEM_FASTCDR wants v2)
# The upstream v2.4.3 logger code does not compile with current spdlog/fmt
# because several endpoint types are logged through fmt without formatters.
# SITL only needs the bridge executable, so the logger profile is disabled.
#
# P2P profile is OFF: with superbuild off it would still try to git-fetch the
# Micro-XRCE-DDS *client* (v2.4.3) for the P2P feature. SITL only needs the plain
# `MicroXRCEAgent udp4 -p 8888` bridge, which doesn't touch P2P.
#
# SEAM TO WATCH on the desktop build (analogous to the gz-plugin one): Fast-DDS
# 2.x installs its cmake config under the `fastrtps` name, but the agent does
# find_package(fastdds). If cmake can't find the package, that's the mismatch —
# the fix is usually that fastrtps's config provides both, or set fastdds_DIR.
{ lib, stdenv, fetchFromGitHub, cmake, fastrtps, fastcdr, asio, tinyxml-2 }:

stdenv.mkDerivation (finalAttrs: {
  pname = "micro-xrce-dds-agent";
  version = "2.4.3";

  src = fetchFromGitHub {
    owner = "eProsima";
    repo = "Micro-XRCE-DDS-Agent";
    rev = "v${finalAttrs.version}";
    hash = "sha256-t2PZurWc8Kbkm3zFyNwHQea4Yj+zHWFXFqZ0E19km54=";
  };

  nativeBuildInputs = [ cmake ];
  buildInputs = [ fastrtps fastcdr asio tinyxml-2 ];

  cmakeFlags = [
    "-DUAGENT_SUPERBUILD=OFF"
    "-DUAGENT_USE_SYSTEM_FASTDDS=ON"
    "-DUAGENT_USE_SYSTEM_FASTCDR=ON"
    "-DUAGENT_LOGGER_PROFILE=OFF"
    "-DUAGENT_P2P_PROFILE=OFF"
    "-DUAGENT_BUILD_EXECUTABLE=ON"
  ];

  meta = {
    description = "eProsima Micro XRCE-DDS Agent (PX4<->ROS2 bridge), pinned to the PX4-compatible v2.x line";
    homepage = "https://github.com/eProsima/Micro-XRCE-DDS-Agent";
    license = lib.licenses.asl20;
    mainProgram = "MicroXRCEAgent";
    platforms = lib.platforms.linux;
  };
})
