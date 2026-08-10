from __future__ import annotations

import socket
import subprocess
import threading
import time

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from flight_interfaces.msg import FleetMode as FleetModeMsg
from flight_interfaces.msg import UwbRange, UwbState
from flight_interfaces.srv import SetFleetMode
from mission10_uwb_protocol import (
    EgoState,
    FleetMode,
    decode_frame,
    encode_clock_reply,
    encode_ego_state,
    encode_fleet_mode,
    frames,
)
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile


class UwbGateway(Node):
    def __init__(self) -> None:
        super().__init__("uwb_gateway")
        self.declare_parameter("vehicle_namespace", "px4_0")
        self.declare_parameter("vehicle_id", 0)
        self.declare_parameter("socket", "/run/uwb/host.sock")
        self.namespace = str(self.get_parameter("vehicle_namespace").value).strip("/")
        self.vehicle_id = int(self.get_parameter("vehicle_id").value)
        self.socket_path = str(self.get_parameter("socket").value)
        if not 0 <= self.vehicle_id <= 3:
            raise ValueError("vehicle_id must be 0..3")

        self.range_pub = self.create_publisher(UwbRange, f"/{self.namespace}/uwb/range", 50)
        mode_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.mode_pub = self.create_publisher(
            FleetModeMsg, f"/{self.namespace}/fleet/mode", mode_qos
        )
        self.diagnostics_pub = self.create_publisher(DiagnosticArray, "/diagnostics", 10)
        # Peer state relayed over UWB stays off /px4_{peer}/uwb/state: that topic
        # carries each drone's own fresh estimate over DDS, and echoing a stale
        # UWB copy back onto it would reset the peers' HOST_STALE detection.
        self.peer_state_pub = self.create_publisher(
            UwbState, f"/{self.namespace}/uwb/peer_state", 20
        )
        self.create_subscription(
            UwbState, f"/{self.namespace}/uwb/state", self._local_state, 20
        )
        self.create_service(
            SetFleetMode, f"/{self.namespace}/uwb/set_fleet_mode", self._set_fleet_mode
        )

        self.stream = None
        self.stream_lock = threading.Lock()
        self.sequence = 0
        self.mode_cond = threading.Condition()
        self.applied_mode = None
        self.requested_mode = None
        self.target_mode = None
        self.apply_error = None
        threading.Thread(target=self._read_loop, name="uwb-reader", daemon=True).start()
        threading.Thread(target=self._apply_loop, name="fleet-mode", daemon=True).start()
        self.create_timer(0.1, self._raise_apply_error)
        self.create_timer(1.0, self._publish_diagnostics)

    def _next_sequence(self) -> int:
        sequence = self.sequence
        self.sequence = (self.sequence + 1) & 0xFFFFFFFF
        return sequence

    def _send(self, frame: bytes) -> None:
        with self.stream_lock:
            if self.stream is None:
                raise ConnectionError("UWB radio service is disconnected")
            self.stream.sendall(frame)

    def _local_state(self, msg: UwbState) -> None:
        if int(msg.vehicle_id) != self.vehicle_id:
            return
        state = EgoState(
            sample_time_us=int(msg.stamp.sec) * 1_000_000 + int(msg.stamp.nanosec) // 1_000,
            sequence=int(msg.sequence),
            frame_epoch=int(msg.frame_epoch),
            phase_mrad=round(float(msg.phase_rad) * 1_000),
            phase_rate_mrad_s=round(float(msg.phase_rate_rad_s) * 1_000),
            yaw_mrad=round(float(msg.yaw_rad) * 1_000),
            position_enu_mm=tuple(round(float(value) * 1_000) for value in msg.position_enu_m),
            velocity_enu_mm_s=tuple(
                round(float(value) * 1_000) for value in msg.velocity_enu_mps
            ),
            mode=int(msg.mode),
            validity=int(msg.validity),
        )
        try:
            self._send(encode_ego_state(self._next_sequence(), state))
        except ConnectionError:
            pass

    def _set_fleet_mode(self, request, response):
        try:
            mode = FleetMode(int(request.master_id), int(request.network))
            self._send(encode_fleet_mode(self._next_sequence(), mode))
        except (ValueError, ConnectionError) as error:
            response.accepted = False
            response.message = str(error)
            return response
        self._queue_mode(mode)
        response.accepted = True
        response.message = "queued"
        return response

    def _read_loop(self) -> None:
        while rclpy.ok():
            try:
                stream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                stream.connect(self.socket_path)
                with self.stream_lock:
                    self.stream = stream
                with stream.makefile("rb", buffering=0) as source:
                    for frame in frames(source):
                        self._event(decode_frame(frame))
            except (ConnectionError, OSError, ValueError) as error:
                self.get_logger().warning(f"UWB host link: {error}")
                time.sleep(1.0)
            finally:
                with self.stream_lock:
                    self.stream = None

    def _event(self, envelope) -> None:
        if envelope.kind == "completed_exchange":
            self._completed_exchange(envelope.fields)
        elif envelope.kind == "fleet_mode_received":
            _, mode = envelope.fields
            self._queue_mode(mode)
        elif envelope.kind == "clock_probe":
            now_us = time.time_ns() // 1_000
            self._send(
                encode_clock_reply(
                    self._next_sequence(), envelope.fields[0], now_us, now_us, 0, 50
                )
            )

    def _completed_exchange(self, fields) -> None:
        peer, exchange_id, _, event_time, millimetres, _, _, state = fields
        stamp = self.get_clock().now().to_msg()
        if event_time.mission_time_us is not None:
            stamp.sec, micros = divmod(int(event_time.mission_time_us), 1_000_000)
            stamp.nanosec = micros * 1_000

        range_msg = UwbRange()
        range_msg.stamp = stamp
        range_msg.sequence = int(exchange_id)
        range_msg.source_id = int(peer)
        range_msg.receiver_id = self.vehicle_id
        range_msg.range_m = float(millimetres) / 1_000.0
        self.range_pub.publish(range_msg)

        peer_id = int(peer)
        state_msg = UwbState()
        state_msg.stamp = stamp
        state_msg.sequence = int(state.sequence)
        state_msg.vehicle_id = peer_id
        state_msg.frame_epoch = int(state.frame_epoch)
        state_msg.validity = int(state.validity)
        state_msg.phase_rad = float(state.phase_mrad) / 1_000.0
        state_msg.phase_rate_rad_s = float(state.phase_rate_mrad_s) / 1_000.0
        state_msg.yaw_rad = float(state.yaw_mrad) / 1_000.0
        state_msg.position_enu_m = [float(value) / 1_000.0 for value in state.position_enu_mm]
        state_msg.velocity_enu_mps = [
            float(value) / 1_000.0 for value in state.velocity_enu_mm_s
        ]
        state_msg.mode = int(state.mode)
        self.peer_state_pub.publish(state_msg)

    def _queue_mode(self, mode: FleetMode) -> None:
        # The radio repeats each broadcast; requested_mode never resets, so a
        # late repeat of an older mode cannot re-queue behind a newer request
        # while a slow nmcli apply is in flight.
        with self.mode_cond:
            if mode == self.requested_mode:
                return
            self.requested_mode = mode
            self.target_mode = mode
            self.mode_cond.notify()
        msg = FleetModeMsg()
        msg.master_id = mode.master
        msg.network = mode.network
        self.mode_pub.publish(msg)

    def _apply_loop(self) -> None:
        while True:
            with self.mode_cond:
                while self.target_mode is None:
                    self.mode_cond.wait()
                mode = self.target_mode
            network = "field" if mode.network == FleetMode.FIELD else "internet"
            try:
                subprocess.run(
                    ["/usr/local/sbin/fleet-network", "set-mode", str(mode.master), network],
                    check=True,
                )
            except subprocess.CalledProcessError as error:
                self.apply_error = error
                return
            with self.mode_cond:
                self.applied_mode = mode
                if self.target_mode == mode:
                    self.target_mode = None

    def _raise_apply_error(self) -> None:
        if self.apply_error is not None:
            raise self.apply_error

    def _publish_diagnostics(self) -> None:
        array = DiagnosticArray()
        array.header.stamp = self.get_clock().now().to_msg()
        status = DiagnosticStatus()
        status.name = "uwb/host_link"
        status.hardware_id = f"dw1000:{self.vehicle_id}"
        status.level = DiagnosticStatus.OK if self.stream is not None else DiagnosticStatus.ERROR
        status.message = "connected" if self.stream is not None else "disconnected"
        status.values = [KeyValue(key="socket", value=self.socket_path)]
        array.status = [status]
        self.diagnostics_pub.publish(array)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = UwbGateway()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
