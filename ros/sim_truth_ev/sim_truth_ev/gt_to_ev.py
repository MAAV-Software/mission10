from __future__ import annotations

import rclpy
from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleOdometry
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from sim_truth_ev.frames import enu_vector_to_ned, ros_enu_flu_to_px4_ned_frd


class GroundTruthToEv(Node):
    def __init__(self):
        super().__init__("gt_to_ev")
        self.declare_parameter("vehicle_namespace", "px4_0")
        self.declare_parameter("odom_topic", "ground_truth/odometry")
        self.declare_parameter("publish", True)
        self.declare_parameter("position_variance", 0.05)
        self.declare_parameter("orientation_variance", 0.02)
        # Fault injection: step the EV frame at runtime
        # (`ros2 param set ... fault_offset_n 5.0`). The step is silent
        # (reset_counter untouched): announced resets get absorbed by EKF2's EV
        # frame-offset handling, while an unannounced step fails the innovation
        # gates until EKF2 hard-resets position onto the biased measurement.
        self.declare_parameter("fault_offset_n", 0.0)
        self.declare_parameter("fault_offset_e", 0.0)
        # VIO emulation: degrade truth the way a visual estimator degrades.
        # fault_scale — monocular scale error, applied about the EV origin.
        # fault_drift_{n,e}_mps — slow frame drift (accumulates from the moment set).
        # publish=false at runtime — tracking loss / dropout (also settable at launch).
        # reset_now=true — announced re-init: bumps reset_counter on the next sample
        # (pair with fault_offset_* to jump the frame the way a VIO re-init does;
        # EKF2 should absorb this via its EV frame-reset handling, unlike the
        # silent step above).
        self.declare_parameter("fault_scale", 1.0)
        self.declare_parameter("fault_drift_n_mps", 0.0)
        self.declare_parameter("fault_drift_e_mps", 0.0)
        self.declare_parameter("reset_now", False)

        self.ns = self.get_parameter("vehicle_namespace").value.strip("/")
        odom_topic = str(self.get_parameter("odom_topic").value)
        self.publish_ev = bool(self.get_parameter("publish").value)
        self.position_variance = float(self.get_parameter("position_variance").value)
        self.orientation_variance = float(self.get_parameter("orientation_variance").value)
        self._fault_n = float(self.get_parameter("fault_offset_n").value)
        self._fault_e = float(self.get_parameter("fault_offset_e").value)
        self._fault_scale = float(self.get_parameter("fault_scale").value)
        self._drift_rate = [
            float(self.get_parameter("fault_drift_n_mps").value),
            float(self.get_parameter("fault_drift_e_mps").value),
        ]
        self._drift_acc = [0.0, 0.0]
        self._drift_t_prev: float | None = None
        self._reset_counter = 0
        self._reset_pending = False
        self.add_on_set_parameters_callback(self._on_param_change)

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self._pub = self.create_publisher(VehicleOdometry, self._topic("in/vehicle_visual_odometry"), qos)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, qos)
        self.get_logger().info(
            f"gt_to_ev up: odom_topic={odom_topic} px4_ns={self.ns} publish={self.publish_ev}"
        )

    def _on_param_change(self, params):
        for p in params:
            if p.name == "fault_offset_n":
                self._fault_n = float(p.value)
            elif p.name == "fault_offset_e":
                self._fault_e = float(p.value)
            elif p.name == "fault_scale":
                self._fault_scale = float(p.value)
                self.get_logger().warn(f"EV scale error = {self._fault_scale:.3f}")
                continue
            elif p.name == "fault_drift_n_mps":
                self._drift_rate[0] = float(p.value)
                self.get_logger().warn(f"EV drift N = {self._drift_rate[0]:+.3f} m/s")
                continue
            elif p.name == "fault_drift_e_mps":
                self._drift_rate[1] = float(p.value)
                self.get_logger().warn(f"EV drift E = {self._drift_rate[1]:+.3f} m/s")
                continue
            elif p.name == "publish":
                self.publish_ev = bool(p.value)
                self.get_logger().warn(f"EV publish = {self.publish_ev} (dropout emulation)")
                continue
            elif p.name == "reset_now":
                if bool(p.value):
                    self._reset_pending = True
                    self.get_logger().warn("EV announced reset queued (reset_counter will bump)")
                continue
            else:
                continue
            self.get_logger().warn(
                f"EV fault offset NE=({self._fault_n:+.2f},{self._fault_e:+.2f}) (silent step)"
            )
        return SetParametersResult(successful=True)

    def _topic(self, suffix: str) -> str:
        return f"/{self.ns}/fmu/{suffix}" if self.ns else f"/fmu/{suffix}"

    def _now_us(self) -> int:
        return int(self.get_clock().now().nanoseconds / 1000)

    def _odom_cb(self, msg: Odometry):
        pose = msg.pose.pose

        out = VehicleOdometry()
        now_us = self._now_us()
        out.timestamp = now_us
        stamp = msg.header.stamp
        sample_us = stamp.sec * 1_000_000 + stamp.nanosec // 1000
        out.timestamp_sample = sample_us if sample_us > 0 else now_us
        out.pose_frame = VehicleOdometry.POSE_FRAME_NED
        # Mocap/QTM gives pose only; the EV airframe (EKF2_EV_CTRL=11) does not fuse
        # EV velocity. Mark velocity absent (NaN + UNKNOWN frame) per PX4 convention.
        out.velocity_frame = VehicleOdometry.VELOCITY_FRAME_UNKNOWN

        t_s = out.timestamp_sample / 1e6
        if self._drift_t_prev is not None:
            dt = t_s - self._drift_t_prev
            if 0.0 < dt < 1.0:
                self._drift_acc[0] += self._drift_rate[0] * dt
                self._drift_acc[1] += self._drift_rate[1] * dt
        self._drift_t_prev = t_s
        if self._reset_pending:
            self._reset_counter += 1
            self._reset_pending = False

        pn, pe, pd = enu_vector_to_ned((pose.position.x, pose.position.y, pose.position.z))
        s = self._fault_scale
        out.position[:] = [
            s * pn + self._fault_n + self._drift_acc[0],
            s * pe + self._fault_e + self._drift_acc[1],
            s * pd,
        ]
        out.q[:] = ros_enu_flu_to_px4_ned_frd((
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ))
        nan = float("nan")
        out.velocity[:] = [nan, nan, nan]
        out.angular_velocity[:] = [nan, nan, nan]
        out.position_variance[:] = [self.position_variance] * 3
        out.orientation_variance[:] = [self.orientation_variance] * 3
        out.velocity_variance[:] = [nan, nan, nan]
        out.reset_counter = self._reset_counter
        out.quality = 100 if self.publish_ev else 0

        if self.publish_ev:
            self._pub.publish(out)
        else:
            p = out.position
            self.get_logger().debug(
                f"EV sample publish=false p_ned=({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})"
            )


def main(args=None):
    rclpy.init(args=args)
    node = GroundTruthToEv()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
