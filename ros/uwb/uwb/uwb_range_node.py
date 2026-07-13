"""Per-drone UWB range node — the real-hardware drop-in for `sim_uwb`.

Unlike `sim_uwb/uwb_range_sim.py` (one central node publishing *every* drone's
topic from gz ground truth), this is **one node per drone**: it owns this
drone's DW1000 radio and publishes only its own `flight_interfaces/msg/UwbRange`
on `/{ns}/uwb/range`. The fields and topic are identical, so consumers
(missions, `flight_lib.rel_localization`) are untouched — sim↔real is a
topic-level swap.

The ranging core (`uwb_core.ranger.Dw1000Ranger`) is ROS-free and runs in a
background thread; its `on_range` callback publishes here. `--fake` swaps in a
synthetic ranger so the node is exercisable in colcon/CI without the radio.
"""
from __future__ import annotations

import threading

import rclpy
from flight_interfaces.msg import UwbRange
from rclpy.node import Node


class _FakeRanger:
    """Emits synthetic ranges at far_rate for hardware-free testing of the
    ROS plumbing (matches the Dw1000Ranger run()/close() interface)."""

    def __init__(self, own_index, peers, on_range, rate_hz, base_m=3.8):
        self.own_index = own_index
        self.peers = peers
        self.on_range = on_range
        self.dt = 1.0 / max(0.1, rate_hz)
        self.base_m = base_m
        self._stop = threading.Event()
        self._seq = 0

    def run(self):
        import math
        import time
        t0 = time.monotonic()
        while not self._stop.wait(self.dt):
            for p in self.peers:
                d = self.base_m + 0.2 * math.sin(time.monotonic() - t0)
                self.on_range(int(p), int(self.own_index), float(d), self._seq)
                self._seq += 1

    def stop(self):
        self._stop.set()

    def close(self):
        self.stop()


class UwbRangeNode(Node):
    def __init__(self):
        super().__init__("uwb_range_node")
        self.declare_parameter("vehicle_namespace", "px4_0")
        self.declare_parameter("drone_index", 0)
        self.declare_parameter("peer_indices", [1])
        self.declare_parameter("spi_bus", 0)
        self.declare_parameter("spi_device", 0)
        self.declare_parameter("irq_pin", 24)
        self.declare_parameter("cs_pin", 8)
        self.declare_parameter("antenna_delay", 16390)
        self.declare_parameter("far_rate_hz", 10.0)
        self.declare_parameter("near_rate_hz", 50.0)
        self.declare_parameter("near_range_m", 3.0)
        self.declare_parameter("fake", False)

        ns = str(self.get_parameter("vehicle_namespace").value)
        self.index = int(self.get_parameter("drone_index").value)
        self.peers = [int(p) for p in self.get_parameter("peer_indices").value]
        far_rate = float(self.get_parameter("far_rate_hz").value)
        fake = bool(self.get_parameter("fake").value)

        # depth 20 matches sim_uwb's publisher, so consumers see one QoS shape
        self.pub = self.create_publisher(UwbRange, f"/{ns}/uwb/range", 20)

        if fake:
            self.get_logger().warn("running with --fake synthetic ranger (no radio)")
            self.ranger = _FakeRanger(self.index, self.peers, self._on_range, far_rate)
        else:
            # lazy import: pulls in spidev/gpiod, only present on the boards
            from uwb_core.ranger import Dw1000Ranger

            def _addr(i):
                return (0xA0 + (i & 0x0F), 0xC0 + (i & 0x0F))

            self.ranger = Dw1000Ranger(
                own_index=self.index,
                own_addr=_addr(self.index),
                peers=[(p, _addr(p)) for p in self.peers],
                on_range=self._on_range,
                irq=int(self.get_parameter("irq_pin").value),
                ss=int(self.get_parameter("cs_pin").value),
                bus=int(self.get_parameter("spi_bus").value),
                device=int(self.get_parameter("spi_device").value),
                antenna_delay=int(self.get_parameter("antenna_delay").value),
                poll_interval_s=1.0 / max(0.1, far_rate),
            )

        # non-daemon so destroy_node() can stop+join it before teardown, rather
        # than letting it publish into a half-destroyed node (codex Medium).
        self._thread = threading.Thread(target=self.ranger.run,
                                        name="dw1000-ranger")
        self._thread.start()
        self.get_logger().info(
            f"uwb_range_node up: ns={ns} index={self.index} peers={self.peers}")

    def _on_range(self, source_id, receiver_id, range_m, seq):
        msg = UwbRange()
        msg.stamp = self.get_clock().now().to_msg()
        msg.sequence = int(seq) & 0xFFFFFFFF
        msg.source_id = int(source_id)
        msg.receiver_id = int(receiver_id)
        msg.range_m = float(range_m)
        self.pub.publish(msg)

    def destroy_node(self):
        try:
            self.ranger.close()                 # signals the loop to stop
            self._thread.join(timeout=1.0)      # ensure no publish mid-teardown
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = UwbRangeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
