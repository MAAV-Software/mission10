"""The ROS rim: one publisher for each gate, one publish for each accepted intent.

This node does not spin. It has no subscriptions, no timers and no services. Thus
it has no callbacks for an executor. DDS discovery operates in the threads of the
middleware. A publish from a Flask request thread is the only ROS operation in
this process.

The publishers use the shared RELIABLE, TRANSIENT_LOCAL, depth-one gate profile.
They live for the life of the process, so a mission node that joins after a gate
fires receives the last value.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

from jarvis_web.grammar import INTENTS, Intent
from px4_offboard.gate_qos import MISSION_GATE_QOS

NODE_NAME = "jarvis_web"


class GatePublisher:
    def __init__(self, expected_fleet_size: int = 1) -> None:
        if expected_fleet_size < 1:
            raise ValueError("expected_fleet_size must be at least 1")
        rclpy.init()
        self._node = Node(NODE_NAME)
        self.expected_fleet_size = expected_fleet_size
        self._publishers = {
            intent.topic: self._node.create_publisher(
                Bool, intent.topic, MISSION_GATE_QOS
            )
            for intent in INTENTS
        }

    @property
    def logger(self):
        return self._node.get_logger()

    def subscriber_count(self, intent: Intent) -> int:
        """The number of subscribers that DDS matched to this gate.

        The webapp checks this against expected_fleet_size before it publishes.
        """
        return self._publishers[intent.topic].get_subscription_count()

    def publish(self, intent: Intent) -> None:
        self._publishers[intent.topic].publish(Bool(data=True))
        self.logger.info(f"{intent.name} -> {intent.topic}")

    def shutdown(self) -> None:
        self._node.destroy_node()
        rclpy.shutdown()
