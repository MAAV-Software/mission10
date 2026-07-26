"""The ROS rim: one publisher for each gate, one publish for each accepted intent.

This node does not spin. It has no subscriptions, no timers and no services. Thus
it has no callbacks for an executor. DDS discovery operates in the threads of the
middleware. A publish from a Flask request thread is the only ROS operation in
this process.

The node publishes one message, not a burst. The mission nodes subscribe RELIABLE
with a depth of 10 (px4_offboard/controller.py and
flight_intelligent/phased_orbits_mission.py). Thus DDS sends the message again
until it arrives. The `--times 5` option in scripts/sitl.sh is necessary for a
different reason. A short-lived CLI publisher can start to publish before DDS
discovery is complete. This node keeps its publishers for the life of the process
and does not have that problem.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

from jarvis_web.grammar import INTENTS, Intent

NODE_NAME = "jarvis_web"
QUEUE_DEPTH = 10


class GatePublisher:
    def __init__(self) -> None:
        rclpy.init()
        self._node = Node(NODE_NAME)
        self._publishers = {
            intent.topic: self._node.create_publisher(Bool, intent.topic, QUEUE_DEPTH)
            for intent in INTENTS
        }

    @property
    def logger(self):
        return self._node.get_logger()

    def subscriber_count(self, intent: Intent) -> int:
        """The number of subscribers that DDS matched to this gate.

        The webapp publishes one message and does not repeat it. Thus a count of
        zero means that the command goes nowhere.
        """
        return self._publishers[intent.topic].get_subscription_count()

    def publish(self, intent: Intent) -> None:
        self._publishers[intent.topic].publish(Bool(data=True))
        self.logger.info(f"{intent.name} -> {intent.topic}")

    def shutdown(self) -> None:
        self._node.destroy_node()
        rclpy.shutdown()
