"""An accepted intent must reach a real subscriber on its own topic.

This is the purpose of the webapp, thus it gets a real node and real DDS. The
tests with fakes cannot show that a topic name is correct or that one publish is
sufficient.

This test is slow, because DDS discovery is slow.
"""

import time
import unittest

from jarvis_web.grammar import INTENTS

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Bool
    from px4_offboard.gate_qos import MISSION_GATE_QOS

    HAVE_ROS = True
except ImportError:  # a machine with no ROS: the pure tests still run
    HAVE_ROS = False

DISCOVERY_TIMEOUT_S = 15.0
DELIVERY_TIMEOUT_S = 5.0
SPIN_S = 0.05


@unittest.skipUnless(HAVE_ROS, "rclpy is not available")
class TestGatePublisher(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from jarvis_web.node import GatePublisher

        # GatePublisher calls rclpy.init(). The listener then uses the same
        # default context.
        cls.gates = GatePublisher()
        cls.listener = None
        cls.received = {intent.topic: [] for intent in INTENTS}

    @classmethod
    def tearDownClass(cls):
        if cls.listener is not None:
            cls.listener.destroy_node()
        cls.gates.shutdown()

    def wait_until(self, done, timeout_s):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if done():
                return True
            rclpy.spin_once(self.listener, timeout_sec=SPIN_S)
        return done()

    def test_a_late_joiner_receives_the_last_gate_on_each_topic(self):
        # Publish before the subscriber exists. The long-lived GatePublisher
        # retains each sample for a later TRANSIENT_LOCAL subscription.
        for intent in INTENTS:
            self.gates.publish(intent)

        self.__class__.listener = Node("jarvis_web_test_late_listener")
        for intent in INTENTS:
            self.listener.create_subscription(
                Bool,
                intent.topic,
                lambda msg, topic=intent.topic: self.received[topic].append(msg.data),
                MISSION_GATE_QOS,
            )

        matched = self.wait_until(
            lambda: all(self.gates.subscriber_count(i) > 0 for i in INTENTS),
            DISCOVERY_TIMEOUT_S,
        )
        self.assertTrue(matched, "DDS did not match the publishers to the listener")

        self.wait_until(
            lambda: all(self.received[i.topic] for i in INTENTS), DELIVERY_TIMEOUT_S
        )

        for intent in INTENTS:
            with self.subTest(intent=intent.name):
                # Exactly one retained message, and its value is True.
                self.assertEqual(self.received[intent.topic], [True])


if __name__ == "__main__":
    unittest.main()
