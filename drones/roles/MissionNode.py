import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from px4_msgs.msg import SensorGps
from rclpy.qos import QoSProfile, ReliabilityPolicy


class MissionNode(Node):
    def __init__(self):
        super().__init__("mission_node")

        self.start_publisher = self.create_publisher(
            Bool,
            "/start_mission",
            10
        )

        qos_profile = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                depth=10
        )

        self.gps_subscription = self.create_subscription(
            SensorGps,
            "/fmu/out/vehicle_gps_position",
            self.gps_callback,
            qos_profile
        )

        self.gps_data = []

    def gps_callback(self, msg):
        gps_point = {
            "timestamp": msg.timestamp,
            "latitude": msg.latitude_deg,
            "longitude": msg.longitude_deg,
            "altitude": msg.altitude_msl_m
        }

        self.gps_data.append(gps_point)

        print(gps_point)

    def start_mission(self):
        msg = Bool()
        msg.data = True

        while self.start_publisher.get_subscription_count() == 0:
            rclpy.spin_once(self, timeout_sec=0.1)

        self.start_publisher.publish(msg)
        self.get_logger().info("Mission started!")
