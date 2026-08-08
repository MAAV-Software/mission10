import rclpy
import queue
from rclpy.node import Node
from std_msgs.msg import Bool
from px4_msgs.msg import SensorGps
from rclpy.qos import QoSProfile, ReliabilityPolicy


class MissionNode(Node):
    def __init__(self, mission_node_lock, mission_node_cv, mission_mode):
        super().__init__("mission_node")

        self.mission_node_lock = mission_node_lock
        self.mission_node_cv = mission_node_cv
        self.mission_mode = mission_mode
        self.timestamp_queue = queue.Queue()
        self.latest_timestamp = 0

        # self.start_publisher = self.create_publisher(
        #     Bool,
        #     "/start_mission",
        #     10
        # )

        # if self.mission_mode == "survey":
        #     self.start_publisher = self.create_publisher(
        #         Bool,
        #         "/begin_survey",
        #         10
        #     )
        # elif self.mission_mode == "orbit":
        #     self.start_publisher = self.create_publisher(
        #         Bool,
        #         "/begin_orbit",
        #         10
        #     )

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

        self.gps_data = {}

    def gps_callback(self, msg):
        print("GPS CALLBACK FIRED")
        print("Waiting for the mission_node_lock")
        with self.mission_node_lock:
            print("Acquired the mission_node_lock")
            gps_point = {
                "timestamp": msg.timestamp,
                "latitude": msg.latitude_deg,
                "longitude": msg.longitude_deg,
                "altitude": msg.altitude_msl_m
            }

            self.gps_data[msg.timestamp] = gps_point
            self.latest_timestamp = msg.timestamp

            print(gps_point)

            with self.mission_node_cv:
                self.mission_node_cv.notify()

    # def start_mission(self):
    #     msg = Bool()
    #     msg.data = True

    #     while self.count_publishers("/fmu/out/vehicle_gps_position") == 0:
    #         rclpy.spin_once(self, timeout_sec=0.1)

    #     self.start_publisher.publish(msg)
    #     self.get_logger().info("Mission started!")

    def start_mission(self):
        self.get_logger().info("Waiting for GPS publisher...")

        while self.count_publishers("/fmu/out/vehicle_gps_position") == 0:
            rclpy.spin_once(self, timeout_sec=0.1)
            print(f"Number of publishers is {self.count_publishers('/fmu/out/vehicle_gps_position')}")

        print("SUBSCRIBERS:", self.count_subscribers("/fmu/out/vehicle_gps_position"))
        print("PUBLISHERS:", self.count_publishers("/fmu/out/vehicle_gps_position"))
        self.get_logger().info("GPS publisher found!")
        self.get_logger().info("Mission started!")
