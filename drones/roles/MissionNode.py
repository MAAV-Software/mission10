import rclpy
import queue
import time
import os
from rclpy.node import Node
from std_msgs.msg import Bool
from px4_msgs.msg import SensorGps
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

px4_path = os.getenv("PX4_NAMESPACE")

class MissionNode(Node):
    def __init__(self, mission_node_lock, mission_node_cv, mission_mode):
        super().__init__("mission_node")

        self.mission_node_lock = mission_node_lock
        self.mission_node_cv = mission_node_cv
        self.mission_mode = mission_mode
        self.timestamp_queue = queue.Queue()
        self.latest_timestamp = 0

        self.orbit_publisher = None
        self.survey_publisher = None

        start_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST
        )

        self.start_publisher = self.create_publisher(
            Bool,
            "/start_mission",
            start_qos_profile
        )

        if self.mission_mode == "survey":
            self.survey_publisher = self.create_publisher(
                Bool,
                "/begin_survey",
                start_qos_profile
            )
        elif self.mission_mode == "orbit":
            self.orbit_publisher = self.create_publisher(
                Bool,
                "/begin_orbit",
                start_qos_profile
            )

        qos_profile = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                depth=10
        )

        self.gps_subscription = self.create_subscription(
            SensorGps,
            f"{px4_path}/fmu/out/vehicle_gps_position",
            self.gps_callback,
            qos_profile
        )

        self.gps_data = {}

    def gps_callback(self, msg):
        # print("GPS CALLBACK FIRED")
        # print("Waiting for the mission_node_lock")
        with self.mission_node_lock:
            # print("Acquired the mission_node_lock")
            gps_point = {
                "timestamp": msg.timestamp,
                "latitude": msg.latitude_deg,
                "longitude": msg.longitude_deg,
                "altitude": msg.altitude_msl_m
            }

            self.gps_data[msg.timestamp] = gps_point
            self.latest_timestamp = msg.timestamp

            # print(gps_point)

            with self.mission_node_cv:
                self.mission_node_cv.notify()

    def start_mission(self):
        msg = Bool()
        msg.data = True

        px4_path = os.getenv("PX4_NAMESPACE")

        while self.count_publishers(f"{px4_path}/fmu/out/vehicle_gps_position") == 0:
            print("Stuck here")
            rclpy.spin_once(self, timeout_sec=0.1)

        self.start_publisher.publish(msg)
        print("published start_mission")
        time.sleep(15)
        if self.mission_mode == "orbit":
            print("Published orbit")
            self.orbit_publisher.publish(msg)
        elif self.mission_mode == "survey":
            print("Published survey")
            self.survey_publisher.publish(msg)
        self.get_logger().info("Mission started!")
