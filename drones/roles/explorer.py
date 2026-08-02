"""Example TCP socket client."""
import socket
import threading
import json
import time
import math
import numpy as np
from pathlib import Path
import subprocess
import queue
import sys
from pyproj import Transformer, CRS

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from roles.MissionNode import MissionNode

# TEMP imports
# import random

bounds_path = Path(__file__).parent.parent.parent / "constants/bounding_boxes.txt"
aggregate_path = Path(__file__).parent.parent / "all_results.csv"
camera_specs_path = Path(__file__).parent.parent.parent / "constants/camera_specs.txt"
base_altitude_path = Path(__file__).parent.parent.parent / "constants/altitude.txt"

class ExploreDrone:
    "Construct an instance of an explorer Drone"

    def __init__(self, host, port, manager_host, manager_port):
        """Construct a Manager instance and start listening for messages."""

        self.host = host
        self.port = port
        self.coords = []
        self.TMP_gps_data = {} # Replace instances of this wth self.mission_node.gps_data
        self.TMP_timestamp_queue = queue.Queue() # Replace instances of this with self.mission_node.timestamp_queue
        self.mission_node = None
        self.send_buffer = []
        self.startup = True
        self.bounds = []
        self.shutdown_flag = False
        self.manager_host = manager_host
        self.manager_port = manager_port
        self.registered = False

        self.coords_lock = threading.Lock()
        self.coords_cv = threading.Condition()

        self.to_spcs = Transformer.from_crs("EPSG:4326", "EPSG:6498", always_xy=True)
        self.from_spcs = Transformer.from_crs("EPSG:6498", "EPSG:4326", always_xy=True) # If we are in Michigan

        # self.to_spcs = Transformer.from_crs("EPSG:4326", "EPSG:6418", always_xy=True) 
        # self.from_spcs = Transformer.from_crs("EPSG:6418", "EPSG:4326", always_xy=True)     # if we were in Huntsville, AL

        with open(bounds_path, "r") as f:
            for line in f:
                line_contents = line.strip().split()
                self.bounds.append(( float(line_contents[0]), float(line_contents[1]) ))
        
        self.camera_HFOV = 0
        self.camera_VFOV = 0
        self.base_altitude = 0

        with open(camera_specs_path, "r") as f:
            for line in f:
                line_contents = line.strip().split()
                self.camera_HFOV= float(line_contents[0])
                self.cammera_VFOV = float(line_contents[1])
                break
        
        with open(base_altitude_path, "r") as f:
            for line in f:
                line_contents = line.strip().split()
                self.base_altitude = float(line_contents[0])
                break

        # print(self.manager_host)
        # print("Finished registering!")
        self.run_drone()

    def rotate_coords(self, pt_latitude, pt_longitude):
    
        def convert_to_spcs(latlon, transformer):
            x, y = transformer.transform(latlon[1], latlon[0])
            return (x, y)

        def convert_from_spcs(spcs, transformer):
            lon, lat = transformer.transform(spcs[0], spcs[1])
            return (lat, lon)

        def rotate_point(pivot, point, theta):
            dx = point[0] - pivot[0]
            dy = point[1] - pivot[1]
            x_rot = dx * math.cos(theta) - dy * math.sin(theta)
            y_rot = dx * math.sin(theta) + dy * math.cos(theta)
            return (x_rot + pivot[0], y_rot + pivot[1])

        def calc_theta(tl, tr):
            return math.atan2(tr[1] - tl[1], tr[0] - tl[0])

        rand_latlon = []
        rand_latlon.append(pt_latitude)
        rand_latlon.append(pt_longitude)

        tl = [self.bounds[0][0], self.bounds[0][1]]
        tr = [self.bounds[1][0], self.bounds[1][1]]

        #convert to SPCS coordinates
        tl_spcs = convert_to_spcs(tl, self.to_spcs)
        tr_spcs = convert_to_spcs(tr, self.to_spcs)

        #calcu theta w top left and top right corners
        theta = calc_theta(tl_spcs, tr_spcs)
        # print("theta: ", math.degrees(theta), " degrees"

        rand_spcs = convert_to_spcs(rand_latlon, self.to_spcs)

        # print("orig point: ", rand_latlon)

        #rotate by theta
        rand_rotated = rotate_point(tl_spcs, rand_spcs, -1 * theta) # Use -1 to rotate against the over-rotation
        rand_rotated_latlon = convert_from_spcs(rand_rotated, self.from_spcs)

        return (rand_rotated_latlon[0], rand_rotated_latlon[1])

    def get_mine_loc_in_img(self, timestamp):

        if True:
            # Will need to reference the timestamp to know which frame to look at
            return (0.5, 0.5, 0.5, 0.5)
        else:
            return (-1, -1, -1, -1)

    def try_connect_to_manager(self):
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.connect((self.manager_host, self.manager_port))
                    return True
            except ConnectionRefusedError:
                print("Manager not started yet")
            time.sleep(0.1)

    def find_mines(self):
        # Initialize the camera interface

        # Start the CV detection pipelien
        while not self.shutdown_flag:
            # If there are detections, append that to self.detected_mine_data

            next_timestamp = 0
            absolute_height = 0
            pt_latitude = 0
            pt_longitude = 0

            with self.coords_cv:
                while self.mission_node is None or self.mission_node.timestamp_queue.qsize() == 0:
                # while self.TMP_timestamp_queue.qsize() == 0:
                    print("Waiting for timestamp to fill up")
                    self.coords_cv.wait()

            with self.coords_lock:
                print("Acquired the lock in find_mines")
                next_timestamp = self.mission_node.timestamp_queue.get()
                absolute_height = self.mission_node.gps_data[next_timestamp]["altitude"]
                pt_latitude = self.mission_node.gps_data[next_timestamp]["latitude"]
                pt_longitude = self.mission_node.gps_data[next_timestamp]["longitude"]

                # next_timestamp = self.TMP_timestamp_queue.get()
                # absolute_height = self.TMP_gps_data[next_timestamp]["altitude"]
                # pt_latitude = self.TMP_gps_data[next_timestamp]["latitude"]
                # pt_longitude = self.TMP_gps_data[next_timestamp]["longitude"]

            # Get the location of the mines within the image (bounding box or smth I dunno)
            print("Got everything that I needed, thanks")
            mine_x_min, mine_x_max, mine_y_min, mine_y_max = self.get_mine_loc_in_img(next_timestamp)

            if mine_x_min == -1:
                continue

            pt_latitude, pt_longitude = self.rotate_coords(pt_latitude, pt_longitude)

            # Get the dimension of the camera frame
            hor_rad = math.radians(self.camera_HFOV)
            img_width_m = 2 * (absolute_height - self.base_altitude) * math.tan(hor_rad) 
            
            vert_rad = math.radians(self.camera_VFOV)
            img_height_m = 2 * (absolute_height - self.base_altitude) * math.tan(vert_rad)

            img_height_cm = (img_height_m) / 100 #convert to cm
            img_width_cm = (img_width_m) / 100
            
            # mine_x_min, mine_y_min, mine_x_max, mine_y_max = (0.11155333116319445, 0.15966543579101564, 0.19914363606770832, 0.22527638753255208)
            mine_x , mine_y = (mine_x_min + mine_x_max ) / 2, (mine_y_min + mine_y_max ) / 2
            mine_x_relative = mine_x - 0.5
            mine_y_relative = mine_y - 0.5
            
            scaled_x = mine_x_relative * img_width_cm
            scaled_y = mine_y_relative * img_height_cm

            #from 4/5 onwards
            scaled_x_meters = scaled_x / 100 #convert to meters
            scaled_y_meters = scaled_y / 100

            change_in_lat = scaled_y_meters/111320 #find change in latitude from center to point
            change_in_long = scaled_x_meters/(111320*np.cos(math.radians(pt_latitude)))

            new_lat = change_in_lat + pt_latitude #calculate new lat/long
            new_long = change_in_long + pt_longitude

            with self.coords_lock:
                print("Acquired the lock in find_mines but lower")
                self.coords.append((new_lat, new_long))
                self.send_buffer.append((new_lat, new_long))
                print(f"The new point is {new_lat}, {new_long} and the number of detected_mines is {len(self.coords)}")

            if len(self.send_buffer) >= 10:
                print("Should try to send soon")
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    self.try_connect_to_manager()
                    sock.connect((self.manager_host, self.manager_port))
                    print("Got the lock and am about to send now")
                    self.sending_state = True
                    for coord in self.send_buffer:
                        message = json.dumps({
                            "message_type": "coordinates",
                            "host": self.host,
                            "port": self.port,
                            "coords": coord
                        })
                        sock.sendall((message + "\n").encode("utf-8"))
                        # sock.sendall(message.encode("utf-8"))
                        print("Sent message")

                self.send_buffer = []

    def tcp_server(self):

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:

            # Bind the socket to the server
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.host, self.port))
            sock.listen()

            with self.coords_lock:
                if not self.registered:
                    self.register()

            # Socket accept() will block for a maximum of 1 second.  If you
            # omit this, it blocks indefinitely, waiting for a connection.
            sock.settimeout(1)

            while not self.shutdown_flag:

                # Wait for a connection for 1s.  The socket library avoids consuming
                # CPU while waiting for a connection.
                try:
                    clientsocket, address = sock.accept()
                except socket.timeout:
                    continue
                print("Connection from", address[0])

                # Socket recv() will block for a maximum of 1 second.  If you omit
                # this, it blocks indefinitely, waiting for packets.
                clientsocket.settimeout(1)

                # Receive data, one chunk at a time.  If recv() times out before we
                # can read a chunk, then go back to the top of the loop and try
                # again.  When the client closes the connection, recv() returns
                # empty data, which breaks out of the loop.  We make a simplifying
                # assumption that the client will always cleanly close the
                # connection.
                with clientsocket:
                    message_chunks = []
                    while True:
                        try:
                            data = clientsocket.recv(4096)
                        except socket.timeout:
                            continue
                        if not data:
                            break
                        message_chunks.append(data)

                # Decode list-of-byte-strings to UTF8 and parse JSON data
                message_bytes = b''.join(message_chunks)
                message_str = message_bytes.decode("utf-8")

                try:
                    message_dict = json.loads(message_str)
                except json.JSONDecodeError:
                    continue
                print(message_dict)
                self.handle_message(message_dict)
    
    def handle_message(self, message_dict):
        if message_dict["message_type"] == "registration_ack":
            self.registered = True
        elif message_dict["message_type"] == "run_drones":
            self.handle_run_drones()
        else:
            print("Message Unknown")

    # def fake_gps_coords_generation(self):
    #     fake_timestamp = 0
    #     print("Entered fake gps coords generation")
    #     while fake_timestamp < 100:
    #         print("Waiting fo the coords_lock")
    #         with self.coords_lock:
    #             print("Acquired the lock in fake_gps_coords_generation")
    #             fake_lat = random.randint(1, 100)
    #             fake_lon = random.randint(1, 100)
    #             fake_alt = random.randint(1, 100)
    #             fake_gps_point = {}
    #             self.TMP_gps_data[fake_timestamp] = {
    #                 "latitude": fake_lat,
    #                 "longitude": fake_lon,
    #                 "altitude": fake_alt
    #             }
    #             self.TMP_timestamp_queue.put(fake_timestamp)
    #         fake_timestamp += 1

    #         print(f"{fake_lat}, {fake_lon}, {fake_alt}")
    #         with self.coords_cv:
    #             self.coords_cv.notify()
    #         time.sleep(0.1)

    def handle_run_drones(self):
        # subprocess.run("ros2", ...) # Run that ros2 command to publish to the start topic
        # time.sleep(5)

        rclpy.init()
        
        node = MissionNode(self.coords_lock, self.coords_cv)
        self.mission_node = node

        while not self.shutdown_flag:
            try:
                # Start the mission
                # node.start_mission()

                # Continue processing GPS messages
                rclpy.spin(node)

            except KeyboardInterrupt:
                pass

            finally:
                print(f"Collected {len(node.gps_data)} GPS points")
                node.destroy_node()
                rclpy.shutdown()

        # self.fake_gps_coords_generation()
    
    
    def register(self):
        if not self.registered:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    self.try_connect_to_manager()
                    sock.connect((self.manager_host, self.manager_port))
                    message = json.dumps({
                        "message_type": "registration",
                        "drone_host": self.host,
                        "drone_port": self.port
                    })
                    sock.sendall((message + "\n").encode("utf-8"))

            except ConnectionRefusedError:
                print("Manager not started yet")
    
    def send_heartbeat(self):
        while not self.shutdown_flag:

            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                # Connect to the UDP socket on server
                # self.try_connect_to_manager()
                sock.connect((self.manager_host, self.manager_port))

                # Send a message
                message = json.dumps({"message_type": "heartbeat",
                                      "drone_host": self.host,
                                      "drone_port": self.port})
                sock.sendall(message.encode('utf-8'))
            time.sleep(1)
        
    def run_drone(self):
        tcp_thread = threading.Thread(target=self.tcp_server)
        # udp_thread = threading.Thread(target=self.send_heartbeat)
        tcp_thread.start()
        # udp_thread.start()

        find_mines_thread = threading.Thread(target=self.find_mines)
        find_mines_thread.start()

        fake_gps_thread = threading.Thread(target=self.handle_run_drones)
        fake_gps_thread.start()

        tcp_thread.join()
        # udp_thread.join()
        find_mines_thread.join()
        fake_gps_thread.join()
