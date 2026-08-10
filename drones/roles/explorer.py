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
import cv2
from picamera2 import Picamera2
import os
import matplotlib
import onnxruntime as ort
import random
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from drones.roles.MissionNode import MissionNode
from ros.sensing.sensing.mine_detector import get_hailo_bounding_boxes

bounds_path = Path(__file__).parent.parent.parent / "constants/bounding_boxes.txt"
aggregate_path = Path(__file__).parent.parent / "all_results.csv"
camera_specs_path = Path(__file__).parent.parent.parent / "constants/camera_specs.txt"
base_altitude_path = Path(__file__).parent.parent.parent / "constants/altitude.txt"
weights_path = Path(__file__).parent.parent / "realpositive-appearance-fold1-epoch9-yolo11m-640.onnx" # Also this model weight sucks ass

class ExploreDrone:
    "Construct an instance of an explorer Drone"

    def __init__(self, host, port, manager_host, manager_port, camera_mode):
        """Construct a Manager instance and start listening for messages."""

        self.host = host
        self.port = port
        self.coords = []
        self.TMP_gps_data = {} # Replace instances of this wth self.mission_node.gps_data
        self.TMP_timestamp_queue = queue.Queue() # OK, ur in the show now cuz we need you
        self.mission_node = None
        self.send_buffer = []
        self.startup = True
        self.bounds = []
        self.shutdown_flag = False
        self.manager_host = manager_host
        self.manager_port = manager_port
        self.registered = False
        self.camera_mode = camera_mode
        self.timestamp_bounding_boxes = queue.Queue()
        self.start_time = time.time()
        self.worker_shutoff_time = 15

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

        rand_spcs = convert_to_spcs(rand_latlon, self.to_spcs)

        #rotate by theta
        rand_rotated = rotate_point(tl_spcs, rand_spcs, -1 * theta) # Use -1 to rotate against the over-rotation
        rand_rotated_latlon = convert_from_spcs(rand_rotated, self.from_spcs)

        return (rand_rotated_latlon[0], rand_rotated_latlon[1])

    def primary_camera_func(self):
        with self.coords_cv:
            while self.mission_node is None:
                print("Waiting for mission to start")
                self.coords_cv.wait()

        get_hailo_bounding_boxes(self.timestamp_bounding_boxes, self.coords_lock)

    def backup_camera_func(self):

        with self.coords_cv:
            while self.mission_node is None:
                print("Waiting for mission to start")
                self.coords_cv.wait()

                if self.shutdown_flag:
                    return

        # Somehow interface the pi-camera module, lead the image into memory, run the yolo, and then save the timestamp for that photo
        cam = Picamera2(0)
        config = cam.create_still_configuration()
        cam.configure(config)
        cam.start()

        session = ort.InferenceSession(weights_path)
        input_name = session.get_inputs()[0].name


        while not self.shutdown_flag:
            # Run Camera picture, and then run the YOLO, and then place those detection values to that timestamp value?
            image = cam.capture_array()

            image_timestamp = 0
            with self.coords_lock:
                image_timestamp = self.mission_node.latest_timestamp

            print(f"Image taken at timestamp {image_timestamp}!")

            img = cv2.resize(image, (256, 256))
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
            img = np.expand_dims(img, axis=0)   # CHW -> NCHW

            outputs = session.run(
                None,
                {input_name: img}
            )

            pred = outputs[0]
            pred = pred[0]
            pred = pred.T

            confidence_threshold = 0.3 # also need to calibrate this threshold

            detections = []

            # Going to have to do some more post-processing for these raw onnx output, but it's a start for sure
            for x, y, w, h, conf in pred:
                if conf > confidence_threshold:
                    detections.append((image_timestamp, x, y, w, h))

            with self.coords_lock:
                for detection in detections:
                    self.timestamp_bounding_boxes.put(detection)

            time.sleep(1) # have it take pictures at this interval, I guess try to do it one time a second I guess?

    def try_connect_to_manager(self):
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.connect((self.manager_host, self.manager_port))
                    return True
            except ConnectionRefusedError:
                print("Manager not started yet")
            time.sleep(0.1)

    def send_coords(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            while True:
                try:
                    sock.connect((self.manager_host, self.manager_port))
                    break
                except ConnectionRefusedError:
                    print("Manager not started yet")
                time.sleep(0.1)

            self.sending_state = True
            # print(f"There are {len(self.send_buffer)} coords in the send buffer!")
            for coord in self.send_buffer:
                coord = [float(x) for x in coord]
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

    def find_mines(self):
        # Initialize the camera interface

        # Start the CV detection pipelien
        while not self.shutdown_flag:
            # If there are detections, append that to self.coords
            
            next_timestamp = 0
            absolute_height = 0
            pt_latitude = 0
            pt_longitude = 0

            mine_x_min = mine_x_max = mine_y_min = mine_y_max = -1
            mine_x_center = mine_y_center = x_width = y_width = -1

            with self.coords_cv:
                # while self.mission_node is None or self.mission_node.timestamp_queue.qsize() == 0:
               while (self.mission_node is None or self.timestamp_bounding_boxes.qsize() == 0):
                    self.coords_cv.wait()

                    if self.shutdown_flag:
                        return

            with self.coords_lock:
                # print("Acquired the lock in find_mines")
                next_timestamp, mine_x_center, mine_y_center, x_width, y_width = self.timestamp_bounding_boxes.get()
                absolute_height = self.mission_node.gps_data[next_timestamp]["altitude"]
                pt_latitude = self.mission_node.gps_data[next_timestamp]["latitude"]
                pt_longitude = self.mission_node.gps_data[next_timestamp]["longitude"]

                mine_x_min = mine_x_center - (x_width / 2)
                mine_x_max = mine_x_center + (x_width / 2)
                mine_y_min = mine_y_center - (y_width / 2)
                mine_y_max = mine_y_center + (y_width / 2)

                # next_timestamp = self.TMP_timestamp_queue.get()
                # absolute_height = self.TMP_gps_data[next_timestamp]["altitude"]
                # pt_latitude = self.TMP_gps_data[next_timestamp]["latitude"]
                # pt_longitude = self.TMP_gps_data[next_timestamp]["longitude"]

            # Get the location of the mines within the image (bounding box or smth I dunno)
            # print(f"The bounding boxes gotten is: {mine_x_min}, {mine_x_max}, {mine_y_min}, {mine_y_max} for timestamp {next_timestamp}")
            if mine_x_min == -1:
                # print("No boxes in img")
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
                # print("Acquired the lock in find_mines but lower")
                self.coords.append((new_lat, new_long))
                self.send_buffer.append((new_lat, new_long))
                print(f"The new point is {new_lat}, {new_long} and the number of detected_mines is {len(self.coords)}")

            if len(self.send_buffer) >= 10:
                # print("Should try to send soon")
                self.send_coords()

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

                if time.time() - self.start_time >= self.worker_shutoff_time:
                    with self.coords_lock:
                        self.shutdown_flag = True
                    continue
                
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

    def fake_gps_coords_generation(self):
        fake_timestamp = 0
        print("Entered fake gps coords generation")
        while fake_timestamp < 100:
            print("Waiting fo the coords_lock")
            with self.coords_lock:
                print("Acquired the lock in fake_gps_coords_generation")
                fake_lat = random.randint(1, 100)
                fake_lon = random.randint(1, 100)
                fake_alt = random.randint(1, 100)
                fake_gps_point = {}
                self.TMP_gps_data[fake_timestamp] = {
                    "latitude": fake_lat,
                    "longitude": fake_lon,
                    "altitude": fake_alt
                }
                self.TMP_timestamp_queue.put(fake_timestamp)
            fake_timestamp += 1

            print(f"{fake_lat}, {fake_lon}, {fake_alt}")
            with self.coords_cv:
                self.coords_cv.notify()
            time.sleep(0.1)

    def run_mission_node(self, mission_mode):

        rclpy.init()

        node = MissionNode(self.coords_lock, self.coords_cv, mission_mode)
        self.mission_node = node

        # while not self.shutdown_flag:
        # Start the mission
        self.mission_node.start_mission()

        # Continue processing GPS messages
        print("Attempting to spin the node")
        while not self.shutdown_flag:
            rclpy.spin_once(self.mission_node, timeout_sec=0.1)

        print("Stopping ROS")
        self.mission_node.destroy_node()
        rclpy.shutdown()
        print(f"Collected {len(node.gps_data)} GPS points")

    def handle_run_drones(self):

        mission_node_thread = threading.Thread(target=self.run_mission_node, args=("survey",))
        mission_node_thread.start()

        # self.fake_gps_coords_generation()
    
    def handle_orbit(self):
        
        mission_node_thread = threading.Thread(target=self.run_mission_node, args=("orbit",))
        mission_node_thread.start()
        
        # self.fake_gps_coords_generation()

    def handle_termination(self):
        pass
    
    
    def register(self):
        if not self.registered:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    while True:
                        try:
                            sock.connect((self.manager_host, self.manager_port))
                            break
                        except ConnectionRefusedError:
                            print("Manager not started yet")
                        time.sleep(0.1)
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

    def send_finished(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            # Connect to the UDP socket on server
            # self.try_connect_to_manager()
            sock.connect((self.manager_host, self.manager_port))

            # Send a message
            message = json.dumps({"message_type": "finished",
                                    "drone_host": self.host,
                                    "drone_port": self.port})
            sock.sendall((message + "\n").encode("utf-8"))
        
    def run_drone(self):
        tcp_thread = threading.Thread(target=self.tcp_server)
        udp_thread = threading.Thread(target=self.send_heartbeat)
        tcp_thread.start()
        udp_thread.start()

        find_mines_thread = threading.Thread(target=self.find_mines)
        find_mines_thread.start()

        if self.camera_mode == "primary":
            take_pictures_thread = threading.Thread(target=self.primary_camera_func)
            take_pictures_thread.start()

        if self.camera_mode == "backup":
            take_pictures_thread = threading.Thread(target=self.backup_camera_func)
            take_pictures_thread.start()

        fake_gps_thread = threading.Thread(target=self.handle_run_drones)
        fake_gps_thread.start()

        # fake_gps_thread = threading.Thread(target=self.fake_gps_coords_generation)
        # fake_gps_thread.start()

        tcp_thread.join()
        udp_thread.join()

        self.send_coords()

        self.send_finished()

        print("Finished!")

        with self.coords_cv:
            self.coords_cv.notify_all()
