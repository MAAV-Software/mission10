"""Example TCP socket server."""
import socket
import json
import threading
from pathlib import Path
import time
import subprocess
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from drones.roles.MissionNode import MissionNode

bounds_path = Path(__file__).parent.parent.parent / "constants/bounding_boxes.txt"
aggregate_path = Path(__file__).parent.parent / "all_results.csv"

class ManagerDrone:
    "Construct an instance of the main drone"

    def __init__(self, host, port):
        """Construct a Manager instance and start listening for messages."""

        self.host = host
        self.port = port
        self.coords = []
        self.detected_mine_data = {}
        self.exp_drones = {}
        self.drone_last_seen = {}
        self.finished_drones = 0
        self.self_finished = False
        self.bounds = []
        self.shutdown_flag = False
        self.mission_status = "pre-mission"
        self.next_action = "scout"

        with open(bounds_path, "r") as f:
            for line in f:
                line_contents = line.strip().split()
                self.bounds.append(( float(line_contents[0]), float(line_contents[1]) ))

        self.run_drone()


    def tcp_server(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:

            # Bind the socket to the server
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.host, self.port))
            sock.listen()

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
    
    def udp_server(self):
        """Test UDP Socket Server."""
        # Create an INET, DGRAM socket, this is UDP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:

            # Bind the UDP socket to the server
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.host, self.port))
            sock.settimeout(1)

            # Receive incoming UDP messages
            while not self.shutdown_flag:
                try:
                    message_bytes = sock.recv(4096)
                except socket.timeout:
                    continue
                message_str = message_bytes.decode("utf-8")
                message_dict = json.loads(message_str)
                # print(message_dict)
                self.handle_message(message_dict)

                time_now = time.monotonic()
                # No sock.listen() since UDP doesn't establish connections like TCP
                for drone_id, last_seen in self.drone_last_seen.items():
                    if time_now - last_seen >= 5:
                        self.exp_drones[drone_id] = "idle"
    
    def handle_message(self, message_dict):
        if message_dict["message_type"] == "coordinates":
            self.handle_coordinates(message_dict)
        elif message_dict["message_type"] == "registration":
            self.handle_registration(message_dict)
        elif message_dict["message_type"] == "finished":
            self.handle_finished(message_dict)
        elif message_dict["message_type"] == "heartbeat":
            self.handle_heartbeat(message_dict)
        elif message_dict["message_type"] == "run_drones":
            self.handle_run_drones()
        else:
            print("Message Unknown")
        
    def handle_heartbeat(self, message_dict):
        worker_host = message_dict["drone_host"]
        worker_port = message_dict["drone_port"]

        drone_key = str(worker_host) + "_" + str(worker_port)
        self.exp_drones[drone_key] = {
            "drone_host": worker_host,
            "drone_port": worker_port,
            "status": "working"
        }
        self.drone_last_seen[drone_key] = time.monotonic()
    
    def handle_run_drones(self):
        self.mission_status = "in_mission"

        for drone_id, drone_status in self.exp_drones.items():
            if drone_status["status"] == "working":
                worker_host, worker_port = drone_id.strip().split("_")
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    print(f"The worker host is {worker_host} and the port is {worker_port}")
                    sock.connect((worker_host, int(worker_port))) 
                    message = json.dumps({
                        "message_type": "run_drones"
                    })
                    sock.sendall(message.encode('utf-8'))
	
	
        rclpy.init()

        node = MissionNode()

        try:
            # Start the mission
            node.start_mission()

            # Continue processing GPS messages
            rclpy.spin(node)

        except KeyboardInterrupt:
            pass

        finally:
            print(f"Collected {len(node.gps_data)} GPS points")
            node.destroy_node()
            rclpy.shutdown()

        

        # Now that the ros has finished running...
        # drones_directory = Path(__file__).parent.parent
        # skib_path = drones_directory / "skib.py"
        # bidi_path = drones_directory / "bidi.py"
        # subprocess.run(["python3", f"{str(skib_path)}"])
        # subprocess.run(["python3", f"{str(bidi_path)}"])
        

    # adds one pair of coords
    def handle_coordinates(self, message_dict):
        worker_host = message_dict["host"]
        worker_port = message_dict["port"]
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((worker_host, worker_port)) 
            self.coords.append(message_dict["coords"])
            message = json.dumps({
                "message_type": "coords_ack"
            })
            sock.sendall(message.encode('utf-8'))
    
    def handle_registration(self, message_dict):
        drone_host = message_dict["drone_host"]
        drone_port = message_dict["drone_port"]
        drone_key = str(drone_host) + "_" + str(drone_port)
        self.exp_drones[drone_key] = {
            "drone_host": drone_host,
            "drone_port": drone_port,
            "status": "idle"
        }
        self.drone_last_seen[drone_key] = time.monotonic()

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((drone_host, drone_port)) 
            message = json.dumps({
                "message_type": "registration_ack"
            })
            sock.sendall(message.encode('utf-8'))

    def handle_finished(self, message_dict):
        drone_host = message_dict["drone_host"]
        drone_port = message_dict["drone_port"]
        drone_key = str(drone_host) + "_" + str(drone_port)
        if self.exp_drones[drone_key]["status"] == "working":
            self.exp_drones[drone_key]["status"] = "finished"
            self.finished_drones += 1
            print(self.finished_drones, len(self.exp_drones))
            if self.finished_drones == len(self.exp_drones):
                self.shutdown_flag = True
        else:
            print("Error: Received a 'finished' message from a finished worker")
            exit(1)

    def run_drone(self):
        udp_thread = threading.Thread(target=self.udp_server)
        udp_thread.start()

        tcp_thread = threading.Thread(target=self.tcp_server)
        tcp_thread.start()

        tcp_thread.join()
        udp_thread.join()
        

        with open(aggregate_path, "w") as f: # Write the coords into it
            for coord in self.coords:
                f.write(f"{coord[0]},{coord[1]}\n")
    
            
