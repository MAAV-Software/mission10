"""Example TCP socket client."""
import socket
import threading
import json
import time
from pathlib import Path
import subprocess
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from drones.roles.MissionNode import MissionNode

class ExploreDrone:
    "Construct an instance of an explorer Drone"

    def __init__(self, host, port, manager_host, manager_port):
        """Construct a Manager instance and start listening for messages."""

        self.host = host
        self.port = port
        self.coords = []
        self.startup = True
        self.shutdown_flag = False
        self.manager_host = manager_host
        self.manager_port = manager_port
        self.registered = False
        self.coords_path = "DervinSmellsLikePoop.csv"

        # print(self.manager_host)
        # print("Finished registering!")
        self.run_drone()

    def try_connect_to_manager(self):
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.connect((self.manager_host, self.manager_port))
                    return True
            except ConnectionRefusedError:
                print("Manager not started yet")
            time.sleep(0.1)

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

                if not self.registered:
                    self.register()

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
        if message_dict["message_type"] == "coords_ack":
            self.handle_coords_ack(message_dict)
        elif message_dict["message_type"] == "registration_ack":
            self.registered = True
        elif message_dict["message_type"] == "run_drones":
            self.handle_run_drones()
        else:
            print("Message Unknown")
        
    # adds one pair of coords
    def handle_coords_ack(self, message_dict):
        self.send_coords()

    def handle_run_drones(self):
        # subprocess.run("ros2", ...) # Run that ros2 command to publish to the start topic
        # time.sleep(5)

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
        drones_directory = Path(__file__).parent.parent
        skib_path = drones_directory / "skib.py"
        bidi_path = drones_directory / "bidi.py"
        subprocess.run(["python3", f"{str(skib_path)}"])
        subprocess.run(["python3", f"{str(bidi_path)}"])

        coords_list = []
        coords_path = drones_directory / self.coords_path
        with open(coords_path, "r") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                line_contents = line.strip().split(',')
                coords_list.append(( float(line_contents[0]), float(line_contents[1]) ))
        self.coords = coords_list

        if self.try_connect_to_manager():
            self.send_coords()

    def send_coords(self):
        if len(self.coords) == 0:
            self.send_finished()
            self.shutdown_flag = True
            return
        
        coord = self.coords.pop(0)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.manager_host, self.manager_port))
            message = json.dumps({
                "message_type": "coordinates",
                "host": self.host,
                "port": self.port,
                "coords": coord
            })
            sock.sendall(message.encode('utf-8'))
    
    
    def register(self):
        if not self.registered:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.connect((self.manager_host, self.manager_port))
                    message = json.dumps({
                        "message_type": "registration",
                        "drone_host": self.host,
                        "drone_port": self.port
                    })
                    sock.sendall(message.encode('utf-8'))

            except ConnectionRefusedError:
                print("Manager not started yet")

    def send_finished(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.manager_host, self.manager_port))
            message = json.dumps({
                "message_type": "finished",
                "drone_host": self.host,
                "drone_port": self.port
            })
            sock.sendall(message.encode('utf-8'))
    
    def send_heartbeat(self):
        while not self.shutdown_flag:

            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                # Connect to the UDP socket on server
                sock.connect((self.manager_host, self.manager_port))

                # Send a message
                message = json.dumps({"message_type": "heartbeat",
                                      "drone_host": self.host,
                                      "drone_port": self.port})
                sock.sendall(message.encode('utf-8'))
            time.sleep(1)
        
    def run_drone(self):
        udp_thread = threading.Thread(target=self.send_heartbeat)
        tcp_thread = threading.Thread(target=self.tcp_server)
        udp_thread.start()
        tcp_thread.start()

        tcp_thread.join()
        udp_thread.join()
