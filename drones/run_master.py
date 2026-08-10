"""Example TCP socket server."""
import socket
import subprocess
from drones.roles.manager import ManagerDrone

def main():
    """Test TCP Socket Server and git access from RPi"""
    # Create an INET, STREAMing socket, this is TCP
    # Note: context manager syntax allows for sockets to automatically be
    # closed when an exception is raised or control flow returns.
    hostname = socket.gethostname()
    print(hostname)
    ManagerDrone("localhost", 8000, "primary")

if __name__ == "__main__":
    main()
