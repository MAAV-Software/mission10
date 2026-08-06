"""Example TCP socket client."""
import socket
from roles.explorer import ExploreDrone

def main():

    hostname = socket.gethostname()
    print(hostname)

    ExploreDrone("35.3.34.40", 8001, "35.3.34.40", 8000, "backup") # For now, just simulating on one laptop, so same hostname for both manager and explorer
    # ExploreDrone("192.168.1.22", 8001, "192.168.1.22", 8000, coords_list)

if __name__ == "__main__":
    main()