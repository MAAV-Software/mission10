#!/bin/bash

cd /home/maav/mission10

source env/bin/activate

source /opt/ros/jazzy/setup.bash

exec python3 -m drones.run_master/worker