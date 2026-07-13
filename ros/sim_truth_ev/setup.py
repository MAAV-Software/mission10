from glob import glob
from setuptools import find_packages, setup

package_name = "sim_truth_ev"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="calgary-kirisame",
    maintainer_email="131201352+calgary-kirisame@users.noreply.github.com",
    description="Convert gz/ROS ground-truth odometry into PX4 vehicle_visual_odometry.",
    license="MPL-2.0",
    entry_points={
        "console_scripts": [
            "gt_to_ev = sim_truth_ev.gt_to_ev:main",
            "world_truth_to_odom = sim_truth_ev.world_truth_to_odom:main",
        ],
    },
)
