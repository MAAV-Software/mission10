from setuptools import find_packages, setup

package_name = "px4_offboard"

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
    description="Reusable PX4 offboard plumbing: handshake, namespacing, setpoint streaming.",
    license="MPL-2.0",
    entry_points={
        "console_scripts": [
            "offboard_controller = px4_offboard.controller:main",
        ],
    },
)
