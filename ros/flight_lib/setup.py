from setuptools import find_packages, setup

package_name = "flight_lib"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="calgary-kirisame",
    maintainer_email="131201352+calgary-kirisame@users.noreply.github.com",
    description="Pure-math flight geometry: phased orbits + hover/land profiles. No ROS, no PX4.",
    license="MPL-2.0",
    entry_points={},
)
