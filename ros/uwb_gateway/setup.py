from setuptools import find_packages, setup

package_name = "uwb_gateway"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="MAAV",
    maintainer_email="maav@umich.edu",
    description="Mission 10 direct-UWB ROS gateway.",
    license="MIT",
    entry_points={"console_scripts": ["uwb_gateway = uwb_gateway.gateway:main"]},
)
