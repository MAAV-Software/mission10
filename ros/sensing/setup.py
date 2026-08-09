from setuptools import find_packages, setup


package_name = "sensing"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    package_data={"sensing": ["config/*.yaml"]},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="calgary-kirisame",
    maintainer_email="131201352+calgary-kirisame@users.noreply.github.com",
    description="Mission-owned camera capture, optical flow, detection, and shared-frame fanout.",
    license="MPL-2.0",
    entry_points={
        "console_scripts": [
            "mission_sensing = sensing.node:main",
        ],
    },
)
