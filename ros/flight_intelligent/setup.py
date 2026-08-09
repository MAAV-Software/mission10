from glob import glob
from setuptools import find_packages, setup

package_name = "flight_intelligent"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="calgary-kirisame",
    maintainer_email="131201352+calgary-kirisame@users.noreply.github.com",
    description="Proof-of-intelligent-flight missions built on px4_offboard and flight_lib.",
    license="MPL-2.0",
    entry_points={
        "console_scripts": [
            "phased_orbits_mission = flight_intelligent.phased_orbits_mission:main",
            "survey_mission = flight_intelligent.survey_mission:main",
            "relative_localization = flight_intelligent.relative_localization_shadow:main",
            "prop_idle_dummy_mission = flight_intelligent.prop_idle_dummy_mission:main",
        ],
    },
)
