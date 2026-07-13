from setuptools import find_packages, setup

setup(
    name="uwb",
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/uwb"]),
        ("share/uwb", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="calgary-kirisame",
    maintainer_email="131201352+calgary-kirisame@users.noreply.github.com",
    description="Per-drone DW1000 DS-TWR UWB range sensor (real-hardware drop-in for sim_uwb).",
    license="MPL-2.0",
    entry_points={"console_scripts": ["uwb_range_node = uwb.uwb_range_node:main"]},
)
