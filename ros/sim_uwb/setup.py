from setuptools import find_packages, setup

setup(
    name="sim_uwb",
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/sim_uwb"]),
        ("share/sim_uwb", ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="calgary-kirisame",
    maintainer_email="131201352+calgary-kirisame@users.noreply.github.com",
    description="gz-truth pairwise UWB range and peer-state simulator.",
    license="MPL-2.0",
    entry_points={"console_scripts": ["uwb_range_sim = sim_uwb.uwb_range_sim:main"]},
)
