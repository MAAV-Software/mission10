from glob import glob

from setuptools import find_packages, setup

package_name = "jarvis_web"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # data_files does not glob on its own. --symlink-install points the
        # installed path back at the source tree, so edits to the page are live.
        ("share/" + package_name + "/static", glob("jarvis_web/static/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="calgary-kirisame",
    maintainer_email="131201352+calgary-kirisame@users.noreply.github.com",
    description="Operator webapp: push-to-talk voice in, mission gates out, result map served.",
    license="MPL-2.0",
    entry_points={
        "console_scripts": [
            "jarvis_web = jarvis_web.app:main",
        ],
    },
)
