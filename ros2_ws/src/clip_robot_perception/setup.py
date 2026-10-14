from setuptools import setup
import os
from glob import glob

package_name = "clip_robot_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Ivy To",
    maintainer_email="ivythecoder@tutamail.com",
    description="CLIP 3D visual grounding for robot arm",
    license="MIT",
    entry_points={
        "console_scripts": [
            "perception_node = clip_robot_perception.perception_node:main",
            "arm_controller_node = clip_robot_perception.arm_controller_node:main",
        ],
    },
)
