from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="clip_robot_perception",
                executable="perception_node",
                name="clip_perception_node",
                parameters=[
                    {"source": "0"},
                    {"query": "red cube"},
                    {"rate_hz": 8.0},
                ],
                output="screen",
            ),
            Node(
                package="clip_robot_perception",
                executable="arm_controller_node",
                name="clip_arm_controller_node",
                parameters=[{"steps": 80}],
                output="screen",
            ),
        ]
    )
