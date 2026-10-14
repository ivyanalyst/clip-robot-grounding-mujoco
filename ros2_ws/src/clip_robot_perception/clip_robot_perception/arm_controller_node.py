import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool, Float32
import numpy as np
from simulation.robot_env import RobotArmEnv
from simulation.controller import SimpleController


class ArmControllerNode(Node):
    def __init__(self):
        super().__init__("clip_arm_controller_node")
        self.declare_parameter("steps", 80)

        self.env = RobotArmEnv(use_viewer=False)
        self.controller = SimpleController(self.env)
        self.steps = self.get_parameter("steps").get_parameter_value().integer_value

        self.sub = self.create_subscription(
            PointStamped, "/clip_robot/target_point", self.target_callback, 10
        )
        self.pub_contact = self.create_publisher(Bool, "/clip_robot/contact", 10)
        self.pub_distance = self.create_publisher(Float32, "/clip_robot/distance", 10)

        self.get_logger().info("MuJoCo arm controller node started")

    def target_callback(self, msg: PointStamped):
        target_3d = np.array([msg.point.x, msg.point.y, msg.point.z])
        info = self.controller.reach(target_3d, steps=self.steps)

        contact_msg = Bool()
        contact_msg.data = bool(info["contact"])
        self.pub_contact.publish(contact_msg)

        dist_msg = Float32()
        dist_msg.data = float(info["final_distance"] or 0.0)
        self.pub_distance.publish(dist_msg)

        self.get_logger().info(
            f"Reached 3D target ({target_3d[0]:.3f},{target_3d[1]:.3f},{target_3d[2]:.3f}) "
            f"dist={info['final_distance']:.3f} contact={info['contact']}"
        )

    def destroy_node(self):
        self.env.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ArmControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
