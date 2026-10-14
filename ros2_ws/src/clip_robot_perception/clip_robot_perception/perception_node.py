import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, PoseStamped
from std_msgs.msg import String, Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
from perception.pipeline import PerceptionPipeline


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("clip_perception_node")
        self.declare_parameter("source", "0")
        self.declare_parameter("query", "red cube")
        self.declare_parameter("rate_hz", 10.0)

        source = self.get_parameter("source").get_parameter_value().string_value
        query = self.get_parameter("query").get_parameter_value().string_value
        rate = self.get_parameter("rate_hz").get_parameter_value().double_value

        self.pipeline = PerceptionPipeline(source=source, query=query)
        self.pipeline.start()
        self.bridge = CvBridge()
        self.query = query

        self.pub_point = self.create_publisher(PointStamped, "/clip_robot/target_point", 10)
        self.pub_pose = self.create_publisher(PoseStamped, "/clip_robot/target_pose", 10)
        self.pub_score = self.create_publisher(Float32, "/clip_robot/ground_score", 10)
        self.pub_label = self.create_publisher(String, "/clip_robot/label", 10)
        self.pub_image = self.create_publisher(Image, "/clip_robot/debug_image", 10)

        self.timer = self.create_timer(1.0 / rate, self.timer_callback)
        self.get_logger().info(f"CLIP 3D perception node started. query={query}")

    def timer_callback(self):
        result = self.pipeline.process(query=self.query)
        if result is None or result.get("target_3d") is None:
            return

        p3d = result["target_3d"]
        now = self.get_clock().now().to_msg()

        pt = PointStamped()
        pt.header.stamp = now
        pt.header.frame_id = "camera_link"
        pt.point.x = float(p3d[0])
        pt.point.y = float(p3d[1])
        pt.point.z = float(p3d[2])
        self.pub_point.publish(pt)

        pose = PoseStamped()
        pose.header = pt.header
        pose.pose.position.x = float(p3d[0])
        pose.pose.position.y = float(p3d[1])
        pose.pose.position.z = float(p3d[2])
        pose.pose.orientation.w = 1.0
        self.pub_pose.publish(pose)

        score_msg = Float32()
        score_msg.data = float(result["ground_score"])
        self.pub_score.publish(score_msg)

        label_msg = String()
        label_msg.data = str(result.get("label", ""))
        self.pub_label.publish(label_msg)

        vis = self.pipeline.draw(result)
        if vis is not None:
            img_msg = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
            img_msg.header = pt.header
            self.pub_image.publish(img_msg)

    def destroy_node(self):
        self.pipeline.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
