from typing import Optional, Dict, Any, List
import numpy as np
import cv2
from .stream_buffer import RTSPStreamBuffer
from .detector import ObjectDetector
from .clip_grounding import CLIPGrounder
from .depth_estimator import DepthEstimator


class PerceptionPipeline:
    def __init__(
        self,
        source: str = "0",
        query: str = "red cube",
        use_yolo: bool = True,
        known_object_size_m: float = 0.08,
    ):
        self.buffer = RTSPStreamBuffer(source=source)
        self.detector = ObjectDetector() if use_yolo else ObjectDetector(model_name=None)
        self.grounder = CLIPGrounder()
        self.depth_estimator = DepthEstimator()
        self.default_query = query
        self.known_object_size_m = known_object_size_m
        self.last_result: Optional[Dict[str, Any]] = None

    def start(self) -> None:
        self.buffer.start()

    def stop(self) -> None:
        self.buffer.stop()

    def process(self, query: Optional[str] = None) -> Optional[Dict[str, Any]]:
        item = self.buffer.get_latest()
        if item is None:
            return None
        timestamp, frame_id, frame = item
        h, w = frame.shape[:2]
        self.depth_estimator.cx = w / 2.0
        self.depth_estimator.cy = h / 2.0
        query = query or self.default_query
        detections = self.detector.detect(frame)
        grounded = self.grounder.ground(frame, detections, query, top_k=1)
        result = {
            "timestamp": timestamp,
            "frame_id": frame_id,
            "frame": frame,
            "detections": detections,
            "grounded": grounded,
            "query": query,
            "image_size": (w, h),
        }
        if grounded:
            g = grounded[0]
            result["target_pixel"] = g["center"]
            result["target_bbox"] = g["bbox"]
            result["ground_score"] = g["ground_score"]
            result["label"] = g["label"]
            point_3d = self.depth_estimator.bbox_to_3d(
                g["bbox"],
                known_size_m=self.known_object_size_m,
                image_size=(w, h),
            )
            result["target_3d"] = point_3d
            result["depth"] = float(point_3d[2])
        self.last_result = result
        return result

    def draw(self, result: Optional[Dict[str, Any]] = None) -> Optional[np.ndarray]:
        result = result or self.last_result
        if result is None:
            return None
        vis = result["frame"].copy()
        for det in result["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(
                vis,
                f"{det['label']} {det['score']:.2f}",
                (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 0),
                1,
            )
        if result.get("grounded"):
            g = result["grounded"][0]
            x1, y1, x2, y2 = g["bbox"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cx, cy = g["center"]
            cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)
            depth = result.get("depth", 0.0)
            p3d = result.get("target_3d")
            label = f"{g['query']} s={g['ground_score']:.3f} d={depth:.2f}m"
            if p3d is not None:
                label += f" 3D=({p3d[0]:.2f},{p3d[1]:.2f},{p3d[2]:.2f})"
            cv2.putText(
                vis,
                label,
                (x1, y2 + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
        return vis

    def health(self) -> dict:
        return self.buffer.is_healthy()
