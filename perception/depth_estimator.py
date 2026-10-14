from typing import Optional, Tuple
import numpy as np
import cv2


class DepthEstimator:
    def __init__(
        self,
        fx: float = 600.0,
        fy: float = 600.0,
        cx: float = 320.0,
        cy: float = 240.0,
        default_depth: float = 0.55,
    ):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.default_depth = default_depth

    def estimate_depth_from_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        known_size_m: float = 0.08,
        image_height: int = 480,
    ) -> float:
        x1, y1, x2, y2 = bbox
        pixel_height = max(1, y2 - y1)
        depth = (known_size_m * self.fy) / pixel_height
        depth = float(np.clip(depth, 0.25, 1.8))
        return depth

    def pixel_to_3d(
        self,
        u: int,
        v: int,
        depth: float,
    ) -> np.ndarray:
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth
        return np.array([x, y, z], dtype=np.float64)

    def bbox_to_3d(
        self,
        bbox: Tuple[int, int, int, int],
        known_size_m: float = 0.08,
        image_size: Tuple[int, int] = (640, 480),
    ) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        u = (x1 + x2) // 2
        v = (y1 + y2) // 2
        depth = self.estimate_depth_from_bbox(bbox, known_size_m, image_size[1])
        return self.pixel_to_3d(u, v, depth)
