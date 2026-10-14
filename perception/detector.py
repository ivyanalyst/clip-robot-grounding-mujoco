from typing import List, Tuple, Optional
import numpy as np
import cv2

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


class ObjectDetector:
    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.4):
        self.conf = conf
        self.model = None
        if HAS_YOLO:
            self.model = YOLO(model_name)

    def detect(self, frame: np.ndarray) -> List[dict]:
        if self.model is None:
            return self._color_fallback(frame)
        results = self.model(frame, conf=self.conf, verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            score = float(box.conf[0])
            label = results.names[cls_id]
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "label": label,
                "score": score,
                "center": ((x1 + x2) // 2, (y1 + y2) // 2),
            })
        return detections

    def _color_fallback(self, frame: np.ndarray) -> List[dict]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ranges = {
            "red": [(0, 100, 80), (10, 255, 255)],
            "blue": [(100, 100, 80), (130, 255, 255)],
            "green": [(40, 100, 80), (80, 255, 255)],
        }
        detections = []
        for name, (lo, hi) in ranges.items():
            mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 800:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append({
                    "bbox": (x, y, x + w, y + h),
                    "label": name,
                    "score": min(1.0, area / 20000.0),
                    "center": (x + w // 2, y + h // 2),
                })
        return detections
