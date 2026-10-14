import cv2
import time
import logging
import threading
from collections import deque
from typing import List, Tuple, Optional, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RTSPStreamBuffer")


class RTSPStreamBuffer:
    def __init__(self, source: str = "0", max_buffer_size: int = 120, reconnect_delay: int = 3):
        self.source = source
        self.max_buffer_size = max_buffer_size
        self.reconnect_delay = reconnect_delay
        self._buffer = deque(maxlen=self.max_buffer_size)
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._is_connected = False
        self._frame_id = 0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(f"Started stream ingestion from: {self.source}")

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _capture_loop(self) -> None:
        while self._running:
            src = int(self.source) if str(self.source).isdigit() else self.source
            cap = cv2.VideoCapture(src)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                self._is_connected = False
                time.sleep(self.reconnect_delay)
                continue
            self._is_connected = True
            while self._running and cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    self._is_connected = False
                    break
                timestamp = time.time()
                with self._lock:
                    self._buffer.append((timestamp, self._frame_id, frame.copy()))
                    self._frame_id += 1
            cap.release()
            self._is_connected = False
            if self._running:
                time.sleep(self.reconnect_delay)

    def get_latest(self) -> Optional[Tuple[float, int, Any]]:
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer[-1]

    def get_recent(self, count: int = 30) -> List[Tuple[float, int, Any]]:
        with self._lock:
            return list(self._buffer)[-count:]

    def is_healthy(self) -> dict:
        with self._lock:
            length = len(self._buffer)
        return {
            "connected": self._is_connected,
            "buffer_length": length,
            "source": self.source,
        }
