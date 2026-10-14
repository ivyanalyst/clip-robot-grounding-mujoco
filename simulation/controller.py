from typing import Tuple, Optional
import numpy as np
from .robot_env import RobotArmEnv


class SimpleController:
    def __init__(self, env: RobotArmEnv):
        self.env = env
        self.kp = np.array([2.5, 2.0, 1.8])

    def ik_approx(self, target: np.ndarray) -> np.ndarray:
        x, y, z = target
        r = np.sqrt(x * x + y * y)
        q1 = np.arctan2(y, x)
        l1, l2 = 0.25, 0.22
        height = max(0.02, z - 0.05)
        cos_q2 = np.clip(
            (r * r + height * height - l1 * l1 - l2 * l2) / (2 * l1 * l2), -1.0, 1.0
        )
        q2 = np.arccos(cos_q2) - 0.35
        q3 = -0.55 - 0.35 * q2
        return np.array([q1, q2, q3])

    def compute(self, target_3d: np.ndarray) -> np.ndarray:
        desired_q = self.ik_approx(target_3d)
        desired_q = np.clip(desired_q, -2.8, 2.8)
        return desired_q

    def reach(
        self,
        target_3d: np.ndarray,
        steps: int = 200,
    ) -> dict:
        x, y, z = target_3d
        self.env.set_target_xy(float(x), float(y), float(max(0.03, z - 0.04)))
        ctrl = self.compute(target_3d)
        history = []
        for i in range(steps):
            info = self.env.step(ctrl)
            history.append(info)
            if info["contact"]:
                break
        return {
            "final_distance": history[-1]["distance"] if history else None,
            "contact": history[-1]["contact"] if history else False,
            "steps": len(history),
            "history": history,
            "target_3d": target_3d.tolist(),
        }
