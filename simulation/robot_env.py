import os
import tempfile
from typing import Optional, Tuple
import numpy as np
import mujoco
import mujoco.viewer


SIMPLE_ARM_XML = """
<mujoco model="simple_arm">
  <compiler angle="radian" coordinate="local"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <default>
    <joint damping="0.8" armature="0.05"/>
    <geom friction="1 0.5 0.5" rgba="0.7 0.7 0.7 1"/>
  </default>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.9 0.9 0.9 1"/>
    <body name="base" pos="0 0 0.05">
      <geom type="cylinder" size="0.08 0.05" rgba="0.3 0.3 0.35 1"/>
      <body name="link1" pos="0 0 0.05">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.25" size="0.035" rgba="0.2 0.5 0.8 1"/>
        <body name="link2" pos="0 0 0.25">
          <joint name="joint2" type="hinge" axis="0 1 0" range="-2.5 2.5"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.22" size="0.03" rgba="0.2 0.6 0.7 1"/>
          <body name="link3" pos="0 0 0.22">
            <joint name="joint3" type="hinge" axis="0 1 0" range="-2.5 2.5"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.18" size="0.025" rgba="0.3 0.7 0.6 1"/>
            <body name="end_effector" pos="0 0 0.18">
              <geom name="ee" type="sphere" size="0.03" rgba="0.9 0.2 0.2 1"/>
              <site name="ee_site" pos="0 0 0" size="0.01"/>
            </body>
          </body>
        </body>
      </body>
    </body>
    <body name="target" pos="0.35 0.0 0.05">
      <geom name="target_geom" type="box" size="0.04 0.04 0.04" rgba="0.9 0.1 0.1 0.8" mass="0.05"/>
      <freejoint name="target_free"/>
    </body>
  </worldbody>
  <actuator>
    <position name="act1" joint="joint1" kp="80" ctrlrange="-3.14 3.14"/>
    <position name="act2" joint="joint2" kp="80" ctrlrange="-2.5 2.5"/>
    <position name="act3" joint="joint3" kp="60" ctrlrange="-2.5 2.5"/>
  </actuator>
</mujoco>
"""


class RobotArmEnv:
    def __init__(self, use_viewer: bool = False):
        self.xml = SIMPLE_ARM_XML
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.use_viewer = use_viewer
        self.viewer = None
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        mujoco.mj_forward(self.model, self.data)

    def reset(self, target_xy: Optional[Tuple[float, float]] = None) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        if target_xy is not None:
            self.data.qpos[3] = target_xy[0]
            self.data.qpos[4] = target_xy[1]
            self.data.qpos[5] = 0.05
        mujoco.mj_forward(self.model, self.data)
        return self.get_ee_pos()

    def get_ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.ee_site_id].copy()

    def get_target_pos(self) -> np.ndarray:
        return self.data.xpos[self.target_body_id].copy()

    def set_target_xy(self, x: float, y: float, z: float = 0.05) -> None:
        self.data.qpos[3] = x
        self.data.qpos[4] = y
        self.data.qpos[5] = z
        mujoco.mj_forward(self.model, self.data)

    def step(self, ctrl: np.ndarray, n_substeps: int = 10) -> dict:
        self.data.ctrl[:] = ctrl
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)
        ee = self.get_ee_pos()
        target = self.get_target_pos()
        dist = np.linalg.norm(ee - target)
        return {
            "ee_pos": ee,
            "target_pos": target,
            "distance": float(dist),
            "contact": dist < 0.08,
        }

    def open_viewer(self) -> None:
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def sync_viewer(self) -> None:
        if self.viewer is not None:
            self.viewer.sync()

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
