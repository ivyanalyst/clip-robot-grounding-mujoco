import time
from typing import Optional, List
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import cv2
import base64
from perception.pipeline import PerceptionPipeline
from simulation.robot_env import RobotArmEnv
from simulation.controller import SimpleController

app = FastAPI(title="CLIP 3D Robot Grounding + MuJoCo")

pipeline: Optional[PerceptionPipeline] = None
env: Optional[RobotArmEnv] = None
controller: Optional[SimpleController] = None


class GroundRequest(BaseModel):
    query: str = "red cube"


class ReachRequest(BaseModel):
    query: str = "red cube"
    steps: int = 180


@app.on_event("startup")
def startup():
    global pipeline, env, controller
    pipeline = PerceptionPipeline(source="0", query="red cube")
    pipeline.start()
    env = RobotArmEnv(use_viewer=False)
    controller = SimpleController(env)
    time.sleep(1.5)


@app.on_event("shutdown")
def shutdown():
    if pipeline:
        pipeline.stop()
    if env:
        env.close()


@app.get("/health")
def health():
    return {
        "perception": pipeline.health() if pipeline else {},
        "sim_ready": env is not None,
        "features": ["3d_depth", "mujoco", "clip_grounding"],
    }


@app.post("/ground")
def ground(req: GroundRequest):
    result = pipeline.process(query=req.query)
    if result is None or not result.get("grounded"):
        return {"success": False, "message": "no target grounded"}
    p3d = result["target_3d"]
    vis = pipeline.draw(result)
    _, buf = cv2.imencode(".jpg", vis)
    jpeg_b64 = base64.b64encode(buf).decode("utf-8")
    return {
        "success": True,
        "query": req.query,
        "pixel": {"x": result["target_pixel"][0], "y": result["target_pixel"][1]},
        "target_3d": {"x": float(p3d[0]), "y": float(p3d[1]), "z": float(p3d[2])},
        "depth_m": float(result["depth"]),
        "score": result["ground_score"],
        "label": result.get("label"),
        "jpeg_base64": jpeg_b64,
    }


@app.post("/reach")
def reach(req: ReachRequest):
    result = pipeline.process(query=req.query)
    if result is None or not result.get("target_3d") is not None:
        return {"success": False, "message": "no 3D target"}
    target_3d = result["target_3d"]
    info = controller.reach(target_3d, steps=req.steps)
    return {
        "success": True,
        "query": req.query,
        "target_3d": {"x": float(target_3d[0]), "y": float(target_3d[1]), "z": float(target_3d[2])},
        "final_distance": info["final_distance"],
        "contact": info["contact"],
        "steps_taken": info["steps"],
    }


@app.get("/")
def root():
    return {
        "service": "CLIP 3D Visual Grounding + MuJoCo Robot Arm",
        "endpoints": ["/health", "/ground", "/reach"],
        "features": ["3D depth estimation", "MuJoCo physics", "CLIP grounding"],
    }
