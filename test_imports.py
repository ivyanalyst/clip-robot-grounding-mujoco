import sys
sys.path.insert(0, ".")

from perception.stream_buffer import RTSPStreamBuffer
from perception.detector import ObjectDetector
from perception.clip_grounding import CLIPGrounder
from perception.pipeline import PerceptionPipeline
from simulation.robot_env import RobotArmEnv
from simulation.controller import SimpleController

print("All modules imported successfully")
env = RobotArmEnv()
ctrl = SimpleController(env)
print("MuJoCo environment and controller created")
print("EE position:", env.get_ee_pos())
info = ctrl.reach((0.35, 0.0), steps=50)
print("Reach test distance:", info["final_distance"], "contact:", info["contact"])
print("Ready")
