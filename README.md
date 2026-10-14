CLIP 3D Visual Grounding + MuJoCo Robot Arm (+ ROS 2)

Real-time computer vision pipeline that extends Industrial-CCTV-Video-RAG into a closed-loop robotics system with 3D perception and physics simulation.

Camera / RTSP
  -> Object detection (YOLO / color)
  -> CLIP visual grounding
  -> Monocular depth estimation -> 3D target point
  -> MuJoCo 3-DOF arm reaches the target (contact-aware)
  -> Optional ROS 2 nodes publish / subscribe the 3D target

Features

- 3D target estimation from single camera (size-based monocular depth)
- Full MuJoCo physics (gravity, freejoint target, contact)
- CLIP zero-shot language grounding
- FastAPI service
- ROS 2 perception node + arm controller node
- Standalone OpenCV demo with optional MuJoCo viewer

Setup (pure Python)

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Run demo

python run_demo.py --source 0 --query "red cube"
python run_demo.py --source 0 --query "red cube" --viewer

Run API

uvicorn app.main:app --reload --port 8000

Endpoints

GET  /health
POST /ground  -> returns pixel + target_3d + depth
POST /reach   -> drives MuJoCo arm to the 3D target

ROS 2 (optional)

Requires ROS 2 Humble (or later) + cv_bridge.

cd ros2_ws
colcon build --packages-select clip_robot_perception
source install/setup.bash
ros2 launch clip_robot_perception clip_robot.launch.py

Topics

/clip_robot/target_point   geometry_msgs/PointStamped
/clip_robot/target_pose    geometry_msgs/PoseStamped
/clip_robot/ground_score   std_msgs/Float32
/clip_robot/label          std_msgs/String
/clip_robot/debug_image    sensor_msgs/Image
/clip_robot/contact        std_msgs/Bool
/clip_robot/distance       std_msgs/Float32

Project layout

clip-robot-grounding-mujoco/
  perception/
    stream_buffer.py
    detector.py
    clip_grounding.py
    depth_estimator.py      <-- 3D
    pipeline.py
  simulation/
    robot_env.py            <-- MuJoCo
    controller.py
  app/
    main.py
  ros2_ws/
    src/clip_robot_perception/
      ...                   <-- ROS 2
  run_demo.py
  requirements.txt
  README.md

Relation to Industrial-CCTV-Video-RAG

Same real-time stream + CLIP grounding core. This project adds monocular 3D, MuJoCo physics, and ROS 2 so the portfolio shows the path from industrial video understanding to embodied robot control.

Limitations

- Monocular depth uses known object size (approximate)
- Simplified 3-DOF arm + analytic IK
- ROS 2 nodes expect the parent perception/simulation packages on PYTHONPATH

Next upgrades

- Real depth camera or stereo
- Learned residual IK / MPC
- Isaac Sim / Isaac Lab parallel training
