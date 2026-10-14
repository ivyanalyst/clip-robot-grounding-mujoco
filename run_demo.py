import argparse
import time
import cv2
import numpy as np
from perception.pipeline import PerceptionPipeline
from simulation.robot_env import RobotArmEnv
from simulation.controller import SimpleController


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    parser.add_argument("--query", default="red cube")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    pipeline = PerceptionPipeline(source=args.source, query=args.query)
    pipeline.start()
    env = RobotArmEnv(use_viewer=args.viewer)
    controller = SimpleController(env)
    if args.viewer:
        env.open_viewer()

    print("3D CLIP Grounding + MuJoCo started. Press q to quit.")
    frame_count = 0
    try:
        while True:
            result = pipeline.process(query=args.query)
            if result is None:
                time.sleep(0.02)
                continue
            vis = pipeline.draw(result)
            if vis is not None:
                cv2.imshow("CLIP 3D Grounding", vis)
            if result.get("target_3d") is not None:
                target_3d = result["target_3d"]
                info = controller.reach(target_3d, steps=50)
                if args.viewer:
                    env.sync_viewer()
                print(
                    f"frame={result['frame_id']} "
                    f"score={result['ground_score']:.3f} "
                    f"3D=({target_3d[0]:.3f},{target_3d[1]:.3f},{target_3d[2]:.3f}) "
                    f"dist={info['final_distance']:.3f} "
                    f"contact={info['contact']}"
                )
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            frame_count += 1
            if args.max_frames > 0 and frame_count >= args.max_frames:
                break
            time.sleep(0.03)
    finally:
        pipeline.stop()
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
