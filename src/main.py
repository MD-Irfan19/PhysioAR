"""PhysioAR main application — Phase 1.

Real-time webcam pose estimation pipeline:

    Webcam → OpenCV Capture → BGR→RGB → MediaPipe Pose Landmarker
        → Pose Landmarks → Skeleton Visualization → OpenCV Window

Run from the project root:

    python -m src.main
"""

import cv2

from src.acquisition import Camera
from src.pose_estimation import PoseEstimator


def main() -> None:
    """Run the PhysioAR real-time pose estimation pipeline.

    Opens the webcam, processes each frame through MediaPipe Pose
    Landmarker, draws detected landmarks and connections, and displays
    the annotated feed. Press 'q' to exit.
    """
    camera = Camera(camera_index=0)
    pose_estimator = PoseEstimator()

    try:
        camera.open()
        print("PhysioAR — Phase 1: Webcam + MediaPipe Pose")
        print("Press 'q' to quit.")

        while True:
            success, frame = camera.read()

            if not success:
                print("Frame capture failed. Exiting.")
                break

            # Process the frame through MediaPipe Pose Landmarker.
            result = pose_estimator.process(frame)

            # Draw pose landmarks and connections if a pose was detected.
            pose_estimator.draw(frame, result)

            # Display the annotated frame.
            cv2.imshow("PhysioAR", frame)

            # Exit on 'q' key press.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        pose_estimator.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
