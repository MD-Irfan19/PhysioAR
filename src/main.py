"""PhysioAR main application — Phase 1.5.

Real-time webcam pose estimation pipeline with EMA landmark smoothing:

    Webcam → OpenCV Capture → BGR→RGB → MediaPipe Pose Landmarker
        → Raw Landmarks → EMA Smoothing → Smoothed Landmarks
        → Skeleton Visualization → OpenCV Window

Run from the project root:

    python -m src.main
"""

import cv2

from src.acquisition import Camera
from src.pose_estimation import PoseEstimator
from src.utils.geometry import calculate_angle


# --- Diagnostic configuration (Phase 1.5 only) ---
# Set to True to display raw vs. smoothed right-elbow angle on screen.
# This is a temporary diagnostic for validating the EMA filter.
_SHOW_SMOOTHING_DIAGNOSTIC = False

# MediaPipe Pose landmark indices for the right elbow angle.
# Right shoulder = 12, Right elbow = 14, Right wrist = 16.
_DIAG_SHOULDER_IDX = 12
_DIAG_ELBOW_IDX = 14
_DIAG_WRIST_IDX = 16


def _compute_angle_safe(landmarks, idx_a: int, idx_b: int, idx_c: int) -> float | None:
    """Compute angle at idx_b from a landmark list, returning None on failure."""
    try:
        a = (landmarks[idx_a].x, landmarks[idx_a].y)
        b = (landmarks[idx_b].x, landmarks[idx_b].y)
        c = (landmarks[idx_c].x, landmarks[idx_c].y)
        return calculate_angle(a, b, c)
    except (ValueError, IndexError, AttributeError):
        return None


def main() -> None:
    """Run the PhysioAR real-time pose estimation pipeline.

    Opens the webcam, processes each frame through MediaPipe Pose
    Landmarker with EMA smoothing, draws detected landmarks and
    connections, and displays the annotated feed. Press 'q' to exit.
    """
    camera = Camera(camera_index=0)
    pose_estimator = PoseEstimator()

    try:
        camera.open()
        print("PhysioAR — Phase 1.5: Webcam + MediaPipe Pose + EMA Smoothing")
        print("Press 'q' to quit.")

        while True:
            success, frame = camera.read()

            if not success:
                print("Frame capture failed. Exiting.")
                break

            # Process the frame through MediaPipe Pose Landmarker + EMA.
            result = pose_estimator.process(frame)

            # Draw pose landmarks and connections if a pose was detected.
            pose_estimator.draw(frame, result)

            # --- Phase 1.5 diagnostic: raw vs. smoothed angle overlay ---
            if _SHOW_SMOOTHING_DIAGNOSTIC and result.pose_detected:
                raw_angle = _compute_angle_safe(
                    result.raw_landmarks,
                    _DIAG_SHOULDER_IDX,
                    _DIAG_ELBOW_IDX,
                    _DIAG_WRIST_IDX,
                )
                smoothed_angle = _compute_angle_safe(
                    result.smoothed_landmarks,
                    _DIAG_SHOULDER_IDX,
                    _DIAG_ELBOW_IDX,
                    _DIAG_WRIST_IDX,
                )

                if raw_angle is not None and smoothed_angle is not None:
                    raw_text = f"Raw elbow:      {raw_angle:.1f} deg"
                    smooth_text = f"Smoothed elbow: {smoothed_angle:.1f} deg"
                    cv2.putText(
                        frame, raw_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                    )
                    cv2.putText(
                        frame, smooth_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                    )

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
