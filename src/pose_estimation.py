"""Pose estimation module for PhysioAR.

This module provides a reusable wrapper around the MediaPipe Pose
Landmarker (Tasks API) for human body landmark detection. It isolates
all MediaPipe-specific functionality from the rest of the application.

No camera access, posture metrics, geometry calculations, or exercise
logic belongs here.

Requires the pose_landmarker_lite.task model file in the data/ directory.
"""

import os
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarkerResult,
    PoseLandmarksConnections,
)
from mediapipe.tasks.python.vision.drawing_utils import draw_landmarks


# Default model path relative to the project root.
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "pose_landmarker_lite.task",
)


class PoseEstimator:
    """Wrapper around MediaPipe Pose Landmarker for real-time landmark detection.

    Uses the MediaPipe Tasks API (PoseLandmarker) with VIDEO running mode
    for sequential frame-by-frame processing with built-in tracking.

    Handles BGR-to-RGB conversion, MediaPipe Image wrapping, timestamped
    processing, and resource cleanup. Returns PoseLandmarkerResult objects
    for downstream use.

    Example::

        estimator = PoseEstimator()
        result = estimator.process(bgr_frame)
        if result.pose_landmarks:
            estimator.draw(bgr_frame, result)
        estimator.close()
    """

    # Pose landmark connections for skeleton drawing.
    POSE_CONNECTIONS = PoseLandmarksConnections.POSE_LANDMARKS

    def __init__(
        self,
        model_path: str | None = None,
        num_poses: int = 1,
        min_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """Initialize the MediaPipe Pose Landmarker.

        Args:
            model_path: Path to the pose_landmarker .task model file.
                Defaults to data/pose_landmarker_lite.task.
            num_poses: Maximum number of poses to detect.
            min_detection_confidence: Minimum confidence for initial
                pose detection.
            min_pose_presence_confidence: Minimum confidence for pose
                presence in each frame.
            min_tracking_confidence: Minimum confidence for landmark
                tracking between frames.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        if model_path is None:
            model_path = _DEFAULT_MODEL_PATH

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Pose landmarker model not found at: {model_path}\n"
                "Download it from: https://storage.googleapis.com/"
                "mediapipe-models/pose_landmarker/pose_landmarker_lite/"
                "float16/latest/pose_landmarker_lite.task\n"
                "and place it in the data/ directory."
            )

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self._landmarker = PoseLandmarker.create_from_options(options)
        self._frame_timestamp_ms = 0

    def process(self, bgr_frame: np.ndarray) -> PoseLandmarkerResult:
        """Process a BGR frame and return pose landmarker results.

        Converts the frame from BGR (OpenCV format) to RGB (MediaPipe
        format), wraps it as a MediaPipe Image, and processes it with
        the pose landmarker using an incrementing timestamp.

        Args:
            bgr_frame: An OpenCV BGR image as a NumPy array.

        Returns:
            A PoseLandmarkerResult. Check ``result.pose_landmarks``
            to determine if a pose was detected (it will be an empty
            list when no person is visible).
        """
        # Convert BGR to RGB for MediaPipe.
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        # Wrap as a MediaPipe Image.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Use monotonically increasing timestamps for VIDEO mode.
        self._frame_timestamp_ms += 1
        result = self._landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)

        return result

    def draw(self, bgr_frame: np.ndarray, result: PoseLandmarkerResult) -> None:
        """Draw detected pose landmarks and connections onto a BGR frame.

        Modifies the frame in-place. Only draws if landmarks are present.

        Args:
            bgr_frame: The OpenCV BGR frame to draw on (modified in-place).
            result: The PoseLandmarkerResult from process().
        """
        if not result.pose_landmarks:
            return

        for landmarks in result.pose_landmarks:
            draw_landmarks(
                image=bgr_frame,
                landmark_list=landmarks,
                connections=self.POSE_CONNECTIONS,
            )

    def close(self) -> None:
        """Release MediaPipe Pose Landmarker resources.

        Safe to call even if the estimator has already been closed.
        """
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
