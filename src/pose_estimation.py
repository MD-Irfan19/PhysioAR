"""Pose estimation module for PhysioAR.

This module provides a reusable wrapper around the MediaPipe Pose
Landmarker (Tasks API) for human body landmark detection. It isolates
all MediaPipe-specific functionality from the rest of the application.

After MediaPipe processing, raw landmark coordinates are passed through
an EMA (Exponential Moving Average) filter to reduce frame-to-frame
numerical jitter. Both raw and smoothed landmarks are made available
to downstream modules.

No camera access, posture metrics, geometry calculations, or exercise
logic belongs here.

Requires the pose_landmarker_lite.task model file in the data/ directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

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

from src.config import SMOOTHING_ALPHA
from src.utils.smoothing import EMAFilter, SmoothedLandmark


# Default model path relative to the project root.
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "pose_landmarker_lite.task",
)


@dataclass
class PoseResult:
    """Application-level pose result containing raw and smoothed landmarks.

    Attributes:
        mediapipe_result: The original PoseLandmarkerResult from MediaPipe.
            Used for drawing and accessing the unmodified detection output.
        raw_landmarks: The raw landmark coordinates from the first detected
            pose, as a list of MediaPipe NormalizedLandmark objects.
            Empty list if no pose was detected.
        smoothed_landmarks: The EMA-smoothed landmark coordinates for the
            first detected pose, as a list of SmoothedLandmark objects.
            Empty list if no pose was detected.
        pose_detected: True if at least one pose was detected in the frame.
    """

    mediapipe_result: PoseLandmarkerResult
    raw_landmarks: list = field(default_factory=list)
    smoothed_landmarks: list[SmoothedLandmark] = field(default_factory=list)
    pose_detected: bool = False


class PoseEstimator:
    """Wrapper around MediaPipe Pose Landmarker for real-time landmark detection.

    Uses the MediaPipe Tasks API (PoseLandmarker) with VIDEO running mode
    for sequential frame-by-frame processing with built-in tracking.

    After MediaPipe processing, raw landmarks are passed through an EMA
    filter (``EMAFilter``) to produce smoothed spatial coordinates. Both
    raw and smoothed landmarks are available via the returned ``PoseResult``.

    The EMA filter:
      - Smooths x, y, z coordinates independently.
      - Does NOT smooth visibility or presence values.
      - Passes the first observation through unchanged.
      - Retains state across frames when no pose is detected.
      - Can be reset via ``reset_smoothing()``.

    Example::

        estimator = PoseEstimator()
        result = estimator.process(bgr_frame)
        if result.pose_detected:
            estimator.draw(bgr_frame, result)
            for lm in result.smoothed_landmarks:
                print(f"smoothed=({lm.x:.3f}, {lm.y:.3f}), "
                      f"raw=({lm.raw_x:.3f}, {lm.raw_y:.3f})")
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
        smoothing_alpha: float | None = None,
    ) -> None:
        """Initialize the MediaPipe Pose Landmarker and EMA filter.

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
            smoothing_alpha: EMA smoothing factor. If None, uses the
                value from ``src.config.SMOOTHING_ALPHA``.

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

        # EMA filter for landmark smoothing.
        alpha = smoothing_alpha if smoothing_alpha is not None else SMOOTHING_ALPHA
        self._ema_filter = EMAFilter(alpha=alpha)

    def process(self, bgr_frame: np.ndarray) -> PoseResult:
        """Process a BGR frame and return raw + smoothed pose results.

        Converts the frame from BGR (OpenCV format) to RGB (MediaPipe
        format), wraps it as a MediaPipe Image, processes it with the
        pose landmarker, and applies EMA smoothing to the detected
        landmark coordinates.

        If no pose is detected, the EMA filter retains its previous
        state and the returned PoseResult has empty landmark lists.

        Args:
            bgr_frame: An OpenCV BGR image as a NumPy array.

        Returns:
            A PoseResult containing the original MediaPipe result,
            raw landmarks, and EMA-smoothed landmarks.
        """
        # Convert BGR to RGB for MediaPipe.
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        # Wrap as a MediaPipe Image.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Use monotonically increasing timestamps for VIDEO mode.
        self._frame_timestamp_ms += 1
        mp_result = self._landmarker.detect_for_video(
            mp_image, self._frame_timestamp_ms
        )

        # Build the application-level result.
        if not mp_result.pose_landmarks:
            # No pose detected — return empty result, keep filter state.
            return PoseResult(
                mediapipe_result=mp_result,
                raw_landmarks=[],
                smoothed_landmarks=[],
                pose_detected=False,
            )

        # Use the first detected pose (index 0).
        raw_landmarks = mp_result.pose_landmarks[0]

        # Apply EMA smoothing to spatial coordinates.
        smoothed_landmarks = self._ema_filter.smooth(raw_landmarks)

        return PoseResult(
            mediapipe_result=mp_result,
            raw_landmarks=list(raw_landmarks),
            smoothed_landmarks=smoothed_landmarks,
            pose_detected=True,
        )

    def draw(self, bgr_frame: np.ndarray, result: PoseResult) -> None:
        """Draw detected pose landmarks and connections onto a BGR frame.

        Uses the original MediaPipe result for drawing, preserving the
        Phase 1 visualization pipeline.

        Modifies the frame in-place. Only draws if landmarks are present.

        Args:
            bgr_frame: The OpenCV BGR frame to draw on (modified in-place).
            result: The PoseResult from process().
        """
        mp_result = result.mediapipe_result
        if not mp_result.pose_landmarks:
            return

        for landmarks in mp_result.pose_landmarks:
            draw_landmarks(
                image=bgr_frame,
                landmark_list=landmarks,
                connections=self.POSE_CONNECTIONS,
            )

    def reset_smoothing(self) -> None:
        """Reset the EMA filter state.

        The next valid landmark observation will pass through unchanged
        as a new filtered baseline.
        """
        self._ema_filter.reset()

    def close(self) -> None:
        """Release MediaPipe Pose Landmarker resources.

        Safe to call even if the estimator has already been closed.
        """
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
