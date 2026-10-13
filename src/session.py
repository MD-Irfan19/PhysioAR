"""PhysioAR Headless Exercise Session Controller — Phase A.

Encapsulates runtime orchestration, neutral-posture calibration,
frame-by-frame exercise analysis, repetition tracking, posture metrics,
compensation evaluation, movement quality scoring, and explainable feedback
into a reusable, headless controller suitable for FastAPI and React integration.

This module contains NO CLI prompts (input) and NO OpenCV GUI windows
(cv2.imshow / cv2.waitKey).
"""

from __future__ import annotations

import math
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Sequence, Union

import cv2
import numpy as np

from src.acquisition import Camera
from src.calibration import (
    CALIBRATION_SECONDS,
    MIN_CALIBRATION_SAMPLES,
    CalibrationResult,
    MetricBaseline,
    compute_frame_metrics,
    validate_required_landmarks,
)
from src.compensation import (
    CompensationResult,
    GenericCompensationResult,
    evaluate_compensation,
    evaluate_generic_checks,
)
from src.config import LANDMARK_VISIBILITY_THRESHOLD
from src.exercises import EXERCISE_REGISTRY, ExerciseDefinition
from src.exercises.base import CameraOrientation, RepState
from src.feedback import generate_all_feedback, generate_feedback
from src.metrics.posture import PostureMetrics, compute_posture_metrics
from src.overlay import draw_hud
from src.pose_estimation import PoseEstimator, PoseResult
from src.quality import (
    FrameSample,
    QualityResult,
    evaluate_rep_quality,
)
from src.rep_detection import RepDetector, RepEvent


class SessionStatus(str, Enum):
    """Lifecycle status of an ExerciseSessionController."""

    IDLE = "idle"
    CALIBRATING = "calibrating"
    ACTIVE = "active"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class CalibrationAccumulator:
    """Headless sample accumulator for neutral posture calibration.

    Encapsulates frame-by-frame sample collection without blocking the caller
    or rendering desktop GUI windows.
    """

    duration_seconds: float = CALIBRATION_SECONDS
    min_samples: int = MIN_CALIBRATION_SAMPLES
    visibility_threshold: float = LANDMARK_VISIBILITY_THRESHOLD
    side: str = "right"

    # Accumulated metric values
    spine_values: list[float] = field(default_factory=list)
    shoulder_values: list[float] = field(default_factory=list)
    neck_values: list[float] = field(default_factory=list)
    hip_values: list[float] = field(default_factory=list)
    hip_rot_values: list[float] = field(default_factory=list)
    lat_lean_values: list[float] = field(default_factory=list)
    shoulder_substitution_values: list[float] = field(default_factory=list)

    valid_samples: int = 0
    skipped_samples: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)
    start_time: float = 0.0
    elapsed_seconds: float = 0.0
    is_started: bool = False

    def start(self) -> None:
        """Start the calibration timer and clear previous samples."""
        self.spine_values.clear()
        self.shoulder_values.clear()
        self.neck_values.clear()
        self.hip_values.clear()
        self.hip_rot_values.clear()
        self.lat_lean_values.clear()
        self.shoulder_substitution_values.clear()
        self.skip_reasons.clear()
        self.valid_samples = 0
        self.skipped_samples = 0
        self.start_time = time.time()
        self.elapsed_seconds = 0.0
        self.is_started = True

    def step(
        self,
        arg1: Any = None,
        arg2: Any = None,
        arg3: Optional[float] = None,
        current_time: Optional[float] = None,
        pose_detected: Optional[bool] = None,
        smoothed_landmarks: Optional[list] = None,
    ) -> bool:
        """Process a single frame for calibration.

        Supports both (pose_detected, smoothed_landmarks, current_time) and
        (smoothed_landmarks, current_time) calling conventions, using positional
        or keyword arguments.

        Returns True if calibration duration has elapsed.
        """
        if not self.is_started:
            self.start()

        if pose_detected is not None:
            detected = pose_detected
            lms = smoothed_landmarks if smoothed_landmarks is not None else []
            time_val = current_time
        elif isinstance(arg1, bool):
            detected = arg1
            lms = arg2 if arg2 is not None else []
            time_val = arg3 if arg3 is not None else current_time
        else:
            lms = arg1 if arg1 is not None else []
            detected = bool(lms)
            time_val = (
                arg2
                if isinstance(arg2, (int, float))
                else (arg3 if arg3 is not None else current_time)
            )

        now = time_val if time_val is not None else time.time()
        self.elapsed_seconds = max(0.0, now - self.start_time)

        if not detected:
            self.skipped_samples += 1
            self.skip_reasons["no_pose_detected"] = (
                self.skip_reasons.get("no_pose_detected", 0) + 1
            )
            return self.elapsed_seconds >= self.duration_seconds

        rejection_reasons = validate_required_landmarks(
            lms,
            threshold=self.visibility_threshold,
        )

        if rejection_reasons:
            self.skipped_samples += 1
            for reason in rejection_reasons:
                landmark_name = reason.split(":")[0].strip()
                key = f"low_visibility_{landmark_name}"
                self.skip_reasons[key] = self.skip_reasons.get(key, 0) + 1
            return self.elapsed_seconds >= self.duration_seconds

        metrics = compute_frame_metrics(
            lms,
            visibility_threshold=self.visibility_threshold,
            side=self.side,
        )

        if metrics is None:
            self.skipped_samples += 1
            self.skip_reasons["degenerate_geometry"] = (
                self.skip_reasons.get("degenerate_geometry", 0) + 1
            )
        else:
            spine, shoulder, neck, hip, hip_rot, lat_lean, upper_arm = metrics
            self.spine_values.append(spine)
            self.shoulder_values.append(shoulder)
            self.neck_values.append(neck)
            self.hip_values.append(hip)
            self.hip_rot_values.append(hip_rot)
            self.lat_lean_values.append(lat_lean)
            if upper_arm is not None:
                self.shoulder_substitution_values.append(upper_arm)
            self.valid_samples += 1

        return self.elapsed_seconds >= self.duration_seconds

    def finish(self) -> CalibrationResult:
        """Calculate and return CalibrationResult.

        Raises RuntimeError if valid_samples < min_samples.
        """
        if self.valid_samples < self.min_samples:
            raise RuntimeError(
                f"Calibration failed: only {self.valid_samples} valid samples "
                f"collected, but {self.min_samples} are required. Ensure the "
                f"user's full body is clearly visible in the camera."
            )

        return CalibrationResult(
            spine_angle=MetricBaseline(
                mean=statistics.mean(self.spine_values),
                std=statistics.stdev(self.spine_values),
            ),
            shoulder_height_difference=MetricBaseline(
                mean=statistics.mean(self.shoulder_values),
                std=statistics.stdev(self.shoulder_values),
            ),
            neck_tilt=MetricBaseline(
                mean=statistics.mean(self.neck_values),
                std=statistics.stdev(self.neck_values),
            ),
            hip_alignment=MetricBaseline(
                mean=statistics.mean(self.hip_values),
                std=statistics.stdev(self.hip_values),
            ),
            hip_rotation=MetricBaseline(
                mean=statistics.mean(self.hip_rot_values),
                std=statistics.stdev(self.hip_rot_values),
            ),
            lateral_trunk_lean=MetricBaseline(
                mean=statistics.mean(self.lat_lean_values),
                std=statistics.stdev(self.lat_lean_values),
            ),
            shoulder_substitution=(
                MetricBaseline(
                    mean=statistics.mean(self.shoulder_substitution_values),
                    std=statistics.stdev(self.shoulder_substitution_values),
                )
                if len(self.shoulder_substitution_values) > 1
                else None
            ),
            valid_samples=self.valid_samples,
            skipped_samples=self.skipped_samples,
            duration_seconds=self.elapsed_seconds,
        )


@dataclass
class SessionSnapshot:
    """Stable, structured representation of the current session state.

    Contains plain Python datatypes suitable for JSON serialization.
    No NumPy arrays or MediaPipe objects are exposed here.
    """

    status: SessionStatus
    exercise_name: Optional[str] = None
    side: Optional[str] = None
    camera_orientation: Optional[str] = None
    is_calibrated: bool = False
    calibration_status: str = "not_calibrated"
    calibration_error: Optional[str] = None
    calibration_samples: int = 0
    calibration_baseline: Optional[dict[str, Any]] = None
    pose_detected: bool = False
    current_angle: Optional[float] = None
    rep_state: Optional[str] = None
    rep_count: int = 0
    rom_target: Optional[float] = None
    rom_progress: Optional[float] = None
    active_compensation_ids: list[str] = field(default_factory=list)
    feedback_messages: list[str] = field(default_factory=list)
    latest_quality: Optional[dict[str, Any]] = None
    last_error: Optional[str] = None
    frame_index: int = 0

    @property
    def selected_side(self) -> Optional[str]:
        return self.side

    @property
    def exercise_id(self) -> Optional[str]:
        if not self.exercise_name:
            return None
        return self.exercise_name.lower().replace(" ", "_").replace("-", "_")

    @property
    def is_pose_detected(self) -> bool:
        return self.pose_detected

    @property
    def joint_angle(self) -> Optional[float]:
        return self.current_angle

    @property
    def rep_phase(self) -> Optional[str]:
        return self.rep_state

    @property
    def feedback(self) -> list[str]:
        return self.feedback_messages

    @property
    def quality(self) -> Optional[dict[str, Any]]:
        return self.latest_quality

    def to_dict(self) -> dict[str, Any]:
        """Convert snapshot to a JSON-serializable dictionary."""
        return {
            "status": self.status.value,
            "exercise_name": self.exercise_name,
            "exercise_id": self.exercise_id,
            "side": self.side,
            "selected_side": self.selected_side,
            "camera_orientation": self.camera_orientation,
            "is_calibrated": self.is_calibrated,
            "calibration_status": self.calibration_status,
            "calibration_error": self.calibration_error,
            "calibration_samples": self.calibration_samples,
            "calibration_baseline": self.calibration_baseline,
            "pose_detected": self.pose_detected,
            "current_angle": (
                round(self.current_angle, 1)
                if self.current_angle is not None
                else None
            ),
            "joint_angle": (
                round(self.current_angle, 1)
                if self.current_angle is not None
                else None
            ),
            "rep_state": self.rep_state,
            "rep_count": self.rep_count,
            "rom_target": self.rom_target,
            "rom_progress": (
                round(self.rom_progress, 3)
                if self.rom_progress is not None
                else None
            ),
            "active_compensation_ids": list(self.active_compensation_ids),
            "feedback_messages": list(self.feedback_messages),
            "latest_quality": self.latest_quality,
            "last_error": self.last_error,
            "frame_index": self.frame_index,
        }


@dataclass
class FrameProcessingResult:
    """Result returned by ExerciseSessionController.process_frame()."""

    snapshot: SessionSnapshot
    rep_event: Optional[RepEvent] = None
    annotated_frame: Optional[np.ndarray] = None
    new_feedback_messages: list[str] = field(default_factory=list)
    posture_metrics: Optional[PostureMetrics] = None
    compensation_result: Optional[CompensationResult] = None
    generic_results: dict[str, GenericCompensationResult] = field(
        default_factory=dict
    )
    calibration_progress: Optional[float] = None

    @property
    def completed_rep(self) -> Optional[RepEvent]:
        return self.rep_event

    @property
    def pose_detected(self) -> bool:
        return self.snapshot.pose_detected

    @property
    def joint_angle(self) -> Optional[float]:
        return self.snapshot.current_angle

    @property
    def rep_phase(self) -> Optional[str]:
        return self.snapshot.rep_state


def resolve_exercise(identifier: Union[ExerciseDefinition, str, int]) -> ExerciseDefinition:
    """Resolve an ExerciseDefinition from an instance, name/slug, or index.

    Raises ValueError if the identifier cannot be resolved.
    """
    if isinstance(identifier, ExerciseDefinition):
        return identifier

    if isinstance(identifier, int):
        if 0 <= identifier < len(EXERCISE_REGISTRY):
            return EXERCISE_REGISTRY[identifier]
        raise ValueError(
            f"Exercise index {identifier} out of bounds (0-{len(EXERCISE_REGISTRY)-1})."
        )

    if isinstance(identifier, str):
        if identifier.strip().isdigit():
            idx = int(identifier.strip()) - 1
            if 0 <= idx < len(EXERCISE_REGISTRY):
                return EXERCISE_REGISTRY[idx]
            raise ValueError(
                f"Exercise index {identifier} out of bounds (1-{len(EXERCISE_REGISTRY)})."
            )

        target = identifier.strip().lower().replace("_", " ").replace("-", " ")
        for ex in EXERCISE_REGISTRY:
            ex_name_norm = ex.name.strip().lower().replace("_", " ").replace("-", " ")
            if target == ex_name_norm:
                return ex
        # Also try matching with/without spaces
        target_compact = target.replace(" ", "")
        for ex in EXERCISE_REGISTRY:
            if target_compact == ex.name.strip().lower().replace(" ", "").replace("_", "").replace("-", ""):
                return ex

        available = [ex.name for ex in EXERCISE_REGISTRY]
        raise ValueError(
            f"Unknown exercise '{identifier}'. Available exercises: {available}"
        )

    raise TypeError(
        f"Exercise identifier must be ExerciseDefinition, str, or int, got {type(identifier)}."
    )


class ExerciseSessionController:
    """Headless session controller for PhysioAR.

    Manages runtime orchestration, calibration baseline lifecycle,
    single-frame processing, repetition detection, posture metrics,
    compensation evaluation, movement quality scoring, and feedback generation.

    Decoupled from camera hardware loops, standard input, and desktop OpenCV windows.
    """

    def __init__(
        self,
        pose_estimator: Optional[PoseEstimator] = None,
        camera: Optional[Camera] = None,
        max_frame_history: int = 600,
    ) -> None:
        """Initialize controller with optional injected dependencies.

        Args:
            pose_estimator: Injected PoseEstimator instance. If None,
                created lazily on demand.
            camera: Optional Camera instance.
            max_frame_history: Maximum capacity of the FrameSample history deque.
        """
        self._pose_estimator = pose_estimator
        self._owns_pose_estimator = pose_estimator is None

        self._camera = camera
        self._owns_camera = camera is not None  # If passed, we hold reference

        self._max_frame_history = max_frame_history
        self._exercise: Optional[ExerciseDefinition] = None
        self._side: Optional[str] = None
        self._status: SessionStatus = SessionStatus.IDLE

        self._calibration_result: Optional[CalibrationResult] = None
        self._calibrator: Optional[CalibrationAccumulator] = None
        self._calibration_error: Optional[str] = None
        self._calibration_status: str = "not_calibrated"

        self._rep_detector: Optional[RepDetector] = None
        self._frame_index: int = 0
        self._frame_history: deque[FrameSample] = deque(maxlen=self._max_frame_history)
        self._last_quality: Optional[QualityResult] = None

        self._active_feedback: list[str] = []
        self._prev_flagged: set[str] = set()

        self._current_angle: Optional[float] = None
        self._pose_detected: bool = False
        self._last_error: Optional[str] = None

        self._last_posture: Optional[PostureMetrics] = None
        self._last_compensation: Optional[CompensationResult] = None
        self._last_generic_results: dict[str, GenericCompensationResult] = {}

    @property
    def status(self) -> SessionStatus:
        """Current session status."""
        return self._status

    @property
    def exercise(self) -> Optional[ExerciseDefinition]:
        """Currently loaded exercise definition."""
        return self._exercise

    @property
    def side(self) -> str:
        """Currently active side."""
        return self._side

    @property
    def is_calibrated(self) -> bool:
        """True if a valid neutral calibration baseline exists."""
        return self._calibration_result is not None

    @property
    def calibration_result(self) -> Optional[CalibrationResult]:
        """Current calibration baseline result."""
        return self._calibration_result

    @property
    def frame_index(self) -> int:
        """Current frame index."""
        return self._frame_index

    @property
    def active_feedback(self) -> list[str]:
        """Currently active feedback messages."""
        return list(self._active_feedback)

    @property
    def last_error(self) -> Optional[str]:
        """Most recent recoverable error message."""
        return self._last_error

    @property
    def last_quality_result(self) -> Optional[QualityResult]:
        """Latest completed repetition quality result."""
        return self._last_quality

    @property
    def calibration_status(self) -> str:
        """Current calibration status string."""
        return self._calibration_status

    @property
    def rep_count(self) -> int:
        """Completed repetition count."""
        if self._rep_detector is not None:
            return self._rep_detector.rep_count
        return 0

    @property
    def rep_state(self) -> Optional[RepState]:
        """Current repetition state machine state."""
        if self._rep_detector is not None:
            return self._rep_detector.state
        return None

    def _get_pose_estimator(self) -> PoseEstimator:
        """Lazy initializer for PoseEstimator if not provided."""
        if self._pose_estimator is None:
            self._pose_estimator = PoseEstimator()
            self._owns_pose_estimator = True
        return self._pose_estimator

    def start_session(
        self,
        exercise: Union[ExerciseDefinition, str, int],
        side: str = "right",
    ) -> SessionSnapshot:
        """Start a new exercise session.

        Validates the exercise and side, cleanly resets session-specific state,
        and initializes rep detector and history buffers.

        Args:
            exercise: ExerciseDefinition instance, name string, or registry index.
            side: "left" or "right" indicating tracked arm/leg.

        Returns:
            Initial SessionSnapshot.
        """
        side_clean = side.strip().lower()
        if side_clean in ("l", "left"):
            side_norm = "left"
        elif side_clean in ("r", "right"):
            side_norm = "right"
        else:
            self._status = SessionStatus.ERROR
            self._last_error = f"Invalid side: '{side}'. Must be 'left' or 'right'."
            raise ValueError(self._last_error)

        try:
            resolved_exercise = resolve_exercise(exercise)
        except Exception as e:
            self._status = SessionStatus.ERROR
            self._last_error = f"Invalid exercise identifier: {e}"
            raise ValueError(self._last_error) from e

        # Clean reset of session-specific state
        self._exercise = resolved_exercise
        self._side = side_norm
        self._frame_index = 0
        self._frame_history.clear()
        self._last_quality = None
        self._active_feedback = []
        self._prev_flagged = set()
        self._current_angle = None
        self._pose_detected = False
        self._last_error = None
        self._last_posture = None
        self._last_compensation = None
        self._last_generic_results = {}

        # Reset rep detector for the new exercise
        self._rep_detector = RepDetector(self._exercise)

        if self._calibration_result is not None:
            self._calibration_status = "calibrated"
        else:
            self._calibration_status = "not_calibrated"
        self._status = SessionStatus.ACTIVE

        return self.get_snapshot()

    def start_calibration(
        self,
        duration_seconds: float = CALIBRATION_SECONDS,
        min_samples: int = MIN_CALIBRATION_SAMPLES,
        visibility_threshold: float = LANDMARK_VISIBILITY_THRESHOLD,
    ) -> None:
        """Initiate neutral-posture calibration.

        Transitions status to CALIBRATING, resets EMA smoothing on the pose
        estimator, and begins accumulating valid posture frames.

        Args:
            duration_seconds: Duration in seconds to collect calibration frames.
            min_samples: Minimum valid frames required to accept baseline.
            visibility_threshold: Minimum raw landmark visibility confidence.
        """
        if self._exercise is None:
            raise RuntimeError("Cannot calibrate: no active session. Call start_session() first.")

        estimator = self._get_pose_estimator()
        estimator.reset_smoothing()

        self._calibrator = CalibrationAccumulator(
            duration_seconds=duration_seconds,
            min_samples=min_samples,
            visibility_threshold=visibility_threshold,
            side=self._side,
        )
        self._calibrator.start()
        self._calibration_error = None
        self._calibration_status = "calibrating"
        self._status = SessionStatus.CALIBRATING

    def finish_calibration(self) -> Optional[CalibrationResult]:
        """Finalize the current calibration run and evaluate the baseline.

        If successful:
            - Replaces the previous calibration result
            - Resets the rep detector and session history
            - Transitions status to ACTIVE
        If failed:
            - Preserves previous valid baseline if one existed
            - Sets calibration_error
            - Transitions to ACTIVE (if previous baseline preserved) or IDLE/ERROR

        Returns:
            The new CalibrationResult if successful, or None on failure.
        """
        if self._calibrator is None:
            return self._calibration_result

        try:
            new_baseline = self._calibrator.finish()
            self._calibration_result = new_baseline
            self._calibration_status = "calibrated"
            self._calibration_error = None
            self._calibrator = None
            self._status = SessionStatus.ACTIVE

            # Reset rep detection and frame history on new calibration
            if self._rep_detector is not None:
                self._rep_detector.reset()
            self._frame_history.clear()
            self._last_quality = None
            self._active_feedback = []
            self._prev_flagged = set()
            return new_baseline

        except RuntimeError as e:
            error_msg = str(e)
            self._calibration_error = error_msg
            self._calibrator = None

            if self._calibration_result is not None:
                # Recalibration failed: preserve previous baseline
                self._calibration_status = "calibrated"
                self._calibration_error = f"Recalibration failed: {error_msg}"
                self._status = SessionStatus.ACTIVE
            else:
                self._calibration_status = "failed"
                self._calibration_error = error_msg
                self._status = SessionStatus.ERROR

            return None

    def set_calibration(self, result: Optional[CalibrationResult]) -> None:
        """Directly set or inject a calibration result (e.g. from saved profile)."""
        self._calibration_result = result
        if result is not None:
            self._calibration_status = "calibrated"
            self._calibration_error = None
            if self._status == SessionStatus.IDLE:
                self._status = SessionStatus.ACTIVE
        else:
            self._calibration_status = "not_calibrated"

    def process_frame(
        self,
        frame: np.ndarray,
        render_overlay: bool = True,
        draw_debug: bool = False,
    ) -> FrameProcessingResult:
        """Process a single camera frame through the PhysioAR pipeline.

        Executes either calibration accumulation or live exercise evaluation
        depending on session status.

        Args:
            frame: OpenCV BGR NumPy array image.
            render_overlay: If True, draws skeleton, HUD, and feedback in-place.
            draw_debug: If True, draws diagnostic landmark coordinates.

        Returns:
            FrameProcessingResult containing session snapshot, rep event,
            and intermediate measurements.
        """
        if self._status == SessionStatus.STOPPED:
            raise RuntimeError("Cannot process frame: session is stopped.")

        if self._exercise is None or self._status == SessionStatus.IDLE:
            raise RuntimeError("Cannot process frame: Session is not active. Call start_session() first.")

        annotated_frame = frame.copy() if render_overlay else frame
        estimator = self._get_pose_estimator()

        # Step 1: MediaPipe Pose Estimation + EMA Smoothing
        try:
            pose_res: PoseResult = estimator.process(frame)
        except Exception as e:
            self._last_error = f"Pose estimation error: {e}"
            self._status = SessionStatus.ERROR
            return FrameProcessingResult(
                snapshot=self.get_snapshot(),
                annotated_frame=annotated_frame,
            )

        self._pose_detected = pose_res.pose_detected

        # Draw skeleton if requested and pose detected
        if render_overlay and pose_res.pose_detected:
            estimator.draw(annotated_frame, pose_res)

        # -------------------------------------------------------------
        # Branch A: Calibration Mode
        # -------------------------------------------------------------
        if self._status == SessionStatus.CALIBRATING:
            if self._calibrator is not None:
                is_done = self._calibrator.step(
                    pose_detected=pose_res.pose_detected,
                    smoothed_landmarks=pose_res.smoothed_landmarks,
                )

                if render_overlay:
                    remaining = max(
                        0.0,
                        self._calibrator.duration_seconds - self._calibrator.elapsed_seconds,
                    )
                    cv2.putText(
                        annotated_frame,
                        f"Calibrating... {remaining:.1f}s remaining",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )
                    if not pose_res.pose_detected:
                        cv2.putText(
                            annotated_frame,
                            "No pose detected - stand in view",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2,
                        )

                if is_done:
                    self.finish_calibration()

            cal_prog = None
            if self._calibrator is not None and self._calibrator.duration_seconds > 0:
                cal_prog = min(1.0, self._calibrator.elapsed_seconds / self._calibrator.duration_seconds)

            return FrameProcessingResult(
                snapshot=self.get_snapshot(),
                annotated_frame=annotated_frame,
                calibration_progress=cal_prog,
            )

        # -------------------------------------------------------------
        # Branch B: Live Exercise Analysis Mode
        # -------------------------------------------------------------
        angle: Optional[float] = None
        if pose_res.pose_detected:
            try:
                angle = self._exercise.angle_calculator(
                    pose_res.smoothed_landmarks, self._side
                )
            except Exception:
                angle = None

        self._current_angle = angle

        # Posture Metrics (Compute-only)
        if pose_res.pose_detected:
            posture = compute_posture_metrics(
                pose_res.smoothed_landmarks, side=self._side
            )
        else:
            posture = PostureMetrics()
        self._last_posture = posture

        # Legacy Compensation Evaluation
        comp = evaluate_compensation(posture, self._calibration_result)
        generic_results: dict[str, GenericCompensationResult] = {}

        rep_state_val = (
            self._rep_detector.state if self._rep_detector is not None else RepState.WAITING_FOR_START
        )

        # Generic Compensation Evaluation
        if comp is not None and self._exercise.generic_checks:
            generic_results = evaluate_generic_checks(
                self._exercise.generic_checks,
                posture,
                self._calibration_result,
                pose_res.smoothed_landmarks,
                self._side,
                LANDMARK_VISIBILITY_THRESHOLD,
                rep_state_val,
            )
            comp.generic_results = generic_results

        self._last_compensation = comp
        self._last_generic_results = generic_results

        # Feedback Generation and De-duplication
        current_flagged: set[str] = set()
        if comp is not None:
            for comp_type in [
                "torso_lean",
                "shoulder_hike",
                "neck_tilt",
                "hip_rotation",
                "lateral_trunk_lean",
                "hip_hike",
                "trunk_lean",
                "shoulder_substitution",
            ]:
                metric_result = getattr(comp, comp_type, None)
                if metric_result and metric_result.flagged:
                    current_flagged.add(comp_type)
            for g_id, g_res in generic_results.items():
                if g_res.flagged:
                    current_flagged.add(g_id)

        new_feedback_messages: list[str] = []
        if current_flagged != self._prev_flagged:
            self._active_feedback = generate_all_feedback(self._exercise, comp)
            newly_flagged = current_flagged - self._prev_flagged
            if newly_flagged and comp is not None:
                for comp_type in sorted(newly_flagged):
                    if comp_type in generic_results:
                        g_res = generic_results[comp_type]
                        val = (
                            g_res.deviation
                            if g_res.deviation is not None
                            else g_res.current_value
                        )
                        msg = generate_feedback(self._exercise, comp_type, val)
                    else:
                        metric = getattr(comp, comp_type, None)
                        if metric and metric.deviation is not None:
                            msg = generate_feedback(
                                self._exercise, comp_type, metric.deviation
                            )
                        else:
                            msg = None
                    if msg:
                        new_feedback_messages.append(msg)
            self._prev_flagged = current_flagged
        elif not current_flagged:
            self._active_feedback = []

        # Record FrameSample for Rep Quality Scoring
        sample = FrameSample(
            frame_index=self._frame_index,
            angle=angle,
            torso_lean=posture.torso_lean,
            shoulder_height_diff=posture.shoulder_height_difference,
            neck_tilt=posture.neck_tilt,
            torso_lean_flagged=comp.torso_lean.flagged if comp is not None else False,
            shoulder_hike_flagged=comp.shoulder_hike.flagged if comp is not None else False,
            neck_tilt_flagged=comp.neck_tilt.flagged if comp is not None else False,
            hip_rotation_flagged=comp.hip_rotation.flagged if comp is not None else False,
            lateral_trunk_lean_flagged=comp.lateral_trunk_lean.flagged if comp is not None else False,
            hip_hike_flagged=comp.hip_hike.flagged if comp is not None else False,
            trunk_lean_flagged=comp.trunk_lean.flagged if comp is not None else False,
            torso_deviation=comp.torso_lean.deviation if comp is not None else None,
            shoulder_deviation=comp.shoulder_hike.deviation if comp is not None else None,
            neck_deviation=comp.neck_tilt.deviation if comp is not None else None,
            hip_rotation_deviation=comp.hip_rotation.deviation if comp is not None else None,
            lateral_trunk_lean_deviation=comp.lateral_trunk_lean.deviation if comp is not None else None,
            hip_hike_deviation=comp.hip_hike.deviation if comp is not None else None,
            trunk_lean_deviation=comp.trunk_lean.deviation if comp is not None else None,
            generic_flags={k: v.flagged for k, v in generic_results.items()},
            generic_deviations={
                k: v.deviation for k, v in generic_results.items() if v.deviation is not None
            },
        )
        self._frame_history.append(sample)

        # Repetition State Machine Update
        rep_event: Optional[RepEvent] = None
        if self._rep_detector is not None:
            rep_event = self._rep_detector.update(angle, frame_index=self._frame_index)

        # Evaluate Quality on Completed Repetition
        if rep_event is not None:
            self._last_quality = evaluate_rep_quality(
                rep_number=rep_event.rep_number,
                frame_history=list(self._frame_history),
                start_frame=rep_event.start_frame,
                top_frame=rep_event.top_frame,
                end_frame=rep_event.end_frame,
                target_rom=self._exercise.rom_target,
            )

        # Overlay Drawing
        if render_overlay:
            rep_state_str = (
                self._rep_detector.state.value.upper()
                if self._rep_detector is not None
                else "WAITING_FOR_START"
            )
            rep_count_val = (
                self._rep_detector.rep_count
                if self._rep_detector is not None
                else 0
            )
            draw_hud(
                annotated_frame,
                angle=angle,
                target_rom=self._exercise.rom_target,
                rep_state=rep_state_str,
                rep_count=rep_count_val,
                feedback_messages=self._active_feedback,
                side=self._side,
            )

            if draw_debug and pose_res.pose_detected:
                self._draw_debug_overlay(annotated_frame, pose_res.smoothed_landmarks)

        self._frame_index += 1

        return FrameProcessingResult(
            snapshot=self.get_snapshot(),
            rep_event=rep_event,
            annotated_frame=annotated_frame,
            new_feedback_messages=new_feedback_messages,
            posture_metrics=posture,
            compensation_result=comp,
            generic_results=generic_results,
        )

    def get_snapshot(self) -> SessionSnapshot:
        """Generate a current, stable snapshot of the session state."""
        rom_prog: Optional[float] = None
        if (
            self._current_angle is not None
            and self._exercise is not None
            and self._exercise.rom_target > 0
        ):
            rom_prog = max(0.0, min(1.0, self._current_angle / self._exercise.rom_target))

        quality_dict: Optional[dict[str, Any]] = None
        if self._last_quality is not None:
            quality_dict = {
                "rep_number": self._last_quality.rep_number,
                "rom_score": round(self._last_quality.rom_score, 1),
                "alignment_score": round(self._last_quality.alignment_score, 1),
                "stability_score": round(self._last_quality.stability_score, 1),
                "compensation_score": round(self._last_quality.compensation_score, 1),
                "overall_score": round(self._last_quality.overall_score, 1),
                "max_angle": (
                    round(self._last_quality.max_angle, 1)
                    if self._last_quality.max_angle is not None
                    else None
                ),
            }

        cal_samples = (
            self._calibrator.valid_samples
            if self._calibrator is not None
            else (
                self._calibration_result.valid_samples
                if self._calibration_result is not None
                else 0
            )
        )

        cal_baseline_dict: Optional[dict[str, Any]] = None
        if self._calibration_result is not None:
            cal_baseline_dict = {}
            for k in [
                "spine_angle",
                "shoulder_height_difference",
                "neck_tilt",
                "hip_alignment",
                "hip_rotation",
                "lateral_trunk_lean",
                "shoulder_substitution",
            ]:
                val = getattr(self._calibration_result, k, None)
                cal_baseline_dict[k] = val.mean if val is not None else None

        return SessionSnapshot(
            status=self._status,
            exercise_name=self._exercise.name if self._exercise else None,
            side=self._side,
            camera_orientation=(
                self._exercise.camera_orientation.value
                if self._exercise
                else None
            ),
            is_calibrated=self.is_calibrated,
            calibration_status=self._calibration_status,
            calibration_error=self._calibration_error,
            calibration_samples=cal_samples,
            calibration_baseline=cal_baseline_dict,
            pose_detected=self._pose_detected,
            current_angle=self._current_angle,
            rep_state=(
                self._rep_detector.state.value
                if self._rep_detector
                else None
            ),
            rep_count=self._rep_detector.rep_count if self._rep_detector else 0,
            rom_target=self._exercise.rom_target if self._exercise else None,
            rom_progress=rom_prog,
            active_compensation_ids=sorted(list(self._prev_flagged)),
            feedback_messages=list(self._active_feedback),
            latest_quality=quality_dict,
            last_error=self._last_error,
            frame_index=self._frame_index,
        )

    def stop_session(self) -> None:
        """Stop the session and cleanly release owned resources.

        Idempotent: safe to call multiple times.
        """
        self._status = SessionStatus.STOPPED
        self._calibrator = None

        if self._owns_pose_estimator and self._pose_estimator is not None:
            self._pose_estimator.close()
            self._pose_estimator = None

        if self._owns_camera and self._camera is not None:
            self._camera.release()
            self._camera = None

    def reset_session_data(self) -> None:
        """Reset repetition count, frame history, and quality without changing exercise."""
        if self._rep_detector is not None:
            self._rep_detector.reset()
        self._frame_index = 0
        self._frame_history.clear()
        self._last_quality = None
        self._active_feedback.clear()
        self._prev_flagged.clear()

    @staticmethod
    def _draw_debug_overlay(frame: np.ndarray, smoothed_landmarks: list) -> None:
        """Internal helper for Phase 2.5 diagnostic landmark labels."""
        from src.main import _draw_debug_overlay
        _draw_debug_overlay(frame, smoothed_landmarks)
