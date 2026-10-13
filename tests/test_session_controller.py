"""Unit tests for src/session.py and the ExerciseSessionController.

Covers:
- Controller initialization without camera hardware or GUI.
- Session start validation (exercise identifier, side).
- Invalid exercise handling and error states without resource leaks.
- Headless frame processing (with and without detected poses).
- Structured snapshot contract (missing values as None, JSON serializability).
- Headless calibration lifecycle (success, insufficient samples, preserving baseline on recalibration failure).
- Repetition detection and quality integration through the controller.
- Idempotent stop_session and resource cleanup.
- State reset between repeated sessions.
- Observable processing error handling.
- CLI compatibility layer preservation.
"""

import json
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.calibration import CalibrationResult, MetricBaseline
from src.exercises import EXERCISE_REGISTRY, SHOULDER_ABDUCTION
from src.pose_estimation import PoseResult
from src.rep_detection import RepState
from src.session import (
    CalibrationAccumulator,
    ExerciseSessionController,
    FrameProcessingResult,
    SessionSnapshot,
    SessionStatus,
)
from src.utils.smoothing import SmoothedLandmark


# ==============================================================================
# Test Fixtures & Helpers
# ==============================================================================

def make_mock_landmarks(coords_dict=None, default_vis=0.99):
    """Create a list of 33 SmoothedLandmark objects for testing."""
    landmarks = []
    for idx in range(33):
        if coords_dict and idx in coords_dict:
            x, y = coords_dict[idx]
        else:
            x, y = 0.5, 0.5
        landmarks.append(
            SmoothedLandmark(
                x=x,
                y=y,
                z=0.0,
                raw_x=x,
                raw_y=y,
                raw_z=0.0,
                visibility=default_vis,
                presence=default_vis,
            )
        )
    return landmarks


def make_mock_pose_result(coords_dict=None, detected=True, default_vis=0.99):
    """Create a PoseResult with deterministic mock landmarks."""
    if not detected:
        return PoseResult(
            mediapipe_result=MagicMock(),
            raw_landmarks=[],
            smoothed_landmarks=[],
            pose_detected=False,
        )
    lms = make_mock_landmarks(coords_dict, default_vis)
    return PoseResult(
        mediapipe_result=MagicMock(),
        raw_landmarks=[MagicMock() for _ in range(33)],
        smoothed_landmarks=lms,
        pose_detected=True,
    )


def make_dummy_frame(height=480, width=640):
    """Create a synthetic blank BGR frame."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def make_sample_calibration_result():
    """Create a valid dummy CalibrationResult."""
    return CalibrationResult(
        spine_angle=MetricBaseline(mean=5.0, std=0.5),
        shoulder_height_difference=MetricBaseline(mean=0.01, std=0.002),
        neck_tilt=MetricBaseline(mean=2.0, std=0.3),
        hip_rotation=MetricBaseline(mean=0.02, std=0.001),
        lateral_trunk_lean=MetricBaseline(mean=1.0, std=0.1),
    )


# ==============================================================================
# 1. Controller Initialization & Lifecycle
# ==============================================================================

class TestControllerInitialization:
    """Tests verifying controller creation and safe initialization."""

    def test_init_without_camera_or_gui(self):
        """Controller initializes without opening camera hardware or GUI."""
        mock_estimator = MagicMock()
        controller = ExerciseSessionController(pose_estimator=mock_estimator)

        assert controller.status == SessionStatus.IDLE
        assert controller.exercise is None
        assert controller.calibration_status in ("not_calibrated", "uncalibrated")
        assert controller.frame_index == 0
        assert controller.rep_count == 0

    def test_stop_session_is_idempotent(self):
        """stop_session can be called multiple times safely without errors."""
        mock_estimator = MagicMock()
        controller = ExerciseSessionController(pose_estimator=mock_estimator)
        controller.start_session("Shoulder Abduction", side="right")
        assert controller.status == SessionStatus.ACTIVE

        controller.stop_session()
        assert controller.status == SessionStatus.STOPPED

        # Subsequent calls must be safe no-ops
        controller.stop_session()
        controller.stop_session()
        assert controller.status == SessionStatus.STOPPED

    def test_owned_estimator_is_closed_on_stop(self):
        """If the controller instantiates the PoseEstimator, it closes it on stop."""
        with patch("src.session.PoseEstimator") as MockEstimatorCls:
            mock_inst = MagicMock()
            MockEstimatorCls.return_value = mock_inst

            controller = ExerciseSessionController()
            assert controller._owns_pose_estimator is True
            # Instantiate lazily
            _ = controller._get_pose_estimator()

            controller.stop_session()
            mock_inst.close.assert_called_once()

            # Second stop should not call close again
            controller.stop_session()
            assert mock_inst.close.call_count == 1

    def test_injected_estimator_is_not_closed_by_default(self):
        """If estimator is injected externally, controller does not close it."""
        mock_estimator = MagicMock()
        controller = ExerciseSessionController(pose_estimator=mock_estimator)
        assert controller._owns_pose_estimator is False

        controller.stop_session()
        mock_estimator.close.assert_not_called()


# ==============================================================================
# 2. Session Start & Input Validation
# ==============================================================================

class TestSessionStartValidation:
    """Tests exercise and side validation on start_session."""

    def test_start_session_by_name(self):
        mock_estimator = MagicMock()
        controller = ExerciseSessionController(pose_estimator=mock_estimator)
        snap = controller.start_session("Shoulder Abduction", side="right")

        assert controller.status == SessionStatus.ACTIVE
        assert snap.status == "active"
        assert snap.exercise_name == "Shoulder Abduction"
        assert snap.selected_side == "right"
        assert snap.camera_orientation == "front"

    def test_start_session_by_index_int(self):
        mock_estimator = MagicMock()
        controller = ExerciseSessionController(pose_estimator=mock_estimator)
        snap = controller.start_session(0, side="left")

        assert controller.status == SessionStatus.ACTIVE
        assert snap.exercise_id == "shoulder_abduction"
        assert snap.selected_side == "left"

    def test_start_session_by_index_str(self):
        mock_estimator = MagicMock()
        controller = ExerciseSessionController(pose_estimator=mock_estimator)
        # "1" corresponds to 0th exercise (1-based index)
        snap = controller.start_session("1", side="right")

        assert controller.status == SessionStatus.ACTIVE
        assert snap.exercise_id == "shoulder_abduction"

    def test_start_session_case_insensitive(self):
        mock_estimator = MagicMock()
        controller = ExerciseSessionController(pose_estimator=mock_estimator)
        snap = controller.start_session("seated knee extension", side="LEFT")

        assert snap.exercise_id == "seated_knee_extension"
        assert snap.selected_side == "left"

    def test_start_session_all_registered_exercises(self):
        exercises = EXERCISE_REGISTRY
        mock_estimator = MagicMock()
        controller = ExerciseSessionController(pose_estimator=mock_estimator)

        for ex in exercises:
            snap = controller.start_session(ex.name, side="right")
            assert snap.exercise_name == ex.name

    def test_invalid_side_raises_and_sets_error_status(self):
        mock_estimator = MagicMock()
        controller = ExerciseSessionController(pose_estimator=mock_estimator)

        with pytest.raises(ValueError, match="Invalid side"):
            controller.start_session("Shoulder Abduction", side="center")

        assert controller.status == SessionStatus.ERROR
        snap = controller.get_snapshot()
        assert snap.status == "error"
        assert "Invalid side" in snap.last_error

    def test_invalid_exercise_name_raises_and_sets_error_status(self):
        mock_estimator = MagicMock()
        controller = ExerciseSessionController(pose_estimator=mock_estimator)

        with pytest.raises(ValueError, match="Invalid exercise identifier"):
            controller.start_session("NonExistentExercise", side="right")

        assert controller.status == SessionStatus.ERROR
        snap = controller.get_snapshot()
        assert snap.status == "error"
        assert "Invalid exercise identifier" in snap.last_error

    def test_invalid_exercise_index_raises_and_sets_error_status(self):
        mock_estimator = MagicMock()
        controller = ExerciseSessionController(pose_estimator=mock_estimator)

        with pytest.raises(ValueError, match="Invalid exercise identifier"):
            controller.start_session(99, side="right")

        assert controller.status == SessionStatus.ERROR


# ==============================================================================
# 3. Headless Frame Processing & Overlay
# ==============================================================================

class TestFrameProcessing:
    """Tests processing single frames through the controller pipeline."""

    def test_process_frame_without_active_session_raises(self):
        mock_estimator = MagicMock()
        controller = ExerciseSessionController(pose_estimator=mock_estimator)
        dummy_frame = make_dummy_frame()

        with pytest.raises(RuntimeError, match="Session is not active"):
            controller.process_frame(dummy_frame)

    def test_process_frame_no_pose_detected(self):
        mock_estimator = MagicMock()
        mock_estimator.process.return_value = make_mock_pose_result(detected=False)

        controller = ExerciseSessionController(pose_estimator=mock_estimator)
        controller.start_session("Shoulder Abduction", side="right")

        result = controller.process_frame(make_dummy_frame())

        assert isinstance(result, FrameProcessingResult)
        assert result.pose_detected is False
        assert result.joint_angle is None
        assert result.rep_phase in ("waiting_for_start", "down")
        assert result.snapshot.is_pose_detected is False
        assert result.snapshot.joint_angle is None

    def test_process_frame_pose_detected_computes_angle(self):
        mock_estimator = MagicMock()
        # Arm straight down: shoulder at (0.5, 0.4), elbow at (0.5, 0.6)
        coords = {
            12: (0.5, 0.4),  # RIGHT_SHOULDER
            14: (0.5, 0.6),  # RIGHT_ELBOW
            11: (0.3, 0.4),  # LEFT_SHOULDER
            23: (0.3, 0.8),  # LEFT_HIP
            24: (0.5, 0.8),  # RIGHT_HIP
            0: (0.4, 0.2),   # NOSE
        }
        mock_estimator.process.return_value = make_mock_pose_result(coords, detected=True)

        controller = ExerciseSessionController(pose_estimator=mock_estimator)
        controller.start_session("Shoulder Abduction", side="right")

        result = controller.process_frame(make_dummy_frame())

        assert result.pose_detected is True
        assert result.joint_angle is not None
        assert abs(result.joint_angle - 0.0) < 5.0
        assert result.snapshot.joint_angle is not None

    def test_process_frame_with_overlay_disabled(self):
        mock_estimator = MagicMock()
        mock_estimator.process.return_value = make_mock_pose_result(detected=False)

        controller = ExerciseSessionController(pose_estimator=mock_estimator)
        controller.start_session("Shoulder Abduction", side="right")

        input_frame = make_dummy_frame()
        result = controller.process_frame(input_frame, render_overlay=False)

        # Frame should be returned unmodified
        np.testing.assert_array_equal(result.annotated_frame, input_frame)

    def test_process_frame_with_overlay_enabled(self):
        mock_estimator = MagicMock()
        mock_estimator.process.return_value = make_mock_pose_result(detected=False)

        controller = ExerciseSessionController(pose_estimator=mock_estimator)
        controller.start_session("Shoulder Abduction", side="right")

        input_frame = make_dummy_frame()
        result = controller.process_frame(input_frame, render_overlay=True)

        assert result.annotated_frame is not None
        assert result.annotated_frame.shape == input_frame.shape


# ==============================================================================
# 4. Structured Snapshot Contract
# ==============================================================================

class TestStructuredSnapshot:
    """Tests the structured snapshot contract, missing values as None, and JSON safety."""

    def test_idle_snapshot_defaults(self):
        controller = ExerciseSessionController(pose_estimator=MagicMock())
        snap = controller.get_snapshot()

        assert snap.status == "idle"
        assert snap.exercise_name is None
        assert snap.exercise_id is None
        assert snap.selected_side is None
        assert snap.joint_angle is None
        assert snap.rom_progress is None
        assert snap.quality is None
        assert snap.last_error is None
        assert snap.rep_count == 0
        assert snap.is_pose_detected is False

    def test_missing_measurements_are_none_not_zero(self):
        mock_estimator = MagicMock()
        mock_estimator.process.return_value = make_mock_pose_result(detected=False)

        controller = ExerciseSessionController(pose_estimator=mock_estimator)
        controller.start_session("Shoulder Abduction", side="right")
        controller.process_frame(make_dummy_frame())

        snap = controller.get_snapshot()
        # Verify explicitly that missing angles/rom/quality are None, not 0 or 0.0
        assert snap.joint_angle is None
        assert snap.rom_progress is None
        assert snap.quality is None

    def test_snapshot_to_dict_is_json_serializable(self):
        controller = ExerciseSessionController(pose_estimator=MagicMock())
        controller.start_session("Shoulder Abduction", side="right")
        snap = controller.get_snapshot()

        snap_dict = snap.to_dict()
        assert isinstance(snap_dict, dict)

        # Must serialize cleanly to JSON string
        json_str = json.dumps(snap_dict)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["exercise_name"] == "Shoulder Abduction"
        assert parsed["status"] == "active"
        assert parsed["joint_angle"] is None


# ==============================================================================
# 5. Headless Calibration
# ==============================================================================

class TestHeadlessCalibration:
    """Tests calibration accumulation, success, and recalibration baseline protection."""

    def test_calibration_success(self):
        mock_estimator = MagicMock()
        # Define complete body landmarks for calibration
        coords = {
            0: (0.5, 0.1),   # NOSE
            11: (0.4, 0.3),  # LEFT_SHOULDER
            12: (0.6, 0.3),  # RIGHT_SHOULDER
            23: (0.4, 0.7),  # LEFT_HIP
            24: (0.6, 0.7),  # RIGHT_HIP
        }
        mock_estimator.process.return_value = make_mock_pose_result(coords, detected=True)

        controller = ExerciseSessionController(pose_estimator=mock_estimator)
        controller.start_session("Shoulder Abduction", side="right")

        # Start calibration expecting min 5 samples
        controller.start_calibration(duration_seconds=10.0, min_samples=5)
        assert controller.status == SessionStatus.CALIBRATING
        assert controller.calibration_status == "calibrating"

        for _ in range(6):
            res = controller.process_frame(make_dummy_frame())
            assert res.calibration_progress is not None

        # Finish calibration
        cal_res = controller.finish_calibration()
        assert cal_res is not None
        assert controller.calibration_status == "calibrated"
        assert controller.status == SessionStatus.ACTIVE

        snap = controller.get_snapshot()
        assert snap.calibration_status == "calibrated"
        assert snap.calibration_baseline is not None
        assert "spine_angle" in snap.calibration_baseline

    def test_calibration_insufficient_samples_fails(self):
        mock_estimator = MagicMock()
        coords = {
            0: (0.5, 0.1),
            11: (0.4, 0.3),
            12: (0.6, 0.3),
            23: (0.4, 0.7),
            24: (0.6, 0.7),
        }
        mock_estimator.process.return_value = make_mock_pose_result(coords, detected=True)

        controller = ExerciseSessionController(pose_estimator=mock_estimator)
        controller.start_session("Shoulder Abduction", side="right")

        controller.start_calibration(duration_seconds=10.0, min_samples=10)
        # Only process 3 frames
        for _ in range(3):
            controller.process_frame(make_dummy_frame())

        cal_res = controller.finish_calibration()
        assert cal_res is None
        assert controller.calibration_status == "failed"
        snap = controller.get_snapshot()
        assert snap.calibration_status == "failed"
        assert snap.calibration_error is not None
        assert "valid samples collected" in snap.calibration_error

    def test_recalibration_failure_preserves_previous_baseline(self):
        """Failed recalibration preserves the previously established valid baseline."""
        mock_estimator = MagicMock()
        controller = ExerciseSessionController(pose_estimator=mock_estimator)
        controller.start_session("Shoulder Abduction", side="right")

        # Set an initial successful baseline
        initial_baseline = make_sample_calibration_result()
        controller.set_calibration(initial_baseline)
        assert controller.calibration_result is initial_baseline
        assert controller.calibration_status == "calibrated"

        # Attempt recalibration with no frames processed
        controller.start_calibration(duration_seconds=5.0, min_samples=10)
        failed_res = controller.finish_calibration()

        assert failed_res is None
        # Previous baseline MUST be preserved
        assert controller.calibration_result is initial_baseline
        assert controller.calibration_status == "calibrated"
        snap = controller.get_snapshot()
        assert snap.calibration_status == "calibrated"
        assert "Recalibration failed" in snap.calibration_error

    def test_calibration_accumulator_step_directly(self):
        """Direct test of the CalibrationAccumulator helper."""
        acc = CalibrationAccumulator(duration_seconds=1.0, min_samples=3)
        coords = {
            0: (0.5, 0.1),
            11: (0.4, 0.3),
            12: (0.6, 0.3),
            23: (0.4, 0.7),
            24: (0.6, 0.7),
        }
        lms = make_mock_landmarks(coords)

        # Step 1: None landmarks
        acc.step(None, 0.0)
        assert acc.valid_samples == 0

        # Step 2: Valid landmarks
        acc.step(lms, 0.2)
        assert acc.valid_samples == 1

        # Step 3: Valid landmarks
        acc.step(lms, 0.4)
        acc.step(lms, 0.6)

        result = acc.finish()
        assert result is not None
        assert isinstance(result, CalibrationResult)


# ==============================================================================
# 6. Repetition & Quality Integration
# ==============================================================================

class TestRepetitionAndQuality:
    """Tests rep phase transitions and quality evaluation within the controller."""

    def test_rep_completion_updates_count_and_quality(self):
        mock_estimator = MagicMock()
        controller = ExerciseSessionController(pose_estimator=mock_estimator)
        controller.start_session("Shoulder Abduction", side="right")

        # Sequence of angles simulating a repetition:
        # rest (0°) -> rising (45°) -> peak (95°) -> lowering (45°) -> rest (0°)
        def make_arm_lms(deg):
            rad = np.radians(deg)
            # shoulder at (0.5, 0.5), elbow displaced by length 0.3
            ex = 0.5 + 0.3 * np.sin(rad)
            ey = 0.5 + 0.3 * np.cos(rad)
            return {
                12: (0.5, 0.5),  # RIGHT_SHOULDER
                14: (float(ex), float(ey)),  # RIGHT_ELBOW
                11: (0.3, 0.5),  # LEFT_SHOULDER
                23: (0.3, 0.8),  # LEFT_HIP
                24: (0.5, 0.8),  # RIGHT_HIP
                0: (0.4, 0.2),   # NOSE
            }

        angle_steps = [
            0, 0, 0,        # REST
            40, 60, 95,     # RISING / PEAK
            60, 40,         # LOWERING
            5, 0, 0         # REST (Rep completed!)
        ]

        completed_rep = None
        for deg in angle_steps:
            mock_estimator.process.return_value = make_mock_pose_result(make_arm_lms(deg))
            res = controller.process_frame(make_dummy_frame())
            if res.completed_rep is not None:
                completed_rep = res.completed_rep

        assert completed_rep is not None
        assert controller.rep_count == 1
        assert controller.last_quality_result is not None
        assert controller.last_quality_result.overall_score > 0
        snap = controller.get_snapshot()
        assert snap.rep_count == 1
        assert snap.quality is not None


# ==============================================================================
# 7. State Reset Across Sessions
# ==============================================================================

class TestSessionReset:
    """Tests that starting a new session resets all per-session state."""

    def test_reset_session_data_clears_all_counters(self):
        mock_estimator = MagicMock()
        controller = ExerciseSessionController(pose_estimator=mock_estimator)
        controller.start_session("Shoulder Abduction", side="right")

        # Fake some state
        controller._rep_count = 5
        controller._active_feedback = ["Keep your shoulders level"]
        controller._frame_index = 100

        # Reset
        controller.reset_session_data()

        assert controller.rep_count == 0
        assert controller.frame_index == 0
        assert len(controller.active_feedback) == 0
        assert controller.last_quality_result is None

    def test_new_session_resets_previous_session_state(self):
        mock_estimator = MagicMock()
        controller = ExerciseSessionController(pose_estimator=mock_estimator)
        controller.start_session("Shoulder Abduction", side="right")

        # Mutate state
        controller._rep_count = 3
        controller._active_feedback = ["Test feedback"]

        # Start new session with different exercise
        snap = controller.start_session("Seated Knee Extension", side="left")

        assert snap.exercise_name == "Seated Knee Extension"
        assert snap.selected_side == "left"
        assert snap.rep_count == 0
        assert len(snap.feedback) == 0
        assert controller.rep_count == 0


# ==============================================================================
# 8. Observable Error Handling
# ==============================================================================

class TestErrorHandling:
    """Tests that processing exceptions are caught and surfaced without crashing."""

    def test_estimator_failure_surfaces_in_status_and_snapshot(self):
        mock_estimator = MagicMock()
        mock_estimator.process.side_effect = RuntimeError("MediaPipe pipeline crash")

        controller = ExerciseSessionController(pose_estimator=mock_estimator)
        controller.start_session("Shoulder Abduction", side="right")

        # process_frame should not crash the caller
        result = controller.process_frame(make_dummy_frame())

        assert controller.status == SessionStatus.ERROR
        assert controller.last_error is not None
        assert "MediaPipe pipeline crash" in controller.last_error

        snap = controller.get_snapshot()
        assert snap.status == "error"
        assert "MediaPipe pipeline crash" in snap.last_error


# ==============================================================================
# 9. CLI Compatibility Layer
# ==============================================================================

class TestCliCompatibility:
    """Tests that the CLI layer functions and delegates to the controller correctly."""

    def test_cli_functions_are_importable(self):
        from src.main import (
            FrameSample,
            _attempt_calibration,
            _draw_debug_overlay,
            get_debug_landmark_info,
            select_exercise,
            select_side,
        )

        assert callable(select_exercise)
        assert callable(select_side)
        assert callable(_attempt_calibration)
        assert callable(_draw_debug_overlay)
        assert callable(get_debug_landmark_info)
        assert FrameSample is not None

    def test_cli_select_exercise_and_side_mock(self):
        from src.main import select_exercise, select_side

        with patch("builtins.input", return_value="1"):
            exercise = select_exercise()
            assert exercise.name == "Shoulder Abduction"

        with patch("builtins.input", return_value="right"):
            side = select_side()
            assert side == "right"

    def test_main_cli_execution_with_immediate_quit(self):
        """CLI main() can run end-to-end with mocked camera and immediate 'q' key."""
        from src.main import main

        mock_camera = MagicMock()
        mock_camera.is_opened.return_value = True
        mock_camera.read.return_value = (True, make_dummy_frame())

        mock_estimator = MagicMock()
        mock_estimator.process.return_value = make_mock_pose_result(detected=False)

        # Mock dependencies in src.main
        with patch("src.main.select_exercise") as mock_sel_ex, \
             patch("src.main.select_side", return_value="right"), \
             patch("src.main.Camera", return_value=mock_camera), \
             patch("src.main.PoseEstimator", return_value=mock_estimator), \
             patch("src.main._attempt_calibration", return_value=None), \
             patch("src.main.cv2") as mock_cv2:

            mock_sel_ex.return_value = SHOULDER_ABDUCTION
            # Simulate user pressing 'q' on first frame
            mock_cv2.waitKey.return_value = ord("q")

            # Run main()
            main()

            mock_camera.release.assert_called_once()
            mock_cv2.destroyAllWindows.assert_called_once()


# ==============================================================================
# 10. Exercise Selection Regression Suite
# ==============================================================================

class TestExerciseSelectionLifecycle:
    """Regression tests verifying exercise selection and persistence across the lifecycle."""

    @pytest.mark.parametrize(
        "exercise_slug,expected_name,expected_joint",
        [
            ("shoulder_abduction", "Shoulder Abduction", "shoulder"),
            ("seated_knee_extension", "Seated Knee Extension", "knee"),
            ("standing_hip_abduction", "Standing Hip Abduction", "hip"),
            ("elbow_flexion_extension", "Elbow Flexion-Extension", "elbow"),
            ("sit_to_stand", "Sit to Stand", "knee"),
        ],
    )
    def test_start_session_all_five_exercise_slugs(
        self, exercise_slug, expected_name, expected_joint
    ):
        """1-3. start_session accepts all 5 IDs, associates implementations, and reports in snapshot."""
        mock_estimator = MagicMock()
        controller = ExerciseSessionController(pose_estimator=mock_estimator)
        snap = controller.start_session(exercise_slug, side="right")

        assert controller.exercise is not None
        assert controller.exercise.name == expected_name
        assert controller.exercise.target_joint == expected_joint

        assert snap.exercise_id == exercise_slug
        assert snap.exercise_name == expected_name
        assert snap.status == "active"

    def test_calibration_preserves_active_exercise_id(self):
        """4. Calibration does not change the active exercise ID."""
        mock_estimator = MagicMock()
        controller = ExerciseSessionController(pose_estimator=mock_estimator)
        controller.start_session("seated_knee_extension", side="left")
        assert controller.exercise.name == "Seated Knee Extension"

        controller.start_calibration(duration_seconds=1.0, min_samples=5)
        assert controller.status == SessionStatus.CALIBRATING
        snap = controller.get_snapshot()
        assert snap.exercise_id == "seated_knee_extension"
        assert snap.exercise_name == "Seated Knee Extension"
        assert snap.side == "left"

    def test_reset_session_data_preserves_active_exercise(self):
        """5. Resetting session measurements does not unexpectedly change selected exercise."""
        mock_estimator = MagicMock()
        controller = ExerciseSessionController(pose_estimator=mock_estimator)
        controller.start_session("sit_to_stand", side="right")
        assert controller.exercise.name == "Sit to Stand"

        controller._rep_count = 4
        controller._frame_index = 50
        controller._active_feedback = ["Drive through your heels"]

        controller.reset_session_data()

        assert controller.rep_count == 0
        assert controller.frame_index == 0
        assert len(controller.active_feedback) == 0
        assert controller.exercise.name == "Sit to Stand"
        snap = controller.get_snapshot()
        assert snap.exercise_id == "sit_to_stand"
        assert snap.exercise_name == "Sit to Stand"

    def test_invalid_exercise_id_raises_validation_error(self):
        """6. Invalid exercise IDs produce the expected validation error without falling back to default."""
        mock_estimator = MagicMock()
        controller = ExerciseSessionController(pose_estimator=mock_estimator)

        with pytest.raises(ValueError, match="Invalid exercise identifier"):
            controller.start_session("unsupported_exercise", side="right")

        assert controller.status == SessionStatus.ERROR
        snap = controller.get_snapshot()
        assert snap.status == "error"
        assert snap.exercise_id is None
        assert snap.exercise_name is None
        assert "Invalid exercise identifier" in snap.last_error

    def test_shared_registry_normalization(self):
        """7. Test shared registry normalization and resolution for all 5 exercises."""
        from src.session import resolve_exercise

        slugs = [
            ("shoulder_abduction", "Shoulder Abduction"),
            ("seated_knee_extension", "Seated Knee Extension"),
            ("standing_hip_abduction", "Standing Hip Abduction"),
            ("elbow_flexion_extension", "Elbow Flexion-Extension"),
            ("sit_to_stand", "Sit to Stand"),
        ]
        for slug, expected_name in slugs:
            resolved = resolve_exercise(slug)
            assert resolved.name == expected_name

        with pytest.raises(ValueError, match="Unknown exercise"):
            resolve_exercise("random_nonexistent_exercise")

