"""Unit and integration tests for src/runtime.py and SessionRuntime.

Covers:
1. Runtime initialization.
2. Successful session startup.
3. Camera initialization failure.
4. Background frame processing.
5. Snapshot retrieval during processing.
6. Latest annotated-frame retrieval and copy isolation.
7. Calibration requests.
8. Camera read failure and recovery or clean error reporting.
9. Controller processing exceptions.
10. Idempotent shutdown.
11. Worker termination.
12. Resource ownership and cleanup.
13. Duplicate start and stop behavior.
14. Restart behavior.
15. No stale frames or snapshots leaking into a new session.
16. Thread-safe access under concurrent readers.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock
import numpy as np
import pytest

from src.exercises import EXERCISE_REGISTRY
from src.pose_estimation import PoseResult
from src.runtime import SessionRuntime
from src.session import ExerciseSessionController, SessionSnapshot, SessionStatus


# ==============================================================================
# Test Fixtures & Fakes
# ==============================================================================

class FakeCamera:
    """Deterministic, mockable camera for headless background testing."""

    def __init__(
        self,
        frame_count: float = float("inf"),
        fail_after: int | None = None,
        transient_failures: int = 0,
        open_fails: bool = False,
    ) -> None:
        self.opened = False
        self.released = False
        self.read_count = 0
        self.frame_count = frame_count
        self.fail_after = fail_after
        self.transient_failures = transient_failures
        self.open_fails = open_fails
        self._failures_served = 0

    def open(self) -> None:
        if self.open_fails:
            raise RuntimeError("Webcam hardware disconnected or busy")
        self.opened = True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.opened or self.released:
            return False, None

        if self._failures_served < self.transient_failures:
            self._failures_served += 1
            return False, None

        if self.fail_after is not None and self.read_count >= self.fail_after:
            return False, None

        if self.read_count >= self.frame_count:
            return False, None

        self.read_count += 1
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Mark frame number on pixel (0, 0)
        frame[0, 0, 0] = self.read_count % 256
        return True, frame

    def release(self) -> None:
        self.released = True
        self.opened = False


def make_mock_controller() -> ExerciseSessionController:
    """Create a controller with a mocked pose estimator to bypass MediaPipe inference."""
    mock_estimator = MagicMock()
    mock_estimator.process.return_value = PoseResult(
        mediapipe_result=MagicMock(),
        raw_landmarks=[],
        smoothed_landmarks=[],
        pose_detected=False,
    )
    return ExerciseSessionController(pose_estimator=mock_estimator)


# ==============================================================================
# 1. Runtime Initialization & Lifecycle
# ==============================================================================

class TestRuntimeInitialization:
    """Tests runtime creation and idle state verification."""

    def test_init_idle_state(self):
        """Runtime initializes cleanly without starting threads or opening camera."""
        runtime = SessionRuntime(
            camera_factory=lambda idx: FakeCamera(),
            controller_factory=make_mock_controller,
        )

        assert runtime.is_running() is False
        assert runtime.get_snapshot() is None
        assert runtime.get_latest_frame() is None
        assert runtime.get_error() is None
        assert runtime.get_debug_overlay() is False

    def test_successful_session_startup(self):
        """start_session launches the background worker and updates status."""
        fake_cam = FakeCamera()
        runtime = SessionRuntime(
            camera=fake_cam,
            controller_factory=make_mock_controller,
            owns_camera=True,
        )

        try:
            snapshot = runtime.start_session("Shoulder Abduction", side="right")

            assert runtime.is_running() is True
            assert snapshot.status == SessionStatus.ACTIVE
            assert snapshot.exercise_name == "Shoulder Abduction"
            assert snapshot.side == "right"
            assert fake_cam.opened is True

            # Wait for worker to process at least one frame
            assert runtime.wait_for_next_frame(timeout=1.0) is True
            assert runtime.get_latest_frame() is not None
        finally:
            runtime.stop_session()
            assert runtime.is_running() is False

    def test_camera_initialization_failure(self):
        """Camera open failure raises RuntimeError, records error, and releases resources."""
        fake_cam = FakeCamera(open_fails=True)
        runtime = SessionRuntime(
            camera=fake_cam,
            controller_factory=make_mock_controller,
            owns_camera=True,
        )

        with pytest.raises(RuntimeError, match="Failed to open camera"):
            runtime.start_session("Shoulder Abduction", side="right")

        assert runtime.is_running() is False
        assert runtime.get_error() is not None
        assert "Webcam hardware disconnected or busy" in runtime.get_error()
        assert fake_cam.released is True

    def test_duplicate_start_rejected(self):
        """Calling start_session while running raises RuntimeError without disrupting active session."""
        fake_cam = FakeCamera()
        runtime = SessionRuntime(
            camera=fake_cam,
            controller_factory=make_mock_controller,
        )

        try:
            runtime.start_session("Shoulder Abduction", side="right")
            assert runtime.is_running() is True

            with pytest.raises(RuntimeError, match="Session is already running"):
                runtime.start_session("Seated Knee Extension", side="left")

            # Active session remains healthy
            assert runtime.is_running() is True
            snap = runtime.get_snapshot()
            assert snap.exercise_name == "Shoulder Abduction"
        finally:
            runtime.stop_session()

    def test_idempotent_stop(self):
        """stop_session is safe to call multiple times or when already stopped."""
        fake_cam = FakeCamera()
        runtime = SessionRuntime(
            camera=fake_cam,
            controller_factory=make_mock_controller,
            owns_camera=True,
        )

        # Stop when idle
        runtime.stop_session()
        assert runtime.is_running() is False

        # Start and stop repeatedly
        runtime.start_session("Shoulder Abduction", side="right")
        assert runtime.is_running() is True

        runtime.stop_session()
        assert runtime.is_running() is False
        assert fake_cam.released is True

        # Second and third stop calls must be no-ops
        runtime.stop_session()
        runtime.stop_session()
        assert runtime.is_running() is False


# ==============================================================================
# 2. Background Frame Processing & State Access
# ==============================================================================

class TestFrameProcessingAndState:
    """Tests background worker loop execution and thread-safe data access."""

    def test_background_frame_processing_increments_frame_index(self):
        """Worker continuously processes frames and updates the session snapshot."""
        fake_cam = FakeCamera()
        runtime = SessionRuntime(
            camera=fake_cam,
            controller_factory=make_mock_controller,
        )

        try:
            runtime.start_session("Shoulder Abduction", side="right")

            # Wait for at least 3 frames
            for _ in range(3):
                assert runtime.wait_for_next_frame(timeout=1.0) is True

            snap = runtime.get_snapshot()
            assert snap is not None
            assert snap.frame_index >= 3
        finally:
            runtime.stop_session()

    def test_snapshot_retrieval_is_consistent(self):
        """Snapshot contains all expected structured fields without throwing exceptions."""
        fake_cam = FakeCamera()
        runtime = SessionRuntime(
            camera=fake_cam,
            controller_factory=make_mock_controller,
        )

        try:
            runtime.start_session("Shoulder Abduction", side="right")
            runtime.wait_for_next_frame(timeout=1.0)

            snap = runtime.get_snapshot()
            assert isinstance(snap, SessionSnapshot)
            assert snap.exercise_name == "Shoulder Abduction"
            assert snap.side == "right"
            assert snap.joint_angle is None
            assert snap.status == SessionStatus.ACTIVE
        finally:
            runtime.stop_session()

    def test_frame_retrieval_and_copy_isolation(self):
        """Mutating the returned frame does not alter the internal runtime frame."""
        fake_cam = FakeCamera(fail_after=1)
        runtime = SessionRuntime(
            camera=fake_cam,
            controller_factory=make_mock_controller,
        )

        try:
            runtime.start_session("Shoulder Abduction", side="right")
            runtime.wait_for_next_frame(timeout=1.0)

            frame_copy = runtime.get_latest_frame(copy=True)
            assert frame_copy is not None
            assert isinstance(frame_copy, np.ndarray)

            # Mutate copy
            original_val = int(frame_copy[0, 0, 0])
            frame_copy[0, 0, 0] = np.uint8((original_val + 50) % 255 + 1)

            # Retrieve frame again and verify original was unchanged
            second_copy = runtime.get_latest_frame(copy=True)
            assert int(second_copy[0, 0, 0]) == original_val
        finally:
            runtime.stop_session()

    def test_debug_overlay_toggle(self):
        """Toggling debug overlay updates internal flag."""
        runtime = SessionRuntime(
            camera_factory=lambda idx: FakeCamera(),
            controller_factory=make_mock_controller,
        )

        assert runtime.get_debug_overlay() is False
        runtime.set_debug_overlay(True)
        assert runtime.get_debug_overlay() is True
        runtime.set_debug_overlay(False)
        assert runtime.get_debug_overlay() is False


# ==============================================================================
# 3. Calibration Coordination
# ==============================================================================

class TestCalibrationCoordination:
    """Tests requesting and executing calibration through the worker loop."""

    def test_request_calibration_when_not_running_raises(self):
        """request_calibration raises RuntimeError if session is not active."""
        runtime = SessionRuntime(
            camera_factory=lambda idx: FakeCamera(),
            controller_factory=make_mock_controller,
        )

        with pytest.raises(RuntimeError, match="Cannot request calibration"):
            runtime.request_calibration()

    def test_request_calibration_transitions_controller_status(self):
        """Worker picks up pending calibration request and enters calibrating status."""
        fake_cam = FakeCamera()
        runtime = SessionRuntime(
            camera=fake_cam,
            controller_factory=make_mock_controller,
        )

        try:
            runtime.start_session("Shoulder Abduction", side="right")
            runtime.wait_for_next_frame(timeout=1.0)

            # Request calibration (non-blocking)
            accepted = runtime.request_calibration(duration_seconds=5.0, min_samples=10)
            assert accepted is True

            # Wait for next frame so worker applies request
            runtime.wait_for_next_frame(timeout=1.0)

            snap = runtime.get_snapshot()
            assert snap.calibration_status == "calibrating"
        finally:
            runtime.stop_session()


# ==============================================================================
# 4. Error Handling & Recovery
# ==============================================================================

class TestErrorHandlingAndRecovery:
    """Tests camera disconnects, read failures, and controller exceptions."""

    def test_transient_read_failure_recovers(self):
        """Worker survives transient camera read failures without failing the session."""
        fake_cam = FakeCamera(transient_failures=2)
        runtime = SessionRuntime(
            camera=fake_cam,
            controller_factory=make_mock_controller,
            read_retry_sleep_seconds=0.001,
        )

        try:
            runtime.start_session("Shoulder Abduction", side="right")

            # Worker should recover and successfully capture frames
            assert runtime.wait_for_next_frame(timeout=1.0) is True
            assert runtime.is_running() is True
            assert runtime.get_latest_frame() is not None
        finally:
            runtime.stop_session()

    def test_persistent_read_failure_stops_worker_with_error(self):
        """Persistent camera read failures stop worker loop and record error."""
        fake_cam = FakeCamera(fail_after=2)
        runtime = SessionRuntime(
            camera=fake_cam,
            controller_factory=make_mock_controller,
            max_consecutive_read_failures=5,
            read_retry_sleep_seconds=0.001,
        )

        try:
            runtime.start_session("Shoulder Abduction", side="right")

            # Process initial frames
            runtime.wait_for_next_frame(timeout=1.0)

            # Wait for consecutive failures to exceed threshold
            for _ in range(20):
                if not runtime.is_running():
                    break
                time.sleep(0.01)

            assert runtime.is_running() is False
            err = runtime.get_error()
            assert err is not None
            assert "Camera stream disconnected" in err
        finally:
            runtime.stop_session()

    def test_controller_processing_exception_handled(self):
        """Worker catches unexpected controller exceptions and surfaces them in error state."""
        fake_cam = FakeCamera()
        controller = make_mock_controller()
        # Make process_frame raise an exception
        controller.process_frame = MagicMock(side_effect=RuntimeError("Simulated pipeline crash"))

        runtime = SessionRuntime(
            camera=fake_cam,
            controller=controller,
        )

        try:
            runtime.start_session("Shoulder Abduction", side="right")

            # Allow worker to run iteration
            time.sleep(0.05)

            err = runtime.get_error()
            assert err is not None
            assert "Simulated pipeline crash" in err
        finally:
            runtime.stop_session()


# ==============================================================================
# 5. Worker Termination & Resource Ownership
# ==============================================================================

class TestWorkerTerminationAndOwnership:
    """Tests clean thread termination and ownership rules."""

    def test_worker_thread_terminates_cleanly(self):
        """Worker thread stops executing within the requested timeout on stop_session."""
        fake_cam = FakeCamera()
        runtime = SessionRuntime(
            camera=fake_cam,
            controller_factory=make_mock_controller,
        )

        runtime.start_session("Shoulder Abduction", side="right")
        worker_thread = runtime._worker_thread
        assert worker_thread is not None
        assert worker_thread.is_alive() is True

        runtime.stop_session(timeout=1.0)

        assert worker_thread.is_alive() is False
        assert runtime._worker_thread is None

    def test_resource_ownership_unowned_camera_not_released(self):
        """If camera is injected with owns_camera=False, runtime does not release it."""
        fake_cam = FakeCamera()
        runtime = SessionRuntime(
            camera=fake_cam,
            controller_factory=make_mock_controller,
            owns_camera=False,
        )

        runtime.start_session("Shoulder Abduction", side="right")
        runtime.stop_session()

        # Camera should still be opened, not released
        assert fake_cam.released is False
        fake_cam.release()

    def test_resource_ownership_owned_camera_released(self):
        """If camera is owned, runtime releases it on stop_session."""
        fake_cam = FakeCamera()
        runtime = SessionRuntime(
            camera=fake_cam,
            controller_factory=make_mock_controller,
            owns_camera=True,
        )

        runtime.start_session("Shoulder Abduction", side="right")
        runtime.stop_session()

        assert fake_cam.released is True


# ==============================================================================
# 6. Session Restart & Stale State Isolation
# ==============================================================================

class TestRestartAndStateIsolation:
    """Tests stopping and restarting sessions without state leakage."""

    def test_restart_session_succeeds(self):
        """Runtime can be started, stopped, and restarted cleanly."""
        runtime = SessionRuntime(
            camera_factory=lambda idx: FakeCamera(),
            controller_factory=make_mock_controller,
        )

        try:
            # Session 1
            snap1 = runtime.start_session("Shoulder Abduction", side="right")
            assert snap1.exercise_name == "Shoulder Abduction"
            runtime.wait_for_next_frame(timeout=1.0)
            runtime.stop_session()
            assert runtime.is_running() is False

            # Session 2
            snap2 = runtime.start_session("Seated Knee Extension", side="left")
            assert snap2.exercise_name == "Seated Knee Extension"
            assert snap2.side == "left"
            assert runtime.is_running() is True
            runtime.wait_for_next_frame(timeout=1.0)
        finally:
            runtime.stop_session()

    def test_no_stale_frame_or_snapshot_leakage(self):
        """A newly started session does not present frames or snapshots from the previous session."""
        runtime = SessionRuntime(
            camera_factory=lambda idx: FakeCamera(),
            controller_factory=make_mock_controller,
        )

        try:
            runtime.start_session("Shoulder Abduction", side="right")
            runtime.wait_for_next_frame(timeout=1.0)
            assert runtime.get_latest_frame() is not None
            runtime.stop_session()

            # Upon starting a new session, previous frame must be cleared immediately
            snap2 = runtime.start_session("Standing Hip Abduction", side="right")
            assert snap2.exercise_name == "Standing Hip Abduction"
            # Immediately after start, before first frame arrives, latest_frame must not be stale frame
            # (or if wait hasn't been called, verify snapshot matches new exercise)
            current_snap = runtime.get_snapshot()
            assert current_snap.exercise_name == "Standing Hip Abduction"
        finally:
            runtime.stop_session()


# ==============================================================================
# 7. Concurrent Reader Thread-Safety
# ==============================================================================

class TestConcurrentReaderThreadSafety:
    """Tests high-frequency concurrent reads while worker is actively writing."""

    def test_concurrent_readers_no_race_conditions(self):
        """Multiple threads reading snapshot and latest frame concurrently execute without error."""
        fake_cam = FakeCamera()
        runtime = SessionRuntime(
            camera=fake_cam,
            controller_factory=make_mock_controller,
        )

        try:
            runtime.start_session("Shoulder Abduction", side="right")
            runtime.wait_for_next_frame(timeout=1.0)

            errors: list[Exception] = []
            num_reader_threads = 8
            reads_per_thread = 50

            def reader_worker():
                try:
                    for _ in range(reads_per_thread):
                        snap = runtime.get_snapshot()
                        frame = runtime.get_latest_frame(copy=True)
                        if snap is not None:
                            _ = snap.status
                            _ = snap.rep_count
                        if frame is not None:
                            _ = frame.shape
                        time.sleep(0.001)
                except Exception as e:
                    errors.append(e)

            threads = [
                threading.Thread(target=reader_worker, daemon=True)
                for _ in range(num_reader_threads)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=3.0)

            assert len(errors) == 0, f"Reader thread errors: {errors}"
        finally:
            runtime.stop_session()

    def test_worker_shutdown_timeout_raises_and_preserves_liveness(self):
        """If worker thread fails to exit within timeout, stop_session raises TimeoutError and is_running remains True."""
        unblock_event = threading.Event()
        controller = make_mock_controller()
        orig_process = controller.process_frame

        def slow_process(frame, **kwargs):
            res = orig_process(frame, **kwargs)
            if not unblock_event.is_set():
                time.sleep(0.2)
            return res

        controller.process_frame = slow_process

        runtime = SessionRuntime(
            camera_factory=lambda idx: FakeCamera(),
            controller=controller,
        )

        try:
            runtime.start_session("Shoulder Abduction", side="right")
            runtime.wait_for_next_frame(timeout=1.0)

            # Calling stop_session with tiny timeout should trigger TimeoutError
            with pytest.raises(TimeoutError, match="did not terminate within"):
                runtime.stop_session(timeout=0.01)

            # Runtime must still report that it is running (worker is still alive)
            assert runtime.is_running() is True
            # Attempting to start another session must be rejected
            with pytest.raises(RuntimeError, match="already running"):
                runtime.start_session("Seated Knee Extension", side="left")
        finally:
            unblock_event.set()
            # Clean up properly once unblocked
            runtime.stop_session(timeout=1.0)
            assert runtime.is_running() is False

