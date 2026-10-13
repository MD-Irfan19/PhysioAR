"""Session runtime module for PhysioAR.

Provides a thread-safe, non-blocking background runtime (SessionRuntime) that
coordinates:
- Camera hardware acquisition in a dedicated background worker thread.
- Continuous frame analysis via ExerciseSessionController.
- Thread-safe, non-blocking access to the latest SessionSnapshot and annotated frame.
- Calibration requests coordinated through the worker thread.
- Robust error handling (camera disconnection, read failures, processing exceptions).
- Idempotent lifecycle management (start, stop, restart) with unambiguous resource ownership.

This module is designed to be consumed by FastAPI / WebSocket / SSE backends in Phase C
without holding global locks during camera I/O or inference.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import numpy as np

from src.acquisition import Camera
from src.config import (
    CALIBRATION_SECONDS,
    LANDMARK_VISIBILITY_THRESHOLD,
    MIN_CALIBRATION_SAMPLES,
)
from src.exercises.base import ExerciseDefinition
from src.session import (
    ExerciseSessionController,
    FrameProcessingResult,
    SessionSnapshot,
    SessionStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class CalibrationRequest:
    """Internal message describing a calibration initiation request."""

    duration_seconds: float = CALIBRATION_SECONDS
    min_samples: int = MIN_CALIBRATION_SAMPLES
    visibility_threshold: float = LANDMARK_VISIBILITY_THRESHOLD
    completed_event: Optional[threading.Event] = None
    success: Optional[bool] = None
    error_message: Optional[str] = None


class SessionRuntime:
    """Coordinates camera capture, background processing, and thread-safe session state.

    The runtime decouples camera acquisition and MediaPipe/exercise analysis from
    presentation and web API route handlers. A dedicated background thread continuously
    captures frames and updates the controller. External callers can read the latest
    snapshot and annotated frame without waiting for inference or camera I/O.

    Resource Ownership:
    - If Camera is instantiated internally by the runtime, the runtime owns it and releases it on stop.
    - If Camera is injected with owns_camera=False, the runtime will not release it on stop.
    - If ExerciseSessionController is instantiated internally, the runtime owns it and stops it on stop.
    - If ExerciseSessionController is injected with owns_controller=False, the runtime will not stop it.
    """

    def __init__(
        self,
        camera: Optional[Camera] = None,
        controller: Optional[ExerciseSessionController] = None,
        camera_factory: Optional[Callable[[int], Camera]] = None,
        controller_factory: Optional[Callable[[], ExerciseSessionController]] = None,
        owns_camera: Optional[bool] = None,
        owns_controller: Optional[bool] = None,
        max_consecutive_read_failures: int = 30,
        read_retry_sleep_seconds: float = 0.01,
    ) -> None:
        """Initialize SessionRuntime.

        Args:
            camera: Optional injected Camera instance.
            controller: Optional injected ExerciseSessionController instance.
            camera_factory: Optional callable (int -> Camera) used to instantiate cameras.
            controller_factory: Optional callable (() -> ExerciseSessionController) used to create controllers.
            owns_camera: Whether this runtime owns and should release the camera. Defaults to True if
                instantiated internally, False if injected.
            owns_controller: Whether this runtime owns and should stop the controller. Defaults to True if
                instantiated internally, False if injected.
            max_consecutive_read_failures: Maximum consecutive frame-read failures before marking runtime error.
            read_retry_sleep_seconds: Sleep duration in seconds between failed camera reads to prevent CPU spinning.
        """
        self._camera = camera
        self._controller = controller
        self._camera_factory = camera_factory or (lambda idx: Camera(camera_index=idx))
        self._controller_factory = controller_factory or (lambda: ExerciseSessionController(camera=None))

        self._owns_camera = owns_camera if owns_camera is not None else (camera is None)
        self._owns_controller = owns_controller if owns_controller is not None else (controller is None)

        self._max_consecutive_read_failures = max_consecutive_read_failures
        self._read_retry_sleep_seconds = read_retry_sleep_seconds

        # Concurrency & synchronization primitives
        self._state_lock = threading.Lock()
        self._controller_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._frame_condition = threading.Condition(self._state_lock)

        # Background worker thread
        self._worker_thread: Optional[threading.Thread] = None
        self._is_running = False

        # Published state (protected by _state_lock)
        self._latest_snapshot: Optional[SessionSnapshot] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._runtime_error: Optional[str] = None
        self._debug_overlay_enabled: bool = False
        self._frame_count: int = 0

        # Calibration state
        self._active_calibration_request: Optional[CalibrationRequest] = None

    @property
    def controller(self) -> Optional[ExerciseSessionController]:
        """Access the underlying controller instance."""
        return self._controller

    @property
    def camera(self) -> Optional[Camera]:
        """Access the underlying camera instance."""
        return self._camera

    def is_running(self) -> bool:
        """Check whether the background worker is currently active."""
        with self._state_lock:
            worker_alive = self._worker_thread is not None and self._worker_thread.is_alive()
            return self._is_running or worker_alive

    def get_error(self) -> Optional[str]:
        """Retrieve the most recent runtime error message, if any."""
        with self._state_lock:
            return self._runtime_error

    def get_snapshot(self) -> Optional[SessionSnapshot]:
        """Thread-safe access to the latest SessionSnapshot.

        Does not block waiting for camera acquisition or inference. Returns None
        if no session is active or no frame has been processed yet.
        """
        with self._state_lock:
            return self._latest_snapshot

    def get_latest_frame(self, copy: bool = True) -> Optional[np.ndarray]:
        """Thread-safe access to the latest annotated BGR frame.

        Args:
            copy: If True (default), returns an isolated copy of the frame to ensure
                consumers cannot mutate the internal frame buffer. If False, returns
                the internal reference for zero-copy read-only operations.

        Returns:
            NumPy BGR array or None if no frame has been captured yet.
        """
        with self._state_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy() if copy else self._latest_frame

    def set_debug_overlay(self, enabled: bool) -> None:
        """Toggle landmark diagnostic overlay rendering in the background worker."""
        with self._state_lock:
            self._debug_overlay_enabled = enabled

    def get_debug_overlay(self) -> bool:
        """Check whether the landmark diagnostic overlay is currently enabled."""
        with self._state_lock:
            return self._debug_overlay_enabled

    def wait_for_next_frame(self, timeout: Optional[float] = None) -> bool:
        """Wait until at least one new frame is processed by the background worker.

        Useful for test synchronization and clients waiting for the next update.

        Args:
            timeout: Maximum wait time in seconds.

        Returns:
            True if a new frame was processed, False if timed out.
        """
        with self._state_lock:
            start_count = self._frame_count
            return self._frame_condition.wait_for(
                lambda: self._frame_count > start_count or not self._is_running,
                timeout=timeout,
            )

    def start_session(
        self,
        exercise: Union[ExerciseDefinition, str, int],
        side: str = "right",
        camera_index: int = 0,
    ) -> SessionSnapshot:
        """Start a new exercise session and launch the background worker.

        Validates inputs, initializes resources, cleanly clears any stale state from
        prior sessions, opens the camera, and spawns the background worker thread.

        Args:
            exercise: ExerciseDefinition instance, name string, or registry index.
            side: "left" or "right" indicating tracked arm/leg.
            camera_index: Webcam device index (default: 0).

        Returns:
            Initial SessionSnapshot representing the newly started session.

        Raises:
            RuntimeError: If a session is already running or if the camera cannot be opened.
            ValueError: If the exercise or side is invalid.
        """
        with self._lifecycle_lock:
            if self.is_running():
                raise RuntimeError(
                    "Session is already running. Call stop_session() before starting a new session."
                )

            # 1. Cleanly clear any stale session artifacts
            with self._state_lock:
                self._latest_frame = None
                self._latest_snapshot = None
                self._runtime_error = None
                self._frame_count = 0
                self._active_calibration_request = None

            self._stop_event.clear()

            # 2. Instantiate or reset controller
            if self._controller is None:
                self._controller = self._controller_factory()
                self._owns_controller = True

            # start_session on controller validates exercise & side
            try:
                initial_snapshot = self._controller.start_session(exercise, side=side)
            except Exception as e:
                with self._state_lock:
                    self._runtime_error = str(e)
                raise

            with self._state_lock:
                self._latest_snapshot = initial_snapshot

            # 3. Instantiate and open camera
            if self._camera is None:
                self._camera = self._camera_factory(camera_index)
                self._owns_camera = True

            try:
                self._camera.open()
            except Exception as e:
                error_msg = f"Failed to open camera: {e}"
                logger.error(error_msg)
                with self._state_lock:
                    self._runtime_error = error_msg
                if self._owns_camera and self._camera is not None:
                    self._camera.release()
                    self._camera = None
                if self._owns_controller and self._controller is not None:
                    self._controller.stop_session()
                    self._controller = None
                raise RuntimeError(error_msg) from e

            # 4. Launch background worker thread
            with self._state_lock:
                self._is_running = True

            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name="PhysioAR-RuntimeWorker",
                daemon=True,
            )
            self._worker_thread.start()

            logger.info(
                f"Session started for {initial_snapshot.exercise_name} ({initial_snapshot.side})"
            )
            return initial_snapshot

    def request_calibration(
        self,
        duration_seconds: float = CALIBRATION_SECONDS,
        min_samples: int = MIN_CALIBRATION_SAMPLES,
        visibility_threshold: float = LANDMARK_VISIBILITY_THRESHOLD,
        wait: bool = False,
        timeout: Optional[float] = None,
    ) -> bool:
        """Request neutral posture calibration or recalibration.

        Initiates calibration through the controller and coordinates progress.

        Args:
            duration_seconds: Duration to collect calibration samples.
            min_samples: Minimum valid samples required.
            visibility_threshold: Landmark confidence threshold.
            wait: If True, blocks until calibration completes or times out.
            timeout: Maximum seconds to wait when wait=True.

        Returns:
            True if request was accepted (and succeeded if wait=True), False otherwise.

        Raises:
            RuntimeError: If no session is currently running.
        """
        if not self.is_running():
            raise RuntimeError("Cannot request calibration: session is not running.")

        req = CalibrationRequest(
            duration_seconds=duration_seconds,
            min_samples=min_samples,
            visibility_threshold=visibility_threshold,
            completed_event=threading.Event() if wait else None,
        )

        with self._controller_lock:
            if self._controller is not None:
                try:
                    self._controller.start_calibration(
                        duration_seconds=duration_seconds,
                        min_samples=min_samples,
                        visibility_threshold=visibility_threshold,
                    )
                    with self._state_lock:
                        self._latest_snapshot = self._controller.get_snapshot()
                        self._active_calibration_request = req
                except Exception as e:
                    logger.error(f"Failed to start calibration: {e}")
                    req.success = False
                    req.error_message = str(e)
                    if req.completed_event:
                        req.completed_event.set()
                    return False

        if wait and req.completed_event is not None:
            signaled = req.completed_event.wait(timeout=timeout)
            if not signaled:
                logger.warning("Calibration request timed out waiting for completion.")
                return False
            return req.success is True

        return True

    def stop_session(self, timeout: float = 2.0) -> None:
        """Stop the session and cleanly release owned resources.

        Idempotent: safe to call multiple times or when no session is active.

        Args:
            timeout: Maximum seconds to wait for worker thread termination.
        """
        with self._lifecycle_lock:
            # Signal stop to background worker
            self._stop_event.set()

            with self._state_lock:
                self._frame_condition.notify_all()

            # Wait for worker thread to exit
            if self._worker_thread is not None and self._worker_thread.is_alive():
                if threading.current_thread() != self._worker_thread:
                    self._worker_thread.join(timeout=timeout)
                    if self._worker_thread.is_alive():
                        error_msg = (
                            f"Worker thread did not terminate within {timeout}s timeout."
                        )
                        logger.error(error_msg)
                        with self._state_lock:
                            self._runtime_error = error_msg
                        raise TimeoutError(error_msg)

            self._worker_thread = None

            # Release camera if owned
            if self._owns_camera and self._camera is not None:
                try:
                    self._camera.release()
                except Exception as e:
                    logger.warning(f"Error releasing camera: {e}")
                self._camera = None

            # Stop controller if owned
            if self._owns_controller and self._controller is not None:
                try:
                    with self._controller_lock:
                        self._controller.stop_session()
                except Exception as e:
                    logger.warning(f"Error stopping controller: {e}")
                self._controller = None

            with self._state_lock:
                self._is_running = False
                self._frame_condition.notify_all()

            logger.info("Session stopped cleanly.")

    def _worker_loop(self) -> None:
        """Main background worker loop executing continuous capture and analysis."""
        consecutive_read_failures = 0
        has_logged_read_failure = False

        while not self._stop_event.is_set():
            # Step 1: Camera Capture (NO LOCK HELD)
            if self._camera is None:
                break

            try:
                success, frame = self._camera.read()
            except Exception as e:
                success, frame = False, None
                if not has_logged_read_failure:
                    logger.error(f"Camera read exception: {e}")
                    has_logged_read_failure = True

            if not success or frame is None:
                consecutive_read_failures += 1
                if not has_logged_read_failure:
                    logger.warning("Camera frame read failed.")
                    has_logged_read_failure = True

                if consecutive_read_failures >= self._max_consecutive_read_failures:
                    error_msg = (
                        f"Camera stream disconnected or frame capture failed "
                        f"persistently after {consecutive_read_failures} attempts."
                    )
                    logger.error(error_msg)
                    with self._state_lock:
                        self._runtime_error = error_msg
                        if self._latest_snapshot is not None:
                            self._latest_snapshot.status = SessionStatus.ERROR
                            self._latest_snapshot.last_error = error_msg
                    break

                # Sleep briefly to avoid 100% CPU spin while camera is unavailable
                time.sleep(self._read_retry_sleep_seconds)
                continue

            # Successful read resets failure counter
            if consecutive_read_failures > 0:
                consecutive_read_failures = 0
                has_logged_read_failure = False

            # Step 2: Frame Processing (PROTECTED BY CONTROLLER LOCK, NO STATE LOCK)
            with self._state_lock:
                draw_debug = self._debug_overlay_enabled

            if self._controller is None:
                break

            try:
                with self._controller_lock:
                    res: FrameProcessingResult = self._controller.process_frame(
                        frame,
                        render_overlay=True,
                        draw_debug=draw_debug,
                    )
            except Exception as e:
                error_msg = f"Controller processing error: {e}"
                logger.error(error_msg)
                with self._state_lock:
                    self._runtime_error = error_msg
                    if self._latest_snapshot is not None:
                        self._latest_snapshot.last_error = error_msg
                continue

            # Step 3: Check if active calibration completed during this frame
            with self._state_lock:
                active_req = self._active_calibration_request
                if active_req is not None and res.snapshot.calibration_status != "calibrating":
                    self._active_calibration_request = None
                    active_req.success = (res.snapshot.calibration_status == "calibrated")
                    active_req.error_message = res.snapshot.calibration_error
                    if active_req.completed_event:
                        active_req.completed_event.set()

            # Step 4: Atomic State Update (SHORT CRITICAL SECTION)
            with self._state_lock:
                self._latest_snapshot = res.snapshot
                self._latest_frame = res.annotated_frame
                self._frame_count += 1
                if res.snapshot.last_error:
                    self._runtime_error = res.snapshot.last_error
                self._frame_condition.notify_all()

        # Mark running false when loop exits
        with self._state_lock:
            self._is_running = False
            self._frame_condition.notify_all()
