"""Camera acquisition module for PhysioAR.

This module provides a simple, reusable abstraction over OpenCV webcam
capture. It is responsible exclusively for camera access — no pose
estimation, metric calculation, or MediaPipe code belongs here.
"""

import cv2
import numpy as np


class Camera:
    """Encapsulates OpenCV webcam capture.

    Provides a clean interface for opening a webcam, reading frames,
    and releasing the camera resource.

    Args:
        camera_index: Index of the webcam device to use. Defaults to 0
            (the system's default webcam).

    Example::

        camera = Camera()
        camera.open()
        success, frame = camera.read()
        camera.release()
    """

    def __init__(self, camera_index: int = 0) -> None:
        self._camera_index = camera_index
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        """Open the webcam for frame capture.

        Creates an OpenCV VideoCapture object and verifies that the
        camera is accessible.

        Raises:
            RuntimeError: If the webcam cannot be opened. This typically
                means the camera is not connected, permissions are not
                granted, or another application is using the webcam.
        """
        self._capture = cv2.VideoCapture(self._camera_index)

        if not self._capture.isOpened():
            self._capture.release()
            self._capture = None
            raise RuntimeError(
                f"Could not open webcam at index {self._camera_index}. "
                "Check that the camera is connected, camera permissions "
                "are enabled, and no other application is currently using "
                "the webcam."
            )

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Capture a single frame from the webcam.

        Returns:
            A tuple of (success, frame) where:
                - success: True if a frame was captured, False otherwise.
                - frame: The captured BGR image as a NumPy array when
                  successful, or None when capture fails.
        """
        if self._capture is None:
            return False, None

        success, frame = self._capture.read()
        if not success:
            return False, None

        return True, frame

    def release(self) -> None:
        """Release the webcam resource.

        Safe to call even if the camera was never opened or has already
        been released. Ensures the webcam is not left locked after the
        application exits.
        """
        if self._capture is not None:
            self._capture.release()
            self._capture = None
