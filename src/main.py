"""PhysioAR main application — Phase 2.5.

Real-time webcam pose estimation pipeline with EMA smoothing,
neutral-posture calibration, runtime recalibration, and
temporary landmark debug overlay:

    Webcam → OpenCV Capture → BGR→RGB → MediaPipe Pose Landmarker
        → Raw Landmarks → EMA Smoothing → Smoothed Landmarks
        → Calibration (neutral posture baseline)
        → Skeleton Visualization → OpenCV Window

Controls:
    D / d  — Toggle landmark debug overlay (Phase 2.5 diagnostic)
    R / r  — Start a new calibration session (recalibrate)
    Q / q  — Quit the application

Run from the project root:

    python -m src.main
"""

import cv2

from src.acquisition import Camera
from src.config import LANDMARK_VISIBILITY_THRESHOLD
from src.pose_estimation import PoseEstimator
from src.calibration import run_calibration
from src.utils.geometry import calculate_midpoint


# ============================================================
# Phase 2.5 diagnostic — temporary landmark validation
# ============================================================

# MediaPipe landmark indices used by the debug overlay.
_DEBUG_NOSE = 0
_DEBUG_LEFT_SHOULDER = 11
_DEBUG_RIGHT_SHOULDER = 12
_DEBUG_LEFT_ELBOW = 13
_DEBUG_RIGHT_ELBOW = 14
_DEBUG_LEFT_HIP = 23
_DEBUG_RIGHT_HIP = 24

# Landmarks to display: (label, index).
# "Neck" is a computed midpoint, handled separately.
_DEBUG_LANDMARKS = [
    ("L-Shoulder", _DEBUG_LEFT_SHOULDER),
    ("R-Shoulder", _DEBUG_RIGHT_SHOULDER),
    ("L-Elbow", _DEBUG_LEFT_ELBOW),
    ("R-Elbow", _DEBUG_RIGHT_ELBOW),
    ("L-Hip", _DEBUG_LEFT_HIP),
    ("R-Hip", _DEBUG_RIGHT_HIP),
]


def get_debug_landmark_info(
    smoothed_landmarks: list,
    visibility_threshold: float | None = None,
) -> list[tuple[str, float | None, float | None]]:
    """Compute the visibility-gated debug overlay data for Phase 2.5.

    Phase 2.5 diagnostic — temporary landmark validation.

    Returns a list of (label, x_or_none, y_or_none) tuples for seven
    landmarks: L-Shoulder, R-Shoulder, L-Elbow, R-Elbow, L-Hip,
    R-Hip, and Neck.

    A landmark is considered displayable only when:
      1. Its index exists in smoothed_landmarks.
      2. Its raw MediaPipe visibility (SmoothedLandmark.visibility,
         which is NOT smoothed by EMA) meets the threshold.

    When displayable, x and y are the SMOOTHED normalized coordinates
    from the Phase 1.5 EMA pipeline.

    When not displayable (missing, or visibility below threshold),
    x and y are None — meaning "unavailable".

    The Neck point is the midpoint of L-Shoulder and R-Shoulder.
    Neck is available only when BOTH shoulders pass the visibility
    check. When available, the neck coordinates are computed from
    the smoothed shoulder coordinates using calculate_midpoint().

    The visibility threshold is an engineering/debugging threshold
    for the Phase 2.5 diagnostic. It is NOT a clinical threshold.

    Args:
        smoothed_landmarks: List of SmoothedLandmark objects from
            PoseResult.smoothed_landmarks.
        visibility_threshold: Minimum raw visibility value for a
            landmark to be considered displayable. Defaults to
            LANDMARK_VISIBILITY_THRESHOLD from config.

    Returns:
        A list of 7 tuples:
            [(label, x_or_none, y_or_none), ...]
        Where x_or_none is the smoothed x coordinate (float) if the
        landmark is displayable, or None if unavailable.
        y_or_none follows the same convention.
    """
    if visibility_threshold is None:
        visibility_threshold = LANDMARK_VISIBILITY_THRESHOLD

    landmark_count = len(smoothed_landmarks)
    results: list[tuple[str, float | None, float | None]] = []

    # Track shoulder availability for Neck calculation.
    l_shoulder_xy: tuple[float, float] | None = None
    r_shoulder_xy: tuple[float, float] | None = None

    for label, idx in _DEBUG_LANDMARKS:
        # Case C: Landmark completely unavailable (index out of range).
        if idx >= landmark_count:
            results.append((label, None, None))
            continue

        lm = smoothed_landmarks[idx]

        # Case B: Landmark present but not sufficiently visible.
        # Uses the raw MediaPipe visibility (NOT smoothed by EMA).
        if lm.visibility < visibility_threshold:
            results.append((label, None, None))
            continue

        # Case A: Landmark available and sufficiently visible.
        # Coordinates come from the SMOOTHED landmark (EMA output).
        results.append((label, lm.x, lm.y))

        # Track shoulders for Neck.
        if idx == _DEBUG_LEFT_SHOULDER:
            l_shoulder_xy = (lm.x, lm.y)
        elif idx == _DEBUG_RIGHT_SHOULDER:
            r_shoulder_xy = (lm.x, lm.y)

    # Neck = midpoint(L-Shoulder, R-Shoulder).
    # Available only when BOTH shoulders pass the visibility check.
    if l_shoulder_xy is not None and r_shoulder_xy is not None:
        neck_xy = calculate_midpoint(l_shoulder_xy, r_shoulder_xy)
        results.append(("Neck", neck_xy[0], neck_xy[1]))
    else:
        results.append(("Neck", None, None))

    return results


def _draw_debug_overlay(frame, smoothed_landmarks) -> None:
    """Draw Phase 2.5 landmark debug labels onto the frame.

    Phase 2.5 diagnostic — temporary landmark validation.

    Displays seven landmark labels with visibility-gated smoothed
    (x, y) normalized coordinates near their corresponding positions.

    Visibility gating uses the raw MediaPipe landmark visibility
    (SmoothedLandmark.visibility, NOT smoothed by EMA) against
    LANDMARK_VISIBILITY_THRESHOLD from config.

    Only landmarks that pass the visibility gate are displayed with
    coordinates. Landmarks that fail show "unavailable" — no stale
    or fabricated coordinates.

    Args:
        frame: The OpenCV BGR frame to draw on (modified in-place).
        smoothed_landmarks: List of SmoothedLandmark objects from
            PoseResult.smoothed_landmarks.
    """
    h, w = frame.shape[:2]

    # Phase 2.5 diagnostic — "Debug: ON" indicator.
    cv2.putText(
        frame, "Debug: ON",
        (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
    )

    # Get visibility-gated landmark data.
    landmark_info = get_debug_landmark_info(smoothed_landmarks)

    # Draw each landmark.
    unavailable_offset = 0
    for label, x, y in landmark_info:
        if x is not None and y is not None:
            _draw_landmark_label(frame, label, x, y, w, h)
        else:
            _draw_unavailable_label(frame, label, h, unavailable_offset)
            unavailable_offset += 1


def _draw_landmark_label(frame, label: str, x: float, y: float,
                         frame_w: int, frame_h: int) -> None:
    """Draw a single landmark label at its pixel position.

    Phase 2.5 diagnostic — temporary landmark validation.

    Args:
        frame: The OpenCV BGR frame to draw on (modified in-place).
        label: The landmark name (e.g., "L-Shoulder").
        x: Smoothed normalized x coordinate (0→1).
        y: Smoothed normalized y coordinate (0→1).
        frame_w: Frame width in pixels.
        frame_h: Frame height in pixels.
    """
    # Convert normalized coordinates to pixel coordinates for positioning.
    px = int(x * frame_w)
    py = int(y * frame_h)

    # Format label with normalized coordinates.
    text = f"{label} ({x:.3f}, {y:.3f})"

    # Draw with a dark background for readability.
    (text_w, text_h), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1,
    )
    # Offset label slightly to avoid occluding the landmark dot.
    label_x = px + 8
    label_y = py - 5

    # Clamp to frame bounds.
    label_x = max(0, min(label_x, frame_w - text_w - 4))
    label_y = max(text_h + 4, min(label_y, frame_h - 4))

    # Background rectangle for contrast.
    cv2.rectangle(
        frame,
        (label_x - 2, label_y - text_h - 2),
        (label_x + text_w + 2, label_y + 4),
        (0, 0, 0), cv2.FILLED,
    )
    # Label text.
    cv2.putText(
        frame, text,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1,
    )
    # Small dot at the landmark position.
    cv2.circle(frame, (px, py), 4, (0, 255, 0), cv2.FILLED)


def _draw_unavailable_label(frame, label: str, frame_h: int,
                            offset: int) -> None:
    """Draw an 'unavailable' label in the bottom-left corner.

    Phase 2.5 diagnostic — temporary landmark validation.

    Args:
        frame: The OpenCV BGR frame to draw on (modified in-place).
        label: The landmark name.
        frame_h: Frame height in pixels.
        offset: Vertical stacking index (0 = bottom, 1 = above, etc.).
    """
    text = f"{label}: unavailable"
    y_pos = frame_h - 20 - (offset * 20)
    cv2.putText(
        frame, text,
        (10, y_pos),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1,
    )


# ============================================================
# End Phase 2.5 diagnostic code
# ============================================================


def main() -> None:
    """Run the PhysioAR pipeline with calibration and recalibration.

    1. Opens the webcam and initializes pose estimation.
    2. Runs initial neutral-posture calibration (~10 seconds).
    3. Enters live pose visualization mode.
    4. Press 'D' to toggle landmark debug overlay (Phase 2.5).
    5. Press 'R' to recalibrate at any time.
    6. Press 'Q' to exit.

    Recalibration reuses the same Camera and PoseEstimator instances.
    It does NOT create a new camera, MediaPipe model, or EMA filter.
    The EMA filter state is reset at the start of each calibration
    so that calibration begins from fresh observations.

    If recalibration fails (insufficient valid samples), the previous
    calibration result is preserved and the application continues.
    """
    camera = Camera(camera_index=0)
    pose_estimator = PoseEstimator()

    # Phase 2.5 diagnostic — debug overlay state (OFF by default).
    debug_overlay_enabled = False

    try:
        camera.open()
        print("PhysioAR — Phase 2.5: Landmark Debug Overlay")
        print("Press 'D' for debug overlay, 'R' to recalibrate, 'Q' to quit.")
        print()

        # --- Initial calibration ---
        calibration_result = _attempt_calibration(camera, pose_estimator)

        # --- Live visualization loop ---
        _print_live_mode_instructions()

        while True:
            success, frame = camera.read()

            if not success:
                print("Frame capture failed. Exiting.")
                break

            # Process the frame through MediaPipe Pose Landmarker + EMA.
            result = pose_estimator.process(frame)

            # Draw pose landmarks and connections if a pose was detected.
            pose_estimator.draw(frame, result)

            # Phase 2.5 diagnostic — draw debug overlay if enabled.
            if debug_overlay_enabled and result.pose_detected:
                _draw_debug_overlay(frame, result.smoothed_landmarks)
            elif debug_overlay_enabled and not result.pose_detected:
                h, w = frame.shape[:2]
                cv2.putText(
                    frame, "Debug: ON (no pose detected)",
                    (w - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2,
                )

            # Live mode overlay.
            cv2.putText(
                frame, "D: Debug | R: Recalibrate | Q: Quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
            )

            # Display the annotated frame.
            cv2.imshow("PhysioAR", frame)

            # Handle key presses.
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                break
            elif key == ord("r") or key == ord("R"):
                # --- Runtime recalibration ---
                new_result = _attempt_calibration(camera, pose_estimator)
                if new_result is not None:
                    calibration_result = new_result
                else:
                    print("Previous calibration preserved.")

                _print_live_mode_instructions()

            elif key == ord("d") or key == ord("D"):
                # Phase 2.5 diagnostic — toggle debug overlay.
                debug_overlay_enabled = not debug_overlay_enabled
                state = "ON" if debug_overlay_enabled else "OFF"
                print(f"  Landmark debug overlay: {state}")

    finally:
        pose_estimator.close()
        camera.release()
        cv2.destroyAllWindows()


def _attempt_calibration(camera, pose_estimator):
    """Attempt a calibration, returning the result or None on failure."""
    try:
        return run_calibration(camera, pose_estimator)
    except RuntimeError as e:
        print(f"Calibration error: {e}")
        print("Continuing without updating calibration baseline.")
        return None


def _print_live_mode_instructions():
    """Print live-mode instructions to the console."""
    print()
    print("-" * 40)
    print("  Live mode active.")
    print("  Press 'D' to toggle debug overlay.")
    print("  Press 'R' to recalibrate.")
    print("  Press 'Q' to quit.")
    print("-" * 40)
    print()


if __name__ == "__main__":
    main()
