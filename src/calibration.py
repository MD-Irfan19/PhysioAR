"""User neutral-posture calibration module for PhysioAR.

Captures the user's neutral standing posture over a configurable
duration and calculates a session baseline (mean and sample standard
deviation) for four posture metrics:

1. Spine angle (torso lean from vertical)
2. Shoulder height difference
3. Neck tilt (head tilt from vertical)
4. Hip alignment (hip height asymmetry)

The calibration consumes **smoothed** landmarks from the Phase 1.5
EMA pipeline. It does NOT perform a second smoothing operation.

Landmark validity gating (Phase 2.1):
    Before computing metrics, each required landmark is checked
    against a configurable visibility threshold. If any required
    landmark falls below the threshold, the entire frame is rejected.
    This prevents unreliable landmark observations (e.g., hips
    inferred when only upper body is visible) from entering the
    calibration baseline.

    The visibility field from SmoothedLandmark is used for gating.
    This value is preserved from the original MediaPipe landmark
    and is NOT smoothed by the EMA filter.

    The threshold is an engineering validity gate, not a clinical
    guarantee. MediaPipe visibility does not guarantee anatomical
    correctness.

Coordinate system (MediaPipe normalized):
    x: 0→1, left to right
    y: 0→1, top to bottom
    Upward direction in image-space = (0, -1)

Vertical reference rule:
    Vertical reference points are constructed as vertex + (0, -1),
    never from the opposite endpoint. This is mandatory for correct
    angle computation with calculate_angle(point_a, point_b, point_c),
    which calculates the angle AT point_b.

The calibration baseline represents the user's observed neutral
posture during this session and is NOT a clinical assessment.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import cv2

from src.config import (
    CALIBRATION_SECONDS,
    LANDMARK_VISIBILITY_THRESHOLD,
    MIN_CALIBRATION_SAMPLES,
)
from src.utils.geometry import calculate_angle, calculate_midpoint


# MediaPipe pose landmark indices.
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

# All landmark indices required for full-body calibration.
# A calibration frame is accepted only if ALL of these landmarks
# pass the visibility threshold.
REQUIRED_LANDMARKS = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]

# Human-readable names for diagnostic messages.
_LANDMARK_NAMES = {
    NOSE: "NOSE",
    LEFT_SHOULDER: "LEFT_SHOULDER",
    RIGHT_SHOULDER: "RIGHT_SHOULDER",
    LEFT_HIP: "LEFT_HIP",
    RIGHT_HIP: "RIGHT_HIP",
}


# ============================================================
# Result data structures
# ============================================================


@dataclass
class MetricBaseline:
    """Mean and sample standard deviation for a single posture metric.

    Attributes:
        mean: The arithmetic mean of the valid per-frame observations.
        std: The sample standard deviation of the valid per-frame
            observations.
    """

    mean: float
    std: float


@dataclass
class CalibrationResult:
    """Session baseline computed from neutral-posture calibration.

    Contains mean and sample standard deviation for each posture
    metric, along with capture statistics.

    Attributes:
        spine_angle: Torso lean from vertical (degrees).
        shoulder_height_difference: Absolute vertical shoulder
            asymmetry (normalized image units).
        neck_tilt: Head/neck tilt from vertical (degrees).
        hip_alignment: Absolute vertical hip asymmetry (normalized
            image units).
        valid_samples: Number of frames that contributed metric values.
        skipped_samples: Number of frames that were rejected.
        duration_seconds: Actual duration of the calibration capture.
    """

    spine_angle: MetricBaseline
    shoulder_height_difference: MetricBaseline
    neck_tilt: MetricBaseline
    hip_alignment: MetricBaseline
    valid_samples: int
    skipped_samples: int
    duration_seconds: float


# ============================================================
# Landmark validity checking
# ============================================================


def validate_required_landmarks(
    smoothed_landmarks: list,
    required_indices: list[int] | None = None,
    threshold: float | None = None,
) -> list[str]:
    """Check whether all required landmarks are present and sufficiently visible.

    Uses the ``visibility`` field of each SmoothedLandmark. This value
    is preserved from the original MediaPipe landmark and is NOT
    smoothed by the EMA filter.

    The visibility threshold is an engineering validity gate intended
    to prevent obviously unreliable landmark observations from entering
    the calibration baseline. It does not guarantee anatomical
    correctness.

    Args:
        smoothed_landmarks: List of SmoothedLandmark objects from
            PoseResult.smoothed_landmarks.
        required_indices: Landmark indices to validate. Defaults to
            REQUIRED_LANDMARKS (all five calibration landmarks).
        threshold: Minimum visibility value for a landmark to be
            considered valid. Defaults to LANDMARK_VISIBILITY_THRESHOLD
            from config.

    Returns:
        A list of human-readable rejection reasons. An empty list
        means all required landmarks are valid.
    """
    if required_indices is None:
        required_indices = REQUIRED_LANDMARKS
    if threshold is None:
        threshold = LANDMARK_VISIBILITY_THRESHOLD

    rejection_reasons: list[str] = []
    landmark_count = len(smoothed_landmarks)

    for idx in required_indices:
        name = _LANDMARK_NAMES.get(idx, f"LANDMARK_{idx}")

        # Check index existence.
        if idx >= landmark_count:
            rejection_reasons.append(f"{name} (index {idx}): missing")
            continue

        lm = smoothed_landmarks[idx]

        # Check visibility against threshold.
        visibility = getattr(lm, "visibility", None)
        if visibility is None:
            rejection_reasons.append(f"{name}: no visibility data")
            continue

        if visibility < threshold:
            rejection_reasons.append(
                f"{name}: visibility {visibility:.3f} < threshold {threshold}"
            )

    return rejection_reasons


# ============================================================
# Metric computation (pure functions — unchanged from Phase 2)
# ============================================================


def compute_spine_angle(
    left_shoulder_xy: tuple[float, float],
    right_shoulder_xy: tuple[float, float],
    left_hip_xy: tuple[float, float],
    right_hip_xy: tuple[float, float],
) -> float:
    """Calculate spine angle — torso lean from vertical.

    The torso direction is hip_midpoint → shoulder_midpoint.
    The angle is measured between this direction and the vertical
    axis at the hip_midpoint vertex.

    Vertical reference = hip_midpoint + (0, -1).

    Args:
        left_shoulder_xy: (x, y) of the left shoulder.
        right_shoulder_xy: (x, y) of the right shoulder.
        left_hip_xy: (x, y) of the left hip.
        right_hip_xy: (x, y) of the right hip.

    Returns:
        Spine angle in degrees (0 = vertical, larger = more lean).

    Raises:
        ValueError: If the geometry is degenerate (e.g., midpoints
            coincide).
    """
    hip_midpoint = calculate_midpoint(left_hip_xy, right_hip_xy)
    shoulder_midpoint = calculate_midpoint(left_shoulder_xy, right_shoulder_xy)

    # Vertical reference from the vertex (hip_midpoint).
    vertical_reference = (hip_midpoint[0], hip_midpoint[1] - 1.0)

    return calculate_angle(shoulder_midpoint, hip_midpoint, vertical_reference)


def compute_shoulder_height_difference(
    left_shoulder_xy: tuple[float, float],
    right_shoulder_xy: tuple[float, float],
) -> float:
    """Calculate absolute vertical shoulder height difference.

    Args:
        left_shoulder_xy: (x, y) of the left shoulder.
        right_shoulder_xy: (x, y) of the right shoulder.

    Returns:
        Absolute y-difference in normalized image units.
        0 = shoulders are level.
    """
    return abs(left_shoulder_xy[1] - right_shoulder_xy[1])


def compute_neck_tilt(
    left_shoulder_xy: tuple[float, float],
    right_shoulder_xy: tuple[float, float],
    nose_xy: tuple[float, float],
) -> float:
    """Calculate neck tilt — head deviation from vertical.

    The neck/head direction is shoulder_midpoint → nose.
    The angle is measured between this direction and the vertical
    axis at the shoulder_midpoint vertex.

    Vertical reference = shoulder_midpoint + (0, -1).

    Args:
        left_shoulder_xy: (x, y) of the left shoulder.
        right_shoulder_xy: (x, y) of the right shoulder.
        nose_xy: (x, y) of the nose.

    Returns:
        Neck tilt in degrees (0 = vertical, larger = more tilt).

    Raises:
        ValueError: If the geometry is degenerate.
    """
    shoulder_midpoint = calculate_midpoint(left_shoulder_xy, right_shoulder_xy)

    # Vertical reference from the vertex (shoulder_midpoint).
    vertical_reference = (shoulder_midpoint[0], shoulder_midpoint[1] - 1.0)

    return calculate_angle(nose_xy, shoulder_midpoint, vertical_reference)


def compute_hip_alignment(
    left_hip_xy: tuple[float, float],
    right_hip_xy: tuple[float, float],
) -> float:
    """Calculate absolute vertical hip height difference.

    Args:
        left_hip_xy: (x, y) of the left hip.
        right_hip_xy: (x, y) of the right hip.

    Returns:
        Absolute y-difference in normalized image units.
        0 = hips are level.
    """
    return abs(left_hip_xy[1] - right_hip_xy[1])


# ============================================================
# Per-frame metric extraction
# ============================================================


def compute_frame_metrics(
    smoothed_landmarks: list,
    visibility_threshold: float | None = None,
) -> tuple[float, float, float, float] | None:
    """Compute all four posture metrics from a single frame's smoothed landmarks.

    Performs landmark validity checking BEFORE metric calculation:

    1. Checks landmark count/index availability.
    2. Checks visibility of all required landmarks against threshold.
    3. If any required landmark is invalid, returns None.
    4. Converts valid landmarks to (x, y) coordinates.
    5. Calculates all four metrics.
    6. If calculate_angle() raises ValueError, returns None.

    Uses the smoothed (x, y) coordinates from the EMA-filtered
    landmarks. Visibility values are checked but NOT smoothed.

    Args:
        smoothed_landmarks: List of SmoothedLandmark objects from
            PoseResult.smoothed_landmarks.
        visibility_threshold: Minimum visibility value for each
            required landmark. Defaults to LANDMARK_VISIBILITY_THRESHOLD
            from config. Pass explicitly for testing.

    Returns:
        A tuple of (spine_angle, shoulder_height_diff, neck_tilt,
        hip_alignment) if all metrics were computed successfully,
        or None if the frame should be skipped.
    """
    # Step 1+2: Validate required landmarks (index + visibility).
    reasons = validate_required_landmarks(
        smoothed_landmarks,
        required_indices=REQUIRED_LANDMARKS,
        threshold=visibility_threshold,
    )
    if reasons:
        return None

    # Step 3-6: Calculate metrics (geometry errors caught).
    try:
        nose = smoothed_landmarks[NOSE]
        l_shoulder = smoothed_landmarks[LEFT_SHOULDER]
        r_shoulder = smoothed_landmarks[RIGHT_SHOULDER]
        l_hip = smoothed_landmarks[LEFT_HIP]
        r_hip = smoothed_landmarks[RIGHT_HIP]

        nose_xy = (nose.x, nose.y)
        l_shoulder_xy = (l_shoulder.x, l_shoulder.y)
        r_shoulder_xy = (r_shoulder.x, r_shoulder.y)
        l_hip_xy = (l_hip.x, l_hip.y)
        r_hip_xy = (r_hip.x, r_hip.y)

        spine = compute_spine_angle(l_shoulder_xy, r_shoulder_xy, l_hip_xy, r_hip_xy)
        shoulder = compute_shoulder_height_difference(l_shoulder_xy, r_shoulder_xy)
        neck = compute_neck_tilt(l_shoulder_xy, r_shoulder_xy, nose_xy)
        hip = compute_hip_alignment(l_hip_xy, r_hip_xy)

        return (spine, shoulder, neck, hip)

    except (ValueError, IndexError, AttributeError):
        # Degenerate geometry or missing data — skip the frame.
        return None


# ============================================================
# Calibration runner
# ============================================================


def run_calibration(
    camera,
    pose_estimator,
    duration_seconds: float | None = None,
    min_samples: int | None = None,
    visibility_threshold: float | None = None,
) -> CalibrationResult:
    """Run the neutral-posture calibration capture.

    Captures frames for approximately ``duration_seconds``, validates
    required landmark visibility, computes posture metrics on valid
    frames using smoothed landmarks, and produces a session baseline
    (mean + sample standard deviation).

    Calibration uses smoothed landmarks from Phase 1.5 and does not
    perform a second smoothing operation.

    At the start of calibration, the EMA filter is reset so that the
    calibration begins from fresh observations rather than inheriting
    coordinate history from previous usage.

    Args:
        camera: An opened Camera instance from src.acquisition.
        pose_estimator: An initialized PoseEstimator instance from
            src.pose_estimation.
        duration_seconds: Capture duration in seconds. Defaults to
            CALIBRATION_SECONDS from config.
        min_samples: Minimum valid samples required. Defaults to
            MIN_CALIBRATION_SAMPLES from config.
        visibility_threshold: Minimum landmark visibility to accept
            a frame. Defaults to LANDMARK_VISIBILITY_THRESHOLD from
            config.

    Returns:
        A CalibrationResult containing the baseline metrics.

    Raises:
        RuntimeError: If fewer than ``min_samples`` valid frames are
            collected, indicating an unreliable calibration.
    """
    if duration_seconds is None:
        duration_seconds = CALIBRATION_SECONDS
    if min_samples is None:
        min_samples = MIN_CALIBRATION_SAMPLES
    if visibility_threshold is None:
        visibility_threshold = LANDMARK_VISIBILITY_THRESHOLD

    # Reset EMA filter so calibration starts from fresh observations.
    pose_estimator.reset_smoothing()

    # Per-metric sample collectors.
    spine_values: list[float] = []
    shoulder_values: list[float] = []
    neck_values: list[float] = []
    hip_values: list[float] = []

    valid_count = 0
    skipped_count = 0
    skip_reason_counts: dict[str, int] = {}

    # Throttle skip diagnostics to avoid flooding the terminal.
    _last_skip_print_time = 0.0
    _SKIP_PRINT_INTERVAL = 1.0  # Print skip reasons at most once per second.

    # Display calibration instructions.
    print()
    print("=" * 60)
    print("  CALIBRATION — Neutral Posture Capture")
    print("=" * 60)
    print()
    print("  Stand naturally in your neutral posture.")
    print("  Keep your FULL BODY visible in the camera.")
    print(f"  Calibration duration: ~{duration_seconds} seconds.")
    print(f"  Landmark visibility threshold: {visibility_threshold}")
    print()
    print("  Calibration starting...")
    print()

    start_time = time.time()

    while True:
        elapsed = time.time() - start_time
        if elapsed >= duration_seconds:
            break

        success, frame = camera.read()
        if not success:
            skipped_count += 1
            skip_reason_counts["frame_capture_failed"] = (
                skip_reason_counts.get("frame_capture_failed", 0) + 1
            )
            continue

        # Process through the existing pose estimation + EMA pipeline.
        result = pose_estimator.process(frame)

        # Draw skeleton for visual feedback during calibration.
        pose_estimator.draw(frame, result)

        remaining = max(0, duration_seconds - elapsed)

        if not result.pose_detected:
            skipped_count += 1
            skip_reason_counts["no_pose_detected"] = (
                skip_reason_counts.get("no_pose_detected", 0) + 1
            )
            # Show frame even without pose.
            cv2.putText(
                frame, f"Calibrating... {remaining:.1f}s remaining",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )
            cv2.putText(
                frame, "No pose detected - stand in view",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
            )
            cv2.imshow("PhysioAR", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # Validate landmark visibility and compute metrics.
        rejection_reasons = validate_required_landmarks(
            result.smoothed_landmarks,
            threshold=visibility_threshold,
        )

        if rejection_reasons:
            skipped_count += 1
            # Track which landmarks caused rejections.
            for reason in rejection_reasons:
                landmark_name = reason.split(":")[0].strip()
                key = f"low_visibility_{landmark_name}"
                skip_reason_counts[key] = skip_reason_counts.get(key, 0) + 1

            # Throttled diagnostic output.
            now = time.time()
            if now - _last_skip_print_time >= _SKIP_PRINT_INTERVAL:
                reasons_str = "; ".join(rejection_reasons)
                print(f"  Frame skipped: {reasons_str}")
                _last_skip_print_time = now

            # Show overlay indicating low confidence.
            cv2.putText(
                frame, f"Calibrating... {remaining:.1f}s remaining",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )
            cv2.putText(
                frame, "Low landmark confidence - adjust position",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2,
            )
            cv2.imshow("PhysioAR", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # All required landmarks valid — compute metrics.
        metrics = compute_frame_metrics(
            result.smoothed_landmarks,
            visibility_threshold=visibility_threshold,
        )

        if metrics is None:
            # Degenerate geometry despite valid landmarks.
            skipped_count += 1
            skip_reason_counts["degenerate_geometry"] = (
                skip_reason_counts.get("degenerate_geometry", 0) + 1
            )
        else:
            spine, shoulder, neck, hip = metrics
            spine_values.append(spine)
            shoulder_values.append(shoulder)
            neck_values.append(neck)
            hip_values.append(hip)
            valid_count += 1

            # Per-frame diagnostic output.
            print(
                f"  Calibration frame {valid_count:03d}  "
                f"Spine: {spine:6.2f}°  "
                f"Shoulder: {shoulder:.4f}  "
                f"Neck: {neck:6.2f}°  "
                f"Hip: {hip:.4f}"
            )

        # Show annotated frame with calibration overlay.
        cv2.putText(
            frame, f"Calibrating... {remaining:.1f}s remaining",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )
        cv2.imshow("PhysioAR", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    actual_duration = time.time() - start_time

    # Print skip reason summary.
    if skip_reason_counts:
        print()
        print("  Skip reasons:")
        for reason, count in sorted(skip_reason_counts.items()):
            print(f"    {reason}: {count}")

    # Check minimum sample requirement.
    if valid_count < min_samples:
        print()
        print(f"  CALIBRATION FAILED: Only {valid_count} valid samples "
              f"collected (minimum: {min_samples}).")
        print(f"  Skipped frames: {skipped_count}")
        print(f"  Duration: {actual_duration:.1f}s")
        print()
        raise RuntimeError(
            f"Calibration failed: only {valid_count} valid samples "
            f"collected, but {min_samples} are required. Ensure the "
            f"user's full body is clearly visible in the camera for "
            f"the entire calibration duration."
        )

    # Calculate mean and sample standard deviation.
    calibration_result = CalibrationResult(
        spine_angle=MetricBaseline(
            mean=statistics.mean(spine_values),
            std=statistics.stdev(spine_values),
        ),
        shoulder_height_difference=MetricBaseline(
            mean=statistics.mean(shoulder_values),
            std=statistics.stdev(shoulder_values),
        ),
        neck_tilt=MetricBaseline(
            mean=statistics.mean(neck_values),
            std=statistics.stdev(neck_values),
        ),
        hip_alignment=MetricBaseline(
            mean=statistics.mean(hip_values),
            std=statistics.stdev(hip_values),
        ),
        valid_samples=valid_count,
        skipped_samples=skipped_count,
        duration_seconds=actual_duration,
    )

    # Print calibration summary.
    print()
    print("=" * 60)
    print("  CALIBRATION COMPLETE")
    print("=" * 60)
    print()
    print(f"  Duration:        {actual_duration:.1f}s")
    print(f"  Valid frames:    {valid_count}")
    print(f"  Skipped frames:  {skipped_count}")
    print()
    print(f"  Spine angle:              "
          f"mean = {calibration_result.spine_angle.mean:.2f}°, "
          f"std = {calibration_result.spine_angle.std:.2f}°")
    print(f"  Shoulder height diff:     "
          f"mean = {calibration_result.shoulder_height_difference.mean:.4f}, "
          f"std = {calibration_result.shoulder_height_difference.std:.4f}")
    print(f"  Neck tilt:                "
          f"mean = {calibration_result.neck_tilt.mean:.2f}°, "
          f"std = {calibration_result.neck_tilt.std:.2f}°")
    print(f"  Hip alignment:            "
          f"mean = {calibration_result.hip_alignment.mean:.4f}, "
          f"std = {calibration_result.hip_alignment.std:.4f}")
    print()

    return calibration_result
