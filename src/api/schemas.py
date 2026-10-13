"""Pydantic schemas and serialization utilities for PhysioAR REST and WebSocket API."""

from __future__ import annotations

from typing import Any, Optional
import numpy as np
from pydantic import BaseModel, Field, field_validator

from src.session import SessionSnapshot, SessionStatus


def sanitize_value(val: Any) -> Any:
    """Recursively convert NumPy scalars/arrays and enums to standard Python types."""
    if val is None:
        return None
    if isinstance(val, (np.floating, float)):
        return float(val)
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if hasattr(val, "value"):  # Enum support
        return val.value
    if isinstance(val, dict):
        return {k: sanitize_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple, set)):
        return [sanitize_value(v) for v in val]
    return val


def snapshot_to_dict(snapshot: Optional[SessionSnapshot]) -> dict[str, Any]:
    """Convert a SessionSnapshot into a JSON-safe dictionary with no NumPy objects."""
    if snapshot is None:
        return {
            "status": SessionStatus.IDLE.value,
            "exercise_name": None,
            "exercise_id": None,
            "side": None,
            "selected_side": None,
            "camera_orientation": None,
            "is_calibrated": False,
            "calibration_status": "not_calibrated",
            "calibration_error": None,
            "calibration_samples": 0,
            "calibration_baseline": None,
            "pose_detected": False,
            "current_angle": None,
            "joint_angle": None,
            "rep_state": None,
            "rep_phase": None,
            "rep_count": 0,
            "rom_target": None,
            "rom_progress": None,
            "active_compensation_ids": [],
            "feedback_messages": [],
            "latest_quality": None,
            "last_error": None,
            "frame_index": 0,
        }

    raw_dict = snapshot.to_dict()
    # Ensure redundant alias properties are populated for client convenience
    raw_dict["rep_phase"] = snapshot.rep_phase
    raw_dict["joint_angle"] = raw_dict.get("joint_angle")
    raw_dict["selected_side"] = raw_dict.get("selected_side")
    return sanitize_value(raw_dict)


# ==============================================================================
# Request Schemas
# ==============================================================================

class StartSessionRequest(BaseModel):
    """Payload for starting an exercise session."""

    exercise_id: str = Field(
        ...,
        description="Exercise identifier (e.g., 'shoulder_abduction', 'sit_to_stand', 'elbow_flexion_extension')",
        examples=["shoulder_abduction"],
    )
    side: str = Field(
        default="right",
        description="Tracking side: 'left' or 'right'",
        examples=["right"],
    )
    camera_index: int = Field(
        default=0,
        ge=0,
        description="Webcam hardware device index",
    )

    @field_validator("side")
    @classmethod
    def validate_side(cls, v: str) -> str:
        cleaned = v.strip().lower()
        if cleaned in ("l", "left"):
            return "left"
        if cleaned in ("r", "right"):
            return "right"
        raise ValueError(f"Invalid side '{v}'. Must be 'left' or 'right'.")

    @field_validator("exercise_id")
    @classmethod
    def validate_exercise_id(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("exercise_id cannot be blank.")
        return cleaned


class CalibrateRequest(BaseModel):
    """Payload for requesting neutral posture calibration or recalibration."""

    duration_seconds: float = Field(
        default=3.0,
        gt=0.0,
        description="Duration in seconds to collect calibration frames",
    )
    min_samples: int = Field(
        default=30,
        ge=1,
        description="Minimum valid posture frames required to produce a baseline",
    )


# ==============================================================================
# Response Schemas
# ==============================================================================

class HealthResponse(BaseModel):
    """API health status and active runtime information."""

    status: str = "ok"
    runtime_running: bool
    runtime_error: Optional[str] = None
    camera_active: bool
    version: str = "1.0.0"


class SessionStatusResponse(BaseModel):
    """Compact status report for the runtime and active session."""

    status: str
    is_running: bool
    exercise_name: Optional[str] = None
    exercise_id: Optional[str] = None
    side: Optional[str] = None
    error: Optional[str] = None


class SessionSnapshotResponse(BaseModel):
    """Structured snapshot of the exercise session state."""

    status: str
    exercise_name: Optional[str] = None
    exercise_id: Optional[str] = None
    side: Optional[str] = None
    selected_side: Optional[str] = None
    camera_orientation: Optional[str] = None
    is_calibrated: bool = False
    calibration_status: str = "not_calibrated"
    calibration_error: Optional[str] = None
    calibration_samples: int = 0
    calibration_baseline: Optional[dict[str, Any]] = None
    pose_detected: bool = False
    current_angle: Optional[float] = None
    joint_angle: Optional[float] = None
    rep_state: Optional[str] = None
    rep_phase: Optional[str] = None
    rep_count: int = 0
    rom_target: Optional[float] = None
    rom_progress: Optional[float] = None
    active_compensation_ids: list[str] = Field(default_factory=list)
    feedback_messages: list[str] = Field(default_factory=list)
    latest_quality: Optional[dict[str, Any]] = None
    last_error: Optional[str] = None
    frame_index: int = 0


class ExerciseItemResponse(BaseModel):
    """Metadata for a registered exercise."""

    id: str
    name: str
    target_joint: str
    camera_orientation: str
    rom_target: float


class ExerciseListResponse(BaseModel):
    """List of all available registered exercises."""

    exercises: list[ExerciseItemResponse]


class ErrorResponse(BaseModel):
    """Standardized error response payload."""

    detail: str
    error_code: Optional[str] = None
