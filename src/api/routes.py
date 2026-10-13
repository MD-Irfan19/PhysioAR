"""REST API routes for PhysioAR."""

from __future__ import annotations

import logging
from typing import Any
import anyio
from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.api.schemas import (
    CalibrateRequest,
    ExerciseItemResponse,
    ExerciseListResponse,
    HealthResponse,
    SessionSnapshotResponse,
    SessionStatusResponse,
    StartSessionRequest,
    snapshot_to_dict,
)
from src.exercises import EXERCISE_REGISTRY
from src.runtime import SessionRuntime
from src.session import SessionSnapshot, SessionStatus, resolve_exercise

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["PhysioAR"])


def get_runtime(request: Request) -> SessionRuntime:
    """Dependency providing the shared SessionRuntime instance."""
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Runtime is not configured on application state.",
        )
    return runtime


@router.get("/health", response_model=HealthResponse)
async def get_health(runtime: SessionRuntime = Depends(get_runtime)) -> HealthResponse:
    """Check API and background runtime health."""
    is_running = runtime.is_running()
    error = runtime.get_error()
    camera_active = bool(
        is_running
        and runtime.camera is not None
        and getattr(runtime.camera, "opened", False)
    )
    return HealthResponse(
        status="ok",
        runtime_running=is_running,
        runtime_error=error,
        camera_active=camera_active,
        version="1.0.0",
    )


@router.get("/exercises", response_model=ExerciseListResponse)
async def list_exercises() -> ExerciseListResponse:
    """List all registered physiotherapy exercises."""
    items = [
        ExerciseItemResponse(
            id=ex.name.lower().replace(" ", "_").replace("-", "_"),
            name=ex.name,
            target_joint=ex.target_joint,
            camera_orientation=ex.camera_orientation.value,
            rom_target=float(ex.rom_target),
        )
        for ex in EXERCISE_REGISTRY
    ]
    return ExerciseListResponse(exercises=items)


@router.get("/session", response_model=SessionSnapshotResponse)
async def get_session(
    runtime: SessionRuntime = Depends(get_runtime),
) -> SessionSnapshotResponse:
    """Retrieve the current structured session snapshot."""
    snapshot = runtime.get_snapshot()
    data = snapshot_to_dict(snapshot)
    return SessionSnapshotResponse(**data)


@router.get("/session/status", response_model=SessionStatusResponse)
async def get_session_status(
    runtime: SessionRuntime = Depends(get_runtime),
) -> SessionStatusResponse:
    """Retrieve runtime lifecycle and session status."""
    is_running = runtime.is_running()
    snap = runtime.get_snapshot()
    error = runtime.get_error()

    if snap is not None:
        st = snap.status.value
        ex_name = snap.exercise_name
        ex_id = snap.exercise_id
        side = snap.side
    else:
        st = SessionStatus.ACTIVE.value if is_running else SessionStatus.IDLE.value
        ex_name = None
        ex_id = None
        side = None

    return SessionStatusResponse(
        status=st,
        is_running=is_running,
        exercise_name=ex_name,
        exercise_id=ex_id,
        side=side,
        error=error,
    )


@router.post("/session/start", response_model=SessionSnapshotResponse)
async def start_session(
    payload: StartSessionRequest,
    runtime: SessionRuntime = Depends(get_runtime),
) -> SessionSnapshotResponse:
    """Start an exercise session with the specified exercise, side, and camera."""
    # 1. Validate exercise
    try:
        resolved = resolve_exercise(payload.exercise_id)
    except (ValueError, TypeError) as e:
        logger.warning(f"Invalid exercise requested: {payload.exercise_id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    # 2. Check duplicate / active session
    if runtime.is_running():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A session is already running. Call /api/session/stop before starting a new session.",
        )

    # 3. Start session offloaded to thread pool to avoid blocking async event loop
    try:
        snapshot = await anyio.to_thread.run_sync(
            runtime.start_session,
            resolved,
            payload.side,
            payload.camera_index,
        )
    except RuntimeError as e:
        err_msg = str(e)
        logger.error(f"Failed to start session: {err_msg}")
        if "camera" in err_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Camera initialization failed: {err_msg}",
            )
        if "already running" in err_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=err_msg,
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start session: {err_msg}",
        )
    except Exception as e:
        logger.exception("Unexpected error starting session")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        )

    return SessionSnapshotResponse(**snapshot_to_dict(snapshot))


@router.post("/session/stop")
async def stop_session(
    runtime: SessionRuntime = Depends(get_runtime),
) -> dict[str, str]:
    """Safely stop the active session."""
    if not runtime.is_running():
        return {
            "status": "stopped",
            "message": "No active session to stop.",
        }

    try:
        await anyio.to_thread.run_sync(runtime.stop_session)
    except TimeoutError as e:
        logger.error(f"Worker shutdown timed out: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Session shutdown timeout: {str(e)}",
        )
    except Exception as e:
        logger.exception("Unexpected error stopping session")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error stopping session: {str(e)}",
        )

    return {
        "status": "stopped",
        "message": "Session stopped successfully.",
    }


@router.post("/session/calibrate")
async def calibrate_session(
    payload: CalibrateRequest,
    runtime: SessionRuntime = Depends(get_runtime),
) -> dict[str, Any]:
    """Request neutral posture calibration or recalibration."""
    if not runtime.is_running():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot request calibration: no active session.",
        )

    try:
        accepted = await anyio.to_thread.run_sync(
            runtime.request_calibration,
            payload.duration_seconds,
            payload.min_samples,
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Calibration request error: {str(e)}",
        )

    if not accepted:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Calibration request could not be accepted.",
        )

    return {
        "status": "calibrating",
        "message": "Calibration requested successfully.",
    }
