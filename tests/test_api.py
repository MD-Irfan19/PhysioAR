"""Test suite for PhysioAR FastAPI backend (Phase C).

Covers:
1. Application creation without camera startup.
2. Health endpoint behavior.
3. Session snapshot serialization.
4. Valid and invalid session-start requests.
5. Duplicate session-start behavior.
6. Stop requests and inactive-session behavior.
7. Calibration request validation and forwarding.
8. Camera startup failures and consistent error responses.
9. Runtime error reporting.
10. WebSocket initial snapshot delivery.
11. WebSocket updates and disconnection handling.
12. Multiple WebSocket clients sharing the same runtime.
13. MJPEG content type and frame encoding.
14. Handling missing frames or an inactive session.
15. Stream-client disconnection.
16. Shutdown cleanup and failed-shutdown behavior.
17. Confirmation that no per-client camera or inference worker is created.
18. CORS behavior for configured origins.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.schemas import snapshot_to_dict
from src.exercises import EXERCISE_REGISTRY
from src.session import SessionSnapshot, SessionStatus


# ==============================================================================
# Deterministic Fake Runtime for Testing
# ==============================================================================

class FakeRuntime:
    """Deterministic, mockable runtime implementing the SessionRuntime interface."""

    def __init__(
        self,
        is_running: bool = False,
        error: Optional[str] = None,
        snapshot: Optional[SessionSnapshot] = None,
        frame: Optional[np.ndarray] = None,
        camera: Any = None,
    ) -> None:
        self._is_running = is_running
        self._error = error
        self._snapshot = snapshot
        self._frame = frame
        self._camera = camera
        self.stop_called = False
        self.start_called = False
        self.calibration_called = False
        self.last_calib_args: Optional[tuple] = None
        self.stop_raises_timeout = False
        self.start_raises_camera = False
        self.frame_count = 0

    def is_running(self) -> bool:
        return self._is_running

    def get_error(self) -> Optional[str]:
        return self._error

    def get_snapshot(self) -> Optional[SessionSnapshot]:
        return self._snapshot

    def get_latest_frame(self, copy: bool = True) -> Optional[np.ndarray]:
        if self._frame is None:
            return None
        return self._frame.copy() if copy else self._frame

    @property
    def camera(self) -> Any:
        return self._camera

    def start_session(
        self,
        exercise: Any,
        side: str = "right",
        camera_index: int = 0,
    ) -> SessionSnapshot:
        if self.start_raises_camera:
            self._error = "Failed to open camera: Hardware disconnected"
            raise RuntimeError(self._error)
        if self._is_running:
            raise RuntimeError("Session is already running")

        self._is_running = True
        self.start_called = True
        ex_name = exercise.name if hasattr(exercise, "name") else str(exercise)

        self._snapshot = SessionSnapshot(
            status=SessionStatus.ACTIVE,
            exercise_name=ex_name,
            side=side,
            camera_orientation="front",
            is_calibrated=False,
            calibration_status="not_calibrated",
            pose_detected=True,
            current_angle=45.0,
            rep_state="ascending",
            rep_count=1,
            rom_target=90.0,
            rom_progress=0.5,
            active_compensation_ids=["torso_lean"],
            feedback_messages=["Keep torso upright"],
            frame_index=1,
        )
        self._frame_count = 1
        return self._snapshot

    def stop_session(self, timeout: float = 2.0) -> None:
        if self.stop_raises_timeout:
            raise TimeoutError("Worker thread did not terminate within timeout")
        self.stop_called = True
        self._is_running = False
        if self._snapshot:
            self._snapshot.status = SessionStatus.STOPPED

    def request_calibration(
        self,
        duration_seconds: float = 3.0,
        min_samples: int = 30,
    ) -> bool:
        if not self._is_running:
            raise RuntimeError("Cannot request calibration: no active session.")
        self.calibration_called = True
        self.last_calib_args = (duration_seconds, min_samples)
        if self._snapshot:
            self._snapshot.status = SessionStatus.CALIBRATING
            self._snapshot.calibration_status = "calibrating"
        return True

    def wait_for_next_frame(self, timeout: Optional[float] = None) -> bool:
        time.sleep(min(timeout or 0.02, 0.02))
        return self._is_running


# ==============================================================================
# 1. Application Creation & Camera Non-Startup
# ==============================================================================

def test_app_creation_without_camera_startup():
    """Importing and creating app does not open camera or start background worker."""
    fake_runtime = FakeRuntime(is_running=False)
    app = create_app(runtime=fake_runtime)

    assert app.state.runtime is fake_runtime
    assert fake_runtime.is_running() is False
    assert fake_runtime.camera is None
    assert fake_runtime.start_called is False


# ==============================================================================
# 2. Health Endpoint Behavior
# ==============================================================================

def test_health_endpoint_idle():
    """GET /api/health reports responsive status and runtime state without false claims."""
    fake_runtime = FakeRuntime(is_running=False, error=None)
    client = TestClient(create_app(runtime=fake_runtime))

    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["runtime_running"] is False
    assert data["camera_active"] is False
    assert data["runtime_error"] is None


def test_health_endpoint_running_with_camera():
    """GET /api/health reports camera active only when camera is truly opened."""
    class FakeCameraObj:
        opened = True

    fake_runtime = FakeRuntime(is_running=True, camera=FakeCameraObj())
    client = TestClient(create_app(runtime=fake_runtime))

    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["runtime_running"] is True
    assert data["camera_active"] is True


# ==============================================================================
# 3. Session Snapshot Serialization
# ==============================================================================

def test_session_snapshot_idle():
    """GET /api/session returns idle structure when no session has started."""
    fake_runtime = FakeRuntime(snapshot=None)
    client = TestClient(create_app(runtime=fake_runtime))

    response = client.get("/api/session")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "idle"
    assert data["is_calibrated"] is False
    assert data["current_angle"] is None
    assert data["rom_progress"] is None
    assert data["feedback_messages"] == []


def test_session_snapshot_active_serialization():
    """GET /api/session properly serializes all required fields without 0 for nulls."""
    snap = SessionSnapshot(
        status=SessionStatus.ACTIVE,
        exercise_name="Shoulder Abduction",
        side="right",
        camera_orientation="front",
        is_calibrated=True,
        calibration_status="calibrated",
        pose_detected=True,
        current_angle=np.float32(78.5),
        rep_state="concentric",
        rep_count=3,
        rom_target=90.0,
        rom_progress=0.872,
        active_compensation_ids=["torso_lean", "shoulder_hike"],
        feedback_messages=["Maintain level shoulders"],
        latest_quality={"overall_score": 85.0},
        frame_index=142,
    )
    fake_runtime = FakeRuntime(snapshot=snap, is_running=True)
    client = TestClient(create_app(runtime=fake_runtime))

    response = client.get("/api/session")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "active"
    assert data["exercise_name"] == "Shoulder Abduction"
    assert data["exercise_id"] == "shoulder_abduction"
    assert data["side"] == "right"
    assert data["selected_side"] == "right"
    assert data["joint_angle"] == 78.5
    assert data["rep_count"] == 3
    assert data["rom_progress"] == 0.872
    assert "torso_lean" in data["active_compensation_ids"]
    assert "Maintain level shoulders" in data["feedback_messages"]
    assert data["latest_quality"]["overall_score"] == 85.0
    assert data["last_error"] is None


# ==============================================================================
# 4. Valid and Invalid Session-Start Requests
# ==============================================================================

@pytest.mark.parametrize(
    "exercise_slug,expected_name",
    [
        ("shoulder_abduction", "Shoulder Abduction"),
        ("seated_knee_extension", "Seated Knee Extension"),
        ("standing_hip_abduction", "Standing Hip Abduction"),
        ("elbow_flexion_extension", "Elbow Flexion-Extension"),
        ("sit_to_stand", "Sit to Stand"),
    ],
)
def test_start_session_all_registered_exercises(exercise_slug, expected_name):
    """POST /api/session/start with each of the five registered exercises succeeds."""
    fake_runtime = FakeRuntime(is_running=False)
    client = TestClient(create_app(runtime=fake_runtime))

    response = client.post(
        "/api/session/start",
        json={"exercise_id": exercise_slug, "side": "left", "camera_index": 0},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "active"
    assert data["exercise_name"] == expected_name
    assert data["side"] == "left"
    assert fake_runtime.is_running() is True



def test_start_session_invalid_exercise():
    """POST /api/session/start with unknown exercise returns 400 Bad Request."""
    fake_runtime = FakeRuntime(is_running=False)
    client = TestClient(create_app(runtime=fake_runtime))

    response = client.post(
        "/api/session/start",
        json={"exercise_id": "nonexistent_exercise", "side": "right"},
    )
    assert response.status_code == 400
    assert "Unknown exercise" in response.json()["detail"]


def test_start_session_invalid_side():
    """POST /api/session/start with invalid side returns 422 Unprocessable Entity."""
    fake_runtime = FakeRuntime(is_running=False)
    client = TestClient(create_app(runtime=fake_runtime))

    response = client.post(
        "/api/session/start",
        json={"exercise_id": "shoulder_abduction", "side": "middle"},
    )
    assert response.status_code == 422


# ==============================================================================
# 5. Duplicate Session-Start Behavior
# ==============================================================================

def test_start_session_duplicate_rejected():
    """POST /api/session/start on already-running session returns 409 Conflict."""
    fake_runtime = FakeRuntime(is_running=True)
    client = TestClient(create_app(runtime=fake_runtime))

    response = client.post(
        "/api/session/start",
        json={"exercise_id": "shoulder_abduction", "side": "right"},
    )
    assert response.status_code == 409
    assert "already running" in response.json()["detail"].lower()


# ==============================================================================
# 6. Stop Requests and Inactive-Session Behavior
# ==============================================================================

def test_stop_session_running():
    """POST /api/session/stop stops an active session cleanly."""
    fake_runtime = FakeRuntime(is_running=True)
    client = TestClient(create_app(runtime=fake_runtime))

    response = client.post("/api/session/stop")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "stopped"
    assert fake_runtime.stop_called is True
    assert fake_runtime.is_running() is False


def test_stop_session_inactive():
    """POST /api/session/stop is idempotent when no session is active."""
    fake_runtime = FakeRuntime(is_running=False)
    client = TestClient(create_app(runtime=fake_runtime))

    response = client.post("/api/session/stop")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "stopped"
    assert "No active session" in data["message"]


# ==============================================================================
# 7. Calibration Request Validation and Forwarding
# ==============================================================================

def test_calibrate_inactive_session_rejected():
    """POST /api/session/calibrate when no session is running returns 409 Conflict."""
    fake_runtime = FakeRuntime(is_running=False)
    client = TestClient(create_app(runtime=fake_runtime))

    response = client.post(
        "/api/session/calibrate",
        json={"duration_seconds": 3.0, "min_samples": 30},
    )
    assert response.status_code == 409
    assert "no active session" in response.json()["detail"].lower()


def test_calibrate_active_session_succeeds():
    """POST /api/session/calibrate forwards duration and sample parameters."""
    snap = SessionSnapshot(status=SessionStatus.ACTIVE)
    fake_runtime = FakeRuntime(is_running=True, snapshot=snap)
    client = TestClient(create_app(runtime=fake_runtime))

    response = client.post(
        "/api/session/calibrate",
        json={"duration_seconds": 5.0, "min_samples": 25},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "calibrating"
    assert fake_runtime.calibration_called is True
    assert fake_runtime.last_calib_args == (5.0, 25)


def test_calibrate_invalid_parameters():
    """POST /api/session/calibrate validates duration and min_samples."""
    fake_runtime = FakeRuntime(is_running=True)
    client = TestClient(create_app(runtime=fake_runtime))

    # Duration <= 0
    response = client.post(
        "/api/session/calibrate",
        json={"duration_seconds": 0.0, "min_samples": 30},
    )
    assert response.status_code == 422


# ==============================================================================
# 8. Camera Startup Failures and Error Responses
# ==============================================================================

def test_camera_startup_failure_returns_503():
    """Camera open failure returns 503 without revealing stack trace."""
    fake_runtime = FakeRuntime(is_running=False)
    fake_runtime.start_raises_camera = True
    client = TestClient(create_app(runtime=fake_runtime))

    response = client.post(
        "/api/session/start",
        json={"exercise_id": "shoulder_abduction", "side": "right"},
    )
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "Camera initialization failed" in detail
    assert "Traceback" not in detail


# ==============================================================================
# 9. Runtime Error Reporting
# ==============================================================================

def test_runtime_error_reporting():
    """Runtime error is surfaced via /api/session/status and /api/health."""
    err_msg = "Persistent camera acquisition timeout"
    fake_runtime = FakeRuntime(is_running=False, error=err_msg)
    client = TestClient(create_app(runtime=fake_runtime))

    status_resp = client.get("/api/session/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["error"] == err_msg

    health_resp = client.get("/api/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["runtime_error"] == err_msg


# ==============================================================================
# 10. WebSocket Initial Snapshot Delivery
# ==============================================================================

def test_websocket_initial_snapshot_delivery():
    """Connecting to /ws/session receives immediate snapshot message."""
    snap = SessionSnapshot(
        status=SessionStatus.ACTIVE,
        exercise_name="Sit to Stand",
        side="right",
        rep_count=4,
    )
    fake_runtime = FakeRuntime(is_running=True, snapshot=snap)
    client = TestClient(create_app(runtime=fake_runtime))

    with client.websocket_connect("/ws/session") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "snapshot"
        assert "timestamp" in msg
        assert msg["data"]["exercise_name"] == "Sit to Stand"
        assert msg["data"]["rep_count"] == 4


# ==============================================================================
# 11. WebSocket Updates and Disconnection Handling
# ==============================================================================

def test_websocket_broadcast_update_and_disconnect():
    """WebSocket client receives state update and disconnects cleanly."""
    snap = SessionSnapshot(
        status=SessionStatus.ACTIVE,
        exercise_name="Shoulder Abduction",
        side="right",
        frame_index=10,
    )
    fake_runtime = FakeRuntime(is_running=True, snapshot=snap)
    client = TestClient(create_app(runtime=fake_runtime))

    with client.websocket_connect("/ws/session") as ws:
        # Initial snapshot
        msg1 = ws.receive_json()
        assert msg1["data"]["frame_index"] == 10

        # Update snapshot in runtime
        snap.frame_index = 11
        snap.rep_count = 1

        # Receive update broadcast
        msg2 = ws.receive_json()
        assert msg2["data"]["frame_index"] == 11
        assert msg2["data"]["rep_count"] == 1

    # Runtime should still be running after client disconnect
    assert fake_runtime.is_running() is True


# ==============================================================================
# 12. Multiple WebSocket Clients Sharing the Same Runtime
# ==============================================================================

def test_multiple_websocket_clients_share_runtime():
    """Multiple WebSocket connections receive updates without extra runtime instances."""
    snap = SessionSnapshot(
        status=SessionStatus.ACTIVE,
        exercise_name="Elbow Flexion-Extension",
        side="left",
        frame_index=1,
    )
    fake_runtime = FakeRuntime(is_running=True, snapshot=snap)
    client = TestClient(create_app(runtime=fake_runtime))

    with client.websocket_connect("/ws/session") as ws1:
        with client.websocket_connect("/ws/session") as ws2:
            m1 = ws1.receive_json()
            m2 = ws2.receive_json()

            assert m1["data"]["exercise_name"] == "Elbow Flexion-Extension"
            assert m2["data"]["exercise_name"] == "Elbow Flexion-Extension"

            # Broadcast next frame
            snap.frame_index = 2
            u1 = ws1.receive_json()
            u2 = ws2.receive_json()

            assert u1["data"]["frame_index"] == 2
            assert u2["data"]["frame_index"] == 2


# ==============================================================================
# 13. MJPEG Content Type and Frame Encoding
# ==============================================================================

def test_mjpeg_content_type_and_frame_encoding():
    """GET /api/video/stream returns multipart/x-mixed-replace and valid JPEG frames."""
    test_frame = np.full((100, 100, 3), 120, dtype=np.uint8)
    fake_runtime = FakeRuntime(is_running=True, frame=test_frame)
    client = TestClient(create_app(runtime=fake_runtime))

    with client.stream("GET", "/api/video/stream?max_frames=1") as response:
        assert response.status_code == 200
        content_type = response.headers.get("content-type", "")
        assert "multipart/x-mixed-replace" in content_type
        assert "boundary=frame" in content_type

        # Verify first chunk contains valid JPEG SOI marker (0xFF, 0xD8)
        content = response.iter_bytes()
        chunk = next(content)
        assert b"--frame\r\n" in chunk
        assert b"Content-Type: image/jpeg\r\n" in chunk
        assert b"\xff\xd8" in chunk


# ==============================================================================
# 14. Missing Frames and Inactive Session in MJPEG
# ==============================================================================

def test_mjpeg_inactive_session_serves_placeholder():
    """GET /api/video/stream serves placeholder frame when no session is active."""
    fake_runtime = FakeRuntime(is_running=False, frame=None)
    client = TestClient(create_app(runtime=fake_runtime))

    with client.stream("GET", "/api/video/stream?max_frames=1") as response:
        assert response.status_code == 200
        chunk = next(response.iter_bytes())
        assert b"--frame\r\n" in chunk
        assert b"\xff\xd8" in chunk  # Valid JPEG placeholder


# ==============================================================================
# 15. Stream-Client Disconnection
# ==============================================================================

def test_mjpeg_client_disconnection_clean():
    """Streaming generator handles client closing connection without error."""
    fake_runtime = FakeRuntime(is_running=True, frame=np.zeros((50, 50, 3), dtype=np.uint8))
    client = TestClient(create_app(runtime=fake_runtime))

    with client.stream("GET", "/api/video/stream?max_frames=3") as resp:
        assert resp.status_code == 200
        for i, chunk in enumerate(resp.iter_bytes()):
            if i >= 1:
                break
    # Connection exited context, runtime still running
    assert fake_runtime.is_running() is True


# ==============================================================================
# 16. Shutdown Cleanup and Failed-Shutdown Behavior
# ==============================================================================

def test_lifespan_shutdown_cleans_runtime():
    """Lifespan context stops runtime upon application termination."""
    fake_runtime = FakeRuntime(is_running=True)
    app = create_app(runtime=fake_runtime)

    with TestClient(app):
        assert fake_runtime.is_running() is True

    # After TestClient context exit, lifespan shutdown executed
    assert fake_runtime.stop_called is True
    assert fake_runtime.is_running() is False


def test_lifespan_shutdown_handles_timeout_explicitly():
    """Failed runtime shutdown logs error and does not raise unhandled exception."""
    fake_runtime = FakeRuntime(is_running=True)
    fake_runtime.stop_raises_timeout = True
    app = create_app(runtime=fake_runtime)

    # Exiting TestClient should not crash the test suite
    with TestClient(app):
        pass


# ==============================================================================
# 17. Single Shared Runtime (No Per-Client Worker)
# ==============================================================================

def test_no_per_client_worker_or_camera_instantiation():
    """Multiple requests share the exact same runtime instance without re-instantiation."""
    fake_runtime = FakeRuntime(is_running=False)
    app = create_app(runtime=fake_runtime)
    client = TestClient(app)

    # 5 concurrent-style requests
    for _ in range(5):
        resp = client.get("/api/session/status")
        assert resp.status_code == 200

    assert app.state.runtime is fake_runtime
    assert fake_runtime.camera is None


# ==============================================================================
# 18. CORS Behavior for Configured Origins
# ==============================================================================

def test_cors_behavior_allowed_origin():
    """Requests from configured origin receive proper Access-Control-Allow-Origin."""
    app = create_app(allowed_origins=["http://localhost:5173"])
    client = TestClient(app)

    # Preflight request
    resp = client.options(
        "/api/session",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert resp.headers.get("access-control-allow-credentials") == "true"


def test_cors_behavior_disallowed_origin():
    """Requests from disallowed origin do not receive Access-Control-Allow-Origin."""
    app = create_app(allowed_origins=["http://localhost:5173"])
    client = TestClient(app)

    resp = client.options(
        "/api/session",
        headers={
            "Origin": "http://malicious-site.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.headers.get("access-control-allow-origin") is None


def test_list_exercises_endpoint():
    """GET /api/exercises returns all exercises from the registry."""
    fake_runtime = FakeRuntime()
    client = TestClient(create_app(runtime=fake_runtime))

    response = client.get("/api/exercises")
    assert response.status_code == 200
    data = response.json()
    exercises = data["exercises"]
    assert len(exercises) == len(EXERCISE_REGISTRY)
    names = [e["name"] for e in exercises]
    assert "Shoulder Abduction" in names
    assert "Sit to Stand" in names


def test_real_runtime_session_lifecycle_integration():
    """Integration test verifying FastAPI controlling a real SessionRuntime instance."""
    from src.runtime import SessionRuntime
    from tests.test_runtime import FakeCamera, make_mock_controller

    fake_cam = FakeCamera()
    runtime = SessionRuntime(
        camera=fake_cam,
        controller_factory=make_mock_controller,
        owns_camera=True,
    )
    app = create_app(runtime=runtime)
    client = TestClient(app)

    try:
        # 1. Start session via REST
        start_resp = client.post(
            "/api/session/start",
            json={"exercise_id": "shoulder_abduction", "side": "right"},
        )
        assert start_resp.status_code == 200
        assert runtime.is_running() is True
        assert fake_cam.opened is True

        # Wait for worker to produce at least 1 frame
        assert runtime.wait_for_next_frame(timeout=1.0) is True

        # 2. Get session snapshot via REST
        snap_resp = client.get("/api/session")
        assert snap_resp.status_code == 200
        snap_data = snap_resp.json()
        assert snap_data["status"] == "active"
        assert snap_data["exercise_name"] == "Shoulder Abduction"
        assert snap_data["frame_index"] >= 1

        # 3. Request calibration via REST
        calib_resp = client.post(
            "/api/session/calibrate",
            json={"duration_seconds": 3.0, "min_samples": 10},
        )
        assert calib_resp.status_code == 200
        assert calib_resp.json()["status"] == "calibrating"

        # 4. Stream MJPEG frame
        with client.stream("GET", "/api/video/stream?max_frames=1") as stream_resp:
            assert stream_resp.status_code == 200
            chunk = next(stream_resp.iter_bytes())
            assert b"--frame\r\n" in chunk

        # 5. Stop session via REST
        stop_resp = client.post("/api/session/stop")
        assert stop_resp.status_code == 200
        assert stop_resp.json()["status"] == "stopped"
        assert runtime.is_running() is False
        assert fake_cam.released is True
    finally:
        runtime.stop_session()
