"""WebSocket handler and broadcaster for PhysioAR live state streaming."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional
import anyio
from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect

from src.api.schemas import snapshot_to_dict
from src.runtime import SessionRuntime

logger = logging.getLogger(__name__)

router = APIRouter(tags=["WebSocket"])


class WebSocketManager:
    """Manages active WebSocket connections and broadcasts runtime updates."""

    def __init__(self, runtime: SessionRuntime) -> None:
        self.runtime = runtime
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._broadcast_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def connect(self, websocket: WebSocket) -> None:
        """Register a new client and send initial snapshot immediately."""
        await websocket.accept()
        async with self._lock:
            self._clients.add(websocket)

            # Send current snapshot upon connection
            snap = self.runtime.get_snapshot()
            initial_msg = {
                "type": "snapshot",
                "timestamp": time.time(),
                "data": snapshot_to_dict(snap),
            }
            try:
                await websocket.send_json(initial_msg)
            except Exception as e:
                logger.debug(f"Failed to send initial snapshot to WebSocket client: {e}")
                self._clients.discard(websocket)
                return

            if self._broadcast_task is None or self._broadcast_task.done():
                self._stop_event.clear()
                self._broadcast_task = asyncio.create_task(self._broadcast_loop())

    async def disconnect(self, websocket: WebSocket) -> None:
        """Unregister a client and terminate broadcast task if no clients remain."""
        async with self._lock:
            self._clients.discard(websocket)
            if not self._clients and self._broadcast_task is not None:
                self._stop_event.set()
                self._broadcast_task.cancel()
                self._broadcast_task = None

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Send message to all registered clients, pruning broken connections."""
        async with self._lock:
            if not self._clients:
                return

            disconnected = []
            for client in list(self._clients):
                try:
                    await client.send_json(message)
                except Exception:
                    disconnected.append(client)

            for client in disconnected:
                self._clients.discard(client)

    async def stop(self) -> None:
        """Gracefully stop the broadcaster during app shutdown."""
        self._stop_event.set()
        if self._broadcast_task is not None:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass
            self._broadcast_task = None
        async with self._lock:
            for client in list(self._clients):
                try:
                    await client.close()
                except Exception:
                    pass
            self._clients.clear()

    async def _broadcast_loop(self) -> None:
        """Background loop broadcasting runtime state updates."""
        last_frame_index = -1
        last_error_sent: Optional[str] = None

        while not self._stop_event.is_set():
            try:
                if self.runtime.is_running():
                    # Wait for next frame without blocking event loop
                    has_new = await anyio.to_thread.run_sync(
                        self.runtime.wait_for_next_frame, 0.1
                    )
                    snap = self.runtime.get_snapshot()
                    if snap is not None and snap.frame_index != last_frame_index:
                        last_frame_index = snap.frame_index
                        msg = {
                            "type": "snapshot",
                            "timestamp": time.time(),
                            "data": snapshot_to_dict(snap),
                        }
                        await self.broadcast(msg)
                else:
                    # Check for runtime error if not running
                    err = self.runtime.get_error()
                    if err is not None and err != last_error_sent:
                        last_error_sent = err
                        msg = {
                            "type": "error",
                            "timestamp": time.time(),
                            "error": err,
                        }
                        await self.broadcast(msg)

                    await asyncio.sleep(0.15)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in WebSocket broadcast loop: {e}")
                await asyncio.sleep(0.1)


def get_ws_manager(app: FastAPI, runtime: SessionRuntime) -> WebSocketManager:
    """Retrieve or initialize the WebSocketManager on app.state."""
    if not hasattr(app.state, "ws_manager") or app.state.ws_manager is None:
        app.state.ws_manager = WebSocketManager(runtime)
    return app.state.ws_manager


@router.websocket("/ws/session")
async def websocket_session_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint streaming live session snapshots."""
    runtime: SessionRuntime = websocket.app.state.runtime
    manager = get_ws_manager(websocket.app, runtime)

    await manager.connect(websocket)
    try:
        while True:
            # Keep connection open and receive any client messages/pings
            _ = await websocket.receive_text()
    except (WebSocketDisconnect, ConnectionResetError):
        pass
    except Exception as e:
        logger.debug(f"WebSocket client connection ended: {e}")
    finally:
        await manager.disconnect(websocket)
