"""FastAPI application factory and lifecycle orchestration for PhysioAR."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
import anyio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router as rest_router
from src.api.streaming import router as streaming_router
from src.api.websocket import router as ws_router
from src.runtime import SessionRuntime

logger = logging.getLogger(__name__)

DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle."""
    logger.info("PhysioAR API application starting up...")
    # Runtime should already be bound in app.state
    yield

    logger.info("PhysioAR API application shutting down...")
    # 1. Stop WebSocket broadcaster if active
    ws_manager = getattr(app.state, "ws_manager", None)
    if ws_manager is not None:
        try:
            await ws_manager.stop()
        except Exception as e:
            logger.warning(f"Error stopping WebSocket manager during shutdown: {e}")

    # 2. Stop runtime if running
    runtime: Optional[SessionRuntime] = getattr(app.state, "runtime", None)
    if runtime is not None and runtime.is_running():
        logger.info("Stopping active SessionRuntime...")
        try:
            # Offload stop_session to worker thread with bounded timeout
            await anyio.to_thread.run_sync(runtime.stop_session, 2.0)
            logger.info("SessionRuntime stopped successfully.")
        except TimeoutError as e:
            logger.error(
                f"SessionRuntime worker thread failed to stop within timeout: {e}"
            )
        except Exception as e:
            logger.error(f"Unexpected error stopping SessionRuntime on shutdown: {e}")


def create_app(
    runtime: Optional[SessionRuntime] = None,
    allowed_origins: Optional[list[str]] = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Supports dependency injection of a custom or mock SessionRuntime for headless testing.
    Importing or calling create_app does NOT open the camera or start background workers.

    Args:
        runtime: Optional SessionRuntime instance to bind. If None, instantiates a default
            SessionRuntime (which remains idle until start_session is called).
        allowed_origins: Optional list of CORS origins. Defaults to local Vite/React origins
            or values configured in PHYSIOAR_CORS_ORIGINS.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="PhysioAR API",
        description="AI-assisted Physiotherapy Exercise Analysis Engine API",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Attach runtime to app state
    if runtime is None:
        runtime = SessionRuntime()
    app.state.runtime = runtime
    app.state.ws_manager = None

    # Determine CORS origins
    env_origins = os.environ.get("PHYSIOAR_CORS_ORIGINS")
    if allowed_origins is not None:
        origins = allowed_origins
    elif env_origins:
        origins = [o.strip() for o in env_origins.split(",") if o.strip()]
    else:
        origins = DEFAULT_ALLOWED_ORIGINS

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount routers
    app.include_router(rest_router)
    app.include_router(streaming_router)
    app.include_router(ws_router)

    return app


# Default app instance for ASGI servers (e.g., uvicorn src.api.app:app)
app = create_app()
