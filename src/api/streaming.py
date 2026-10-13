"""MJPEG video streaming endpoint for PhysioAR annotated frames."""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator, Optional
import cv2
import numpy as np
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse

from src.api.routes import get_runtime
from src.runtime import SessionRuntime

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/video", tags=["Video"])

# Pre-encoded placeholder frame used when no session is active or no frame has been captured
_PLACEHOLDER_JPEG: bytes | None = None


def get_placeholder_frame() -> bytes:
    """Generate or return a cached JPEG placeholder image for inactive streams."""
    global _PLACEHOLDER_JPEG
    if _PLACEHOLDER_JPEG is None:
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw placeholder background text
        cv2.putText(
            placeholder,
            "PhysioAR Video Stream",
            (140, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            placeholder,
            "Waiting for active session...",
            (155, 260),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 180, 180),
            1,
        )
        success, encoded = cv2.imencode(".jpg", placeholder, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if success:
            _PLACEHOLDER_JPEG = encoded.tobytes()
        else:
            _PLACEHOLDER_JPEG = b""
    return _PLACEHOLDER_JPEG


async def generate_mjpeg_frames(
    runtime: SessionRuntime,
    request: Optional[Request] = None,
    max_frames: Optional[int] = None,
) -> AsyncGenerator[bytes, None]:
    """Asynchronous generator that encodes and yields multipart JPEG frames on demand.

    Retrieves frames from the single shared SessionRuntime buffer. Does not
    spawn background threads or duplicate camera capture.

    Frame delivery is throttled to approximately 30 FPS to conserve bandwidth
    and CPU utilization.
    """
    frame_interval = 0.033  # ~30 FPS max
    frames_yielded = 0
    logger.debug("MJPEG client connected.")

    try:
        while True:
            # Check client disconnection
            if request is not None and await request.is_disconnected():
                logger.debug("MJPEG client disconnected (detected via request).")
                break

            # 1. Fetch latest frame without holding long locks
            frame = runtime.get_latest_frame(copy=True)

            if frame is not None:
                # 2. Encode to JPEG on demand
                success, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 80],
                )
                if success:
                    jpeg_bytes = encoded.tobytes()
                else:
                    jpeg_bytes = get_placeholder_frame()
            else:
                jpeg_bytes = get_placeholder_frame()

            # 3. Format MJPEG multipart chunk
            chunk = (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpeg_bytes)).encode("ascii") + b"\r\n\r\n"
                + jpeg_bytes
                + b"\r\n"
            )

            yield chunk
            frames_yielded += 1

            if max_frames is not None and frames_yielded >= max_frames:
                break

            await asyncio.sleep(frame_interval)

    except (asyncio.CancelledError, GeneratorExit):
        logger.debug("MJPEG client disconnected.")
    except Exception as e:
        logger.warning(f"Error in MJPEG streaming generator: {e}")


@router.get("/stream")
async def video_stream(
    request: Request,
    max_frames: Optional[int] = Query(default=None),
    runtime: SessionRuntime = Depends(get_runtime),
) -> StreamingResponse:
    """Stream real-time annotated camera frames via Motion JPEG (MJPEG).

    Technical notes & browser compatibility:
    - Content-Type: multipart/x-mixed-replace with frame boundary.
    - Consumes the single shared SessionRuntime instance; multiple clients can
      subscribe simultaneously without duplicate capture loops.
    - In standard browsers (Chrome, Firefox, Safari, Edge), this endpoint can be
      used directly as the `src` attribute of an `<img>` tag:
      `<img src="http://localhost:8000/api/video/stream" />`.
    - HTTP/1.1 limits browser connections to ~6 concurrent persistent requests per host.
    - MJPEG streams frames as independent JPEG images without inter-frame temporal
      compression (H.264/WebRTC), which is ideal for low-latency local LAN / localhost
      desktop setups but requires higher bandwidth than video codecs.
    """
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
        "Connection": "close",
    }
    return StreamingResponse(
        generate_mjpeg_frames(runtime, request=request, max_frames=max_frames),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers=headers,
    )
