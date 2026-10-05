"""Visual overlay module for PhysioAR.

This module provides a pure presentation layer for rendering the Phase 8 HUD.
It consumes existing structured runtime outputs (angle, ROM, rep state, feedback)
and draws them onto the live OpenCV video feed.

It does NOT perform any assessment logic, recalculate angles, or generate
feedback text.

Phase 8 — Visual Overlay System.
"""

from __future__ import annotations

import cv2
import numpy as np


def draw_angle(frame: np.ndarray, angle: float | None, side: str = "") -> None:
    """Draw the current tracked exercise angle.

    Displays 'Angle: X°' or 'Angle: N/A' if unavailable.

    Args:
        frame: OpenCV BGR frame (modified in-place).
        angle: The calculated exercise angle in degrees, or None.
        side: Optional side prefix, e.g., 'left' or 'right'.
    """
    if angle is not None:
        text = f"Angle: {angle:.0f}\xb0"
        color = (0, 255, 0)
    else:
        text = "Angle: N/A"
        color = (0, 100, 255)

    if side:
        text = f"{side.capitalize()} {text}"

    # Position in the top right corner.
    h, w = frame.shape[:2]
    cv2.putText(
        frame, text,
        (w - 200, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
    )


def draw_rom_progress(
    frame: np.ndarray,
    current_angle: float | None,
    target_rom: float,
    rep_state: str,
) -> None:
    """Draw a progress bar for the current movement toward target ROM.

    Progress is bounded [0, 1].

    Args:
        frame: OpenCV BGR frame (modified in-place).
        current_angle: The calculated exercise angle in degrees, or None.
        target_rom: Target ROM from the exercise definition.
        rep_state: Current repetition state (e.g. 'RISING', 'DOWN').
    """
    h, w = frame.shape[:2]
    bar_x = 20
    bar_y = h - 60
    bar_w = 200
    bar_h = 20

    # Draw label and state.
    cv2.putText(
        frame, "ROM Progress",
        (bar_x, bar_y - 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
    )
    cv2.putText(
        frame, f"State: {rep_state}",
        (bar_x + 110, bar_y - 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1,
    )

    # Draw background bar.
    cv2.rectangle(
        frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
        (50, 50, 50), -1,
    )

    if current_angle is not None and target_rom > 0:
        # Calculate progress [0, 1].
        progress = max(0.0, min(current_angle / target_rom, 1.0))
        fill_w = int(bar_w * progress)

        # Draw filled portion.
        cv2.rectangle(
            frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
            (0, 255, 0), -1,
        )

        text = f"{current_angle:.0f} / {target_rom:.0f}\xb0"
        color = (255, 255, 255)
    else:
        text = "N/A"
        color = (150, 150, 150)

    # Draw progress text.
    cv2.putText(
        frame, text,
        (bar_x, bar_y - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
    )

    # Draw border.
    cv2.rectangle(
        frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
        (200, 200, 200), 1,
    )


def draw_feedback(
    frame: np.ndarray,
    feedback_messages: list[str],
) -> None:
    """Draw active corrective feedback messages.

    Args:
        frame: OpenCV BGR frame (modified in-place).
        feedback_messages: List of formatted feedback strings.
    """
    if not feedback_messages:
        return

    h, w = frame.shape[:2]
    # Place on the right side, below the angle.
    base_x = w - 450
    base_y = 100

    cv2.putText(
        frame, "CORRECTIONS",
        (base_x, base_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 2,
    )

    for i, msg in enumerate(feedback_messages):
        # We need to wrap text if it's too long, or just render it.
        # Since OpenCV doesn't natively wrap text, we assume messages
        # fit within 450px (~60 chars at scale 0.5), or we can split manually.
        # A simple manual split on ' - ' if present, or just simple wrapping.
        parts = msg.split(" — ")
        y_offset = base_y + 25 + (i * 45)

        if len(parts) == 2:
            title = f"\u26A0 {parts[0]}"
            detail = parts[1]
            cv2.putText(
                frame, title,
                (base_x, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 2,
            )
            cv2.putText(
                frame, detail,
                (base_x + 15, y_offset + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
            )
        else:
            cv2.putText(
                frame, f"\u26A0 {msg}",
                (base_x, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 2,
            )


def draw_rep_info(
    frame: np.ndarray,
    rep_count: int,
) -> None:
    """Draw repetition count.

    Args:
        frame: OpenCV BGR frame (modified in-place).
        rep_count: Current completed repetitions.
    """
    # Position: near ROM bar.
    h, w = frame.shape[:2]
    x = 20
    y = h - 100
    cv2.putText(
        frame, f"Reps: {rep_count}",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
    )


def draw_hud(
    frame: np.ndarray,
    *,
    angle: float | None,
    target_rom: float,
    rep_state: str,
    rep_count: int,
    feedback_messages: list[str],
    side: str = "",
) -> None:
    """Draw the complete Phase 8 HUD.

    Args:
        frame: OpenCV BGR frame (modified in-place).
        angle: The calculated exercise angle in degrees, or None.
        target_rom: Target ROM from the exercise definition.
        rep_state: Current repetition state.
        rep_count: Current completed repetitions.
        feedback_messages: List of formatted feedback strings.
        side: The selected exercise side.
    """
    draw_angle(frame, angle, side)
    draw_rom_progress(frame, angle, target_rom, rep_state)
    draw_rep_info(frame, rep_count)
    draw_feedback(frame, feedback_messages)
