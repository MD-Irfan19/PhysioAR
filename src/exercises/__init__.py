"""PhysioAR exercise definitions package.

Provides:
  - ``EXERCISE_REGISTRY``: A list of all available ExerciseDefinition
    instances. New exercises are added here.
  - Re-exports of base types for convenience.

Phase 3 — Exercise Definition Structure.
"""

from src.exercises.base import CameraOrientation, ExerciseDefinition
from src.exercises.shoulder_abduction import SHOULDER_ABDUCTION
from src.exercises.seated_knee_extension import SEATED_KNEE_EXTENSION

# Central registry of all available exercises.
# New exercises should be appended here.
EXERCISE_REGISTRY: list[ExerciseDefinition] = [
    SHOULDER_ABDUCTION,
    SEATED_KNEE_EXTENSION,
]

__all__ = [
    "CameraOrientation",
    "ExerciseDefinition",
    "EXERCISE_REGISTRY",
    "SHOULDER_ABDUCTION",
    "SEATED_KNEE_EXTENSION",
]
