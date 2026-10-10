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
from src.exercises.standing_hip_abduction import STANDING_HIP_ABDUCTION
from src.exercises.elbow_flexion_extension import ELBOW_FLEXION_EXTENSION
from src.exercises.sit_to_stand import SIT_TO_STAND

# Central registry of all available exercises.
# New exercises should be appended here.
EXERCISE_REGISTRY: list[ExerciseDefinition] = [
    SHOULDER_ABDUCTION,
    SEATED_KNEE_EXTENSION,
    STANDING_HIP_ABDUCTION,
    ELBOW_FLEXION_EXTENSION,
    SIT_TO_STAND,
]

__all__ = [
    "CameraOrientation",
    "ExerciseDefinition",
    "EXERCISE_REGISTRY",
    "SHOULDER_ABDUCTION",
    "SEATED_KNEE_EXTENSION",
    "STANDING_HIP_ABDUCTION",
    "ELBOW_FLEXION_EXTENSION",
    "SIT_TO_STAND",
]
