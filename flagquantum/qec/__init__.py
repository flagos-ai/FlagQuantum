"""Experimental quantum-error-correction domain."""

from .decoders import Decoder, RepetitionLookupDecoder
from .repetition import run_repetition_memory_experiment
from .types import (
    Correction,
    DecodeResult,
    DetectionEvent,
    ErrorEvent,
    ErrorSchedule,
    PauliFrame,
    RepetitionMemoryResult,
    RepetitionMemoryShot,
    SyndromeRound,
)

__all__ = (
    "Correction",
    "DecodeResult",
    "Decoder",
    "DetectionEvent",
    "ErrorEvent",
    "ErrorSchedule",
    "PauliFrame",
    "RepetitionLookupDecoder",
    "RepetitionMemoryResult",
    "RepetitionMemoryShot",
    "SyndromeRound",
    "run_repetition_memory_experiment",
)
