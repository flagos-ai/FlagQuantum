"""Experimental quantum-error-correction domain."""

from .decoders import Decoder, RepetitionLookupDecoder
from .repetition import run_repetition_memory_experiment
from .types import (
    Correction,
    DetectionEvent,
    RepetitionMemoryResult,
    RepetitionMemoryShot,
    SyndromeRound,
)

__all__ = (
    "Correction",
    "Decoder",
    "DetectionEvent",
    "RepetitionLookupDecoder",
    "RepetitionMemoryResult",
    "RepetitionMemoryShot",
    "SyndromeRound",
    "run_repetition_memory_experiment",
)
