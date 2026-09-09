"""Experimental quantum-error-correction domain."""

from .decoders import Decoder, RepetitionLookupDecoder
from .noise import RepetitionNoiseProfile, run_repetition_memory_noise_sweep
from .repetition import run_repetition_memory_experiment
from .types import (
    Correction,
    DecodeResult,
    DetectionEvent,
    ErrorEvent,
    ErrorSchedule,
    NoiseSweepPoint,
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
    "NoiseSweepPoint",
    "PauliFrame",
    "RepetitionNoiseProfile",
    "RepetitionLookupDecoder",
    "RepetitionMemoryResult",
    "RepetitionMemoryShot",
    "SyndromeRound",
    "run_repetition_memory_experiment",
    "run_repetition_memory_noise_sweep",
)
