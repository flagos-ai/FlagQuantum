"""Experimental quantum-error-correction domain."""

from .circuit import (
    Detector,
    DetectorLayout,
    LogicalObservable,
    MeasurementRef,
    MemoryCircuit,
    ObservableLayout,
    build_memory_circuit,
)
from .codes import CodeCheck, RepetitionCode, StabilizerCode
from .decoders import (
    Decoder,
    RepetitionLookupDecoder,
    RepetitionStreamingLookupDecoder,
    RepetitionTemporalDecoder,
    StreamingDecoder,
)
from .dem import DemError, DemSample, DetectorErrorModel
from .noise import (
    PhenomenologicalNoise,
    RepetitionNoiseProfile,
    run_repetition_memory_noise_sweep,
)
from .pauli import Pauli
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
    "CodeCheck",
    "Correction",
    "DecodeResult",
    "Decoder",
    "DemError",
    "DemSample",
    "DetectionEvent",
    "Detector",
    "DetectorErrorModel",
    "DetectorLayout",
    "ErrorEvent",
    "ErrorSchedule",
    "LogicalObservable",
    "MeasurementRef",
    "MemoryCircuit",
    "NoiseSweepPoint",
    "ObservableLayout",
    "Pauli",
    "PauliFrame",
    "PhenomenologicalNoise",
    "RepetitionCode",
    "RepetitionNoiseProfile",
    "RepetitionLookupDecoder",
    "RepetitionStreamingLookupDecoder",
    "RepetitionTemporalDecoder",
    "RepetitionMemoryResult",
    "RepetitionMemoryShot",
    "StabilizerCode",
    "SyndromeRound",
    "StreamingDecoder",
    "build_memory_circuit",
    "run_repetition_memory_experiment",
    "run_repetition_memory_noise_sweep",
)
