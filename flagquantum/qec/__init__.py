"""Experimental quantum-error-correction domain."""

from .adapters import MatchingDependencyError, PyMatchingDecoder
from .circuit import (
    Detector,
    DetectorLayout,
    LogicalObservable,
    MeasurementRef,
    MemoryCircuit,
    ObservableLayout,
    build_memory_circuit,
)
from .codes import (
    CodeCheck,
    RepetitionCode,
    RotatedSurfaceCode,
    StabilizerCode,
    SteaneCode,
)
from .decoders import (
    Decoder,
    RepetitionLookupDecoder,
    RepetitionStreamingLookupDecoder,
    RepetitionTemporalDecoder,
    StreamingDecoder,
)
from .decoding_graph import DecodingGraph, DecodingGraphEdge
from .dem import DemError, DemMergeRule, DemSample, DetectorErrorModel
from .dem_construction import CssCodeMatrices, css_code_matrices
from .matching import MatchingDecodeResult, MinimumWeightMatchingDecoder
from .noise import (
    PhenomenologicalNoise,
    RepetitionNoiseProfile,
    run_repetition_memory_noise_sweep,
)
from .pauli import Pauli
from .registry import (
    AUTHORITY_NAME,
    CROSS_CHECK_NAME,
    DetectorErrorModelDecoder,
    decoder_names,
    get_decoder,
    register_decoder,
)
from .repetition import run_repetition_memory_experiment
from .sampling import sample_memory_circuit
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
    "AUTHORITY_NAME",
    "CROSS_CHECK_NAME",
    "CodeCheck",
    "CssCodeMatrices",
    "Correction",
    "DecodeResult",
    "Decoder",
    "DecodingGraph",
    "DecodingGraphEdge",
    "DemError",
    "DemMergeRule",
    "DemSample",
    "DetectionEvent",
    "Detector",
    "DetectorErrorModel",
    "DetectorErrorModelDecoder",
    "DetectorLayout",
    "ErrorEvent",
    "ErrorSchedule",
    "LogicalObservable",
    "MatchingDecodeResult",
    "MatchingDependencyError",
    "MeasurementRef",
    "MemoryCircuit",
    "MinimumWeightMatchingDecoder",
    "NoiseSweepPoint",
    "ObservableLayout",
    "Pauli",
    "PauliFrame",
    "PhenomenologicalNoise",
    "PyMatchingDecoder",
    "RepetitionCode",
    "RepetitionNoiseProfile",
    "RepetitionLookupDecoder",
    "RepetitionStreamingLookupDecoder",
    "RepetitionTemporalDecoder",
    "RepetitionMemoryResult",
    "RepetitionMemoryShot",
    "RotatedSurfaceCode",
    "StabilizerCode",
    "SteaneCode",
    "SyndromeRound",
    "StreamingDecoder",
    "build_memory_circuit",
    "css_code_matrices",
    "decoder_names",
    "get_decoder",
    "register_decoder",
    "run_repetition_memory_experiment",
    "run_repetition_memory_noise_sweep",
    "sample_memory_circuit",
)
