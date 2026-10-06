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
    ancilla_bands,
    code_names,
    get_code,
    register_code,
)
from .context import (
    DecoderContext,
    DecoderInputs,
    MeasurementMap,
    decoder_context_from_memory_circuit,
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
from .sampling import (
    MeasurementSamples,
    sample_memory_circuit,
    sample_memory_measurements,
)
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
    "DecoderContext",
    "DecoderInputs",
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
    "MeasurementMap",
    "MeasurementSamples",
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
    "ancilla_bands",
    "build_memory_circuit",
    "code_names",
    "css_code_matrices",
    "decoder_context_from_memory_circuit",
    "decoder_names",
    "get_code",
    "get_decoder",
    "register_code",
    "register_decoder",
    "run_repetition_memory_experiment",
    "run_repetition_memory_noise_sweep",
    "sample_memory_circuit",
    "sample_memory_measurements",
)
