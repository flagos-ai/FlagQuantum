"""Experimental quantum-error-correction domain."""

from .adapters import MatchingDependencyError, PyMatchingDecoder
from .bposd import (
    BeliefPropagationOsdDecoder,
    BeliefPropagationOsdDecodeResult,
)
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
    CssCode,
    RepetitionCode,
    StabilizerCode,
    SteaneCode,
    toric_code,
    triangular_colour_code,
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
from .dem_circuit import (
    circuit_from_detector_error_model,
    detector_error_model_from_circuit,
)
from .dem_construction import CssCodeMatrices, css_code_matrices
from .logical import certify_logical_product, derive_anticommuting_logical_product
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
    SLIDING_WINDOW_NAME,
    DetectorErrorModelDecoder,
    decoder_names,
    get_decoder,
    register_decoder,
)
from .repetition import run_repetition_memory_experiment
from .sampling import sample_memory_circuit
from .sliding_window import SlidingWindowMatchingDecoder
from .surface import RotatedSurfaceCode, ZxxzSurfaceCode
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
    "BeliefPropagationOsdDecodeResult",
    "BeliefPropagationOsdDecoder",
    "CROSS_CHECK_NAME",
    "CodeCheck",
    "CssCode",
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
    "SLIDING_WINDOW_NAME",
    "SlidingWindowMatchingDecoder",
    "StabilizerCode",
    "SteaneCode",
    "SyndromeRound",
    "StreamingDecoder",
    "build_memory_circuit",
    "certify_logical_product",
    "circuit_from_detector_error_model",
    "derive_anticommuting_logical_product",
    "decoder_context_from_memory_circuit",
    "css_code_matrices",
    "decoder_names",
    "detector_error_model_from_circuit",
    "get_decoder",
    "register_decoder",
    "run_repetition_memory_experiment",
    "run_repetition_memory_noise_sweep",
    "sample_memory_circuit",
    "toric_code",
    "triangular_colour_code",
    "ZxxzSurfaceCode",
)
