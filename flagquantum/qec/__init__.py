"""Experimental quantum-error-correction domain."""

from flagquantum.qec.circuits.memory import (
    Detector,
    DetectorLayout,
    LogicalObservable,
    MeasurementRef,
    MemoryCircuit,
    ObservableLayout,
    build_memory_circuit,
)
from flagquantum.qec.codes.bicycle import bivariate_bicycle_code
from flagquantum.qec.codes.css import CssCode
from flagquantum.qec.codes.floquet import (
    FloquetCode,
    MeasurementPhase,
    ring_floquet_code,
)
from flagquantum.qec.codes.matrices import CssCodeMatrices, css_code_matrices
from flagquantum.qec.codes.qldpc import qldpc_code
from flagquantum.qec.codes.reed_muller import reed_muller_code
from flagquantum.qec.codes.subsystem import (
    SubsystemCode,
    tesseract_code,
    tesseract_column_swap,
    tesseract_free_cnot_pairs,
)
from flagquantum.qec.codes.surface.zxxz import ZxxzSurfaceCode
from flagquantum.qec.decoders.belief_propagation import (
    BeliefPropagationDecoder,
    BeliefPropagationDecodeResult,
)
from flagquantum.qec.decoders.bposd import (
    BeliefPropagationOsdDecoder,
    BeliefPropagationOsdDecodeResult,
)
from flagquantum.qec.decoders.matching import (
    MatchingDecodeResult,
    MinimumWeightMatchingDecoder,
)
from flagquantum.qec.decoders.registry import (
    AUTHORITY_NAME,
    BELIEF_PROPAGATION_NAME,
    BELIEF_PROPAGATION_OSD_NAME,
    CROSS_CHECK_NAME,
    SLIDING_WINDOW_NAME,
    DetectorErrorModelDecoder,
    DetectorErrorModelDecodeResult,
    GraphlikeDetectorErrorModelDecoder,
    decoder_names,
    get_decoder,
    register_decoder,
)
from flagquantum.qec.decoders.sliding_window import (
    SlidingWindowDecoder,
    SlidingWindowDecodeResult,
    SlidingWindowMatchingDecoder,
    dem_chunks_to_d_sparse,
    dem_chunks_to_o_sparse,
    dem_chunks_to_pcm,
)
from flagquantum.qec.dem.chunks import (
    ChunkLayout,
    DemChunk,
    DemChunkSpec,
    DemChunksSpec,
    DemSeam,
    PhaseId,
    SeamId,
    dem_chunk_from_spec,
    dem_chunks_from_spec,
    dem_close,
    dem_close_all,
    dem_stitch,
    dem_stitch_all,
    dem_stitch_merged,
)
from flagquantum.qec.dem.circuit import (
    circuit_from_detector_error_model,
    detector_error_model_from_circuit,
)
from flagquantum.qec.dem.graph import DecodingGraph, DecodingGraphEdge
from flagquantum.qec.execution.context import (
    DecoderContext,
    DecoderInputs,
    MeasurementMap,
    decoder_context_from_memory_circuit,
)
from flagquantum.qec.execution.repetition import run_repetition_memory_experiment
from flagquantum.qec.execution.sampling import (
    MeasurementSamples,
    MemorySampler,
    compile_memory_sampler,
    sample_memory_circuit,
    sample_memory_measurements,
)

from .adapters import MatchingDependencyError, PyMatchingDecoder
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
    toric_code,
    triangular_colour_code,
)
from .decoders import (
    Decoder,
    RepetitionLookupDecoder,
    RepetitionStreamingLookupDecoder,
    RepetitionTemporalDecoder,
    StreamingDecoder,
)
from .dem import DemError, DemMergeRule, DemSample, DetectorErrorModel
from .logical import certify_logical_product, derive_anticommuting_logical_product
from .noise import (
    PhenomenologicalNoise,
    RepetitionNoiseProfile,
    run_repetition_memory_noise_sweep,
)
from .pauli import Pauli
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
    "BELIEF_PROPAGATION_NAME",
    "BELIEF_PROPAGATION_OSD_NAME",
    "CROSS_CHECK_NAME",
    "SLIDING_WINDOW_NAME",
    "BeliefPropagationDecodeResult",
    "BeliefPropagationDecoder",
    "BeliefPropagationOsdDecodeResult",
    "BeliefPropagationOsdDecoder",
    "ChunkLayout",
    "CodeCheck",
    "Correction",
    "CssCode",
    "CssCodeMatrices",
    "DecodeResult",
    "Decoder",
    "DecoderContext",
    "DecoderInputs",
    "DecodingGraph",
    "DecodingGraphEdge",
    "DemChunk",
    "DemChunkSpec",
    "DemChunksSpec",
    "DemError",
    "DemMergeRule",
    "DemSample",
    "DemSeam",
    "DetectionEvent",
    "Detector",
    "DetectorErrorModel",
    "DetectorErrorModelDecodeResult",
    "DetectorErrorModelDecoder",
    "DetectorLayout",
    "ErrorEvent",
    "ErrorSchedule",
    "FloquetCode",
    "GraphlikeDetectorErrorModelDecoder",
    "LogicalObservable",
    "MatchingDecodeResult",
    "MatchingDependencyError",
    "MeasurementMap",
    "MeasurementPhase",
    "MeasurementRef",
    "MeasurementSamples",
    "MemoryCircuit",
    "MinimumWeightMatchingDecoder",
    "NoiseSweepPoint",
    "ObservableLayout",
    "Pauli",
    "PauliFrame",
    "PhaseId",
    "PhenomenologicalNoise",
    "PyMatchingDecoder",
    "RepetitionCode",
    "RepetitionLookupDecoder",
    "RepetitionMemoryResult",
    "RepetitionMemoryShot",
    "RepetitionNoiseProfile",
    "RepetitionStreamingLookupDecoder",
    "RepetitionTemporalDecoder",
    "RotatedSurfaceCode",
    "SeamId",
    "SlidingWindowDecodeResult",
    "SlidingWindowDecoder",
    "SlidingWindowMatchingDecoder",
    "StabilizerCode",
    "SteaneCode",
    "StreamingDecoder",
    "SubsystemCode",
    "SyndromeRound",
    "ZxxzSurfaceCode",
    "ancilla_bands",
    "bivariate_bicycle_code",
    "build_memory_circuit",
    "certify_logical_product",
    "circuit_from_detector_error_model",
    "code_names",
    "css_code_matrices",
    "decoder_context_from_memory_circuit",
    "decoder_names",
    "dem_chunk_from_spec",
    "dem_chunks_from_spec",
    "dem_chunks_to_d_sparse",
    "dem_chunks_to_o_sparse",
    "dem_chunks_to_pcm",
    "dem_close",
    "dem_close_all",
    "dem_stitch",
    "dem_stitch_all",
    "dem_stitch_merged",
    "derive_anticommuting_logical_product",
    "detector_error_model_from_circuit",
    "get_code",
    "get_decoder",
    "qldpc_code",
    "reed_muller_code",
    "register_code",
    "register_decoder",
    "ring_floquet_code",
    "run_repetition_memory_experiment",
    "run_repetition_memory_noise_sweep",
    "sample_memory_circuit",
    "sample_memory_measurements",
    "tesseract_code",
    "tesseract_column_swap",
    "tesseract_free_cnot_pairs",
    "toric_code",
    "triangular_colour_code",
    "MemorySampler",
    "compile_memory_sampler",
)
