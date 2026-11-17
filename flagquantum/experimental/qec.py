"""Experimental QEC workflows using the bundled Sparse Blossom solver."""

from __future__ import annotations

from flagquantum.qec.adapters.stim import compile_stim_sampler
from flagquantum.qec.decoders.sparse_blossom import SparseBlossomDecoder
from flagquantum.qec.dem.graph import DecodingGraph
from flagquantum.qec.execution.experiments import (
    StimExperimentResult,
    _collect_fixed_shots,
    _wilson_interval,
)
from flagquantum.qec.execution.sampling import _checked_seed, _checked_shots

__all__ = ("SparseBlossomDecoder", "run_stim_matching_experiment")


def run_stim_matching_experiment(
    circuit_text: str,
    *,
    graph: DecodingGraph,
    shots: int,
    batch_size: int,
    seed: int | None = None,
) -> StimExperimentResult:
    """Estimate logical failures with the bundled decoder and original-circuit shots.

    Provide a matching graph whose detector and observable order agree with the
    annotated Stim circuit. The graph supplies decoding weights. All draws use
    the original circuit's noise model. The existing Stim sampling requirements
    apply. The solver has no runtime dependency on the PyMatching Python package.

    Example:
        >>> from importlib.util import find_spec
        >>> from flagquantum.experimental.qec import run_stim_matching_experiment
        >>> from flagquantum.qec import DecodingGraph, DetectorErrorModel
        >>> source = "R 0\\nX_ERROR(0.1) 0\\nM 0\\nDETECTOR rec[-1]\\nOBSERVABLE_INCLUDE(0) rec[-1]"
        >>> model = DetectorErrorModel.from_stim_text("error(0.1) D0 L0\\ndetector D0\\nlogical_observable L0")
        >>> graph = DecodingGraph.from_detector_error_model(model)
        >>> native = find_spec('flagquantum.qec.decoders._sparse_blossom')
        >>> if native is not None and find_spec('stim') is not None:
        ...     result = run_stim_matching_experiment(source, graph=graph, shots=100, batch_size=32, seed=1)
        ...     assert result.logical_failures == 0
    """
    if not isinstance(circuit_text, str):
        raise TypeError("circuit_text must be a string")
    shots = _checked_shots(shots)
    batch_size = _checked_shots(batch_size)
    seed = _checked_seed(seed)
    decoder = SparseBlossomDecoder(graph)
    sampler = compile_stim_sampler(circuit_text, seed=seed)
    failures, per_observable = _collect_fixed_shots(
        sampler,
        None,
        shots=shots,
        batch_size=batch_size,
        detectors=graph.num_detectors,
        observables=graph.num_observables,
        predict_dense=decoder.decode_batch,
    )
    return StimExperimentResult(
        shots=shots,
        logical_failures=failures,
        logical_error_rate=failures / shots,
        observable_failures=per_observable,
        confidence_interval=_wilson_interval(failures, shots),
        seed=seed,
        sampling_source="stim_circuit",
    )
