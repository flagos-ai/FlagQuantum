"""Fixed-shot memory experiments composed from existing QEC boundaries."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral
from statistics import NormalDist
from typing import Literal

import torch

from flagquantum.core.noise import NoiseModel

from ...errors import ValidationError
from ...simulation.stabilizer._validation import _validate_seed
from ..adapters.stim import (
    _stim_circuit_to_matching_dem_text,
    compile_stim_sampler,
    memory_to_stim_text,
    stim_circuit_to_dem_text,
)
from ..circuits.memory import MemoryCircuit
from ..decoders.batch import decode_batch
from ..decoders.registry import DetectorErrorModelDecoder, decoder_names, get_decoder
from ..decoders.sparse_blossom import SparseBlossomDecoder
from ..dem import DetectorErrorModel
from ..dem.graph import DecodingGraph
from ..noise import PhenomenologicalNoise
from .sampling import MemorySampler, _checked_seed, _checked_shots

__all__ = (
    "MemoryExperimentResult",
    "run_memory_experiment",
    "StimExperimentResult",
    "run_stim_experiment",
)

_SPARSE_BLOSSOM = "sparse_blossom"


@dataclass(frozen=True)
class MemoryExperimentResult:
    """Completed fixed-shot counts and a two-sided 95% Wilson interval.

    Counts describe any-logical and per-observable prediction disagreements.
    Records own immutable scalars only; no persistence schema is promised.
    """

    shots: int
    logical_failures: int
    logical_error_rate: float
    observable_failures: tuple[int, ...]
    confidence_interval: tuple[float, float]
    seed: int | None
    sampling_source: Literal["memory_circuit"]


@dataclass(frozen=True)
class StimExperimentResult:
    """Fixed-shot logical disagreements from an original Stim circuit source."""

    shots: int
    logical_failures: int
    logical_error_rate: float
    observable_failures: tuple[int, ...]
    confidence_interval: tuple[float, float]
    seed: int | None
    sampling_source: Literal["stim_circuit"]


def _wilson_interval(failures: int, shots: int) -> tuple[float, float]:
    """Two-sided 95% score interval without continuity correction."""
    for name, value in (("failures", failures), ("shots", shots)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if shots <= 0 or not 0 <= failures <= shots:
        raise ValueError(
            "an interval requires positive shots and 0 <= failures <= shots"
        )
    proportion = failures / shots
    quantile = NormalDist().inv_cdf(0.975)
    squared_quantile = quantile**2
    denominator = 1 + squared_quantile / shots
    center = (proportion + squared_quantile / (2 * shots)) / denominator
    half_width = (
        quantile
        * math.sqrt(
            proportion * (1 - proportion) / shots + squared_quantile / (4 * shots**2)
        )
        / denominator
    )
    lower = 0.0 if failures == 0 else max(0.0, center - half_width)
    upper = 1.0 if failures == shots else min(1.0, center + half_width)
    return lower, upper


def _check_bits(bits: torch.Tensor, shots: int, columns: int) -> None:
    if (
        not isinstance(bits, torch.Tensor)
        or bits.device.type != "cpu"
        or bits.shape != (shots, columns)
        or not bool(((bits == 0) | (bits == 1)).all())
    ):
        raise ValidationError(
            "sample must contain binary CPU tensors of the expected shape"
        )


def run_memory_experiment(
    memory: MemoryCircuit,
    *,
    noise: PhenomenologicalNoise | NoiseModel,
    decoder: str,
    shots: int,
    batch_size: int,
    seed: int | None = None,
) -> MemoryExperimentResult:
    """Sample and decode exactly ``shots`` original memory-circuit repetitions.

    Requires the optional Stim extra; the pymatching selection also needs its
    existing optional dependencies. Decoder names use constructor defaults.
    Select `decoder="sparse_blossom"` for the bundled CPU batch solver. Its
    graphlike decomposition is used for decoding the original-circuit samples.
    Batches advance one seeded stream, including a final partial batch. Different
    batch sizes need not produce identical samples. Storage is bounded by the
    source/model and one batch, apart from the existing decoder's own state.

    Raises:
        TypeError: A carrier or scalar has an invalid type.
        ValueError: A positive count, registered name or declared parity is missing.
        ValidationError: Seed range or backend output violates its contract.
        CapabilityError: The source, model or exact decoder type is unsupported.
        StabilizerDependencyError: The optional Stim dependency is missing.
        MatchingDependencyError: The selected decoder's dependency is missing.

    Existing preparation, decoding and sampling failures propagate unchanged.
    Cleanup failure never replaces a primary failure. No partial result returns.

    Examples:
        >>> from importlib.util import find_spec
        >>> from flagquantum.qec import (
        ...     PhenomenologicalNoise, RepetitionCode, build_memory_circuit,
        ... )
        >>> if find_spec("stim") is not None:
        ...     memory = build_memory_circuit(RepetitionCode(distance=3), rounds=1)
        ...     result = run_memory_experiment(
        ...         memory, noise=PhenomenologicalNoise(),
        ...         decoder="minimum_weight_matching", shots=17, batch_size=8, seed=53,
        ...     )
        ...     assert result.shots == 17 and result.logical_failures == 0
    """
    if not isinstance(memory, MemoryCircuit):
        raise TypeError("memory must be a MemoryCircuit")
    if not isinstance(noise, (PhenomenologicalNoise, NoiseModel)):
        raise TypeError("noise must be a PhenomenologicalNoise or a NoiseModel")
    shots, batch_size, seed = _experiment_options(decoder, shots, batch_size, seed)
    detectors, observables = len(memory.detectors), len(memory.observables)
    if detectors == 0 or observables == 0:
        raise ValueError(
            "memory experiments require at least one detector and observable"
        )

    text = memory_to_stim_text(memory, noise=noise)
    model = _experiment_model(text, decoder)
    if (model.num_detectors, model.num_observables) != (detectors, observables):
        raise ValidationError("strict model dimensions do not match the memory source")
    selected, predict_dense = _experiment_decoder(decoder, model)
    sampler = compile_stim_sampler(text, seed=seed)
    failures, per_observable = _collect_fixed_shots(
        sampler,
        selected,
        shots=shots,
        batch_size=batch_size,
        detectors=detectors,
        observables=observables,
        predict_dense=predict_dense,
    )
    return MemoryExperimentResult(
        shots=shots,
        logical_failures=failures,
        logical_error_rate=failures / shots,
        observable_failures=tuple(per_observable),
        confidence_interval=_wilson_interval(failures, shots),
        seed=seed,
        sampling_source="memory_circuit",
    )


def run_stim_experiment(
    circuit_text: str,
    *,
    decoder: str,
    shots: int,
    batch_size: int,
    seed: int | None = None,
) -> StimExperimentResult:
    """Sample and decode fixed shots from supported original Stim circuit text.

    Uses strict DEM analysis and the selected registered decoder's defaults.
    The sampler reads the original circuit, not its DEM. Unsupported source or
    decoder combinations propagate their existing errors without approximation.
    Standard builds require the Stim extra. Embedded Stim builds provide the
    engine directly. The pymatching decoder requires its own extra.

    Args:
        circuit_text: Text accepted by the existing annotated Stim bridge.
        decoder: Registered independent-shot decoder name or `"sparse_blossom"`
            for bundled batch matching on the graphlike decomposition.
        shots: Positive total number of original-circuit repetitions.
        batch_size: Positive maximum draw size, including the final partial draw.
        seed: Existing optional unsigned 64-bit sampler seed.

    Returns:
        Immutable counts, logical error rate and a two-sided 95% Wilson interval.

    Raises:
        TypeError: A carrier or scalar has an invalid type.
        ValueError: A value or declared detector/observable count is invalid.
        ValidationError: A backend returns invalid sample or prediction data.
        CapabilityError: The source, model or decoder is unsupported.
        StabilizerDependencyError: The optional Stim dependency is missing.
        MatchingDependencyError: The selected decoder dependency is missing.

    Examples:
        >>> from importlib.util import find_spec
        >>> text = "R 0\\nX_ERROR(0.125) 0\\nM 0\\nDETECTOR rec[-1]\\nOBSERVABLE_INCLUDE(0) rec[-1]"
        >>> if find_spec("stim") is not None:
        ...     result = run_stim_experiment(
        ...         text, decoder="minimum_weight_matching",
        ...         shots=35, batch_size=16, seed=53,
        ...     )
        ...     assert result.shots == 35 and result.logical_failures == 0
    """
    if not isinstance(circuit_text, str):
        raise TypeError("circuit_text must be a string")
    shots, batch_size, seed = _experiment_options(decoder, shots, batch_size, seed)
    model = _experiment_model(circuit_text, decoder)
    detectors, observables = model.num_detectors, model.num_observables
    if detectors == 0 or observables == 0:
        raise ValueError(
            "Stim experiments require at least one detector and observable"
        )
    selected, predict_dense = _experiment_decoder(decoder, model)
    sampler = compile_stim_sampler(circuit_text, seed=seed)
    failures, per_observable = _collect_fixed_shots(
        sampler,
        selected,
        shots=shots,
        batch_size=batch_size,
        detectors=detectors,
        observables=observables,
        predict_dense=predict_dense,
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


def _experiment_options(
    decoder: str,
    shots: int,
    batch_size: int,
    seed: int | None,
) -> tuple[int, int, int | None]:
    if not isinstance(decoder, str):
        raise TypeError("decoder must be a registered name")
    shots = _checked_shots(shots)
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral):
        raise TypeError("batch_size must be a positive integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    batch_size = int(batch_size)
    seed = _validate_seed(_checked_seed(seed))
    if decoder != _SPARSE_BLOSSOM and decoder not in decoder_names():
        raise ValueError(f"unknown decoder name: {decoder!r}")
    return shots, batch_size, seed


def _experiment_model(text: str, decoder: str) -> DetectorErrorModel:
    if decoder != _SPARSE_BLOSSOM:
        return DetectorErrorModel.from_stim_text(
            stim_circuit_to_dem_text(text, decompose_errors=False)
        )
    return DetectorErrorModel.from_stim_text(
        _stim_circuit_to_matching_dem_text(text), use_decomp_suggestions=True
    ).merge_duplicate_mechanisms()


def _experiment_decoder(
    name: str, model: DetectorErrorModel
) -> tuple[
    DetectorErrorModelDecoder | None, Callable[[torch.Tensor], torch.Tensor] | None
]:
    if name == _SPARSE_BLOSSOM:
        graph = DecodingGraph.from_detector_error_model(model)
        return None, SparseBlossomDecoder(graph).decode_batch
    selected = get_decoder(name, model)
    decode_batch(selected, ())
    return selected, None


def _collect_fixed_shots(
    sampler: MemorySampler,
    selected: DetectorErrorModelDecoder | None,
    *,
    shots: int,
    batch_size: int,
    detectors: int,
    observables: int,
    predict_dense: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[int, tuple[int, ...]]:
    completed = failures = 0
    per_observable = [0] * observables
    try:
        while completed < shots:
            size = min(batch_size, shots - completed)
            batch = sampler.sample(shots=size)
            _check_bits(batch.detectors, size, detectors)
            _check_bits(batch.observables, size, observables)
            if predict_dense is None:
                assert selected is not None
                predicted = _predict_sparse_batch(
                    selected, batch.detectors, observables
                )
            else:
                predicted = predict_dense(batch.detectors)
                _check_bits(predicted, size, observables)
            disagreement = predicted != batch.observables
            failures += int(disagreement.any(dim=1).sum().item())
            for index, count in enumerate(disagreement.sum(dim=0).tolist()):
                per_observable[index] += int(count)
            completed += size
    except BaseException as primary:
        # This synchronous composition owns the handle even when a backend fails.
        try:
            sampler.close()
        except BaseException as cleanup:
            raise primary from cleanup
        raise
    else:
        sampler.close()
    return failures, tuple(per_observable)


def _predict_sparse_batch(
    selected: DetectorErrorModelDecoder,
    detectors: torch.Tensor,
    observables: int,
) -> torch.Tensor:
    rows = tuple(tuple(row.nonzero().flatten().tolist()) for row in detectors)
    results = decode_batch(selected, rows)
    if len(results) != len(rows):
        raise ValidationError("decoder returned an unexpected number of results")
    predicted = torch.zeros((len(rows), observables), dtype=torch.bool, device="cpu")
    for row_index, result in enumerate(results):
        seen: set[int] = set()
        if not isinstance(result.observables, tuple):
            raise ValidationError("decoder observables must be a sparse tuple")
        for index in result.observables:
            if (
                isinstance(index, bool)
                or not isinstance(index, Integral)
                or not 0 <= index < observables
                or index in seen
            ):
                raise ValidationError("decoder returned invalid observable indices")
            seen.add(index)
            predicted[row_index, index] = True
    return predicted
