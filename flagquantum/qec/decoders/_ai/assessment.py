"""Private paired logical-quality assessment and CPU whole-block timing."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from flagquantum.qec.decoders._ai.predecoder import Predecoder, _positive
from flagquantum.qec.decoders.registry import DetectorErrorModelDecoder


@dataclass(frozen=True)
class PairedCounts:
    """Fail a shot when any logical observable is predicted incorrectly."""

    shots: int
    baseline_failures: int
    hybrid_failures: int
    wins: int
    losses: int


def paired_counts(
    truth: torch.Tensor, baseline: torch.Tensor, hybrid: torch.Tensor
) -> PairedCounts:
    if truth.ndim != 2 or truth.shape[0] == 0 or truth.shape[1] == 0:
        raise ValueError("requires nonempty shot/observable arrays")
    for bits in (truth, baseline, hybrid):
        if bits.shape != truth.shape or not bool(((bits == 0) | (bits == 1)).all()):
            raise ValueError(
                "predictions and truth must be equally shaped binary arrays"
            )
    first = (baseline != truth).any(1)
    second = (hybrid != truth).any(1)
    return PairedCounts(
        truth.shape[0],
        int(first.sum()),
        int(second.sum()),
        int((first & ~second).sum()),
        int((~first & second).sum()),
    )


def rate_interval(failures: int, shots: int, alpha: float) -> tuple[float, float]:
    """Distribution-free Hoeffding interval; exact endpoint bound for zero/all failures."""
    _positive(shots, "shots")
    if (
        isinstance(failures, bool)
        or not isinstance(failures, int)
        or not 0 <= failures <= shots
    ):
        raise ValueError("failures must be an integer between zero and shots")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    radius = math.sqrt(math.log(2 / alpha) / (2 * shots))
    lower, upper = max(0.0, failures / shots - radius), min(
        1.0, failures / shots + radius
    )
    if failures == 0:
        upper = min(upper, -math.expm1(math.log(alpha / 2) / shots))
    if failures == shots:
        lower = max(lower, math.exp(math.log(alpha / 2) / shots))
    return lower, upper


def paired_p_value(wins: int, losses: int) -> float:
    """Exact two-sided discordant-pair binomial test, not an unpaired LER test."""
    for value in (wins, losses):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("discordant counts must be nonnegative integers")
    total = wins + losses
    if total == 0:
        return 1.0
    limit = min(wins, losses)
    logs = [
        math.lgamma(total + 1)
        - math.lgamma(k + 1)
        - math.lgamma(total - k + 1)
        - total * math.log(2)
        for k in range(limit + 1)
    ]
    largest = max(logs)
    return min(
        1.0,
        2 * math.exp(largest) * math.fsum(math.exp(item - largest) for item in logs),
    )


def summarize(counts: PairedCounts, *, alpha: float = 0.05) -> dict[str, object]:
    # Split alpha across both rate intervals; their ratio bound remains valid
    # without assuming the paired baseline and hybrid failures are independent.
    baseline = rate_interval(counts.baseline_failures, counts.shots, alpha / 2)
    hybrid = rate_interval(counts.hybrid_failures, counts.shots, alpha / 2)
    upper = hybrid[1] / baseline[0] if baseline[0] > 0 else None
    return {
        "baseline_ler": counts.baseline_failures / counts.shots,
        "hybrid_ler": counts.hybrid_failures / counts.shots,
        "baseline_interval": baseline,
        "hybrid_interval": hybrid,
        "ratio_upper": upper,
        "paired_p_value": paired_p_value(counts.wins, counts.losses),
        "quality_gate": (
            "supported_10_percent_reduction"
            if upper is not None and upper <= 0.9
            else "inconclusive"
        ),
    }


def baseline_decode(
    detectors: torch.Tensor, backend: DetectorErrorModelDecoder, observables: int
) -> torch.Tensor:
    """Execute only the global decoder; no neural forward or residual-update cost."""
    result = torch.zeros((detectors.shape[0], observables), dtype=torch.int8)
    for shot, row in enumerate(detectors):
        for observable in backend.decode(row.nonzero().flatten().tolist()).observables:
            result[shot, observable] = 1
    return result


def assess(
    model: Predecoder,
    backend: DetectorErrorModelDecoder,
    *,
    shots: int,
    seed: int,
    threshold: float,
    batch_size: int = 128,
    temperature: float = 1.0,
) -> PairedCounts:
    _positive(shots, "shots")
    _positive(batch_size, "batch_size")
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    totals = [0, 0, 0, 0]
    for start in range(0, shots, batch_size):
        batch = model.sample(min(batch_size, shots - start), generator).sample
        first = baseline_decode(batch.detectors, backend, model.logical.shape[0])
        second = model.decode(
            batch.detectors, backend, threshold=threshold, temperature=temperature
        )
        counts = paired_counts(batch.observables, first, second)
        for index, value in enumerate(
            (
                counts.baseline_failures,
                counts.hybrid_failures,
                counts.wins,
                counts.losses,
            )
        ):
            totals[index] += value
    return PairedCounts(shots, *totals)


def block_timings(
    model: Predecoder,
    backend: DetectorErrorModelDecoder,
    *,
    seed: int,
    threshold: float,
    repetitions: int = 32,
) -> dict[str, object]:
    """CPU batch-one block latency, including tensor/list conversion and residual decode.

    Excludes sampling and graph/model construction. Includes the neural forward
    only on the hybrid path. Alternating order reduces systematic order bias.
    This measures a complete offline block, not per-round feedback latency.
    """
    _positive(repetitions, "repetitions")
    if model.h.device.type != "cpu":
        raise ValueError("timing reference supports CPU only")
    model.eval()
    data = model.sample(
        repetitions, torch.Generator().manual_seed(seed)
    ).sample.detectors
    baseline: list[float] = []
    hybrid: list[float] = []
    for _ in range(3):
        baseline_decode(data[:1], backend, model.logical.shape[0])
        model.decode(data[:1], backend, threshold=threshold)
    for index in range(repetitions):
        row = data[index : index + 1]
        order = (False, True) if index % 2 == 0 else (True, False)
        for is_hybrid in order:
            start = time.perf_counter_ns()
            if is_hybrid:
                model.decode(row, backend, threshold=threshold)
            else:
                baseline_decode(row, backend, model.logical.shape[0])
            elapsed = (time.perf_counter_ns() - start) / 1000
            (hybrid if is_hybrid else baseline).append(elapsed)

    def quantiles(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        return {
            name: ordered[math.ceil(q * len(ordered)) - 1]
            for name, q in (("p50_us", 0.5), ("p95_us", 0.95), ("p99_us", 0.99))
        }

    return {
        "scope": "cpu_batch_one_offline_block",
        "warmup": 3,
        "repetitions": repetitions,
        "baseline": quantiles(baseline),
        "hybrid": quantiles(hybrid),
    }


def evaluate(
    model: Predecoder,
    backend: DetectorErrorModelDecoder,
    *,
    shots: int,
    seed: int,
    batch_size: int = 128,
    threshold: float = 0.9,
) -> dict[str, int | float]:
    """Paired held-out logical failure counts; never equate zero failures with zero risk."""
    _positive(shots, "shots")
    _positive(batch_size, "batch_size")
    generator = torch.Generator().manual_seed(seed)
    baseline = hybrid = 0
    model.eval()
    for start in range(0, shots, batch_size):
        batch = model.sample(min(batch_size, shots - start), generator)
        truth = batch.sample.observables
        baseline += int(
            (model.decode(batch.sample.detectors, backend, enabled=False) != truth)
            .any(1)
            .sum()
        )
        hybrid += int(
            (
                model.decode(batch.sample.detectors, backend, threshold=threshold)
                != truth
            )
            .any(1)
            .sum()
        )
    return {
        "shots": shots,
        "baseline_failures": baseline,
        "hybrid_failures": hybrid,
        "baseline_ler": baseline / shots,
        "hybrid_ler": hybrid / shots,
    }
