"""Comparison primitives with one shared CPU global decoding boundary."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from flagquantum.qec.decoders._ai.predecoder import _positive

Correction = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
GlobalDecode = Callable[[torch.Tensor], torch.Tensor]


def on_device(correction: Correction, device: torch.device) -> Correction:
    """Transfer input and both correction outputs across an explicit device boundary.

    The caller places its model on the same device. Blocking CPU outputs include
    D2H completion before global matching; no accelerator fallback is permitted.
    """
    if device.type not in ("cpu", "cuda", "flagos"):
        raise ValueError(
            "comparison supports only explicit CPU or CUDA devices, or FlagOS"
        )

    def run(bits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        transferred = bits.to(device)
        residual, frame = correction(transferred)
        if residual.device != transferred.device or frame.device != residual.device:
            raise ValueError("correction outputs must remain on the selected device")
        return residual.cpu(), frame.cpu()

    return run


@torch.inference_mode()
def require_parity(
    detectors: torch.Tensor,
    reference: Correction,
    candidate: Correction,
    global_decode: GlobalDecode | None = None,
) -> None:
    """Require exact correction bits, plus final predictions for matchable inputs."""
    expected = reference(detectors)
    actual = candidate(detectors)
    for residual, frame in (expected, actual):
        binary_array(residual, width=detectors.shape[1], shots=detectors.shape[0])
        if frame.ndim != 2 or frame.shape[1] < 1:
            raise ValueError("requires a nonempty logical frame")
        binary_array(frame, width=frame.shape[1], shots=detectors.shape[0])
    if not all(torch.equal(a, b) for a, b in zip(expected, actual, strict=True)):
        raise ValueError("CPU/device residual or logical-frame parity failed")
    if global_decode is not None and not torch.equal(
        decode(detectors, reference, global_decode),
        decode(detectors, candidate, global_decode),
    ):
        raise ValueError("CPU/device final-prediction parity failed")


def binary_array(bits: torch.Tensor, *, width: int, shots: int) -> None:
    if bits.device.type != "cpu" or bits.shape != (shots, width):
        raise ValueError("requires equally shaped CPU shot/bit arrays")
    if not bool(((bits == 0) | (bits == 1)).all()):
        raise ValueError("requires binary arrays")


@torch.inference_mode()
def decode(
    detectors: torch.Tensor, correction: Correction, global_decode: GlobalDecode
) -> torch.Tensor:
    """Include input conversion, correction, residual decode and logical XOR."""
    if detectors.ndim != 2 or not all(detectors.shape):
        raise ValueError("requires a nonempty detector matrix")
    binary_array(detectors, width=detectors.shape[1], shots=detectors.shape[0])
    residual, frame = correction(detectors)
    binary_array(residual, width=detectors.shape[1], shots=detectors.shape[0])
    if frame.ndim != 2 or frame.shape[1] < 1:
        raise ValueError("requires a nonempty logical frame")
    binary_array(frame, width=frame.shape[1], shots=detectors.shape[0])
    predicted = global_decode(residual)
    binary_array(predicted, width=frame.shape[1], shots=detectors.shape[0])
    return predicted.to(torch.int8) ^ frame.to(torch.int8)


@dataclass(frozen=True)
class Timing:
    batch_size: int
    warmup: int
    repetitions: int
    p50_block_us: float
    p95_block_us: float
    p99_block_us: float
    shots_per_second: float


@torch.inference_mode()
def measure(
    detectors: torch.Tensor,
    correction: Correction,
    global_decode: GlobalDecode,
    *,
    warmup: int = 3,
    repetitions: int = 32,
    synchronize: Callable[[], None] | None = None,
) -> Timing:
    """Time complete batches; the device owner supplies its synchronization fence."""
    _positive(warmup, "warmup")
    _positive(repetitions, "repetitions")
    expected = decode(detectors, correction, global_decode)
    for _ in range(warmup):
        decode(detectors, correction, global_decode)
    elapsed: list[float] = []
    for _ in range(repetitions):
        if synchronize is not None:
            synchronize()
        start = time.perf_counter_ns()
        actual = decode(detectors, correction, global_decode)
        if synchronize is not None:
            synchronize()
        elapsed.append((time.perf_counter_ns() - start) / 1000)
        if not torch.equal(actual, expected):
            raise ValueError("nondeterministic predictions in timing run")
    ordered = sorted(elapsed)
    p50, p95, p99 = (ordered[math.ceil(q * repetitions) - 1] for q in (0.5, 0.95, 0.99))
    return Timing(
        detectors.shape[0],
        warmup,
        repetitions,
        p50,
        p95,
        p99,
        detectors.shape[0] * repetitions * 1e6 / math.fsum(elapsed),
    )
