"""Generate the original Boolean sketches and finite-size operator statistics."""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

IntegerArray = NDArray[np.int64]
RealArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
TRIALS = 1500


@dataclass(frozen=True)
class Benchmark:
    table: IntegerArray
    demo_samples: IntegerArray
    bell_samples: IntegerArray
    sizes: IntegerArray
    mean_error: RealArray
    mean_operator_error: RealArray
    single_error: float


def angle_samples(
    table: IntegerArray,
    indices: IntegerArray,
    angle: float,
) -> RealArray:
    counts = np.bincount(indices, minlength=table.size)
    return np.asarray(angle * counts * table, dtype=np.float64)


def known_angles(indices: IntegerArray, probs: RealArray, width: int) -> RealArray:
    """Accumulate the per-sample inverse-probability phases."""
    if not indices.size or probs.shape != indices.shape or np.any(probs <= 0):
        raise ValueError("Each active sample needs a positive sampling probability")
    weights = np.pi / (probs * probs.size)
    return np.asarray(
        np.bincount(indices, weights=weights, minlength=1 << width), dtype=np.float64
    )


def benchmark() -> Benchmark:
    sequence = np.random.RandomState(7)
    table = sequence.randint(0, 2, 8)
    demo = sequence.randint(0, 8, 200)
    sequence = np.random.RandomState(0)
    bell = sequence.randint(0, 4, 500)
    single = sequence.randint(0, 8, 4000)
    ideal = 1 - 2 * table
    observed = np.exp(1j * angle_samples(table, single, math.pi * 8 / 4000))
    sizes = np.unique(np.logspace(2, 4.5, 12, dtype=int))
    single_errors = np.empty(sizes.size)
    operator_errors = np.empty(sizes.size)
    for index, count in enumerate(sizes):
        k = int(count)
        operator = np.zeros(8, dtype=np.complex128)
        deviation = 0.0
        for _ in range(TRIALS):
            sample = sequence.randint(0, 8, k)
            sketch = np.exp(1j * angle_samples(table, sample, math.pi * 8 / k))
            operator += sketch
            deviation += float(np.max(np.abs(sketch - ideal)))
        single_errors[index] = deviation / TRIALS
        operator_errors[index] = float(np.max(np.abs(operator / TRIALS - ideal)))
    return Benchmark(
        table,
        demo,
        bell,
        sizes,
        single_errors,
        operator_errors,
        float(np.max(np.abs(observed - ideal))),
    )
