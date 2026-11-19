"""Execute the Boolean oracle demo, both Bell tests, and the original sweep."""

import math
from dataclasses import dataclass

import numpy as np
import torch

import flagquantum as fq
from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_qsvt.overview.phases import (
    CTYPE,
    DTYPE,
    PhaseFit,
)
from flagquantum import synth

from .model import Benchmark, angle_samples, benchmark, known_angles
from .quantum import P_MAX, SHOTS, Corrected, corrected, sampled, sketch, threshold_fit

TABLE_BELL = np.array([0, 0, 1, 1], dtype=np.int64)
UNKNOWN_DISTRIBUTION = (0.10, 0.15, 0.30, 0.45)


@dataclass(frozen=True)
class Result:
    benchmark: Benchmark
    demo: synth.Program
    bell: synth.Program
    demo_error: float
    known_error: float
    bell_error: float
    bell_exact: float
    bell_sampled: float
    threshold: PhaseFit
    threshold_error: float
    corrected: Corrected


def run() -> Result:
    report = benchmark()
    demo_angles = angle_samples(report.table, report.demo_samples, math.pi * 8 / 200)
    bell_angles = angle_samples(TABLE_BELL, report.bell_samples, math.pi * 4 / 500)
    demo = sketch(tuple(map(float, demo_angles)))
    bell = sketch(tuple(map(float, bell_angles)), bell=True)
    demo_vector = fq.run(demo).statevector().flatten()
    bell_vector = fq.run(bell).statevector().flatten()
    expected_demo = torch.exp(1j * torch.tensor(demo_angles, dtype=DTYPE)) / math.sqrt(
        8
    )
    angle = complex(np.exp(1j * bell_angles[3]))
    expected_bell = torch.tensor([(1 + angle) / 2, (1 - angle) / 2, 0, 0], dtype=CTYPE)
    torch.testing.assert_close(demo_vector, expected_demo, atol=1e-12, rtol=0)
    torch.testing.assert_close(bell_vector, expected_bell, atol=1e-12, rtol=0)
    known = np.array([2, 3, 3, 2, 3], dtype=np.int64)
    known_probability = np.array(
        [UNKNOWN_DISTRIBUTION[i] for i in known], dtype=np.float64
    )
    weighted = known_angles(known, known_probability, 2)
    known_program = sketch(tuple(map(float, weighted)))
    known_state = fq.run(known_program).statevector().flatten()
    diag = torch.eye(4, dtype=CTYPE)
    for word, p in zip(known, known_probability, strict=True):
        diag[int(word), int(word)] *= np.exp(1j * np.pi / (p * len(known)))
    ref_circuit = fq.Circuit(2, dtype=CTYPE).h(0).h(1).any(0, 1, matrix=diag)
    expect_known = fq.run(ref_circuit).statevector().flatten()
    torch.testing.assert_close(known_state, expect_known, atol=1e-12, rtol=0)
    bell_probs = bell_vector.abs().square()
    count = sampled(bell_probs, 136)
    threshold, response_error = threshold_fit()
    samples = np.random.RandomState(0).choice(
        np.arange(4), size=20, p=UNKNOWN_DISTRIBUTION
    )
    phases = tuple(map(float, angle_samples(TABLE_BELL, samples, 1 / (P_MAX * 20))))
    corrected_bell = corrected(phases, threshold)
    return Result(
        report,
        demo,
        bell,
        float((demo_vector - expected_demo).abs().max()),
        float((known_state - expect_known).abs().max()),
        float((bell_vector - expected_bell).abs().max()),
        float(bell_probs[1]),
        int(count[1]) / SHOTS,
        threshold,
        response_error,
        corrected_bell,
    )
