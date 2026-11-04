"""Verify coherent scalar control with independent complex amplitudes."""

import math

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QBit,
    allocate,
    control,
    hadamard_transform,
    phase,
    qfunc,
)

pytestmark = pytest.mark.integration
HALF_PI = math.pi / 2


@qfunc
def conditional_phase(
    condition: Output[QBit],
    target: Output[QBit],
    left: Output[QBit],
    right: Output[QBit],
) -> None:
    allocate(condition)
    allocate(target)
    allocate(left)
    allocate(right)
    hadamard_transform(condition)
    hadamard_transform(target)
    hadamard_transform(left)
    hadamard_transform(right)
    control(condition, lambda: phase(target, HALF_PI))


@qfunc
def interference(condition: Output[QBit]) -> None:
    allocate(condition)
    hadamard_transform(condition)
    control(condition, lambda: phase(1.0, math.pi))
    hadamard_transform(condition)


@qfunc
def controlled_h(condition: Output[QBit], target: Output[QBit]) -> None:
    allocate(condition)
    allocate(target)
    hadamard_transform(condition)
    control(condition, lambda: hadamard_transform(target))


@qfunc
def overlap(condition: Output[QBit]) -> None:
    allocate(condition)
    control(condition, lambda: phase(condition, 1.0))


def test_controlled_target_phase_checks_all_sixteen_amplitudes():
    expected = torch.full((16,), 0.25, dtype=torch.complex128)
    expected[12:] = 0.25j
    state = fq.run(synth.synthesize(conditional_phase)).statevector().flatten()
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)


def test_controlled_constant_phase_interferes():
    result = synth.synthesize(interference)
    state = fq.run(result).statevector().flatten()
    expected = torch.tensor([0, 1], dtype=torch.complex128)
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)


def test_budget_counts_the_lowered_controlled_operation():
    result = synth.synthesize(conditional_phase, max_operations=5)
    assert len(result.circuit.instructions) == 5
    with pytest.raises(ValueError, match="operation"):
        synth.synthesize(conditional_phase, max_operations=4)


def test_controlled_hadamard_amplitudes():
    state = fq.run(synth.synthesize(controlled_h)).statevector().flatten()
    expected = torch.tensor([2**-0.5, 0, 0.5, 0.5], dtype=torch.complex128)
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)


def test_condition_target_overlap_is_rejected():
    with pytest.raises(ValueError, match="condition and target must be disjoint"):
        synth.synthesize(overlap)
