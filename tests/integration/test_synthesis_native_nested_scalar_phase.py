"""Candidate regressions for implicit scalar phases under borrowed controls."""

import math

import pytest
import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from tests.integration.test_synthesis_native_exact_numbers import definition

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("outer_value", [0, 1])
@pytest.mark.parametrize("inner_value", [0, 1])
@pytest.mark.parametrize("angle", [0.37, -0.51, math.pi])
def test_nested_scalar_phase_keeps_complete_interference(
    outer_value: int, inner_value: int, angle: float
) -> None:
    outer = f"outer == {outer_value}"
    inner = f"target == {inner_value}"
    helper = (
        "@qfunc\ndef mark(target:QBit):\n"
        f"    control({inner},lambda:phase(1,{angle!r}))\n"
    )
    program = synth.synthesize(
        definition(
            "outer:Output[QBit],target:Output[QBit]",
            "allocate(outer)\nallocate(target)\n"
            "hadamard_transform(outer)\nhadamard_transform(target)\n"
            "RZ(-0.41,outer)\nRZ(0.23,target)\n"
            f"control({outer},lambda:mark(target))\n"
            "hadamard_transform(outer)",
            helper,
        ),
        max_qubits=8,
        memory_limit_bytes=1024**2,
    )
    reference = fq.Circuit(2, dtype=torch.complex128).h(0).h(1)
    reference.rz(0, theta=-0.41).rz(1, theta=0.23)
    for wire, value in enumerate((outer_value, inner_value)):
        if not value:
            reference.x(wire)
    reference.cphase(0, 1, theta=angle)
    for wire, value in enumerate((outer_value, inner_value)):
        if not value:
            reference.x(wire)
    reference.h(0)
    compare_states(program, reference, ("outer", "target"))


def test_nested_scalar_phase_refuses_writable_condition_alias() -> None:
    entry = definition(
        "outer:Output[QBit]",
        "allocate(outer)\nhadamard_transform(outer)\n"
        "control(outer,lambda:mark(outer))",
        "@qfunc\ndef mark(target:QBit):\n" "    control(target,lambda:phase(1,0.37))\n",
    )
    with pytest.raises(HybridVerificationError, match="disjoint|alias"):
        synth.synthesize(entry)
