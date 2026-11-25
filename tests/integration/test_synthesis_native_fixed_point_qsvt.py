"""Concrete fixed-point QSVT preserves signed predicates and absolute phases."""

from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.amplitude_amplification.qsvt_fixed_point.comparisons.circuit import (
    reference_circuit,
    rotation_circuit,
)
from examples.synthesis.algorithms.amplitude_amplification.qsvt_fixed_point.main import (
    PHASES,
    build_program,
    main,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.integration.test_synthesis_native_exact_numbers import definition

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


def test_untouched_fixed_point_source_and_native_match_independent_full_state() -> None:
    original = synth.synthesize(
        ROOT / "tests/fixtures/classiq_qmod/corpus/original/algorithms/"
        "amplitude_amplification_and_estimation/qsvt_fixed_point_amplitude_amplification/"
        "qsvt_fixed_point_amplitude_amplification.qmod"
    )
    compare_states(
        original, reference_circuit(), ("state.a", "state.b", "state.c", "aux")
    )
    compare_states(build_program(), reference_circuit(), ("a", "b", "c", "aux"))


@pytest.mark.parametrize("offset", [0.17, -0.29])
def test_two_nonzero_phase_bindings_preserve_all_26_rotations(offset: float) -> None:
    phases = tuple(
        value + offset * (index + 1) / 26 for index, value in enumerate(PHASES)
    )
    compare_states(
        build_program(phases), reference_circuit(phases), ("a", "b", "c", "aux")
    )


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("angle", [0.37, -1.13])
def test_projectors_preserve_all_words_and_both_aux_values_coherently(
    initial: bool,
    angle: float,
) -> None:
    predicate = "(a<1)&(b<1)&(c<1)" if initial else "((a+b)==3)&((a+2)==c)"
    transform = (
        "hadamard_transform(a)\nhadamard_transform(b)\nhadamard_transform(c)\n"
        if initial
        else ""
    )
    program = synth.synthesize(
        definition(
            "a:Output[QNum[2]],b:Output[QNum[1]],c:Output[QNum[3]],aux:Output[QBit]",
            "allocate(a)\nallocate(b)\nallocate(c)\nallocate(aux)\n"
            "hadamard_transform(a)\nhadamard_transform(b)\nhadamard_transform(c)\nH(aux)\n"
            "phase(0.13*a+0.17*b+0.19*c+0.23*aux,1)\n"
            + transform
            + "with within():\n    marked:QNum[1] = "
            + predicate
            + "\n"
            + f"with apply():\n    phase(aux-0.5,{angle})\n    phase((1-2*aux)*marked,{angle})\n"
            + transform,
        )
    )
    reference = fq.Circuit(7, dtype=torch.complex128)
    for wire in range(7):
        reference.h(wire)
    for wire, coefficient in enumerate((0.26, 0.13, 0.17, 0.76, 0.38, 0.19, 0.23)):
        reference.phase(wire, theta=coefficient)
    reference.compose(rotation_circuit(angle, initial))
    compare_states(program, reference, ("a", "b", "c", "aux"))


@pytest.mark.parametrize("constraint", ["operations", "qubits", "depth", "memory"])
def test_fixed_point_program_budget_refusals(constraint: str) -> None:
    bindings = {"phases": list(PHASES)}
    with pytest.raises((synth.SynthesisError, HybridVerificationError, PlanningError)):
        if constraint == "operations":
            synth.synthesize(main, classical_inputs=bindings, max_operations=100)
        elif constraint == "qubits":
            synth.synthesize(main, classical_inputs=bindings, max_qubits=6)
        elif constraint == "depth":
            synth.synthesize(main, classical_inputs=bindings, max_depth=10)
        else:
            synth.synthesize(main, classical_inputs=bindings, memory_limit_bytes=16)


def test_phase_count_is_not_silently_truncated() -> None:
    with pytest.raises(ValueError, match="26 phases"):
        build_program(PHASES[:-1])
