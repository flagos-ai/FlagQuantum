"""The concrete QSVT source retains comparison XOR and projected RZ phases."""

import cmath
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.qsvt.comparisons.circuit import reference_circuit
from examples.synthesis.functions.qsvt.main import PHASES, build_program, main
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.integration.test_synthesis_native_exact_numbers import definition

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
HELPER = "from examples.synthesis.functions.qsvt.main import project_zero\n"


def test_untouched_qsvt_source_native_and_independent_full_state() -> None:
    original = synth.synthesize(
        ROOT
        / "tests/fixtures/classiq_qmod/corpus/original/functions/qmod_library_reference/classiq_open_library/qsvt/qsvt.qmod"
    )
    native = build_program()
    for program in (original, native):
        compare_states(
            program, reference_circuit(), ("state", "ref", "ind", "qsvt_aux")
        )
    assert len(native.registers["state"]) == len(native.registers["ref"]) == 4
    assert len(native.registers["ind"]) == len(native.registers["qsvt_aux"]) == 1


@pytest.mark.parametrize("angle", [0.37, -1.13])
@pytest.mark.parametrize("word", range(4))
@pytest.mark.parametrize("marker", [0, 1])
def test_projected_rotation_preserves_every_nonzero_marker_and_absolute_phase(
    angle: float,
    word: int,
    marker: int,
) -> None:
    program = synth.synthesize(
        definition(
            "word:Output[QNum[2]],aux:Output[QBit]",
            f"word |= {word}\naux |= {marker}\nproject_zero({angle},aux,word)",
            HELPER,
        ),
        max_workspace=0,
    )
    reference = fq.Circuit(3, dtype=torch.complex128)
    for bit in range(2):
        if word & (1 << bit):
            reference.x(1 - bit)
    if marker:
        reference.x(2)
    value = cmath.exp(1j * angle * ((marker ^ int(word == 0)) - 0.5))
    reference.any(2, matrix=value * torch.eye(2, dtype=torch.complex128))
    compare_states(program, reference, ("word", "aux"))


def test_projector_keeps_coherent_relative_phase_across_all_branches() -> None:
    program = synth.synthesize(
        definition(
            "word:Output[QNum[2]],aux:Output[QBit]",
            "allocate(word)\nallocate(aux)\nhadamard_transform(word)\nH(aux)\n"
            "phase(0.13*word+0.17*aux,1)\nproject_zero(-0.43,aux,word)",
            HELPER,
        ),
        max_workspace=0,
    )
    values = [
        cmath.exp(
            1j
            * (0.13 * word + 0.17 * marker - 0.43 * ((marker ^ int(word == 0)) - 0.5))
        )
        for word in range(4)
        for marker in range(2)
    ]
    reference = fq.Circuit(3, dtype=torch.complex128).h(0).h(1).h(2)
    reference.any(
        0, 1, 2, matrix=torch.diag(torch.tensor(values, dtype=torch.complex128))
    )
    compare_states(program, reference, ("word", "aux"))


@pytest.mark.parametrize("constraint", ["operations", "qubits", "depth", "memory"])
def test_concrete_qsvt_resource_limits_remain_enforced(constraint: str) -> None:
    bindings = {"phases": list(PHASES)}
    with pytest.raises((synth.SynthesisError, HybridVerificationError, PlanningError)):
        if constraint == "operations":
            synth.synthesize(main, classical_inputs=bindings, max_operations=100)
        elif constraint == "qubits":
            synth.synthesize(main, classical_inputs=bindings, max_qubits=9)
        elif constraint == "depth":
            synth.synthesize(main, classical_inputs=bindings, max_depth=10)
        else:
            synth.synthesize(main, classical_inputs=bindings, memory_limit_bytes=16)


def test_concrete_polynomial_phase_count_is_explicit() -> None:
    with pytest.raises(ValueError, match="six phases"):
        build_program(PHASES[:-1])
