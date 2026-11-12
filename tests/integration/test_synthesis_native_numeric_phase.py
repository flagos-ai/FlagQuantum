"""Signed and binary-scaled phase reads retain exact value weights and phase."""

import cmath
from fractions import Fraction
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.qpe.comparisons.circuit import reference_circuit
from examples.synthesis.functions.qpe.main import build_program as qpe_program
from examples.synthesis.functions.qpe.qpe_flexible.main import (
    build_program as flexible_program,
)
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.integration.test_synthesis_native_exact_numbers import definition

pytestmark = pytest.mark.integration
ERRORS = (synth.SynthesisError, HybridVerificationError)


@pytest.mark.parametrize("signed,fraction", [(True, 0), (True, 1), (False, 2)])
def test_every_numeric_basis_weight_and_global_phase(
    signed: bool, fraction: int
) -> None:
    entry = definition(
        f"a:Output[QNum[3,{signed},{fraction}]]",
        "allocate(a)\nhadamard_transform(a)\nphase(a*a-0.5*a+0.17,0.37)",
    )
    program = synth.synthesize(entry, max_workspace=0)
    values = []
    for raw in range(8):
        value = (raw - 8 if signed and raw >= 4 else raw) / (1 << fraction)
        values.append(cmath.exp(0.37j * (value * value - 0.5 * value + 0.17)))
    reference = fq.Circuit(3, dtype=torch.complex128).h(0).h(1).h(2)
    reference.any(
        0, 1, 2, matrix=torch.diag(torch.tensor(values, dtype=torch.complex128))
    )
    compare_states(program, reference, ("a",))
    assert program.report["workspace_qubits"] == 0
    decoded = program.numeric_probabilities(fq.run(program), registers=("a",))
    assert set(decoded) == {
        (Fraction(raw - 8 if signed and raw >= 4 else raw, 1 << fraction),)
        for raw in range(8)
    }


def test_mixed_words_preserve_all_coherent_cross_products() -> None:
    entry = definition(
        "a:Output[QNum[2,True,1]],b:Output[QNum[2,False,2]]",
        "allocate(a)\nallocate(b)\nhadamard_transform(a)\nhadamard_transform(b)\n"
        "phase(a*b+a-b+0.23,-0.41)",
    )
    values = []
    for raw_a in range(4):
        a = (raw_a - 4 if raw_a >= 2 else raw_a) / 2
        for raw_b in range(4):
            b = raw_b / 4
            values.append(cmath.exp(-0.41j * (a * b + a - b + 0.23)))
    reference = fq.Circuit(4, dtype=torch.complex128).h(0).h(1).h(2).h(3)
    reference.any(
        *range(4), matrix=torch.diag(torch.tensor(values, dtype=torch.complex128))
    )
    program = synth.synthesize(entry, max_workspace=0)
    compare_states(program, reference, ("a", "b"))


def test_cancellation_keeps_uninitialized_read_witnesses() -> None:
    with pytest.raises(ERRORS, match="initialized"):
        synth.synthesize(definition("a:Output[QNum[3,True,1]]", "phase(a-a+1,0.3)"))


@pytest.mark.parametrize(
    "body,diagnostic",
    [
        ("phase(a<0,0.2)", "numeric comparisons"),
        ("control(a,lambda:phase(1,0.2))", "numeric reads"),
        ("phase(a,a)", "phase coefficients"),
        ("phase(a**8,0.2)", "monomials|products"),
    ],
)
def test_unadmitted_numeric_paths_and_expansion_fail_closed(
    body: str, diagnostic: str
) -> None:
    with pytest.raises(ERRORS, match=diagnostic):
        synth.synthesize(
            definition("a:Output[QNum[10,True,1]]", "allocate(a)\n" + body)
        )


@pytest.mark.parametrize("constraint", ["operations", "qubits", "memory"])
def test_numeric_phase_obeys_actual_budgets(constraint: str) -> None:
    entry = definition(
        "a:Output[QNum[3,True,1]]",
        "allocate(a)\nhadamard_transform(a)\nphase(a*a,0.37)",
    )
    with pytest.raises((*ERRORS, PlanningError)):
        if constraint == "operations":
            synth.synthesize(entry, max_operations=1)
        elif constraint == "qubits":
            synth.synthesize(entry, max_qubits=2)
        else:
            synth.synthesize(entry, memory_limit_bytes=16)


@pytest.mark.parametrize("name", ["qpe", "qpe_flexible"])
def test_original_diagonal_qpe_keeps_fractional_output_and_full_phase(
    name: str,
) -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "fixtures/classiq_qmod/corpus/original/functions/qmod_library_reference/classiq_open_library/qpe"
        / (name + ".qmod")
    )
    original = synth.synthesize(path)
    native = qpe_program() if name == "qpe" else flexible_program()
    reference = reference_circuit()
    for program in (original, native):
        compare_states(program, reference, ("state", "phase"))
        assert program.report["workspace_qubits"] == 0
    expected = {
        (Fraction(0), Fraction(0)): 0.25,
        (Fraction(1), Fraction(1, 2)): 0.25,
        (Fraction(2), Fraction(1, 4)): 0.25,
        (Fraction(3), Fraction(3, 4)): 0.25,
    }
    assert native.numeric_probabilities(
        fq.run(native), registers=("state", "phase")
    ) == pytest.approx(expected)
    original_raw = original.probabilities(
        fq.run(original), registers=("state", "phase")
    )
    assert {
        (Fraction(state), Fraction(phase, 4)): value
        for (state, phase), value in original_raw.items()
    } == pytest.approx(expected)
