"""Select compact supported segments without losing checked wide arithmetic."""

import cmath
from collections.abc import Callable
from fractions import Fraction

import pytest
import torch

import flagquantum as fq
from benchmarks.synthesis.verification import classical_apply, encode
from flagquantum import synth
from flagquantum.compiler._construction import register_lowering
from flagquantum.compiler._construction.boolean_lowering import BooleanCandidateError
from flagquantum.compiler._construction.frontend_program import capture_program
from flagquantum.compiler._construction.frontend_python import _QuantumFunction
from flagquantum.compiler._construction.selection import select_register_program
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.integration.test_synthesis_native_exact_numbers import definition

pytestmark = pytest.mark.integration


def program_definition(
    relation: str = ">=", *, coherent: bool = False
) -> _QuantumFunction:
    preparation = (
        "hadamard_transform(x)\nhadamard_transform(y)\nX(marker)\nH(marker)\nS(marker)\nphase(1,0.31)\n"
        if coherent
        else ""
    )
    return definition(
        "x:Output[QNum[2]],y:Output[QNum[5]],value:Output[QNum[4]],marker:Output[QBit]",
        "allocate(x)\nallocate(y)\nallocate(marker)\n"
        + preparation
        + f"value |= x*x+1\nmarker ^= value {relation} y",
    )


def test_mixed_candidate_retains_individual_implementation_and_refusal() -> None:
    root, functions = capture_program(program_definition())
    selection = select_register_program(root, functions, include_boolean=True)
    assert selection.complete and selection.selected is not None
    assert selection.selected.strategy == "boolean"
    candidate = selection.selected.arithmetic
    assert isinstance(candidate, register_lowering.RegisterCandidate)
    assert {segment.implementation for segment in candidate.segments} == {
        "boolean",
        "conservative",
    }
    refusals = [
        diagnostic.message
        for segment in candidate.segments
        for diagnostic in segment.implementation_errors
    ]
    assert any("active inputs" in message for message in refusals)


@pytest.mark.parametrize("relation", [">=", "<"])
def test_predicate_xor_reads_signed_and_differently_scaled_words(relation: str) -> None:
    entry = definition(
        "x:Output[QNum[2,True,1]],y:Output[QNum[2,False,2]],marker:Output[QBit]",
        "allocate(x)\nallocate(y)\nallocate(marker)\n"
        "hadamard_transform(x)\nhadamard_transform(y)\n"
        "X(marker)\nH(marker)\nS(marker)\n"
        f"marker ^= x {relation} y",
    )
    program = synth.synthesize(entry)
    matrix = torch.zeros((32, 32), dtype=torch.complex128)
    for x in range(4):
        value = Fraction(x if x < 2 else x - 4, 2)
        for y in range(4):
            marked = (
                value >= Fraction(y, 4) if relation == ">=" else value < Fraction(y, 4)
            )
            for marker in (0, 1):
                before = 2 * (4 * x + y) + marker
                after = 2 * (4 * x + y) + (marker ^ int(marked))
                matrix[after, before] = 1
    circuit = fq.Circuit(5, dtype=torch.complex128)
    for wire in range(4):
        circuit.h(wire)
    circuit.x(4).h(4).s(4).any(*range(5), matrix=matrix)
    actual = fq.run(program).statevector().reshape(32, -1)
    torch.testing.assert_close(
        actual[:, 0], fq.run(circuit).statevector().flatten(), atol=1e-12, rtol=0
    )
    assert float(actual[:, 1:].abs().square().sum()) < 1e-22


@pytest.mark.parametrize(
    "relation,oracle", [(">=", lambda a, b: a >= b), ("<", lambda a, b: a < b)]
)
def test_every_input_and_nonzero_xor_target_with_clean_workspace(
    relation: str, oracle: Callable[[int, int], bool]
) -> None:
    program = synth.synthesize(program_definition(relation), max_qubits=30)
    wires = tuple(program.registers[name] for name in ("x", "y", "value", "marker"))
    for x in range(4):
        for y in range(32):
            for marker in (0, 1):
                initial = encode((x, y, 0, marker), wires)
                expected = encode(
                    (x, y, x * x + 1, marker ^ int(oracle(x * x + 1, y))), wires
                )
                assert classical_apply(program.circuit, initial) == expected


@pytest.mark.parametrize("relation", [">=", "<"])
def test_coherence_and_absolute_phase_against_independent_permutation_circuit(
    relation: str,
) -> None:
    program = synth.synthesize(
        program_definition(relation, coherent=True),
        max_qubits=30,
        memory_limit_bytes=4 * 1024**3,
    )
    compute = torch.zeros((64, 64), dtype=torch.complex128)
    for x in range(4):
        for value in range(16):
            compute[16 * x + (value ^ (x * x + 1)), 16 * x + value] = 1
    compare = torch.zeros((1024, 1024), dtype=torch.complex128)
    for value in range(16):
        for y in range(32):
            marked = value >= y if relation == ">=" else value < y
            for marker in (0, 1):
                before = 2 * (32 * value + y) + marker
                after = 2 * (32 * value + y) + (marker ^ int(marked))
                compare[after, before] = 1
    reference = fq.Circuit(12, dtype=torch.complex128)
    for wire in range(7):
        reference.h(wire)
    reference.x(11).h(11).s(11)
    reference.any(0, matrix=cmath.exp(0.31j) * torch.eye(2, dtype=torch.complex128))
    reference.any(0, 1, 7, 8, 9, 10, matrix=compute)
    reference.any(7, 8, 9, 10, 2, 3, 4, 5, 6, 11, matrix=compare)
    state = fq.run(program).statevector().reshape(4096, -1)
    torch.testing.assert_close(
        state[:, 0], fq.run(reference).statevector().flatten(), atol=1e-12, rtol=0
    )
    assert float(state[:, 1:].abs().square().sum()) < 1e-22


def test_exhausted_boolean_search_does_not_become_an_arithmetic_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def exhausted(*args: object, **kwargs: object) -> object:
        raise BooleanCandidateError("construction budget exhausted", exhausted=True)

    monkeypatch.setattr(register_lowering, "lower_boolean_value", exhausted)
    root, functions = capture_program(program_definition())
    with pytest.raises(BooleanCandidateError, match="construction budget exhausted"):
        register_lowering.schedule_register_program(
            root, functions, boolean_values=True
        )


@pytest.mark.parametrize("limit", ["qubits", "workspace", "operations", "memory"])
def test_mixed_program_still_enforces_actual_resource_limits(limit: str) -> None:
    with pytest.raises((synth.SynthesisError, HybridVerificationError, PlanningError)):
        if limit == "qubits":
            synth.synthesize(program_definition(), max_qubits=10)
        elif limit == "workspace":
            synth.synthesize(program_definition(), max_workspace=0)
        elif limit == "operations":
            synth.synthesize(program_definition(), max_operations=10)
        else:
            synth.synthesize(program_definition(), memory_limit_bytes=16)
