"""Static array-bit Boolean XOR preserves arbitrary targets and coherence."""

import cmath
from collections.abc import Callable
from itertools import product

import pytest
import torch

import flagquantum as fq
from benchmarks.synthesis.verification import classical_apply, encode
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from tests.integration.test_synthesis_native_exact_numbers import definition

pytestmark = pytest.mark.integration
CASES: tuple[tuple[str, Callable[[int, int, int], int]], ...] = (
    ("s[0] & s[1]", lambda a, b, c: a & b),
    ("s[0] | s[1]", lambda a, b, c: a | b),
    ("s[0] ^ s[1] ^ s[2]", lambda a, b, c: a ^ b ^ c),
    ("~s[0]", lambda a, b, c: 1 - a),
    ("(s[0] | s[1]) & (~s[2] ^ 1)", lambda a, b, c: (a | b) & c),
    ("s[0] ^ s[0] ^ 1", lambda a, b, c: 1),
)


@pytest.mark.parametrize("expression,oracle", CASES)
def test_all_basis_inputs_nonzero_target_and_absolute_coherence(
    expression: str, oracle: Callable[[int, int, int], int]
) -> None:
    body = "allocate(3,s)\nallocate(t)\n"
    ports = "s:Output[QArray[QBit]],t:Output[QBit]"
    program = synth.synthesize(definition(ports, body + f"t ^= {expression}"))
    wires = (program.registers["s"], program.registers["t"])
    for a, b, c, target in product(range(2), repeat=4):
        source = a + 2 * b + 4 * c
        assert classical_apply(
            program.circuit, encode((source, target), wires)
        ) == encode((source, target ^ oracle(a, b, c)), wires)
    coherent = synth.synthesize(
        definition(
            ports,
            body
            + "hadamard_transform(s)\nH(t)\nphase(s[0]*0.17+s[1]*0.23+s[2]*0.31+t*0.43+0.19,1)\n"
            + f"t ^= {expression}",
        )
    )
    matrix = torch.zeros((16, 16), dtype=torch.complex128)
    for a, b, c, target in product(range(2), repeat=4):
        before = 8 * c + 4 * b + 2 * a + target
        after = 8 * c + 4 * b + 2 * a + (target ^ oracle(a, b, c))
        matrix[after, before] = torch.tensor(
            cmath.exp(1j * (a * 0.17 + b * 0.23 + c * 0.31 + target * 0.43 + 0.19)),
            dtype=torch.complex128,
        )
    reference = fq.Circuit(4, dtype=torch.complex128)
    for bit in range(4):
        reference.h(bit)
    reference.any(*range(4), matrix=matrix)
    compare_states(coherent, reference, ("s", "t"))
    assert program.circuit.n_wires <= 5
    assert program.report["workspace_qubits"] == program.circuit.n_wires - 4


def test_helper_static_index_slice_and_partial_target_borrow() -> None:
    helper = "@qfunc\ndef marker(s:QArray[QBit],t:QBit,i:int):\n    t ^= s[i] & s[1]\n"
    program = synth.synthesize(
        definition(
            "s:Output[QArray[QBit]]",
            "allocate(5,s)\nhadamard_transform(s)\nphase(s[4],0.37)\nmarker(s[1:4],s[4],0)",
            helper,
        )
    )
    reference = fq.Circuit(5, dtype=torch.complex128)
    for bit in range(5):
        reference.h(bit)
    reference.any(
        0,
        matrix=torch.diag(torch.tensor([1, cmath.exp(0.37j)], dtype=torch.complex128)),
    ).ccx(3, 2, 0)
    compare_states(program, reference, ("s",))


@pytest.mark.parametrize("expression", ["t & s[0]", "s[3]", "s[-4]"])
def test_alias_and_bounds_rejected(expression: str) -> None:
    with pytest.raises((synth.SynthesisError, HybridVerificationError)):
        synth.synthesize(
            definition(
                "s:Output[QArray[QBit]],t:Output[QBit]",
                "allocate(3,s)\nallocate(t)\n" + f"t ^= {expression}",
            )
        )


def test_helper_cross_formal_alias_is_rejected_even_when_cancelled() -> None:
    helper = "@qfunc\ndef marker(s:QArray[QBit],t:QBit):\n    t ^= s[0] ^ s[0]\n"
    with pytest.raises(
        (synth.SynthesisError, HybridVerificationError), match="alias|disjoint"
    ):
        synth.synthesize(
            definition(
                "s:Output[QArray[QBit]]",
                "allocate(3,s)\nmarker(s,s[0])",
                helper,
            )
        )


@pytest.mark.parametrize("constraint", ["qubits", "operations", "memory"])
def test_boolean_xor_uses_actual_resource_budgets(constraint: str) -> None:
    entry = definition(
        "s:Output[QArray[QBit]],t:Output[QBit]",
        "allocate(8,s)\nallocate(t)\nt ^= s[0]&s[1]&s[2]&s[3]&s[4]&s[5]&s[6]&s[7]",
    )
    from flagquantum.errors import PlanningError

    with pytest.raises((synth.SynthesisError, HybridVerificationError, PlanningError)):
        if constraint == "qubits":
            synth.synthesize(entry, max_qubits=8)
        elif constraint == "operations":
            synth.synthesize(entry, max_operations=1)
        else:
            synth.synthesize(entry, memory_limit_bytes=16)


def test_boolean_xor_ninth_input_fails_before_sparse_phase_fallback() -> None:
    expression = " & ".join(f"s[{bit}]" for bit in range(9))
    with pytest.raises((synth.SynthesisError, HybridVerificationError), match="eight"):
        synth.synthesize(
            definition(
                "s:Output[QArray[QBit]],t:Output[QBit]",
                f"allocate(9,s)\nallocate(t)\nt ^= {expression}",
            )
        )


def test_const_target_and_uninitialized_reads_fail_closed() -> None:
    helper = "@qfunc\ndef marker(s:QArray[QBit],t:Const[QBit]):\n    t ^= s[0]\n"
    with pytest.raises((synth.SynthesisError, HybridVerificationError)):
        synth.synthesize(
            definition(
                "s:Output[QArray[QBit]],t:Output[QBit]",
                "allocate(3,s)\nallocate(t)\nmarker(s,t)",
                helper,
            )
        )
    with pytest.raises((synth.SynthesisError, HybridVerificationError)):
        synth.synthesize(
            definition(
                "s:Output[QArray[QBit]],t:Output[QBit]",
                "allocate(t)\nt ^= s[0]",
            )
        )


@pytest.mark.parametrize("workspace", [0, 4])
def test_eight_input_marker_respects_workspace_choice(workspace: int) -> None:
    program = synth.synthesize(
        definition(
            "s:Output[QArray[QBit]],t:Output[QBit]",
            "allocate(8,s)\nallocate(t)\nhadamard_transform(s)\n"
            "t ^= s[0]&s[1]&s[2]&s[3]&s[4]&s[5]&s[6]&s[7]",
        ),
        max_workspace=workspace,
        max_qubits=9 + workspace,
    )
    actual = fq.run(program).statevector().flatten()
    work = program.circuit.n_wires - 9
    expected = torch.zeros((512, 1 << work), dtype=torch.complex128)
    for word in range(256):
        expected[2 * word + int(word == 255), 0] = 1 / 16
    torch.testing.assert_close(actual, expected.flatten(), atol=2e-11, rtol=0)
    used_workspace = program.report["workspace_qubits"]
    assert isinstance(used_workspace, int)
    assert used_workspace <= workspace
