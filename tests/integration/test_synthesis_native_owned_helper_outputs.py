"""A helper may initialize caller-owned fresh storage without publishing it."""

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate
from tests.integration.test_synthesis_native_exact_numbers import definition

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("angle", [0.0, 0.73, -1.21])
def test_fresh_local_helper_preparation_preserves_the_entangled_environment(
    angle: float,
) -> None:
    helper = f"@qfunc\ndef prepare(out:Output[QBit]):\n    allocate(out)\n    H(out)\n    PHASE({angle},out)\n"
    entry = definition(
        "q:Output[QBit]",
        "allocate(q)\ns:QBit=QBit()\nprepare(s)\nCX(s,q)\ndrop(s)",
        helper,
    )
    program = synth.synthesize(entry, max_workspace=0)
    result = fq.run(program)
    reference = fq.Circuit(2, dtype=torch.complex128)
    reference.h(1).phase(1, angle).cx(1, 0)
    actual = result.statevector().flatten()
    expected = fq.run(reference).statevector().flatten()
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    assert program.registers == {"q": (0,)}
    assert dict(program.probabilities(result, registers=("q",))) == pytest.approx(
        {(0,): 0.5, (1,): 0.5}
    )
    assert float(actual.reshape(2, 2)[:, 1].abs().square().sum()) == pytest.approx(0.5)
    reduced = actual.reshape(2, 2)
    torch.testing.assert_close(
        reduced @ reduced.mH,
        torch.eye(2, dtype=torch.complex128) / 2,
        atol=1e-12,
        rtol=0,
    )


def test_two_fresh_local_outputs_compose_without_ownership_transfer() -> None:
    helper = "@qfunc\ndef copy(x:Const[QBit],out:Output[QBit]):\n    out |= x\n"
    program = synth.synthesize(
        definition(
            "q:Output[QBit]",
            "allocate(q)\nH(q)\na:QBit=QBit()\nb:QBit=QBit()\ncopy(q,a)\ncopy(a,b)\ndrop(a)\ndrop(b)",
            helper,
        ),
        max_workspace=0,
    )
    reference = fq.Circuit(3, dtype=torch.complex128).h(0).cx(0, 1).cx(1, 2)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )
    assert tuple(program.registers) == ("q",)


def test_array_helper_initialization_retains_the_entire_private_array() -> None:
    helper = "@qfunc\ndef prepare(out:Output[QArray[QBit]]):\n    allocate(2,out)\n    H(out[0])\n    PHASE(0.37,out[0])\n    CX(out[0],out[1])\n"
    program = synth.synthesize(
        definition(
            "q:Output[QBit]",
            "allocate(q)\ns:QArray[QBit]=QArray()\nprepare(s)\ndrop(s)",
            helper,
        ),
        max_workspace=0,
    )
    reference = fq.Circuit(3, dtype=torch.complex128).h(2).phase(2, 0.37).cx(2, 1)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )
    assert program.registers == {"q": (0,)}
    assert program.circuit.n_wires == 3


def test_exact_signed_fractional_helper_output_keeps_its_inferred_format() -> None:
    helper = "@qfunc\ndef negate(x:Const[QNum],out:Output[QNum]):\n    out |= -x\n"
    program = synth.synthesize(
        definition(
            "x:Output[QNum[2,True,1]]",
            "allocate(x)\nhadamard_transform(x)\ns:QNum=QNum()\nnegate(x,s)\ndrop(s)",
            helper,
        )
    )
    reference = fq.Circuit(program.circuit.n_wires, dtype=torch.complex128).h(0).h(1)
    for bit in range(3):
        matrix = torch.eye(8, dtype=torch.complex128)
        for raw in range(4):
            signed = raw - 4 if raw >= 2 else raw
            if ((-signed) & 7) >> bit & 1:
                matrix[[2 * raw, 2 * raw + 1]] = matrix[[2 * raw + 1, 2 * raw]]
        reference.any(0, 1, 4 - bit, matrix=matrix)
    actual = fq.run(program).statevector().flatten()
    torch.testing.assert_close(
        actual, fq.run(reference).statevector().flatten(), atol=1e-12, rtol=0
    )
    assert program._checked.numeric_formats == (("x", True, 1),)
    workspace = program.report["workspace_qubits"]
    assert isinstance(workspace, int)
    assert program.circuit.n_wires == 5 + workspace
    assert float(actual.reshape(32, -1)[:, 1:].abs().square().sum()) < 1e-24


def test_fresh_addition_helper_keeps_every_coherent_input_and_hidden_sum() -> None:
    helper = "@qfunc\ndef add(x:Const[QNum[2]],y:Const[QNum[2]],out:Output[QNum]):\n    out |= x+y\n"
    program = synth.synthesize(
        definition(
            "x:Output[QNum[2]],y:Output[QNum[2]]",
            "allocate(x)\nallocate(y)\nhadamard_transform(x)\nhadamard_transform(y)\ns:QNum=QNum()\nadd(x,y,s)\ndrop(s)",
            helper,
        )
    )
    reference = fq.Circuit(program.circuit.n_wires, dtype=torch.complex128)
    for wire in range(4):
        reference.h(wire)
    for bit in range(3):
        matrix = torch.eye(32, dtype=torch.complex128)
        for word in range(16):
            if ((word >> 2) + (word & 3)) >> bit & 1:
                matrix[[2 * word, 2 * word + 1]] = matrix[[2 * word + 1, 2 * word]]
        reference.any(0, 1, 2, 3, 6 - bit, matrix=matrix)
    actual = fq.run(program).statevector().flatten()
    expected = fq.run(reference).statevector().flatten()
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    assert float(actual.reshape(128, -1)[:, 1:].abs().square().sum()) < 1e-24
    a, b = actual.reshape(16, -1), expected.reshape(16, -1)
    torch.testing.assert_close(a @ a.mH, b @ b.mH, atol=1e-12, rtol=0)
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    workspace = program.report["workspace_qubits"]
    assert isinstance(workspace, int)
    assert resources.n_qubits == 7 + workspace


@pytest.mark.parametrize(
    "body",
    [
        "allocate(q)\ns:QBit\nallocate(s)\ninit(s)\ndrop(s)",
        "allocate(q)\ns:QBit\ninit(s)\ninit(s)\ndrop(s)",
        "allocate(q)\ns:QBit\ninit(s)\ndrop(s)\ninit(s)",
        "allocate(q)\ns:QBit\ninit(s)",
        "allocate(q)\ns:QBit\ncontrol(q,lambda:init(s))\ndrop(s)",
        "allocate(q)\ns:QBit\nwith adjoint():\n    init(s)\ndrop(s)",
        "allocate(q)\ns:QBit\ninit(s)\nreturn s",
    ],
)
def test_only_fresh_owned_forward_outputs_are_admitted(body: str) -> None:
    helper = "@qfunc\ndef init(out:Output[QBit]):\n    allocate(out)\n"
    with pytest.raises(ValueError):
        synth.synthesize(definition("q:Output[QBit]", body, helper))


def test_borrowed_const_and_mutable_aliases_cannot_be_initialized_by_a_helper() -> None:
    cases = [
        "@qfunc\ndef outer(q:Const[QBit]):\n    init(q)\n",
        "@qfunc\ndef outer(q:QBit):\n    init(q)\n",
    ]
    for outer in cases:
        helper = "@qfunc\ndef init(out:Output[QBit]):\n    allocate(out)\n" + outer
        with pytest.raises(ValueError):
            synth.synthesize(
                definition("q:Output[QBit]", "allocate(q)\nouter(q)", helper)
            )


def test_read_output_alias_and_format_mismatch_stay_rejected() -> None:
    helper = "@qfunc\ndef negate(x:Const[QNum],out:Output[QNum]):\n    out |= -x\n"
    bodies = [
        "s:QNum[1]=QNum()\nnegate(s,s)\nallocate(q)\ndrop(s)",
        "allocate(q)\ns:QNum[2,False,1]=QNum()\nnegate(q,s)\ndrop(s)",
    ]
    for body in bodies:
        with pytest.raises(ValueError):
            synth.synthesize(definition("q:Output[QNum[1]]", body, helper))
