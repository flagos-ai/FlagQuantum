"""Unsigned borrowed words retain ordered storage, modular updates and phase."""

from collections.abc import Callable
from itertools import product

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler._construction.frontend_python import _QuantumFunction
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from tests.integration.test_synthesis_native_exact_numbers import definition
from tests.integration.test_synthesis_native_inplace_preparation import (
    independent_unitary,
)
from tests.team.compiler.test_construction_python_program import program_definition

pytestmark = pytest.mark.integration

HELPERS = """
from flagquantum.synth import control, adjoint, within, apply, X, RZ
@qfunc
def increment(word: QNum) -> None:
    word += 1
@qfunc
def decrement(word: QNum) -> None:
    word -= 1
@qfunc
def nested(word: QNum) -> None:
    increment(word)
    increment(word)
"""


def _definition(
    body: str, *, coherent: bool = False, initial: tuple[int, int, int] = (0, 0, 0)
) -> _QuantumFunction:

    initialization = (
        "hadamard_transform(x)\nhadamard_transform(flag)\nhadamard_transform(c)\nRZ(0.37,flag)\nRZ(-0.23,c)"
        if coherent
        else f"x ^= {initial[0]}\nflag ^= {initial[1]}\nc ^= {initial[2]}"
    )
    return program_definition(
        HELPERS
        + "\n@qfunc\ndef main(x: Output[QNum[3]], flag: Output[QBit], c: Output[QBit]) -> None:\n"
        + "\n".join(
            "    " + line
            for line in (
                "allocate(x)\nallocate(flag)\nallocate(c)\n"
                + initialization
                + "\n"
                + body
            ).splitlines()
        ),
    )


def _permutation(transform: Callable[[int], int]) -> torch.Tensor:
    matrix = torch.zeros((32, 32), dtype=torch.complex128)
    for column in range(32):
        matrix[transform(column), column] = 1
    return matrix


def _update(basis: int, delta: int, controlled: bool = False) -> int:
    # MSB physical order is x[2],x[1],x[0],flag,c; packed LSB order
    # is flag,x[1],x[2]. x[0] remains an independent surrounding bit.
    if controlled and not basis & 1:
        return basis
    word = ((basis >> 1) & 1) | (((basis >> 3) & 3) << 1)
    updated = (word + delta) % 8
    return (basis & 5) | ((updated & 1) << 1) | ((updated >> 1) << 3)


@pytest.mark.parametrize(
    "body,delta,controlled",
    [
        ("increment([flag,x[1:3]])", 1, False),
        ("decrement((flag,x[1:3]))", -1, False),
        ("nested([flag,x[1:3]])", 2, False),
        ("control(c,lambda:increment([flag,x[1:3]]))", 1, True),
        ("control(c,lambda:decrement([flag,x[1:3]]))", -1, True),
    ],
)
def test_every_basis_word_and_nonzero_surrounding_bits(
    body: str, delta: int, controlled: bool
) -> None:
    for x, flag, control in product(range(8), range(2), range(2)):
        program = synth.synthesize(
            _definition(body, initial=(x, flag, control)),
        )
        state = fq.run(program).statevector().reshape(32, -1)
        expected = torch.zeros_like(state)
        expected[_update((x << 2) | (flag << 1) | control, delta, controlled), 0] = 1
        torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("inverse", [False, True])
def test_coherent_control_and_adjoint_preserve_absolute_phase(inverse: bool) -> None:
    body = "control(c,lambda:increment([flag,x[1:3]]))"
    if inverse:
        body = "with adjoint():\n    " + body
    program = synth.synthesize(_definition(body, coherent=True))
    reference = fq.Circuit(5, dtype=torch.complex128)
    for wire in range(5):
        reference.h(wire)
    reference.rz(3, 0.37).rz(4, -0.23)
    reference.any(
        *range(5),
        matrix=_permutation(lambda basis: _update(basis, -1 if inverse else 1, True)),
    )
    actual = fq.run(program).statevector().reshape(32, -1)
    torch.testing.assert_close(
        actual[:, 0], reference.state().flatten(), atol=1e-12, rtol=0
    )
    assert torch.count_nonzero(actual[:, 1:]) == 0


@pytest.mark.parametrize(
    "body",
    [
        "increment([flag,flag])",
        "increment([x,x[1:3]])",
        "increment([x[1:3],x[2]])",
        "control(x[0],lambda:increment([flag,x[1:3]]))",
        "increment(x[3:4])",
    ],
)
def test_duplicate_overlap_condition_dependency_and_invalid_views_refuse(
    body: str,
) -> None:
    with pytest.raises((synth.SynthesisError, HybridVerificationError)):
        synth.synthesize(_definition(body, coherent=True))


@pytest.mark.parametrize(
    "view,expected",
    [
        ("x[1:3]", lambda x: x >> 1),
        ("[x[2],x[0]]", lambda x: ((x >> 2) & 1) | ((x & 1) << 1)),
    ],
)
def test_sliced_numeric_helper_predicates_use_formal_bit_order(
    view: str, expected: Callable[[int], int]
) -> None:
    helpers = f"""{HELPERS}
@qfunc
def mark(word: QNum, flag: QBit) -> None:
    control(word == 2, lambda: X(flag))
"""
    definition = program_definition(
        f"""{helpers}
@qfunc
def main(x: Output[QNum[3]], flag: Output[QBit]) -> None:
    allocate(x)
    allocate(flag)
    hadamard_transform(x)
    hadamard_transform(flag)
    RZ(0.37,flag)
    mark({view},flag)
""",
    )
    program = synth.synthesize(definition)
    reference = fq.Circuit(4, dtype=torch.complex128)
    for wire in range(4):
        reference.h(wire)
    reference.rz(3, 0.37)
    matrix = torch.zeros((16, 16), dtype=torch.complex128)
    for value in range(16):
        matrix[value ^ int(expected(value >> 1) == 2), value] = 1
    reference.any(*range(4), matrix=matrix)
    actual = fq.run(program).statevector().reshape(16, -1)
    torch.testing.assert_close(
        actual[:, 0], reference.state().flatten(), atol=1e-12, rtol=0
    )
    assert torch.count_nonzero(actual[:, 1:]) == 0


def test_nested_slice_is_relative_to_borrowed_formal_and_keeps_work_zero() -> None:
    helpers = f"""{HELPERS}
@qfunc
def mark(word: QNum, flag: QBit) -> None:
    control(word[1:3] == 2, lambda: X(flag))
@qfunc
def nested_mark(word: QNum, flag: QBit) -> None:
    mark(word[1:4],flag)
"""
    definition = program_definition(
        f"""{helpers}
@qfunc
def main(x: Output[QNum[6]], flag: Output[QBit]) -> None:
    allocate(x)
    allocate(flag)
    hadamard_transform(x)
    nested_mark(x[1:5],flag)
""",
    )
    program = synth.synthesize(definition)
    actual = fq.run(program).statevector().reshape(128, -1)
    expected = torch.zeros_like(actual)
    for x in range(64):
        expected[(x << 1) | int(((x >> 3) & 3) == 2), 0] = 1 / 8
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("composition", ["direct", "inverse", "within", "controlled"])
def test_partial_preparation_on_entangled_nonzero_storage(composition: str) -> None:
    vector = (0.5, -0.5, 0.5, 0.5)
    helper = """
@qfunc
def prepare(word: QNum) -> None:
    inplace_prepare_amplitudes([0.5,-0.5,0.5,0.5],0,word)
@qfunc
def selected(c: QBit, word: QNum) -> None:
    control(c,lambda:prepare(word))
"""
    call = "prepare([x[2],x[0]])"
    if composition == "inverse":
        call = "with adjoint():\n    " + call
    elif composition == "within":
        call = "with within():\n    " + call + "\nwith apply():\n    RZ(0.41,x[0])"
    elif composition == "controlled":
        call = "control(c,lambda:selected(d,[x[2],x[0]]))"
    program = synth.synthesize(
        definition(
            "x:Output[QNum[3]],c:Output[QBit],d:Output[QBit]",
            "allocate(x)\nallocate(c)\nallocate(d)\n"
            "hadamard_transform(x)\nH(c)\nH(d)\n"
            "RZ(0.37,x[1])\nRZ(-0.61,c)\nCX(x[0],d)\n" + call,
            helper,
        )
    )
    reference = fq.Circuit(5, dtype=torch.complex128)
    for wire in range(5):
        reference.h(wire)
    reference.rz(1, 0.37).rz(3, -0.61).cx(2, 4)
    matrix = independent_unitary(vector)
    if composition == "inverse":
        matrix = matrix.conj().T
    elif composition == "within":
        matrix = (
            matrix.conj().T
            @ torch.diag(
                torch.exp(0.205j * torch.tensor([-1, -1, 1, 1], dtype=torch.float64))
            )
            @ matrix
        )
    if composition == "controlled":
        controlled = torch.eye(16, dtype=torch.complex128)
        controlled[12:, 12:] = matrix
        reference.any(3, 4, 2, 0, matrix=controlled)
    else:
        reference.any(2, 0, matrix=matrix)
    actual = fq.run(program).statevector().reshape(32, -1)
    torch.testing.assert_close(
        actual[:, 0], reference.state().flatten(), atol=1e-12, rtol=0
    )
    assert torch.count_nonzero(actual[:, 1:]) == 0


@pytest.mark.parametrize(
    "case", ["const", "uninitialized", "width", "signed", "fractional", "escape"]
)
def test_borrow_ownership_formats_and_output_escape_refuse(case: str) -> None:
    helper = HELPERS
    ports = "x:Output[QNum[3]]"
    body = "allocate(x)\n"
    call = "increment(x[1:3])"
    if case == "const":
        helper += "@qfunc\ndef protected(x:Const[QNum]):\n    increment(x[1:3])\n"
        call = "protected(x)"
    elif case == "uninitialized":
        body = ""
    elif case == "width":
        helper += "@qfunc\ndef fixed(x:QNum[3]):\n    x += 1\n"
        call = "fixed(x[1:3])"
    elif case in {"signed", "fractional"}:
        ports = (
            "x:Output[QNum[3,True,0]]"
            if case == "signed"
            else "x:Output[QNum[3,False,1]]"
        )
    else:
        helper += "@qfunc\ndef escape(x:Output[QNum[2]]):\n    allocate(x)\n"
        call = "escape(x[1:3])"
    with pytest.raises((synth.SynthesisError, HybridVerificationError)):
        synth.synthesize(definition(ports, body + call, helper))


def test_borrowed_increment_within_apply_matches_independent_permutation() -> None:
    program = synth.synthesize(
        _definition(
            "with within():\n    increment([flag,x[1:3]])\nwith apply():\n    RZ(0.41,x[2])",
            coherent=True,
        )
    )
    reference = fq.Circuit(5, dtype=torch.complex128)
    for wire in range(5):
        reference.h(wire)
    reference.rz(3, 0.37).rz(4, -0.23)
    reference.any(*range(5), matrix=_permutation(lambda b: _update(b, 1)))
    reference.rz(0, 0.41)
    reference.any(*range(5), matrix=_permutation(lambda b: _update(b, -1)))
    actual = fq.run(program).statevector().reshape(32, -1)
    torch.testing.assert_close(
        actual[:, 0], reference.state().flatten(), atol=1e-12, rtol=0
    )
    assert torch.count_nonzero(actual[:, 1:]) == 0


@pytest.mark.parametrize("limit", ["operations", "qubits", "workspace"])
def test_borrowed_predicate_respects_actual_resource_budgets(
    limit: str,
) -> None:
    helper = (
        "@qfunc\ndef mark(word:QNum,flag:QBit):\n    control(word==2,lambda:X(flag))\n"
    )
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(
            definition(
                "x:Output[QNum[4]],flag:Output[QBit]",
                "allocate(x)\nallocate(flag)\nhadamard_transform(x)\nmark(x[1:4],flag)",
                helper,
            ),
            max_operations=1 if limit == "operations" else None,
            max_qubits=4 if limit == "qubits" else None,
            max_workspace=0 if limit == "workspace" else None,
        )


@pytest.mark.parametrize("basis", range(8))
def test_partial_preparation_preserves_every_column_and_surrounding_bit(
    basis: int,
) -> None:
    helper = "@qfunc\ndef prepare(word:QNum):\n    inplace_prepare_amplitudes([0.5,-0.5,0.5,0.5],0,word)\n"
    program = synth.synthesize(
        definition(
            "x:Output[QNum[3]]",
            f"allocate(x)\nx ^= {basis}\nprepare(x[1:3])",
            helper,
        ),
        max_workspace=0,
    )
    reference = fq.Circuit(3, dtype=torch.complex128)
    for wire in range(3):
        if basis & (1 << (2 - wire)):
            reference.x(wire)
    reference.any(0, 1, matrix=independent_unitary((0.5, -0.5, 0.5, 0.5)))
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(),
        reference.state().flatten(),
        atol=1e-12,
        rtol=0,
    )


def test_nested_preparation_rejects_ninth_indexed_location() -> None:
    helpers = "@qfunc\ndef level7(word:QNum):\n    inplace_prepare_amplitudes([0.5,-0.5,0.5,0.5],0,word)\n"
    for index in reversed(range(7)):
        formals = ",".join(f"c{k}:QBit" for k in range(index, 7))
        actuals = ",".join([*(f"c{k}" for k in range(index + 1, 7)), "word"])
        helpers += f"@qfunc\ndef level{index}({formals},word:QNum):\n    control(c{index},lambda:level{index+1}({actuals}))\n"
    ports = ",".join(
        [*(f"c{k}:Output[QBit]" for k in range(7)), "word:Output[QNum[2]]"]
    )
    body = "\n".join(
        [
            *(f"allocate(c{k})" for k in range(7)),
            "allocate(word)",
            "level0(c0,c1,c2,c3,c4,c5,c6,word)",
        ]
    )
    with pytest.raises(HybridVerificationError, match="eight indexed locations"):
        synth.synthesize(definition(ports, body, helpers))
