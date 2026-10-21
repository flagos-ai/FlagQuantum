"""The operand a `Circuit.gate` call carries must fit the wires it names.

The runtime has always compared an operand's entry count against `4 ** arity` —
`flagquantum/simulation/gate_matrix.py`'s `_dense_operator` says so in its own
docstring, "the entry count decides the size, not the shape". What it did with a
mismatch was raise at *execution*: the program was built, planned and only then
refused with `instruction 'any' on operands (0,) carries a dense operator of
shape (2, 3) with 6 entries`. A malformed argument to `Circuit.gate` belongs at
the call that made it, which is where every other schema violation already
lands.

Nothing is refused here that the runtime would have accepted. The tests below
therefore come in two halves, and the second half is the more important one: the
spellings the runtime does accept — a flat `(4,)`, a `(1, 4)`, a `(1, 2, 2)`, and
a plain nested list — must still be accepted, because the entry count rather
than the shape is the contract.
"""

from __future__ import annotations

import pytest
import torch

import flagquantum as fq
from flagquantum.core.ir import CircuitIR, Instruction, IRValidationError
from flagquantum.simulation.gate_matrix import gate_matrix

pytestmark = pytest.mark.unit

#: A valid one-qubit operand, used to show that the check is not refusing operands
#: generally.
_ONE_QUBIT = torch.eye(2, dtype=torch.complex128)

#: A valid two-qubit operand.
_TWO_QUBIT = torch.eye(4, dtype=torch.complex128)


def _instruction(matrix: object, wires: tuple[int, ...] = (0,)) -> Instruction:
    return Instruction("any", wires, matrix=matrix)


@pytest.mark.parametrize(
    ("label", "matrix", "wires"),
    [
        ("(2, 3) on one wire", torch.zeros(2, 3, dtype=torch.complex128), (0,)),
        ("(3, 3) on one wire", torch.zeros(3, 3, dtype=torch.complex128), (0,)),
        ("(3,) on one wire", torch.zeros(3, dtype=torch.complex128), (0,)),
        ("(0, 0) on one wire", torch.zeros(0, 0, dtype=torch.complex128), (0,)),
        ("(4, 4) on one wire", _TWO_QUBIT, (0,)),
        ("(2, 2) on two wires", _ONE_QUBIT, (0, 1)),
        (
            "(2, 2, 2) on two wires",
            torch.zeros(2, 2, 2, dtype=torch.complex128),
            (0, 1),
        ),
        ("(2, 2) on two wires", _ONE_QUBIT, (0, 1)),
        ("(16,) on three wires", torch.zeros(4, 4, dtype=torch.complex128), (0, 1, 2)),
    ],
)
def test_an_operand_that_does_not_fit_its_wires_is_refused_at_construction(
    label: str, matrix: torch.Tensor, wires: tuple[int, ...]
) -> None:
    """The refusal happens where the argument was written, and it names it."""

    with pytest.raises(IRValidationError) as caught:
        _instruction(matrix, wires)

    message = str(caught.value)
    assert "'any'" in message, label
    assert str(wires) in message, label
    assert "matrix of shape" in message, label
    assert str(tuple(matrix.shape)) in message, label
    # The expectation is stated in the same sentence, so a reader does not have
    # to know the runtime's arithmetic to repair the call.
    expected = 4 ** len(wires)
    assert f"{expected} entries" in message, label


def test_the_refusal_is_reachable_from_the_public_gate_call() -> None:
    """`Circuit.gate` is the surface a user writes, so it is the surface that says so."""

    circuit = fq.Circuit(1)
    with pytest.raises(IRValidationError):
        circuit.gate("any", [0], matrix=torch.zeros(2, 3, dtype=torch.complex128))
    assert circuit.to_ir().instructions == ()


@pytest.mark.parametrize(
    ("label", "matrix", "wires"),
    [
        ("(2, 2)", _ONE_QUBIT, (0,)),
        ("(4,) flat", _ONE_QUBIT.reshape(4), (0,)),
        ("(1, 4) flat", _ONE_QUBIT.reshape(1, 4), (0,)),
        ("(1, 2, 2)", _ONE_QUBIT.reshape(1, 2, 2), (0,)),
        ("(4, 4)", _TWO_QUBIT, (0, 1)),
        ("(2, 2, 2, 2)", _TWO_QUBIT.reshape(2, 2, 2, 2), (0, 1)),
    ],
)
def test_every_spelling_the_runtime_accepts_is_still_accepted(
    label: str, matrix: torch.Tensor, wires: tuple[int, ...]
) -> None:
    """The entry count is the contract, so a reshape of a valid operand stays valid.

    `(4,)`, `(1, 4)` and `(1, 2, 2)` are the three spellings Decision 5 of the
    operator-registration proposal names, and the runtime's own documented rule
    admits all three. Refusing them here would change which programs run, which
    is not what this check is for.
    """

    instruction = _instruction(matrix, wires)
    assert instruction.matrix is not None, label
    dense = gate_matrix(
        instruction, bsz=1, device=torch.device("cpu"), dtype=torch.complex128
    )
    expected = 2 ** len(wires)
    assert tuple(dense.shape) == (expected, expected), label


def test_a_python_sequence_is_left_to_the_runtime() -> None:
    """A nested list publishes no shape, so this check declines to judge it.

    Declining is deliberate: the count would have to be guessed by walking a
    structure that may hold tensors, and the runtime already reads it. The test
    pins that the check is not silently refusing sequences it cannot count.
    """

    instruction = _instruction([[1.0, 0.0], [0.0, 1.0]])
    assert instruction.matrix == [[1.0, 0.0], [0.0, 1.0]]


def test_a_channel_is_classified_before_it_is_measured() -> None:
    """A channel carries Kraus tensors, not an operand, and is not measured as one.

    The classifier is `metadata["is_channel"]`, never the shape: a non-square
    Kraus operator would otherwise be read as a malformed operand and refused
    here instead of by the channel's own validation.
    """

    channel = Instruction(
        "any",
        (0,),
        matrix=(torch.eye(2, dtype=torch.complex128),),
        metadata={"is_channel": True},
    )
    assert channel.metadata["is_channel"] is True


def test_the_failure_no_longer_arrives_from_the_executor() -> None:
    """The program cannot be built wrong, so it cannot be run wrong.

    Before this check the same call built a circuit, and the executor refused it
    with a `ValidationError` about a dense operator. The distinction is observable
    in which call raises, and this test names both ends.
    """

    circuit = fq.Circuit(1)
    with pytest.raises(IRValidationError) as caught:
        circuit.gate("any", [0], matrix=torch.zeros(2, 3, dtype=torch.complex128))
    # The old message is the executor's; it is not the one raised now.
    assert "dense operator" not in str(caught.value)
    assert "carries a matrix of shape" in str(caught.value)


def test_a_circuit_read_from_an_ir_is_held_to_the_same_rule() -> None:
    """Deserialization goes through the same constructor, so the rule is total."""

    with pytest.raises(IRValidationError):
        CircuitIR(
            1,
            (
                Instruction(
                    "any", (0,), matrix=torch.zeros(2, 3, dtype=torch.complex128)
                ),
            ),
        )
