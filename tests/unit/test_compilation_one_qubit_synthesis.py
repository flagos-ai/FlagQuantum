"""Unit tests for one-qubit Euler synthesis into a published z-rotation basis."""

from __future__ import annotations

import math

import pytest
import torch

from flagquantum.compiler.one_qubit_synthesis import (
    synthesize_one_qubit,
    synthesize_one_qubit_matrix,
)
from flagquantum.core.ir import Instruction
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS
from flagquantum.simulation.gate_matrix import gate_matrix

pytestmark = pytest.mark.unit

_ANGLES = {"theta": 0.7137, "phi": -0.4211, "lbd": 1.9073}
_EDGE_ANGLES = (0.0, math.pi / 4, math.pi / 2, math.pi, -math.pi / 2, 2 * math.pi)
_SINGLE_QUBIT_UNITARIES = tuple(
    sorted(
        name
        for name, schema in OPERATOR_SCHEMAS.items()
        if schema.unitary and schema.arity == 1
    )
)
_Z_ROTATIONS = ("rz", "phase", "u1")


def _unitary(instruction: Instruction) -> torch.Tensor:
    matrix = gate_matrix(instruction, bsz=1, device="cpu", dtype=torch.complex128)
    return matrix.reshape(2, 2)


def _product(leaves: tuple[Instruction, ...]) -> torch.Tensor:
    total = torch.eye(2, dtype=torch.complex128)
    for leaf in leaves:
        total = _unitary(leaf) @ total
    return total


def _assert_same_unitary(leaves: tuple[Instruction, ...], source: Instruction) -> None:
    """The leaves must equal the source up to one global phase.

    An empty sequence is the identity, which is a correct answer for `i` and for
    a rotation whose angles all vanish.
    """

    product = _product(leaves) @ _unitary(source).conj().mT
    torch.testing.assert_close(
        product,
        product[0, 0] * torch.eye(2, dtype=torch.complex128),
        rtol=0.0,
        atol=1e-12,
    )


def _angles_for(name: str, **overrides: object) -> dict[str, object]:
    schema = OPERATOR_SCHEMAS[name]
    angles: dict[str, object] = {item: _ANGLES[item] for item in schema.parameters}
    angles.update(overrides)
    return angles


@pytest.mark.parametrize("z_rotation", _Z_ROTATIONS)
def test_every_declared_single_qubit_unitary_synthesizes(z_rotation: str) -> None:
    """Every declared arity-1 unitary reaches the basis, with a pinned count."""

    synthesized: list[str] = []
    rewritten: list[str] = []
    for name in _SINGLE_QUBIT_UNITARIES:
        source = Instruction(name, (0,), params=_angles_for(name))
        leaves = synthesize_one_qubit(source, z_rotation=z_rotation)
        if name == z_rotation:
            assert leaves is None
            continue
        assert leaves is not None, f"{name} has no synthesis into {z_rotation}"
        synthesized.append(name)
        _assert_same_unitary(leaves, source)
        if leaves:
            rewritten.append(name)
        assert {leaf.name for leaf in leaves} <= {z_rotation, "sx"}

    # Non-vacuity: one opcode is the basis's own z-rotation, one is the identity,
    # so an 18-opcode schema table can only reach 17 and only rewrite 16.
    assert len(_SINGLE_QUBIT_UNITARIES) == 18
    assert len(synthesized) == 17
    assert len(rewritten) == 16
    assert "i" not in rewritten


@pytest.mark.parametrize("name", ("h", "s", "sdg", "t", "tdg", "sx", "sxdg", "x", "y"))
def test_parameter_free_gates_use_the_tabulated_angles(name: str) -> None:
    """Tabulated opcodes need no parameters and stay in short form."""

    source = Instruction(name, (0,))
    leaves = synthesize_one_qubit(source, z_rotation="rz")
    assert leaves is not None
    _assert_same_unitary(leaves, source)
    assert len(leaves) <= 3


@pytest.mark.parametrize("z_rotation", _Z_ROTATIONS)
def test_an_rx_pulse_replaces_sx_without_changing_the_unitary(z_rotation: str) -> None:
    """`RX(pi/2)` *is* `SX`, so a rotation-only basis reaches the same group.

    A basis that publishes `rx` and a z-rotation but no `sx` is no weaker, and the
    pulse angle has to be bound because `rx` declares `theta`.
    """

    for name in _SINGLE_QUBIT_UNITARIES:
        if name == z_rotation:
            continue
        source = Instruction(name, (0,), params=_angles_for(name))
        leaves = synthesize_one_qubit(source, z_rotation=z_rotation, pulse_opcode="rx")
        assert leaves is not None, name
        assert {leaf.name for leaf in leaves} <= {z_rotation, "rx"}
        for leaf in leaves:
            if leaf.name == "rx":
                assert leaf.params["theta"] == math.pi / 2
        _assert_same_unitary(leaves, source)


def test_parameter_free_table_covers_exactly_the_opcodes_that_need_it() -> None:
    """The table is not a second source of truth for parameterized opcodes."""

    from flagquantum.compiler.one_qubit_synthesis import _FIXED_EULER_ANGLES

    tabulated = set(_FIXED_EULER_ANGLES)
    parameter_free = {
        name
        for name in _SINGLE_QUBIT_UNITARIES
        if not OPERATOR_SCHEMAS[name].parameters
    }
    assert tabulated == parameter_free


@pytest.mark.parametrize("polar", _EDGE_ANGLES)
def test_edge_polar_angles_reproduce_the_rotation(polar: float) -> None:
    """Short forms for 0, +/-pi/2, +/-pi stay correct, not merely shorter."""

    source = Instruction("rx", (0,), params={"theta": polar})
    leaves = synthesize_one_qubit(source, z_rotation="rz")
    assert leaves is not None
    _assert_same_unitary(leaves, source)
    assert len(leaves) <= 5
    if polar in (0.0, -math.pi / 2, math.pi):
        assert len(leaves) < 5
        assert not any(leaf.params.get("theta") == 0 for leaf in leaves)


def test_i_becomes_an_empty_sequence_and_zero_leaves_are_dropped() -> None:
    """A gate that is the identity on the nose emits nothing to execute."""

    assert synthesize_one_qubit(Instruction("i", (0,)), z_rotation="rz") == ()
    assert (
        synthesize_one_qubit(
            Instruction("rz", (0,), params={"theta": 0.0}), z_rotation="phase"
        )
        == ()
    )
    leaves = synthesize_one_qubit(
        Instruction("u2", (0,), params={"phi": -math.pi / 2, "lbd": math.pi / 2}),
        z_rotation="rz",
    )
    assert leaves is not None
    assert all(leaf.params.get("theta") != 0 for leaf in leaves)


def test_trainable_angles_stay_attached_to_the_gate_that_uses_them() -> None:
    """A trainable rotation must not turn into a constant during synthesis."""

    theta = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    leaves = synthesize_one_qubit(
        Instruction("rz", (0,), params={"theta": theta}), z_rotation="phase"
    )

    assert leaves is not None
    assert tuple(leaf.name for leaf in leaves) == ("phase",)
    angle = leaves[0].params["theta"]
    assert isinstance(angle, torch.Tensor) and angle.requires_grad
    torch.testing.assert_close(angle, theta, rtol=0.0, atol=0.0)
    angle.backward()
    assert theta.grad is not None and float(theta.grad) == 1.0


@pytest.mark.parametrize(
    ("name", "wires"),
    (("cx", (0, 1)), ("swap", (0, 1)), ("barrier", (0,)), ("measure", (0,))),
)
def test_other_arity_and_meta_instructions_are_left_to_the_caller(
    name: str, wires: tuple[int, ...]
) -> None:
    """Two-wire and meta instructions are not one-qubit synthesis."

    ``barrier``, ``measure``, and ``reset`` are outside ``OPERATOR_SCHEMAS``, so
    a synthesized gate could never stand in for them; the IR only accepts them
    as dynamic operations.
    """

    instruction = Instruction(name, wires, metadata={"is_dynamic": True})
    assert synthesize_one_qubit(instruction, z_rotation="rz") is None


def test_a_matrix_entry_point_reaches_the_same_group_as_the_opcode_entry_point() -> (
    None
):
    """Every declared arity-1 unitary is reachable from its matrix alone.

    The matrix path is what two-qubit KAK hands its local factors to, so it has
    to accept a plain nested sequence as well as a tensor and reach the same
    group the opcode path does.
    """

    reached: list[str] = []
    for name in _SINGLE_QUBIT_UNITARIES:
        if name == "rz":
            continue
        source = Instruction(name, (0,), params=_angles_for(name))
        matrix = _unitary(source)
        leaves = synthesize_one_qubit_matrix(matrix, wire=0, z_rotation="rz")
        assert leaves is not None, name
        reached.append(name)
        assert {leaf.name for leaf in leaves} <= {"rz", "sx"}
        assert all(leaf.wires == (0,) for leaf in leaves)
        product = _product(leaves) @ matrix.conj().mT
        torch.testing.assert_close(
            product,
            product[0, 0] * torch.eye(2, dtype=torch.complex128),
            rtol=0.0,
            atol=1e-12,
        )
        nested = synthesize_one_qubit_matrix(
            [[complex(item) for item in row] for row in matrix.tolist()],
            wire=0,
            z_rotation="rz",
        )
        assert nested is not None
        assert [(leaf.name, leaf.params) for leaf in nested] == [
            (leaf.name, leaf.params) for leaf in leaves
        ]

    # Non-vacuity: 18 declared arity-1 unitaries, one of which is `rz` itself.
    assert len(_SINGLE_QUBIT_UNITARIES) == 18
    assert len(reached) == 17


def test_the_matrix_entry_point_accepts_an_rx_pulse_and_another_wire() -> None:
    source = Instruction("h", (3,))
    leaves = synthesize_one_qubit_matrix(
        _unitary(source), wire=3, z_rotation="phase", pulse_opcode="rx"
    )
    assert leaves is not None
    assert {leaf.name for leaf in leaves} <= {"phase", "rx"}
    assert all(leaf.wires == (3,) for leaf in leaves)
    for leaf in leaves:
        if leaf.name == "rx":
            assert leaf.params["theta"] == math.pi / 2


def test_the_matrix_entry_point_refuses_what_it_cannot_express() -> None:
    """Fail closed, and say which argument was wrong."""

    identity = [[1.0, 0.0], [0.0, 1.0]]

    assert synthesize_one_qubit_matrix(identity, wire=0, z_rotation="h") is None
    assert (
        synthesize_one_qubit_matrix(identity, wire=0, z_rotation="rz", pulse_opcode="h")
        is None
    )
    # The identity and a phase-only diagonal both need no leaves at all.
    assert synthesize_one_qubit_matrix(identity, wire=0, z_rotation="rz") == ()
    assert (
        synthesize_one_qubit_matrix([[1j, 0.0], [0.0, 1j]], wire=0, z_rotation="rz")
        == ()
    )

    for bad in (
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[2.0, 0.0], [0.0, 1.0]],
        "not a matrix",
        7,
    ):
        with pytest.raises(ValueError):
            synthesize_one_qubit_matrix(bad, wire=0, z_rotation="rz")

    for wire in (-1, 1.5, True):
        with pytest.raises(ValueError):
            synthesize_one_qubit_matrix(identity, wire=wire, z_rotation="rz")
