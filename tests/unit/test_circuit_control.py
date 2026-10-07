"""Conformance tests for ``Circuit.control`` and the opcode control declaration.

A controlled program is defined by one equation: the receiver runs when every control
qubit is set and is skipped otherwise. The tests here measure that equation against the
receiver's own dense operator, embedded in the wider register, so a ladder that is
instruction-for-instruction different but operator-identical is accepted and a ladder that
changes the operator is not. The second group covers the emitted program's shape -- the
one-control shortcut, the added qubits, the carried input state, and the fact that the
receiver is not consumed. The third group covers the trainable path, which is the reason
the ladder is emitted rather than a matrix. The fourth group covers the refusals.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum.core import (
    CONTROL_PARTNERS,
    CONTROL_RULES,
    MAX_LADDER_LEVEL,
    OPERATOR_SCHEMAS,
    Parameter,
    control_ladder_level,
)
from flagquantum.core.ir import Instruction
from flagquantum.errors import CapabilityError

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]

#: One legal value per parameter name the registry declares, held strictly inside the
#: range where a basis change and a phase ladder both do visible work. A value of `0` or a
#: multiple of `pi/2` would let several wrong ladders agree with the right one.
_ANGLE = {"theta": 0.37, "phi": 0.61, "lbd": -0.23}

#: The registered opcodes that are unitary and therefore have a controlled form, keyed by
#: opcode with their arity and parameters. Derived from the registry rather than tabulated,
#: so a new unitary opcode is exercised here without an edit -- and the count is asserted
#: so that a registry change is visible rather than silently skipped.
_UNITARY_OPCODES: dict[str, tuple[int, dict[str, float]]] = {
    opcode: (schema.arity, {name: _ANGLE[name] for name in schema.parameters})
    for opcode, schema in OPERATOR_SCHEMAS.items()
    if schema.unitary and schema.opcode == opcode
}

#: The control counts every numeric test is measured at. `1` reaches the registered
#: partner shortcut and `2..4` reach the ladder, including the multi-qubit rungs.
_CONTROL_COUNTS = (1, 2, 3, 4)


def _receiver(opcode: str) -> fq.Circuit:
    arity, params = _UNITARY_OPCODES[opcode]
    return fq.Circuit(arity, dtype=torch.complex128).gate(
        opcode, tuple(range(arity)), params=params
    )


#: One, in the dtype every test in this file compiles at, so an amplitude comparison does
#: not compare a `complex128` state against a default-precision literal.
_ONE = torch.tensor(1.0 + 0j, dtype=torch.complex128)


def _basis_inputs(index: int, n_qubits: int) -> torch.Tensor:
    state = torch.zeros(1, 2**n_qubits, dtype=torch.complex128)
    state[0, index] = 1.0
    return state


def _program_operator(program: fq.Circuit) -> torch.Tensor:
    """Return a program's own dense operator, read back from its own simulation.

    Built by running the program on every computational basis vector, so the matrix is
    what the simulator executes rather than a second implementation of the same gates.
    Column `i` is the image of basis vector `i`.
    """

    n_qubits = program.n_qubits
    columns = []
    for index in range(2**n_qubits):
        carrier = fq.Circuit(
            n_qubits, dtype=torch.complex128, inputs=_basis_inputs(index, n_qubits)
        )
        carrier.compose(program)
        columns.append(carrier.state().detach()[0])
    return torch.stack(columns, dim=1)


def _definition_operator(receiver: fq.Circuit, n_controls: int) -> torch.Tensor:
    """Return ``C^n_controls(receiver)`` from its definition, as a dense matrix.

    The definition is read directly, on qubit labels: for every computational basis
    state, the receiver's own operator is applied when every qubit outside the receiver's
    range is set, and nothing happens otherwise. The control qubits keep their values. The
    receiver's operator is the one `_program_operator` reads back from its own simulation,
    so this is the definition applied to a measured gate rather than a second ladder.
    """

    base = _program_operator(receiver)
    width = receiver.n_qubits
    dimension = width + n_controls
    size = 2**dimension
    operator = torch.zeros((size, size), dtype=torch.complex128)
    for index in range(size):
        bits = [(index >> (dimension - 1 - qubit)) & 1 for qubit in range(dimension)]
        controls = bits[width:]
        if not all(controls):
            operator[index, index] = 1.0
            continue
        source = 0
        for bit in bits[:width]:
            source = (source << 1) | bit
        for image in range(2**width):
            amplitude = base[image, source]
            if amplitude == 0:
                continue
            image_bits = [(image >> (width - 1 - qubit)) & 1 for qubit in range(width)]
            out = 0
            for bit in image_bits + controls:
                out = (out << 1) | bit
            operator[out, index] += amplitude
    return operator


# --------------------------------------------------------------------------------------
# The definition
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("opcode", sorted(_UNITARY_OPCODES))
def test_a_controlled_gate_runs_only_when_every_control_is_set(opcode: str) -> None:
    """The definition, measured against the receiver's own operator.

    Run for every registered unitary opcode at control counts one through four, which is
    where the diagonal ladder, the basis change, the target's own arity and the phase
    correction all have to be right at once.
    """

    receiver = _receiver(opcode)
    arity = receiver.n_qubits
    for n_controls in _CONTROL_COUNTS:
        controls = tuple(range(arity, arity + n_controls))
        controlled = receiver.control(n_controls, ctrl_qubits=controls)
        assert torch.allclose(
            _program_operator(controlled),
            _definition_operator(receiver, n_controls),
            atol=1e-12,
        ), f"{opcode} at {n_controls} control qubit(s)"


def test_the_identity_is_controlled_to_the_empty_program() -> None:
    """`i` is the one unitary whose controlled form is no instruction at all."""

    controlled = fq.Circuit(1).i(0).control(3, ctrl_qubits=(1, 2, 3))
    assert controlled.to_ir().instructions == ()
    # The width still grows, because the controls are programmed even when nothing is
    # conditional on them. A silently narrower result would drop the caller's qubits.
    assert controlled.n_qubits == 4


def test_a_ladder_agrees_with_the_partner_it_does_not_use() -> None:
    """Two routes to the same three-control gate, built by different mechanisms.

    The registered partner route (`z -> cz -> ccx`-style partners) and the ladder route
    are separate code paths in the emitter, so measuring one against the other is a
    cross-check rather than a restatement.
    """

    direct = (
        fq.Circuit(1).x(0).control(1, ctrl_qubits=(1,)).control(1, ctrl_qubits=(2,))
    )
    laddered = fq.Circuit(1).x(0).control(2, ctrl_qubits=(1, 2))
    assert torch.allclose(
        _program_operator(direct), _program_operator(laddered), atol=1e-12
    )


# --------------------------------------------------------------------------------------
# The emitted program
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("opcode", "partner"),
    [("x", "cx"), ("cx", "ccx"), ("swap", "cswap"), ("rz", "crz")],
)
def test_one_control_on_a_gate_with_a_partner_is_one_instruction(
    opcode: str, partner: str
) -> None:
    """The shortcut is taken where the registry declares a partner, and nowhere else."""

    receiver = _receiver(opcode)
    arity = receiver.n_qubits
    controlled = receiver.control(1, ctrl_qubits=(arity,))
    emitted = [(item.name, item.wires) for item in controlled.to_ir().instructions]
    assert [name for name, _ in emitted] == [partner]
    # The control qubit is named first, which is the registry's own convention for a
    # controlled two-qubit opcode and the order the compiler and drawer expect.
    assert emitted[0][1][0] == arity
    assert CONTROL_PARTNERS[opcode] == partner


def test_the_control_qubits_are_added_and_start_at_zero() -> None:
    """The controls are new qubits, and they arrive in ``|0>``.

    The receiver's input state carries into the wider register and the added qubits are
    untouched, so a controlled program starts from the receiver's own input rather than
    from a prepared control. Qubit 0 is the most significant amplitude bit and the
    receiver keeps its numbers, so a one-qubit receiver under two controls labelled `7`
    and `9` occupies only index `0` and index `2**9`.
    """

    receiver = fq.Circuit(1, dtype=torch.complex128, inputs=torch.tensor([[0.6, 0.8]]))
    controlled = receiver.control(2, ctrl_qubits=(7, 9))
    # The labels are the caller's, not positions: a control may sit anywhere above the
    # receiver's own range.
    assert controlled.n_qubits == 10
    state = controlled.state().detach()
    assert state.shape == (1, 2**10)
    occupied = {int(index) for index in state[0].abs().nonzero().flatten()}
    assert occupied == {0, 2**9}
    assert torch.allclose(
        state[0, 0], torch.tensor(0.6 + 0j, dtype=torch.complex128), atol=1e-7
    )
    assert torch.allclose(
        state[0, 2**9], torch.tensor(0.8 + 0j, dtype=torch.complex128), atol=1e-7
    )


def test_the_receiver_keeps_its_own_program() -> None:
    """`control` returns a new program; the receiver is read, not consumed."""

    receiver = fq.Circuit(2).h(0).cnot(0, 1)
    before = receiver.to_ir().instructions
    controlled = receiver.control(2, ctrl_qubits=(2, 3))
    assert receiver.to_ir().instructions == before
    assert receiver.n_qubits == 2
    assert len(controlled.to_ir().instructions) > len(before)
    # A second expansion of the same receiver is identical, which a consumed receiver
    # could not be.
    assert (
        controlled.to_ir().instructions
        == receiver.control(2, ctrl_qubits=(2, 3)).to_ir().instructions
    )


def test_every_emitted_instruction_is_a_registered_opcode() -> None:
    """A controlled program stays a program the IR, compiler, and drawer understand."""

    for opcode in sorted(_UNITARY_OPCODES):
        arity = _UNITARY_OPCODES[opcode][0]
        controlled = _receiver(opcode).control(2, ctrl_qubits=(arity, arity + 1))
        for item in controlled.to_ir().instructions:
            assert item.name in OPERATOR_SCHEMAS, (opcode, item.name)
    # No emitted gate carries a matrix: the whole point of the declaration is that the
    # answer is expressible in registered opcodes rather than as a dense operator.
    for item in (
        fq.Circuit(1)
        .u3(0, 0.37, 0.61, -0.23)
        .control(1, ctrl_qubits=(1,))
        .to_ir()
        .instructions
    ):
        assert item.matrix is None


def test_a_ladder_never_touches_a_qubit_outside_the_result() -> None:
    """Qubit labels are unchanged: the receiver's own qubits keep their numbers."""

    receiver = fq.Circuit(2).h(0).cnot(0, 1)
    controlled = receiver.control(3, ctrl_qubits=(5, 6, 7))
    touched = {
        qubit for item in controlled.to_ir().instructions for qubit in item.wires
    }
    assert touched <= {0, 1, 5, 6, 7}
    assert touched & {5, 6, 7}


# --------------------------------------------------------------------------------------
# The trainable path
# --------------------------------------------------------------------------------------


def test_a_declared_angle_stays_declared_through_the_ladder() -> None:
    """A parameter that was symbolic before the expansion is symbolic after it.

    This is the reason the ladder is emitted instead of a matrix: a bound matrix would
    break the autograd graph and freeze the angle at whatever value it had.
    """

    parameter = Parameter("th")
    receiver = fq.Circuit(1).rz(0, theta=parameter)
    controlled = receiver.control(2, ctrl_qubits=(1, 2))
    assert any(
        isinstance(item.params["theta"], Parameter)
        or type(item.params["theta"]).__name__ == "ParameterExpression"
        for item in controlled.to_ir().instructions
        if item.params
    )
    # And binding the symbol afterwards produces exactly the program written with the
    # value: not merely an operator that agrees to a tolerance.
    bound = controlled.bind_parameters({parameter: 0.7})
    literal = fq.Circuit(1).rz(0, theta=0.7).control(2, ctrl_qubits=(1, 2))
    assert bound.to_ir().instructions == literal.to_ir().instructions


def test_gradient_flows_through_a_laddered_angle() -> None:
    """A laddered rotation is differentiable in its own parameter.

    The controls are prepared so the rotation actually runs: with the controls left at
    ``|0>`` the whole expansion is the identity and every gradient is zero, which would
    make the comparison below vacuous. With both controls set, the controlled rotation is
    the plain rotation on the target, so the reference is the rotation written without any
    control at all.
    """

    def build(angle: torch.Tensor, controlled: bool) -> fq.Circuit:
        register = fq.Circuit(3, dtype=torch.complex128)
        register.x(1).x(2)
        block = fq.Circuit(1, dtype=torch.complex128).rx(0, angle)
        if controlled:
            block = block.control(2, ctrl_qubits=(1, 2))
        register.compose(block)
        return register

    angle = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    build(angle, controlled=True).state().real.sum().backward()

    reference = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    build(reference, controlled=False).state().real.sum().backward()
    assert angle.grad is not None
    assert torch.allclose(angle.grad, reference.grad, atol=1e-12)
    # The gradient is not the trivial one, so the comparison above is not two zeros.
    assert abs(float(angle.grad)) > 1e-3


# --------------------------------------------------------------------------------------
# The declared rules and the census
# --------------------------------------------------------------------------------------


def test_the_declaration_covers_every_registered_opcode() -> None:
    """Every opcode answers one of the declared rules, so none falls through."""

    rules = {opcode: schema.control for opcode, schema in OPERATOR_SCHEMAS.items()}
    assert set(rules.values()) <= set(CONTROL_RULES)
    assert all(rules.values())
    assert len(OPERATOR_SCHEMAS) == 35
    # The opcodes with no controlled form are exactly the noise channels, which is a
    # statement about the registry rather than a list maintained here.
    assert {opcode for opcode, rule in rules.items() if rule == "not_available"} == {
        opcode for opcode, schema in OPERATOR_SCHEMAS.items() if schema.channel
    }
    assert len(_UNITARY_OPCODES) == 31


def test_the_ladder_level_is_arity_aware() -> None:
    """A wider target needs deeper rungs, and the depth is the declared formula."""

    for arity in (1, 2, 3):
        for n_controls in (1, 2, 3, 4):
            assert control_ladder_level(arity, n_controls) == n_controls + arity - 1
    # A three-qubit target at eight controls is the widest route the ceiling admits.
    assert control_ladder_level(3, 8) == MAX_LADDER_LEVEL
    assert control_ladder_level(3, 9) == MAX_LADDER_LEVEL + 1


def test_the_emitted_count_is_the_declared_bound() -> None:
    """The diagonal opcodes have no wrapper, so the recursion's cost is measurable.

    `T(level) = 4 * 3 ** (level - 1) - 3` is the published cost of the ancilla-free
    ladder. It is asserted on the opcodes whose whole controlled form *is* the primitive,
    because everywhere else a basis change or a correction term wraps it and the count
    would be measuring the wrapper.
    """

    for opcode in ("z", "s", "sdg", "t", "phase"):
        arity = _UNITARY_OPCODES[opcode][0]
        for n_controls in range(1, 7):
            level = control_ladder_level(arity, n_controls)
            controlled = _receiver(opcode).control(
                n_controls, ctrl_qubits=tuple(range(arity, arity + n_controls))
            )
            assert len(controlled.to_ir().instructions) == 4 * 3 ** (level - 1) - 3, (
                opcode,
                n_controls,
            )


# --------------------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("opcode", "parameter"),
    [
        ("bit_flip", "probability"),
        ("phase_flip", "probability"),
        ("depolarizing", "probability"),
        ("amplitude_damping", "gamma"),
    ],
)
def test_a_channel_has_no_controlled_form(opcode: str, parameter: str) -> None:
    """A channel is not a unitary, so conditioning it is not expressible here."""

    receiver = fq.Circuit(1).gate(opcode, (0,), params={parameter: 0.1})
    with pytest.raises(CapabilityError, match="noise channel"):
        receiver.control(1, ctrl_qubits=(1,))
    # The refusal names the receiver, not the ladder that was never built.
    assert [item.name for item in receiver.to_ir().instructions] == [opcode]


@pytest.mark.parametrize(
    ("metadata", "phrase"),
    [
        ({"is_dynamic": True}, "dynamic"),
        ({"conditions": (("c", 1),)}, "conditioned"),
    ],
)
def test_a_non_unitary_instruction_is_refused(
    metadata: dict[str, object], phrase: str
) -> None:
    """A dynamic or already-conditioned instruction is refused, not silently copied."""

    circuit = fq.Circuit(2)
    circuit._instructions.append(Instruction("x", (1,), metadata=metadata))
    with pytest.raises(CapabilityError, match=phrase):
        circuit.control(1, ctrl_qubits=(2,))


def test_an_instruction_carrying_a_matrix_is_refused() -> None:
    """A carried matrix is not a declaration, so it has no controlled form."""

    circuit = fq.Circuit(2)
    circuit._instructions.append(
        Instruction("custom", (1,), matrix=((1.0, 0.0), (0.0, 1.0)))
    )
    with pytest.raises(CapabilityError):
        circuit.control(1, ctrl_qubits=(2,))


def test_the_ladder_ceiling_is_enforced() -> None:
    """Beyond the ceiling the emitter refuses by name instead of running forever."""

    receiver = fq.Circuit(1).t(0)
    deepest = receiver.control(MAX_LADDER_LEVEL, ctrl_qubits=tuple(range(1, 11)))
    assert len(deepest.to_ir().instructions) == 4 * 3 ** (MAX_LADDER_LEVEL - 1) - 3
    with pytest.raises(ValueError, match="deepest supported ladder"):
        receiver.control(MAX_LADDER_LEVEL + 1, ctrl_qubits=tuple(range(1, 12)))
    # A wider target reaches the ceiling sooner, which is what "arity aware" means.
    with pytest.raises(ValueError, match="deepest supported ladder"):
        fq.Circuit(3).ccx(0, 1, 2).control(
            MAX_LADDER_LEVEL - 1, ctrl_qubits=tuple(range(3, 12))
        )


def test_a_control_qubit_inside_the_receiver_is_refused() -> None:
    """A control is added to the circuit, so it cannot be one of its own qubits."""

    with pytest.raises(ValueError, match="outside the receiver's own range"):
        fq.Circuit(2).h(0).control(1, ctrl_qubits=(0,))


def test_control_is_a_method_and_not_a_root_export() -> None:
    """The family stays a family of methods on `Circuit`, so no export is added."""

    assert callable(fq.Circuit.control)
    assert "control" not in fq.__all__
    # The recorded Stable Core is the manifest's business, not this test's, so the
    # claim is made against the manifest rather than against a transcription of its
    # cardinality. `density_matrix` joined the exports under PR #541, which landed
    # before #552 cut this file; the literal count this test used to carry was
    # already stale on the branch that wrote it.
    recorded = json.loads(
        (ROOT / "docs" / "public_api_v1.json").read_text(encoding="utf-8")
    )
    assert "control" not in recorded["stable_exports"], (
        "`control` is a method on `Circuit`; the recorded Stable Core must not "
        "list it as an export"
    )


# --------------------------------------------------------------------------------------
# A real workflow
# --------------------------------------------------------------------------------------


def test_a_multi_controlled_rotation_inside_a_wider_program() -> None:
    """The journey the capability exists for: a conditional block in a larger circuit.

    A phase kickback, which is the step phase estimation is built from: the counting
    register is put in superposition and the phase lands on the branch where every control
    is set. The block is placed by `compose`, so this exercises the composition family the
    way a user would write it, and the expected state is what the definition predicts
    rather than what the ladder emits.
    """

    target, controls, angle = 0, (1, 2), 0.37

    def run(preparation: tuple[int, ...]) -> torch.Tensor:
        register = fq.Circuit(3, dtype=torch.complex128)
        for qubit in preparation:
            register.x(qubit)
        register.compose(
            fq.Circuit(1).phase(target, theta=angle).control(2, ctrl_qubits=controls)
        )
        return register.state().detach()[0]

    # The kickback: `h` on both controls and the target in `|1>`, so three branches keep
    # the amplitude and the all-set branch carries the phase.
    kickback = fq.Circuit(3, dtype=torch.complex128).h(1).h(2).x(target)
    kickback.compose(
        fq.Circuit(1).phase(target, theta=angle).control(2, ctrl_qubits=controls)
    )
    expected = torch.zeros(8, dtype=torch.complex128)
    for index in (0b100, 0b101, 0b110):
        expected[index] = 0.5
    expected[0b111] = 0.5 * complex(math.cos(angle), math.sin(angle))
    assert torch.allclose(kickback.state().detach()[0], expected, atol=1e-12)

    # Nothing at all, so nothing happens.
    assert torch.allclose(run(())[0], _ONE, atol=1e-12)
    # The controls are not all set, so the rotation is skipped even though the target is
    # in `|1>` where the phase gate would have done something.
    assert torch.allclose(run((target,))[0b100], _ONE, atol=1e-12)
    # `phase` leaves `|0>` alone, so a set control over an unset target changes nothing.
    idle_target = run((1, 2))
    assert torch.allclose(idle_target[0b011], _ONE, atol=1e-12)
    assert idle_target.imag.abs().max() < 1e-12
