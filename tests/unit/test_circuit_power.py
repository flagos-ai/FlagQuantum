"""Conformance tests for ``Circuit.power`` and the opcode power declaration.

A power is one of the two ways to build a bigger program out of a smaller one that this
repository offers, and it is defined by one equation: applying a program ``k`` times. The
tests here measure that equation against the program's dense operator, so a rewrite that
is instruction-for-instruction different but operator-identical is accepted and a rewrite
that changes the operator is not. The second group covers the single-gate rewrite and its
counterexamples -- the two- and three-angle gates whose angles must *not* be scaled. The
third group covers the refusals, including the one that only a negative exponent reaches.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum.core import OPERATOR_SCHEMAS, POWER_RULES, Parameter
from flagquantum.core.ir import Instruction
from flagquantum.core.operator_schema import MAX_POWER_REPEATS
from flagquantum.errors import CapabilityError

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]

#: Every registered opcode with one set of legal parameters, grouped by the rule the
#: declaration answers for it. The keys are the opcodes; the values are the arity and the
#: parameters, so a single table drives the numeric tests.
_UNITARY_OPCODES: dict[str, tuple[int, dict[str, object]]] = {
    "i": (1, {}),
    "x": (1, {}),
    "y": (1, {}),
    "z": (1, {}),
    "h": (1, {}),
    "s": (1, {}),
    "sdg": (1, {}),
    "t": (1, {}),
    "tdg": (1, {}),
    "sx": (1, {}),
    "sxdg": (1, {}),
    "rx": (1, {"theta": 0.37}),
    "ry": (1, {"theta": 0.37}),
    "rz": (1, {"theta": 0.37}),
    "phase": (1, {"theta": 0.37}),
    "u1": (1, {"theta": 0.37}),
    "u2": (1, {"phi": 0.37, "lbd": 1.1}),
    "u3": (1, {"theta": 0.37, "phi": 0.4, "lbd": 1.1}),
    "cx": (2, {}),
    "cy": (2, {}),
    "cz": (2, {}),
    "swap": (2, {}),
    "crx": (2, {"theta": 0.37}),
    "cry": (2, {"theta": 0.37}),
    "crz": (2, {"theta": 0.37}),
    "cphase": (2, {"theta": 0.37}),
    "rxx": (2, {"theta": 0.37}),
    "ryy": (2, {"theta": 0.37}),
    "rzz": (2, {"theta": 0.37}),
    "ccx": (3, {}),
    "cswap": (3, {}),
}

#: The exponents the repetition identity is measured at. `1` is included because a power
#: of one is the case a rewrite is most likely to break, and a negative exponent is
#: covered separately because it is routed through `adjoint`.
_EXPONENTS = (0, 1, 2, 3, 5)
_NEGATIVE_EXPONENTS = (-1, -2, -3)


def dense_operator(program: fq.Circuit) -> torch.Tensor:
    """Return the program's operator as a ``2**n_qubits`` square matrix.

    The matrix is built by running the program on every computational basis vector, so it
    is the operator the simulator executes. Comparing operators rather than instruction
    lists is what lets a rewrite be judged on whether it preserves the program.
    """

    assert program.bsz == 1, "a unitary operator is defined for one batch element"
    n_qubits = program.n_qubits
    if n_qubits == 0:
        return torch.ones((1, 1), dtype=torch.complex128)
    dimension = 2**n_qubits
    columns = []
    for basis in range(dimension):
        carrier = fq.Circuit(n_qubits, dtype=torch.complex128)
        carrier.compose(program)
        state = torch.zeros(1, dimension, dtype=torch.complex128)
        state[0, basis] = 1.0
        carrier._inputs = state
        columns.append(carrier.state().detach()[0])
    return torch.stack(columns, dim=1)


def state_from_basis(program: fq.Circuit) -> torch.Tensor:
    """Return the program applied to ``|0...0>``, one row per batch element."""

    n_qubits = program.n_qubits
    carrier = fq.Circuit(n_qubits, bsz=program.bsz, dtype=torch.complex128)
    carrier.compose(program)
    return carrier.state().detach()


def _sequence_operator(one: torch.Tensor, exponent: int) -> torch.Tensor:
    """``one`` applied ``exponent`` times, or its inverse applied ``-exponent`` times.

    Built by repeated multiplication in the order the program runs, so it is the
    left-to-right function composition a user gets from writing the gates out. It is
    deliberately not `matrix_power`, which would make this reference share the
    implementation under test.
    """

    step = one if exponent >= 0 else torch.linalg.inv(one)
    result = torch.eye(one.shape[0], dtype=torch.complex128)
    for _ in range(abs(exponent)):
        result = result @ step
    return result


def _one_instruction(opcode: str) -> fq.Circuit:
    arity, params = _UNITARY_OPCODES[opcode]
    circuit = fq.Circuit(arity, dtype=torch.complex128)
    return circuit.gate(opcode, tuple(range(arity)), params=params)


@pytest.mark.parametrize("opcode", sorted(_UNITARY_OPCODES))
@pytest.mark.parametrize("exponent", _EXPONENTS)
def test_a_power_matches_the_repeated_program(opcode: str, exponent: int) -> None:
    """The definition: ``p.power(k)`` executes what ``p`` executes ``k`` times."""

    program = _one_instruction(opcode)
    expected = _sequence_operator(dense_operator(program), exponent)

    assert torch.allclose(dense_operator(program.power(exponent)), expected, atol=1e-12)


@pytest.mark.parametrize("opcode", sorted(_UNITARY_OPCODES))
@pytest.mark.parametrize("exponent", _NEGATIVE_EXPONENTS)
def test_a_negative_power_matches_the_inverse_applied(
    opcode: str, exponent: int
) -> None:
    """A negative exponent is the inverse applied, which is the matrix power."""

    program = _one_instruction(opcode)
    expected = _sequence_operator(dense_operator(program), exponent)

    assert torch.allclose(dense_operator(program.power(exponent)), expected, atol=1e-12)


@pytest.mark.parametrize("exponent", [*_EXPONENTS, *_NEGATIVE_EXPONENTS])
def test_a_power_equals_the_matrix_power_of_the_program(exponent: int) -> None:
    """The same claim for a multi-instruction program, against `matrix_power`.

    A one-gate receiver can be rewritten into a single scaled instruction, so it would not
    detect a rewrite that got the order of a longer program wrong. This program cannot be
    collapsed, so the order is what is measured.
    """

    block = fq.Circuit(2, dtype=torch.complex128).h(0).rz(1, 0.7).cnot(0, 1)
    reference = dense_operator(block)
    step = reference if exponent >= 0 else torch.linalg.inv(reference)
    expected = torch.linalg.matrix_power(step, abs(exponent))

    got = dense_operator(block.power(exponent))
    assert (got - expected).abs().max().item() < 1e-12


def test_a_power_is_the_program_repeated_in_order() -> None:
    """The order is the program's own, not the reverse and not interleaved."""

    block = fq.Circuit(2).rx(0, 0.3).cnot(0, 1)
    powered = block.power(2)

    assert [(item.name, item.wires) for item in powered.to_ir().instructions] == [
        ("rx", (0,)),
        ("cx", (0, 1)),
        ("rx", (0,)),
        ("cx", (0, 1)),
    ]
    # A power is not the reverse of a power of the adjoint, which is the mistake a
    # "multiply the matrices together" implementation would make for a non-commuting
    # program. Measured, because the two agree for every single-gate receiver.
    assert not torch.allclose(
        dense_operator(powered), dense_operator(block.adjoint().power(2)), atol=1e-6
    )


def test_the_power_law_holds_for_a_non_commuting_program() -> None:
    """``p.power(a).power(b)`` is ``p.power(a * b)``, including across the sign."""

    block = fq.Circuit(2).h(0).rz(1, 0.7).cnot(0, 1)
    for first, second in ((2, 3), (3, 2), (2, -1), (-1, 2), (1, 4)):
        assert torch.allclose(
            dense_operator(block.power(first).power(second)),
            dense_operator(block.power(first * second)),
            atol=1e-12,
        ), (first, second)


def test_power_zero_is_the_empty_program_of_the_same_shape() -> None:
    shaped = fq.Circuit(3, bsz=2, device="cpu", dtype=torch.complex64).h(1).cnot(1, 2)
    raised = shaped.power(0)

    assert raised.to_ir().instructions == ()
    assert raised.n_qubits == shaped.n_qubits
    assert raised.bsz == shaped.bsz
    assert raised.dtype == shaped.dtype
    assert raised.device == shaped.device
    # The identity, so the state is unchanged, which is the only reading of `power(0)`
    # that makes the power law hold at `a = 0`.
    initial = fq.Circuit(3, bsz=2, device="cpu", dtype=torch.complex64)
    assert torch.allclose(raised.state().detach(), initial.state().detach())
    # And it is the empty program rather than the program it came from, which the state
    # alone would not separate: the two differ once the source is not the identity.
    assert not torch.allclose(raised.state().detach(), shaped.state().detach())


def test_power_of_an_empty_program_is_empty_at_every_exponent() -> None:
    empty = fq.Circuit(2, dtype=torch.complex128)

    for exponent in (0, 1, 4, -3):
        raised = empty.power(exponent)
        assert raised.to_ir().instructions == (), exponent
        assert raised.n_qubits == 2, exponent


def test_power_one_is_the_same_program() -> None:
    """A power of one copies, because scaling every angle by one changes expressions."""

    symbolic = fq.Circuit(1).rx(0, Parameter("t"))
    raised = symbolic.power(1).to_ir().instructions[0]
    assert raised.name == "rx"
    # Not `t * 1`: the identity power must not rewrite the parameter it was given.
    assert raised.params == {"theta": Parameter("t")}

    angle = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    trainable = fq.Circuit(1, dtype=torch.complex128).rx(0, angle)
    copied = trainable.power(1).to_ir().instructions[0].params["theta"]
    # The same tensor object, not a new one behind a multiplication, so a caller who holds
    # the angle keeps holding the angle the program uses.
    assert copied is angle

    block = fq.Circuit(2).rx(0, 0.3).cnot(0, 1)
    assert [
        (item.name, dict(item.params)) for item in block.power(1).to_ir().instructions
    ] == [
        ("rx", {"theta": 0.3}),
        ("cx", {}),
    ]


def test_a_power_does_not_mutate_its_receiver() -> None:
    block = fq.Circuit(2).rx(0, 0.3).cnot(0, 1)
    before = [(item.name, dict(item.params)) for item in block.to_ir().instructions]

    raised = block.power(4)

    assert [(item.name, dict(item.params)) for item in block.to_ir().instructions] == (
        before
    )
    assert raised is not block
    assert len(raised.to_ir().instructions) == 8


def test_the_instruction_bound_is_measured_against_the_whole_program() -> None:
    """The bound counts what is emitted, so a long program reaches it sooner."""

    one = fq.Circuit(1).h(0)
    assert len(one.power(MAX_POWER_REPEATS).to_ir().instructions) == MAX_POWER_REPEATS
    with pytest.raises(ValueError, match=str(MAX_POWER_REPEATS)):
        one.power(MAX_POWER_REPEATS + 1)

    three = fq.Circuit(2).h(0).cnot(0, 1).rz(0, 0.3)
    allowed = MAX_POWER_REPEATS // 3
    assert len(three.power(allowed).to_ir().instructions) == allowed * 3
    with pytest.raises(ValueError, match=str(allowed * 3 + 1)):
        three.power(allowed + 1)


def test_a_power_emits_distinct_instruction_records() -> None:
    """The copies are independent records, so editing one does not reach the others."""

    block = fq.Circuit(1).h(0)
    raised = block.power(3)
    instructions = raised.to_ir().instructions

    assert len(instructions) == 3
    assert len({id(item) for item in instructions}) == 3
    assert all(item is not instructions[0] for item in instructions[1:])


# --------------------------------------------------------------------------------------
# The single-gate rewrite, and the gates it must not touch
# --------------------------------------------------------------------------------------


def test_the_closed_form_rewrite_is_exact_for_every_opcode_it_names() -> None:
    """`U(theta)^k == U(k * theta)`, measured, for the opcodes the rule names.

    This is the claim the declaration makes. It is a claim about the gates, so it is
    measured through the operators rather than read off the field that predicts it.
    """

    closed = sorted(
        opcode
        for opcode, schema in OPERATOR_SCHEMAS.items()
        if schema.power_rule == "scale_single_parameter"
    )
    assert closed, "the rewrite would be dead code"
    for opcode in closed:
        schema = OPERATOR_SCHEMAS[opcode]
        assert len(schema.parameters) == 1, opcode
        name = schema.parameters[0]
        for angle in (0.3, 1.1, -0.7):
            for exponent in (2, 3, 5):
                single = _dense_of(opcode, {name: angle})
                repeated = single
                for _ in range(exponent - 1):
                    repeated = repeated @ single
                assert torch.allclose(
                    repeated, _dense_of(opcode, {name: angle * exponent}), atol=1e-12
                ), (opcode, angle, exponent)


def _program(opcode: str, params: dict[str, object]) -> fq.Circuit:
    schema = OPERATOR_SCHEMAS[opcode]
    return fq.Circuit(schema.arity, dtype=torch.complex128).gate(
        opcode, tuple(range(schema.arity)), params=params
    )


def _dense_of(opcode: str, params: dict[str, object]) -> torch.Tensor:
    return dense_operator(_program(opcode, params))


def test_the_rewrite_collapses_a_single_instruction() -> None:
    """The rewrite is visible: one instruction, with the angle multiplied."""

    for opcode in ("rx", "ry", "rz", "phase", "u1"):
        schema = OPERATOR_SCHEMAS[opcode]
        raised = _one_instruction(opcode).power(4)
        instructions = raised.to_ir().instructions
        assert len(instructions) == 1, opcode
        assert instructions[0].name == opcode
        assert instructions[0].params[schema.parameters[0]] == pytest.approx(0.37 * 4)


def test_the_rewrite_does_not_touch_a_multi_angle_gate() -> None:
    """Scaling `u2` or `u3`'s angles is a different operator, so it is not done.

    This is the counterexample that makes the rule per-opcode rather than global. The
    residual is quoted exactly in
    `docs/development/API_CHANGE_PROPOSAL_066_CIRCUIT_COMPOSITION.md`, so it is asserted
    here as the same number rather than as an inequality that any wrong value satisfies.
    """

    for opcode, params, residual in (
        ("u3", {"theta": 0.4, "phi": 0.9, "lbd": 1.3}, 2.191289381e-01),
        ("u2", {"phi": 0.9, "lbd": 1.3}, 4.135343356e-01),
    ):
        schema = OPERATOR_SCHEMAS[opcode]
        assert schema.power_rule == "repeat_instruction"
        once = _dense_of(opcode, params)
        squared = once @ once
        scaled = _dense_of(opcode, {key: 2.0 * value for key, value in params.items()})
        assert float((squared - scaled).abs().max()) == pytest.approx(
            residual, abs=1e-9
        ), opcode
        # The repetition is right, so the two are not merely different: squaring is what
        # the emitted program does, and it is what the operator confirms.
        assert torch.allclose(
            squared, dense_operator(_program(opcode, params).power(2))
        )


def test_the_rewrite_does_not_touch_a_parameter_free_gate() -> None:
    assert OPERATOR_SCHEMAS["h"].power_rule == "repeat_instruction"
    assert [item.name for item in fq.Circuit(1).h(0).power(3).to_ir().instructions] == [
        "h",
        "h",
        "h",
    ]
    # Two hadamards are the identity, which the operator says and the instruction list
    # does not: the rewrite is an optimization, not a semantic claim about the count.
    assert torch.allclose(
        dense_operator(fq.Circuit(1).h(0).power(2)),
        torch.eye(2, dtype=torch.complex128),
        atol=1e-12,
    )


def test_the_rewrite_is_not_applied_to_a_longer_program() -> None:
    """In a longer program the copies of one gate are separated and do not compose."""

    block = fq.Circuit(2).rx(0, 0.3).cnot(0, 1)
    raised = block.power(2)
    assert [item.name for item in raised.to_ir().instructions] == [
        "rx",
        "cx",
        "rx",
        "cx",
    ]
    # The wrong form -- collapsing the two `rx` gates into `rx(0.6)` first -- is a
    # different operator, so it is excluded numerically as well as by name.
    wrong = fq.Circuit(2).rx(0, 0.6).cnot(0, 1).cnot(0, 1)
    assert not torch.allclose(dense_operator(raised), dense_operator(wrong), atol=1e-6)


def test_the_rewrite_does_not_touch_an_operation_carrying_a_matrix() -> None:
    """The matrix is what executes, so it suppresses the rewrite."""

    matrix = torch.tensor([[0.0, 0.6], [0.6, 0.0]], dtype=torch.complex128)
    custom = fq.Circuit(1).any(0, unitary=matrix)

    raised = custom.power(2)
    instructions = raised.to_ir().instructions
    assert [item.name for item in instructions] == ["any", "any"]
    # The opcode `any` carries no parameter to scale, so the rewrite has nothing to touch
    # even if the matrix route did not come first.
    assert all(item.params == {} for item in instructions)
    # Repeated through the matrix, the result is the matrix product, not a scaled matrix.
    expected = matrix @ matrix
    assert torch.allclose(dense_operator(raised), expected, atol=1e-12)
    assert not torch.allclose(dense_operator(raised), matrix, atol=1e-6)


def test_the_rewrite_keeps_the_declared_angle_forms_working() -> None:
    """Per-batch angles and symbolic parameters both survive the multiplication."""

    batched = fq.Circuit(1, bsz=3, dtype=torch.complex128).rx(0, [0.1, 0.2, 0.3])
    raised = batched.power(2)
    assert raised.to_ir().instructions[0].params["theta"] == pytest.approx(
        [0.2, 0.4, 0.6]
    )
    # A batch is not a unitary, so the two programs are compared the way a user compares
    # them: the same state for the same input, on an initial state that is not symmetric.
    reference = fq.Circuit(1, bsz=3, dtype=torch.complex128).rx(0, [0.2, 0.4, 0.6])
    assert torch.allclose(state_from_basis(raised), state_from_basis(reference))

    symbolic = fq.Circuit(1).rx(0, Parameter("t")).power(3)
    parameter = symbolic.to_ir().instructions[0].params["theta"]
    # The symbolic form is multiplied rather than evaluated, so the parameter stays a
    # parameter of the program and can still be bound to a value.
    assert getattr(parameter, "op", None) == "mul"
    assert parameter.args == (Parameter("t"), 3)
    assert symbolic.parameter_names == ("t",)
    assert symbolic.bind_parameters({"t": 0.2}).to_ir().instructions[0].params[
        "theta"
    ] == pytest.approx(0.6)


def test_gradient_flows_through_a_scaled_angle() -> None:
    """The rewrite keeps the angle a differentiable input, which is the point of it."""

    angle = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    program = fq.Circuit(1, dtype=torch.complex128).rx(0, angle).power(2)
    # A trainable program is run through `fq.run` in practice; the state is read here so
    # the test depends on no optimizer or training loop.
    program.state().real.sum().backward()

    assert angle.grad is not None
    # `d/dtheta cos(2 * theta) * 2`-scale quantity: the two applications are visible in
    # the derivative, which a wrongly-scaled angle would change.
    reference = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    fq.Circuit(1, dtype=torch.complex128).rx(
        0, 2.0 * reference
    ).state().real.sum().backward()
    assert torch.allclose(angle.grad, reference.grad, atol=1e-12)


# --------------------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("exponent", [2.5, 0.5, -0.5, "2", None, True, False, [2]])
def test_a_non_integer_exponent_is_refused(exponent: object) -> None:
    """A fractional power is a matrix power, which the IR cannot express.

    `bool` is refused with the rest rather than read as `0`/`1`: `power(True)` written
    deliberately would be `power(1)`, but `power(flag)` where `flag` came from a
    comparison is a bug that would otherwise silently succeed.
    """

    block = fq.Circuit(1).h(0)
    with pytest.raises(TypeError, match="must be an integer, got"):
        block.power(exponent)
    # The refusal leaves the receiver untouched.
    assert [item.name for item in block.to_ir().instructions] == ["h"]


def test_integer_like_types_are_accepted() -> None:
    """A count that is an integer is accepted, without a lossy conversion.

    ``numpy`` is not a declared dependency of this repository and no lane installs it,
    so the integer that is not an ``int`` is a local ``__index__`` carrier here. A
    numpy integer takes the same path, but importing numpy to say so would fail
    collection in every lane instead of testing anything. The rule being measured is
    the interpreter's ``__index__`` protocol, not ``isinstance(value, int)`` -- an
    ``int`` subclass would pass that and prove nothing.
    """

    class IntegerLike:
        """An integer that is not an ``int``, which is the shape numpy supplies."""

        def __index__(self) -> int:
            return 3

    block = fq.Circuit(1).h(0)
    assert len(block.power(2).to_ir().instructions) == 2
    assert len(block.power(IntegerLike()).to_ir().instructions) == 3
    assert len(block.power(torch.tensor(2)).to_ir().instructions) == 2
    # A float that is exactly integral is still refused rather than truncated: the
    # argument is a count, and `2.0` in a variable usually came from arithmetic.
    with pytest.raises(TypeError, match="must be an integer, got"):
        block.power(2.0)


def test_a_negative_power_of_a_channel_is_refused() -> None:
    """A channel has no unitary inverse, and the refusal is `adjoint`'s own."""

    channel = fq.Circuit(1).depolarizing(0, 0.1)
    with pytest.raises(CapabilityError, match="noise channel"):
        channel.power(-1)
    assert [item.name for item in channel.to_ir().instructions] == ["depolarizing"]


def test_a_positive_power_of_a_channel_is_a_program() -> None:
    """Two noise events in order are expressible, so they are not refused."""

    channel = fq.Circuit(1, dtype=torch.complex128).depolarizing(0, 0.1)
    raised = channel.power(2)
    assert [item.name for item in raised.to_ir().instructions] == [
        "depolarizing",
        "depolarizing",
    ]
    assert OPERATOR_SCHEMAS["depolarizing"].power_rule == "repeat_instruction"


def test_a_negative_power_of_a_non_invertible_instruction_is_refused() -> None:
    """A dynamic operation has no inverse, so it cannot appear under a negative power."""

    circuit = fq.Circuit(2)
    circuit._instructions.append(
        Instruction("reset", (1,), metadata={"is_dynamic": True})
    )
    with pytest.raises(CapabilityError):
        circuit.power(-2)
    assert [item.name for item in circuit.to_ir().instructions] == ["reset"]


def test_power_rules_cover_every_registered_opcode() -> None:
    """Every opcode answers one of the declared rules, so none falls through."""

    rules = {opcode: schema.power_rule for opcode, schema in OPERATOR_SCHEMAS.items()}
    assert set(rules.values()) <= set(POWER_RULES)
    assert all(rules.values())
    # `scale_single_parameter` is exactly "a unitary declaring one parameter", which is
    # the derivation rather than a second table.
    assert {
        opcode for opcode, rule in rules.items() if rule == "scale_single_parameter"
    } == {
        opcode
        for opcode, schema in OPERATOR_SCHEMAS.items()
        if schema.unitary and len(schema.parameters) == 1
    }


def test_power_is_a_method_and_not_a_root_export() -> None:
    """The family stays a family of methods on `Circuit`, so no export is added."""

    assert callable(fq.Circuit.power)
    assert "power" not in fq.__all__
    # The recorded Stable Core is the manifest's business, not this test's, so the
    # claim is made against the manifest rather than against a transcription of its
    # cardinality. The literal count this test used to carry (`37`) was correct when
    # this slice was measured and stale once `fq.jacobian`, `fq.jvp`, and `fq.vjp`
    # landed: a cardinality this file does not own had to be re-typed by a branch that
    # added no export here. This is the same repair `test_circuit_control.py` took, for
    # the same reason, when `fq.density_matrix` moved the count to 37.
    recorded = json.loads(
        (ROOT / "docs" / "public_api_v1.json").read_text(encoding="utf-8")
    )
    assert "power" not in recorded["stable_exports"], (
        "`power` is a method on `Circuit`; the recorded Stable Core must not list it "
        "as an export"
    )
