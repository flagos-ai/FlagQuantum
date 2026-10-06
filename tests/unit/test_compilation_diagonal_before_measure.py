"""Check the diagonal-before-measure pass against the gate matrices and the runtime.

`diagonal_before_measure.remove_diagonal_gates_before_measure` removes a diagonal
gate when the next instruction on every one of its wires is an unconditional
measurement. Two different kinds of claim hold that rule up, and this file tests
them separately because they fail differently.

The **single-qubit** half rests on a theorem, not a list: `U3(theta, phi, lam)` is
diagonal exactly when `sin(theta / 2)` vanishes, and `one_qubit_synthesis` is
already the module that tabulates that polar angle for all eighteen declared
single-qubit unitaries. So the test here does not assert an opcode set; it *measures*
each of the eighteen out of the runtime gate matrices at several angles and asserts
the pass's verdict equals the measured matrix shape. That is what makes
`u3(0, phi, lam)`, `rx(0)` and `ry(0)` removable by this pass and not by
`remove_identity_gates`.

The **two-wire** half rests on a declared set, because no module states a two-wire
diagonality and `two_qubit_synthesis` works in Weyl coordinates that do not
distinguish a diagonal operator from a locally equivalent one. The test therefore
measures the whole declared two-wire unitary set the same way and asserts the
declared set *equals* the measured one, in both directions, so a new Core opcode
fails here rather than being silently declined by the pass.

The third question is whether a removal is a no-op at run time. Every diagonal gate
is a phase on each basis state and moves no amplitude, so the outcome bits must be
identical *shot for shot* -- not merely equal in distribution -- and the tests below
assert that over a fixed seed. A control then shows the instrument can see the
difference it is looking for: deleting a non-diagonal gate from a Bell pair, or a
`ry` from a bare register, moves thousands of shots where deleting a diagonal gate
moves none.
"""

from __future__ import annotations

import ast
import random
from pathlib import Path

import pytest
import torch

import flagquantum.compiler.pipeline as pipeline_module
from flagquantum.compiler import optimize
from flagquantum.compiler.diagonal_before_measure import (
    _DIAGONAL_TWO_WIRE,
    remove_diagonal_gates_before_measure,
)
from flagquantum.compiler.one_qubit_synthesis import is_diagonal_one_qubit
from flagquantum.compiler.optimization_levels import (
    DEFAULT_OPTIMIZATION_LEVEL,
    OPTIMIZATION_LEVEL_STAGES,
)
from flagquantum.core.ir import CircuitIR, Instruction, IRValidationError
from flagquantum.core.operator_schema import (
    OPERATOR_SCHEMAS,
    canonical_opcode,
    get_operator_schema,
)
from flagquantum.runtime.dynamic import DynamicCircuit, run_dynamic
from flagquantum.simulation.gate_matrix import gate_matrix

pytestmark = pytest.mark.unit

#: A matrix is diagonal when its every off-diagonal entry is this small. The runtime
#: gate matrices are complex128, so this is rounding rather than a convention.
_ATOL = 1.0e-12

#: Angle triples the classification is measured at. The first is a generic point; the
#: second drives a polar angle of zero, which is the one value that makes a generally
#: non-diagonal opcode diagonal.
_GENERIC = {"theta": 0.7137, "phi": -0.4211, "lbd": 1.9073}
_ZERO_POLAR = {"theta": 0.0, "phi": -0.4211, "lbd": 1.9073}

_SHOTS = 4000
_SEED = 20261117

_SINGLE_QUBIT_UNITARIES = tuple(
    sorted(
        name
        for name, schema in OPERATOR_SCHEMAS.items()
        if schema.unitary and schema.arity == 1
    )
)
_TWO_WIRE_UNITARIES = tuple(
    sorted(
        name
        for name, schema in OPERATOR_SCHEMAS.items()
        if schema.unitary and schema.arity == 2
    )
)


def _angles_for(name: str, point: dict[str, float]) -> dict[str, float]:
    schema = get_operator_schema(name)
    assert schema is not None
    return {key: point[key] for key in schema.parameters}


def _matrix(
    name: str, wires: tuple[int, ...], params: dict[str, float]
) -> torch.Tensor:
    width = 2 ** len(wires)
    return gate_matrix(
        Instruction(name, wires, params=params),
        bsz=1,
        device="cpu",
        dtype=torch.complex128,
    ).reshape(width, width)


def _measured_diagonal(
    name: str, wires: tuple[int, ...], params: dict[str, float]
) -> bool:
    matrix = _matrix(name, wires, params)
    return float((matrix - torch.diag(torch.diagonal(matrix))).abs().max()) <= _ATOL


def _names(ir: CircuitIR) -> list[str]:
    return [instruction.name for instruction in ir]


def _measure(wire: int, bit: int = 0, **metadata: object) -> Instruction:
    return Instruction(
        "measure",
        (wire,),
        metadata={"is_dynamic": True, "classical_bit": bit, **metadata},
    )


def _gate(name: str, wires: tuple[int, ...], **params: float) -> Instruction:
    return Instruction(name, wires, params=params)


def _reset(wire: int, **metadata: object) -> Instruction:
    return Instruction("reset", (wire,), metadata={"is_dynamic": True, **metadata})


def _outcomes(
    ir: CircuitIR, shots: int = _SHOTS, seed: int = _SEED
) -> list[tuple[int, ...]]:
    """The classical bits of every shot, so two programs compare shot for shot."""

    result = run_dynamic(DynamicCircuit.from_ir(ir), shots=shots, seed=seed)
    return [tuple(row) for row in result.classical_bits.tolist()]


def _differing_shots(left: list[tuple[int, ...]], right: list[tuple[int, ...]]) -> int:
    return sum(1 for a, b in zip(left, right, strict=True) if a != b)


def test_the_single_qubit_verdict_equals_the_matrix_shape_at_a_generic_point() -> None:
    """The theorem, not a list: every declared single-qubit unitary is measured."""

    measured = {
        name: _measured_diagonal(name, (0,), _angles_for(name, _GENERIC))
        for name in _SINGLE_QUBIT_UNITARIES
    }

    assert len(measured) == 18
    assert sum(1 for value in measured.values() if value) == 9
    for name, value in measured.items():
        verdict = is_diagonal_one_qubit(
            Instruction(name, (0,), params=_angles_for(name, _GENERIC))
        )
        assert verdict is value, f"{name}: measured {value}, reported {verdict}"


def test_a_zero_polar_angle_makes_the_general_forms_diagonal_too() -> None:
    """This is reach `remove_identity_gates` does not have, and the reason is exact.

    `u3`, `rx` and `ry` at a zero polar angle are diagonal and are not the identity,
    so the identity pass leaves them and this one takes them. `u2` at any angle is
    not diagonal, because its polar angle is the literal pi/2.
    """

    removable = {"u3": _ZERO_POLAR, "rx": {"theta": 0.0}, "ry": {"theta": 0.0}}
    for name, params in removable.items():
        assert _measured_diagonal(name, (0,), params), name
        assert is_diagonal_one_qubit(Instruction(name, (0,), params=params)), name

    assert not is_diagonal_one_qubit(
        Instruction("u2", (0,), params={"phi": 0.3, "lbd": 0.9})
    )
    assert not _measured_diagonal("u2", (0,), _ZERO_POLAR)


def test_a_trainable_polar_angle_is_never_reported_diagonal() -> None:
    """A rotation initialized at zero must not be dropped out of the autograd graph.

    This refusal applies to the opcodes whose polar angle *is* their parameter. For
    `rz`, `phase` and `u1` the tabulated polar angle is the literal zero, because
    those opcodes are diagonal at every value; reporting a trainable one diagonal
    detaches nothing a measurement could have read.
    """

    angle = torch.zeros((), dtype=torch.float64, requires_grad=True)
    assert not is_diagonal_one_qubit(Instruction("rx", (0,), params={"theta": angle}))
    assert not is_diagonal_one_qubit(Instruction("ry", (0,), params={"theta": angle}))
    assert not is_diagonal_one_qubit(
        Instruction("u3", (0,), params={"theta": angle, "phi": 0.3, "lbd": 0.9})
    )
    assert is_diagonal_one_qubit(Instruction("rx", (0,), params={"theta": 0.0}))
    assert is_diagonal_one_qubit(Instruction("rz", (0,), params={"theta": angle}))


def test_the_declared_two_wire_set_equals_the_measured_one() -> None:
    """A declared set is the honest answer here, so it is checked in both directions."""

    measured = {
        name
        for name in _TWO_WIRE_UNITARIES
        if _measured_diagonal(name, (0, 1), _angles_for(name, _GENERIC))
    }

    assert len(_TWO_WIRE_UNITARIES) == 11
    assert measured == {"cz", "cphase", "crz", "rzz"}
    assert measured == set(_DIAGONAL_TWO_WIRE)


def test_an_opcode_the_euler_table_is_silent_about_is_declined() -> None:
    """A channel, a three-wire gate, and a measurement."""

    assert not is_diagonal_one_qubit(
        Instruction("depolarizing", (0,), params={"probability": 0.1})
    )
    assert not is_diagonal_one_qubit(Instruction("ccx", (0, 1, 2)))
    # `measure` and `reset` are not declared opcodes, so the schema lookup is the
    # refusal and no name list is consulted. `ecr` is not declared either, so the IR
    # refuses to build it at all.
    assert get_operator_schema("measure") is None
    assert not is_diagonal_one_qubit(_measure(0))
    assert not is_diagonal_one_qubit(_reset(0))
    assert canonical_opcode("measure") == "measure"
    with pytest.raises(IRValidationError):
        Instruction("ecr", (0, 1))


def test_a_diagonal_gate_the_measurements_read_out_is_removed() -> None:
    ir = CircuitIR(2, (_gate("z", (0,)), _measure(0)))

    assert _names(remove_diagonal_gates_before_measure(ir)) == ["measure"]


def test_a_non_diagonal_gate_before_a_measurement_is_kept() -> None:
    ir = CircuitIR(2, (_gate("h", (0,)), _measure(0)))

    assert _names(remove_diagonal_gates_before_measure(ir)) == ["h", "measure"]


def test_a_gate_with_a_non_measurement_successor_is_kept() -> None:
    """Only the immediately following instruction on the wire licenses a removal."""

    ir = CircuitIR(2, (_gate("z", (0,)), _gate("h", (0,)), _measure(0)))

    assert _names(remove_diagonal_gates_before_measure(ir)) == ["z", "h", "measure"]


def test_a_gate_with_no_measurement_downstream_is_kept() -> None:
    ir = CircuitIR(2, (_gate("z", (0,)),))

    assert remove_diagonal_gates_before_measure(ir) is ir


def test_a_leading_run_collapses_completely() -> None:
    """A removed gate touches nothing, so the gate behind it is read out too."""

    ir = CircuitIR(
        2, (_gate("s", (0,)), _gate("t", (0,)), _gate("z", (0,)), _measure(0))
    )

    assert _names(remove_diagonal_gates_before_measure(ir)) == ["measure"]


def test_a_two_wire_gate_is_removed_only_when_every_wire_is_read_out() -> None:
    both = CircuitIR(2, (_gate("cz", (0, 1)), _measure(0, 0), _measure(1, 1)))
    one = CircuitIR(2, (_gate("cz", (0, 1)), _measure(0, 0)))

    assert _names(remove_diagonal_gates_before_measure(both)) == [
        "measure",
        "measure",
    ]
    assert _names(remove_diagonal_gates_before_measure(one)) == ["cz", "measure"]


def test_a_two_wire_gate_is_unaffected_by_a_third_wire() -> None:
    """A third wire doing something else must not block it; a successor wire must."""

    elsewhere = CircuitIR(
        3, (_gate("cz", (0, 1)), _measure(0, 0), _measure(1, 1), _gate("x", (2,)))
    )
    on_a_successor = CircuitIR(
        2, (_gate("cz", (0, 1)), _measure(0, 0), _gate("h", (1,)), _measure(1, 1))
    )

    assert _names(remove_diagonal_gates_before_measure(elsewhere)) == [
        "measure",
        "measure",
        "x",
    ]
    assert _names(remove_diagonal_gates_before_measure(on_a_successor)) == [
        "cz",
        "measure",
        "h",
        "measure",
    ]


@pytest.mark.parametrize("opcode", sorted(_DIAGONAL_TWO_WIRE))
def test_every_declared_two_wire_opcode_reaches_the_removal(opcode: str) -> None:
    params = dict.fromkeys(_angles_for(opcode, _GENERIC), 0.3)
    ir = CircuitIR(
        2, (Instruction(opcode, (0, 1), params=params), _measure(0, 0), _measure(1, 1))
    )

    assert _names(remove_diagonal_gates_before_measure(ir)) == [
        "measure",
        "measure",
    ]


@pytest.mark.parametrize(
    "opcode",
    sorted(set(_TWO_WIRE_UNITARIES) - set(_DIAGONAL_TWO_WIRE)),
)
def test_a_non_diagonal_two_wire_opcode_is_never_removed(opcode: str) -> None:
    params = dict.fromkeys(_angles_for(opcode, _GENERIC), 0.3)
    ir = CircuitIR(
        2, (Instruction(opcode, (0, 1), params=params), _measure(0, 0), _measure(1, 1))
    )

    assert _names(remove_diagonal_gates_before_measure(ir)) == [
        opcode,
        "measure",
        "measure",
    ]


def test_a_conditional_measurement_blocks_the_removal() -> None:
    """A condition is the one event this pass cannot see the whole program across."""

    conditioned = CircuitIR(2, (_gate("z", (0,)), _measure(0, 0, conditions=((0, 1),))))
    clauses = CircuitIR(
        2, (_gate("z", (0,)), _measure(0, 0, condition_clauses=((0, 1),)))
    )

    assert _names(remove_diagonal_gates_before_measure(conditioned)) == [
        "z",
        "measure",
    ]
    assert _names(remove_diagonal_gates_before_measure(clauses)) == ["z", "measure"]


def test_a_conditional_gate_is_left_alone() -> None:
    conditional = Instruction(
        "z", (0,), metadata={"is_dynamic": True, "conditions": ((0, 1),)}
    )
    ir = CircuitIR(2, (conditional, _measure(0)))

    assert _names(remove_diagonal_gates_before_measure(ir)) == ["z", "measure"]


def test_a_gate_carrying_a_caller_supplied_matrix_is_left_alone() -> None:
    """The tabulated angles describe the opcode, not an operator a caller attached."""

    attached = Instruction("z", (0,), matrix=torch.eye(2, dtype=torch.complex64))
    ir = CircuitIR(2, (attached, _measure(0)))

    assert _names(remove_diagonal_gates_before_measure(ir)) == ["z", "measure"]


def test_a_dynamic_gate_is_left_alone() -> None:
    dynamic = Instruction("z", (0,), metadata={"is_dynamic": True})
    ir = CircuitIR(2, (dynamic, _measure(0)))

    assert _names(remove_diagonal_gates_before_measure(ir)) == ["z", "measure"]


def test_a_program_with_nothing_removable_is_returned_unchanged() -> None:
    ir = CircuitIR(2, (_gate("h", (0,)), _measure(0)))

    assert remove_diagonal_gates_before_measure(ir) is ir


def test_the_pass_is_idempotent() -> None:
    ir = CircuitIR(
        3,
        (
            _gate("z", (0,)),
            _gate("cz", (0, 1)),
            _measure(0, 0),
            _measure(1, 1),
            _gate("h", (2,)),
            _measure(2, 2),
        ),
    )
    once = remove_diagonal_gates_before_measure(ir)

    assert remove_diagonal_gates_before_measure(once) is once


def test_a_removal_needs_the_fixed_point_the_pipeline_runs() -> None:
    """The gain is the loop's: the pair in front has to cancel before `z` is read."""

    ir = CircuitIR(
        2, (_gate("x", (0,)), _gate("x", (0,)), _gate("z", (0,)), _measure(0))
    )

    # One call already sees the measurement, because the pair is *behind* the phase.
    assert _names(remove_diagonal_gates_before_measure(ir)) == ["x", "x", "measure"]
    assert _names(optimize(ir)) == ["measure"]


def test_a_removal_a_pass_above_exposes_needs_the_fixed_point() -> None:
    """The loop, not the ordering, is what earns this: `z` starts behind the pair."""

    ir = CircuitIR(
        2, (_gate("z", (0,)), _gate("x", (0,)), _gate("x", (0,)), _measure(0))
    )

    # `z`'s successor on wire 0 is an `x`, so a single call declines it.
    assert _names(remove_diagonal_gates_before_measure(ir)) == [
        "z",
        "x",
        "x",
        "measure",
    ]
    assert _names(optimize(ir)) == ["measure"]


def test_the_pass_is_wired_into_the_fixed_point_loop_exactly_once() -> None:
    """The loop reads the declared sequence, and names no pass itself.

    The order this pass runs in is declared in `optimization_levels`, so the check
    below pins the declaration and then confirms the loop body neither restates it
    nor skips it.
    """

    composition = OPTIMIZATION_LEVEL_STAGES[DEFAULT_OPTIMIZATION_LEVEL]
    assert composition.count("remove_diagonal_gates_before_measure") == 1

    tree = ast.parse(Path(pipeline_module.__file__).read_text(encoding="utf-8"))
    loop = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_optimize_to_fixed_point"
    )
    called = {
        node.func.id
        for node in ast.walk(loop)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "optimization_level_stages" in called
    assert "remove_diagonal_gates_before_measure" not in called


def test_a_removed_diagonal_gate_leaves_the_outcome_bits_identical() -> None:
    """Shot for shot, not merely in distribution: a diagonal gate moves no amplitude."""

    programs = {
        "phase_on_a_zero_state": (_gate("z", (0,)), _measure(0)),
        "phase_behind_a_hadamard": (
            _gate("h", (0,)),
            _gate("rz", (0,), theta=0.4),
            _measure(0),
        ),
        "t_behind_a_hadamard": (_gate("h", (0,)), _gate("t", (0,)), _measure(0)),
        "general_rotation_at_a_zero_polar_angle": (
            _gate("h", (0,)),
            _gate("u3", (0,), **_ZERO_POLAR),
            _measure(0),
        ),
        "controlled_phase_on_a_bell_pair": (
            _gate("h", (0,)),
            _gate("h", (1,)),
            _gate("cz", (0, 1)),
            _measure(0, 0),
            _measure(1, 1),
        ),
        "controlled_rz_on_a_bell_pair": (
            _gate("h", (0,)),
            _gate("h", (1,)),
            _gate("crz", (0, 1), theta=0.9),
            _measure(0, 0),
            _measure(1, 1),
        ),
        "rzz_on_a_bell_pair": (
            _gate("h", (0,)),
            _gate("h", (1,)),
            _gate("rzz", (0, 1), theta=0.9),
            _measure(0, 0),
            _measure(1, 1),
        ),
    }

    for label, program in programs.items():
        source = CircuitIR(2, program)
        after = optimize(source)
        assert len(after) < len(source), label
        assert _outcomes(source) == _outcomes(after), label


def test_the_outcome_instrument_can_detect_a_wrong_removal() -> None:
    """An instrument that cannot fail is not evidence, so a wrong deletion is driven.

    Deleting a diagonal gate is invisible because a diagonal gate is a phase on each
    basis state. Deleting a *non-diagonal* gate is not: on the Bell pair below the
    single `cx` moves the joint outcome on most shots, and on a bare register a
    single `ry` moves the marginal. Both readings stand next to the zero that the
    correct removals produce.
    """

    prepared = CircuitIR(
        2,
        (
            _gate("h", (0,)),
            _gate("cx", (0, 1)),
            _measure(0, 0),
            _measure(1, 1),
        ),
    )
    entangled_wrong = _differing_shots(
        _outcomes(prepared),
        _outcomes(
            CircuitIR(
                2,
                (
                    _gate("h", (0,)),
                    _measure(0, 0),
                    _measure(1, 1),
                ),
            )
        ),
    )

    tilted = CircuitIR(1, (_gate("ry", (0,), theta=1.8), _measure(0)))
    single_wire_wrong = _differing_shots(
        _outcomes(tilted), _outcomes(CircuitIR(1, (_measure(0),)))
    )

    correct = _differing_shots(
        _outcomes(prepared),
        _outcomes(
            optimize(
                CircuitIR(
                    2,
                    (
                        _gate("h", (0,)),
                        _gate("cx", (0, 1)),
                        _gate("rz", (0,), theta=0.4),
                        _measure(0, 0),
                        _measure(1, 1),
                    ),
                )
            )
        ),
    )

    assert entangled_wrong > _SHOTS // 4
    assert single_wire_wrong > _SHOTS // 4
    assert correct == 0


def test_random_programs_agree_with_the_runtime_shot_for_shot() -> None:
    """A fixed sweep, so the counts below are stable rather than merely nonzero."""

    rng = random.Random(_SEED)
    driven = 0
    removed = 0
    for _ in range(60):
        width = rng.randint(1, 3)
        instructions: list[Instruction] = []
        written: list[int] = []
        for _ in range(rng.randint(1, 12)):
            choice = rng.random()
            wire = rng.randrange(width)
            if choice < 0.45:
                opcode = rng.choice(("z", "s", "t", "sdg", "rz", "phase", "u1", "i"))
                params = (
                    {"theta": rng.choice((0.0, 0.3, -0.7))}
                    if opcode in {"rz", "phase", "u1"}
                    else {}
                )
                instructions.append(Instruction(opcode, (wire,), params=params))
            elif choice < 0.6:
                opcode = rng.choice(("h", "x", "ry", "sx", "u2", "u3"))
                params = {
                    key: rng.choice((0.0, 0.3, -0.7))
                    for key in _angles_for(opcode, _GENERIC)
                }
                instructions.append(Instruction(opcode, (wire,), params=params))
            elif choice < 0.7 and width >= 2:
                opcode = rng.choice(("cz", "crz", "cphase", "rzz", "cx", "rxx"))
                params = {
                    key: rng.choice((0.3, -0.7))
                    for key in _angles_for(opcode, _GENERIC)
                }
                pair = tuple(rng.sample(range(width), 2))
                instructions.append(Instruction(opcode, pair, params=params))
            elif choice < 0.95 or not written:
                bit = len(written)
                instructions.append(_measure(wire, bit))
                written.append(bit)
            else:
                # A condition must read a bit that has already been written, or the
                # runtime refuses the program before the pass is ever reached.
                instructions.append(_measure(wire, 0, conditions=((written[0], 1),)))

        source = CircuitIR(width, tuple(instructions))
        after = optimize(source)
        assert _names(after) == _names(remove_diagonal_gates_before_measure(after))
        assert _outcomes(source) == _outcomes(after)
        driven += 1
        removed += len(source) - len(after)

    assert driven == 60
    # A differential sweep whose programs never change proves nothing.
    assert removed > 100


def test_optimization_never_leaves_a_removable_diagonal_gate_behind() -> None:
    """Whatever the pass declines, no later round of the pipeline may decline more."""

    rng = random.Random(_SEED + 1)
    for _ in range(40):
        width = rng.randint(1, 3)
        instructions: list[Instruction] = []
        for _ in range(rng.randint(1, 10)):
            wire = rng.randrange(width)
            draw = rng.random()
            if draw < 0.6:
                instructions.append(
                    Instruction("rz", (wire,), params={"theta": rng.choice((0.0, 0.4))})
                )
            elif draw < 0.8:
                instructions.append(_measure(wire, 0))
            else:
                instructions.append(Instruction("h", (wire,)))
        after = optimize(CircuitIR(width, tuple(instructions)))
        assert remove_diagonal_gates_before_measure(after) is after


def test_the_pass_only_deletes_and_never_invents_an_instruction() -> None:
    """Every survivor is the identical object, in the order it was given."""

    rng = random.Random(_SEED + 2)
    for _ in range(40):
        width = rng.randint(1, 3)
        instructions: list[Instruction] = []
        for _ in range(rng.randint(1, 8)):
            name = rng.choice(("z", "t", "h", "x", "rz", "phase"))
            schema = get_operator_schema(name)
            assert schema is not None
            params = {key: rng.choice((0.0, 0.3)) for key in schema.parameters}
            instructions.append(
                Instruction(name, (rng.randrange(width),), params=params)
            )
        instructions.append(_measure(0))
        source = CircuitIR(width, tuple(instructions))
        after = remove_diagonal_gates_before_measure(source)
        survivors = list(after)

        assert len(survivors) <= len(source)
        kept = 0
        for instruction in source:
            if kept < len(survivors) and survivors[kept] == instruction:
                kept += 1
        assert kept == len(survivors)
