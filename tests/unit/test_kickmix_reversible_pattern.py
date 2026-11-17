"""Bind locked CircuitPattern and affine-index cases to executable artifacts."""

import pytest

from flagquantum.compiler.artifacts.reversible_pattern import bind_reversible_pattern
from flagquantum.compiler.artifacts.reversible_phase import (
    ReversiblePhaseOperation as Op,
)
from flagquantum.compiler.artifacts.reversible_phase import (
    ReversiblePhaseProgram,
)
from flagquantum.compiler.formats.kmx import encode_kmx
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient,
)
from flagquantum.compiler.registers.primitives._pattern_mux import (
    bind_quantum_enable_pattern,
    bind_z_control_pattern,
)

pytestmark = pytest.mark.unit


def test_forward_reverse_and_reused_mixed_lane_pattern_match_source() -> None:
    recipe = (
        Op("x", (0,)),
        Op("r", (1,)),
        Op("hmr", (0,), result_bit=0),
        Op("push_condition", (), condition_bit=0),
        Op("swap", (0, 1)),
        Op("ccz", (0, 2, 1)),
        Op("bit_invert", (), result_bit=1, condition_bit=0),
        Op("neg", (), condition_bit=0),
        Op("pop_condition", ()),
    )
    quantum = ((0, 1, 2), (5, 4, 3), 6)
    bits = ((1, 2, 3), 0)
    forward = bind_reversible_pattern(recipe, range(1, 3), qubits=quantum, bits=bits)
    reverse = bind_reversible_pattern(
        recipe, range(2, 0, -1), qubits=quantum, bits=bits
    )
    shorter = bind_reversible_pattern(
        (Op("z", (0,)),), range(1, 2), qubits=((5, 4, 3),)
    )
    program = ReversiblePhaseProgram(7, 4, (*forward, *reverse, *shorter), (0,))
    original: list[str] = []
    for wire, aux, bit in ((1, 4, 2), (2, 3, 3), (2, 3, 3), (1, 4, 2)):
        original.extend(
            (
                f"X q{wire}",
                f"R q{aux}",
                f"HMR q{wire} b{bit}",
                f"PUSH_CONDITION if b{bit}",
                f"SWAP q{wire} q{aux}",
                f"CCZ q{wire} q6 q{aux}",
                f"BIT_INVERT b0 if b{bit}",
                f"NEG if b{bit}",
                "POP_CONDITION",
            )
        )
    assert encode_kmx(program).splitlines()[1:] == [*original, "Z q4"]


def test_iota_arithmetic_matches_bounded_physical_indexing() -> None:
    # The locked source uses signed/unsigned 16-, 32- and 64-bit additions,
    # one generic integral overload, and subtraction and unary negation.
    wires = tuple(range(12))
    columns = (
        *tuple(wires[offset:] for offset in range(1, 8)),
        range(-1, 11),
        range(8, -1, -1),
    )
    recipe = tuple(Op("x", (column,)) for column in range(len(columns)))
    emitted = bind_reversible_pattern(recipe, range(1, 4), qubits=columns)
    original = (
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        0,
        7,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        1,
        6,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        2,
        5,
    )
    assert tuple(op.qubits[0] for op in emitted) == original


def test_empty_ranges_and_unbound_indices() -> None:
    pattern = (Op("x", (0,)),)
    assert bind_reversible_pattern(pattern, range(4, 1)) == ()
    with pytest.raises(ValueError, match="no bound column"):
        bind_reversible_pattern(pattern, range(1))
    with pytest.raises(ValueError, match="lane is outside"):
        bind_reversible_pattern(pattern, range(2), qubits=((0,),))
    with pytest.raises(ValueError, match="lanes require nonnegative"):
        bind_reversible_pattern(pattern, range(-1, 0), qubits=((0,),))


@pytest.mark.parametrize("classical", [False, True])
def test_source_homogeneous_mux_preserves_control_and_guard_columns(
    classical: bool,
) -> None:
    controls: tuple[int | ClassicalFieldCoefficient, ...]
    if classical:
        controls = (ClassicalFieldCoefficient(0), ClassicalFieldCoefficient(1))
        expected = [
            "X q4 if b0",
            "Z q4 if b0",
            "NEG if b0",
            "X q4 if b1",
            "Z q4 if b1",
            "NEG if b1",
        ]
    else:
        controls = (2, 3)
        expected = ["CX q2 q4", "CZ q2 q4", "Z q2", "CX q3 q4", "CZ q3 q4", "Z q3"]
    body = tuple(
        operation
        for lane in range(2)
        for operation in (
            *bind_z_control_pattern("cx", controls, range(lane, lane + 1), target=4),
            *bind_z_control_pattern("cz", controls, range(lane, lane + 1), target=4),
            *bind_z_control_pattern("z", controls, range(lane, lane + 1)),
        )
    )
    assert (
        encode_kmx(ReversiblePhaseProgram(5, 2, body, (0, 1))).splitlines()[1:]
        == expected
    )


def test_source_mixed_controls_refuse_without_emitting_a_prefix() -> None:
    mixed = (2, ClassicalFieldCoefficient(1))
    with pytest.raises(ValueError, match="homogeneous"):
        bind_z_control_pattern("cx", mixed, range(2), target=4)
    with pytest.raises(ValueError, match="homogeneous"):
        bind_z_control_pattern("cz", mixed, range(2), target=4)
    with pytest.raises(ValueError, match="homogeneous"):
        bind_z_control_pattern("z", mixed, range(2))


@pytest.mark.parametrize("enable", [True, 5])
def test_source_optional_quantum_gate_order(enable: int) -> None:
    native = (
        Op("r", (2,), condition_bit=0),
        Op("cz", (2, 3), condition_bit=0),
        *bind_quantum_enable_pattern("cswap", enable, (2, 3), range(1), condition=1),
        *bind_quantum_enable_pattern("ccx", enable, (2, 3), range(1), condition=1),
        *bind_quantum_enable_pattern("z", enable, (), range(1), condition=1),
        *bind_quantum_enable_pattern("cswap", enable, (2, 3), range(1)),
        *bind_quantum_enable_pattern("cx", enable, (3,), range(1), condition=1),
        *bind_quantum_enable_pattern("ccx", enable, (2, 3), range(1)),
        *bind_quantum_enable_pattern("ccz", enable, (2, 3), range(1)),
    )
    if enable is True:
        expected = [
            "R q2 if b0",
            "CZ q2 q3 if b0",
            "SWAP q2 q3 if b1",
            "CX q2 q3 if b1",
            "NEG if b1",
            "SWAP q2 q3",
            "X q3 if b1",
            "CX q2 q3",
            "CZ q2 q3",
        ]
    else:
        expected = [
            "R q2 if b0",
            "CZ q2 q3 if b0",
            "CSWAP q5 q2 q3 if b1",
            "CCX q2 q5 q3 if b1",
            "Z q5 if b1",
            "CSWAP q5 q2 q3",
            "CX q5 q3 if b1",
            "CCX q2 q5 q3",
            "CCZ q2 q5 q3",
        ]
    assert (
        encode_kmx(ReversiblePhaseProgram(6, 2, native, (0, 1))).splitlines()[1:]
        == expected
    )
