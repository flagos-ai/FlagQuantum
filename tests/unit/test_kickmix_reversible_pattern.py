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
