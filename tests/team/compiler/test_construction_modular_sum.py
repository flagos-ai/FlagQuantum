"""Fuse equal-width modular sums without losing XOR or shared-value semantics."""

from itertools import product

import pytest

from benchmarks.synthesis.verification import classical_apply, encode
from flagquantum.compiler._construction.codegen.arithmetic import (
    Schedule,
    _emit_arithmetic,
)
from flagquantum.compiler._construction.verification.profile import uint_type
from flagquantum.compiler._hybrid.model import (
    INDEX,
    Block,
    HybridProgram,
    Operation,
    Region,
    Value,
    ValueId,
)
from tests.team.compiler.test_construction_borrowed_input_bits import candidate

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("scheduling", ["retain", "recompute"])
@pytest.mark.parametrize(
    "expression, coefficient_x, coefficient_y",
    [
        ("x+y", 2, 1),
        ("y+x", 2, 1),
        ("(x-y)+y", 2, 0),
        ("x+(y+y)", 2, 2),
        ("(x+y)+(x-y)", 4, 0),
        ("x-y", 2, -1),
        ("x*4+y", 8, 1),
        ("4*x+y", 8, 1),
        ("x*8+y", 16, 1),
        ("x+y*2", 2, 2),
    ],
)
def test_signed_expression_family_every_input_and_arbitrary_xor(
    scheduling: Schedule, expression: str, coefficient_x: int, coefficient_y: int
) -> None:
    compiled = candidate(1, 2, scheduling, optimized=True, expression=expression)
    width = (
        compiled.lifetimes.program.body.blocks[0]
        .operations[-1]
        .operands[0]
        .type.parameters[0]
    )
    registers = ((0,), (1, 2), tuple(range(3, 3 + width)))
    for x, word, target in product(range(2), range(4), range(1 << width)):
        signed = word - 4 if word & 2 else word
        value = (coefficient_x * x + coefficient_y * signed) % (1 << width)
        initial = encode((x, word, target), registers)
        expected = encode((x, word, target ^ value), registers)
        assert classical_apply(compiled.circuit, initial) == expected
        assert classical_apply(compiled.circuit, expected) == initial


def modular_graph(kind: str) -> HybridProgram:
    a = Value(ValueId(kind, 0), uint_type(3))
    b = Value(ValueId(kind, 1), uint_type(2 if kind == "different_moduli" else 3))
    constant = Value(ValueId(kind, 2), INDEX)
    negative = Value(ValueId(kind, 3), b.type)
    result = Value(ValueId(kind, 4), a.type)
    operations = (
        Operation(
            "arith.constant",
            results=(constant,),
            attributes={"value": int(kind == "nonzero_base")},
        ),
        Operation("qint.subtract_mod", (constant, b), (negative,)),
        Operation(
            "qint.subtract_mod",
            (negative if kind == "shared" else a, negative),
            (result,),
        ),
        Operation("qint.return", (result,)),
    )
    return HybridProgram(kind, Region((Block((a, b), operations),)))


@pytest.mark.parametrize("scheduling", ["retain", "recompute"])
@pytest.mark.parametrize("kind", ["shared", "different_moduli", "nonzero_base"])
def test_shared_negative_and_nonmatching_modular_subtractions(
    scheduling: Schedule, kind: str
) -> None:
    graph = modular_graph(kind)
    a, b = graph.body.blocks[0].arguments
    width_b = b.type.parameters[0]
    offset = 3 + width_b
    registers = (
        tuple(range(3)),
        tuple(range(3, offset)),
        tuple(range(offset, offset + 3)),
    )
    compiled = _emit_arithmetic(
        (graph,),
        {a.id: registers[0], b.id: registers[1]},
        registers[2],
        entry=graph.name,
        n_qubits=48,
        workspace=tuple(range(offset + 3, 48)),
        workspace_state="fresh_zero",
        scheduling=scheduling,
        implementation="arithmetic",
    )
    fused = any(
        rule == "terminal_modular_sum" for _, rule in compiled.implementation_rules
    )
    assert fused == (kind == "shared")
    for x, y, target in product(range(8), range(1 << width_b), range(8)):
        negative = (int(kind == "nonzero_base") - y) % (1 << width_b)
        value = 0 if kind == "shared" else (x - negative) % 8
        initial = encode((x, y, target), registers)
        expected = encode((x, y, target ^ value), registers)
        assert classical_apply(compiled.circuit, initial) == expected
        assert classical_apply(compiled.circuit, expected) == initial
