"""Unmodified AddK/XorK/Always call chains and native resource dispatch."""

from dataclasses import fields, replace
from itertools import product

import pytest
import sympy

from flagquantum.compiler._component_trace.arithmetic_call import ArithmeticCall
from flagquantum.compiler._component_trace.call_graph import symbolic_call_graph
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import (
    CliffordCall,
    ConjunctionCall,
)
from flagquantum.compiler.registers.arithmetic_declarations import (
    arithmetic_call_graph,
    arithmetic_callees,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
n, k = sympy.symbols("n k", integer=True, positive=True)
z = sympy.Symbol("z", integer=True)
DIMENSIONS = (
    (1, 0),
    (2, -1),
    (3, -1),
    (3, 0),
    (3, 3),
    (8, 83),
    (3, sympy.Integer(5)),
    (sympy.Integer(3), -1),
    (sympy.Integer(3), sympy.Integer(5)),
    (n, k),
    (n, -1),
    (n, 0),
    (n, 3),
    (3, k),
    (z, -1),
)
CASES = [
    ArithmeticCall(
        "constant_add", (width,), (encoding,), constant=value, adjoint=adjoint
    )
    for encoding, (width, value), adjoint in product(
        ("unsigned", "twos_complement", "montgomery"), DIMENSIONS, (False, True)
    )
]


def source(call):
    from qualtran import QInt, QMontgomeryUInt, QUInt
    from qualtran.bloqs.arithmetic.addition import Add, AddK
    from qualtran.bloqs.arithmetic.bitwise import XorK
    from qualtran.bloqs.basic_gates import CNOT, XGate
    from qualtran.bloqs.bookkeeping.always import Always
    from qualtran.bloqs.mcmt import And

    if isinstance(call, ConjunctionCall):
        result = And(*call.controls)
        return result.adjoint() if call.adjoint else result
    if isinstance(call, CliffordCall):
        return CNOT() if call.opcode == "cx" else XGate()
    word = {"unsigned": QUInt, "twos_complement": QInt, "montgomery": QMontgomeryUInt}
    words = [word[e](w) for w, e in zip(call.widths, call.encodings, strict=True)]
    if call.operation == "constant_add":
        result = AddK(words[0], call.constant)
    elif call.operation == "xor_constant":
        result = XorK(words[0], call.constant)
    else:
        assert call.operation == "add"
        result = Add(*words)
    if call.adjoint:
        result = result.adjoint()
    return Always(result) if call.always else result


@pytest.mark.parametrize("root", CASES)
def test_each_source_row_and_all_recursive_qec_categories(root):
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    rows = arithmetic_call_graph(root)
    for node, row in rows.items():
        actual = [(source(child), count) for child, count in row.items()]
        assert actual == list(source(node).build_call_graph(None).items())
    load, always = rows[root]
    assert load.operation == "xor_constant" and always.always is True
    assert list(rows[root].values()) == [2, 1]
    assert replace(always, always=False) not in rows
    graph = symbolic_call_graph(root, rows)
    assert not any(isinstance(leaf, ArithmeticCall) for leaf in graph.leaf_counts)
    for legacy in (False, True):
        actual = query_qec_gate_cost(root, declared_calls=rows, legacy_shims=legacy)
        expected = get_cost_value(source(root), QECGatesCost(legacy_shims=legacy))
        for field in fields(actual):
            assert (
                sympy.simplify(
                    getattr(actual, field.name) - getattr(expected, field.name)
                )
                == 0
            )


def test_concrete_and_symbolic_equal_constants_cannot_merge_resource_cache_entries():
    py = ArithmeticCall("constant_add", (3,), constant=5)
    symbolic = replace(py, constant=sympy.Integer(5))
    cache = {}
    rows = {**arithmetic_call_graph(py), **arithmetic_call_graph(symbolic)}
    a = query_qec_gate_cost(py, declared_calls=rows, cache=cache)
    b = query_qec_gate_cost(symbolic, declared_calls=rows, cache=cache)
    assert a.clifford == 15 and b.clifford == 17


@pytest.mark.parametrize(
    "encoding,width,value",
    [
        ("unsigned", 3, 8),
        ("montgomery", 3, 9),
        ("twos_complement", 3, 4),
        ("twos_complement", 3, -5),
        ("unsigned", 1, 2),
    ],
)
def test_concrete_source_range_failures_are_not_wrapped_as_valid_constants(
    encoding, width, value
):
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    root = ArithmeticCall("constant_add", (width,), (encoding,), constant=value)
    assert len(arithmetic_callees(root)) == 2
    with pytest.raises(ValueError):
        get_cost_value(source(root), QECGatesCost())
    with pytest.raises(ValueError, match="source .* register range"):
        arithmetic_call_graph(root)
