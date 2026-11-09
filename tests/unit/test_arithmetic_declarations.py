"""Native call recipes compared with unmodified pinned source declarations."""

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
    FixedTCall,
    OnEachCall,
)
from flagquantum.compiler._component_trace.terminal_t_cost import (
    estimate_terminal_t_cost,
)
from flagquantum.compiler.registers.arithmetic.arithmetic_declarations import (
    arithmetic_call_graph,
    arithmetic_callees,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def source(call):
    from qualtran import QInt, QMontgomeryUInt, QUInt
    from qualtran.bloqs.arithmetic._shims import CHalf, CSub, Lt, MultiCToffoli
    from qualtran.bloqs.arithmetic.addition import Add
    from qualtran.bloqs.arithmetic.bitwise import BitwiseNot
    from qualtran.bloqs.arithmetic.controlled_addition import CAdd

    dtypes = {"unsigned": QUInt, "twos_complement": QInt, "montgomery": QMontgomeryUInt}
    words = [dtypes[e](w) for e, w in zip(call.encodings, call.widths, strict=True)]
    if call.operation == "add":
        result = Add(*words)
    elif call.operation == "controlled_add":
        result = CAdd(*words, cv=call.control_value)
    elif call.operation == "bitwise_not":
        result = BitwiseNot(words[0])
    elif call.operation == "less_than":
        result = Lt(call.widths[0], signed=call.signed)
    else:
        result = {
            "multi_controlled_x": MultiCToffoli,
            "controlled_subtraction": CSub,
            "controlled_half": CHalf,
        }[call.operation](call.widths[0])
    return result.adjoint() if call.adjoint else result


def source_leaf(call):
    from qualtran.bloqs.basic_gates import CNOT, Toffoli, XGate
    from qualtran.bloqs.basic_gates.on_each import OnEach
    from qualtran.bloqs.basic_gates.swap import TwoBitCSwap
    from qualtran.bloqs.mcmt import And

    if isinstance(call, ArithmeticCall):
        return source(call)
    if isinstance(call, ConjunctionCall):
        result = And(*call.controls)
        return result.adjoint() if call.adjoint else result
    if isinstance(call, CliffordCall):
        return CNOT() if call.opcode == "cx" else XGate()
    if isinstance(call, FixedTCall):
        return Toffoli() if call.opcode == "toffoli" else TwoBitCSwap()
    if isinstance(call, OnEachCall):
        return OnEach(call.width, source_leaf(call.child))
    raise AssertionError(call)


CASES = [
    ArithmeticCall(
        op,
        (n, n) if op in ("add", "controlled_add") else (n,),
        control_value=cv if op == "controlled_add" else 1,
        adjoint=adjoint,
    )
    for op, n, cv, adjoint in product(
        (
            "add",
            "controlled_add",
            "bitwise_not",
            "multi_controlled_x",
            "controlled_subtraction",
            "less_than",
            "controlled_half",
        ),
        (1, 3, sympy.Integer(3), sympy.Symbol("n", integer=True, positive=True)),
        (0, 1),
        (False, True),
    )
    if op == "controlled_add" or cv == 1
]
CASES += [
    ArithmeticCall("controlled_add", (2, 3), (encoding, encoding), control_value=cv)
    for encoding, cv in product(("montgomery", "twos_complement"), (0, 1))
]


@pytest.mark.parametrize("root", CASES)
def test_each_direct_source_edge_order_parameters_and_recursive_qec_cost(root):
    from qualtran.resource_counting import (
        QECGatesCost,
        SympySymbolAllocator,
        get_cost_value,
    )

    native = arithmetic_call_graph(root)
    for node, row in native.items():
        if not isinstance(node, ArithmeticCall) or node.operation == "bitwise_not":
            continue
        expected = source(node).build_call_graph(SympySymbolAllocator())
        assert list(
            (source_leaf(child), weight) for child, weight in row.items()
        ) == list(expected.items())
    for legacy in (False, True):
        actual = query_qec_gate_cost(root, declared_calls=native, legacy_shims=legacy)
        expected = get_cost_value(source(root), QECGatesCost(legacy_shims=legacy))
        for field in fields(actual):
            assert (
                sympy.simplify(
                    getattr(actual, field.name) - getattr(expected, field.name)
                )
                == 0
            )
    graph = symbolic_call_graph(root, native)
    assert not any(
        isinstance(leaf, (ArithmeticCall, OnEachCall)) for leaf in graph.leaf_counts
    )
    # Expanded terminal estimation must no longer encounter opaque arithmetic.
    estimate_terminal_t_cost(graph.leaf_counts)


def test_subtraction_retains_montgomery_identity_and_symbolic_width_objects():
    n = sympy.Symbol("n", integer=True, positive=True)
    root = ArithmeticCall("controlled_subtraction", (n,))
    row = arithmetic_callees(root)
    add, invert = row
    assert add == ArithmeticCall("controlled_add", (n, n), ("montgomery", "montgomery"))
    assert invert == ArithmeticCall("bitwise_not", (n,), ("montgomery",))
    assert all(w is n for w in add.widths)
    assert list(row.values()) == [1, 3]
    inverse = arithmetic_callees(replace(root, adjoint=True))
    assert list(inverse) == [replace(add, adjoint=True), invert]


def test_zero_width_and_negative_source_declarations_remain_diagnostics():
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    root = ArithmeticCall("controlled_subtraction", (0,))
    assert len(arithmetic_callees(root)) == 2
    with pytest.raises(TypeError):
        get_cost_value(source(root), QECGatesCost())
    with pytest.raises(ValueError, match="zero-width BitwiseNot"):
        arithmetic_call_graph(root)
    diagnostic = ArithmeticCall("multi_controlled_x", (0,))
    assert arithmetic_callees(diagnostic) == {FixedTCall("toffoli"): -2}
    assert (
        query_qec_gate_cost(
            diagnostic, declared_calls=arithmetic_call_graph(diagnostic)
        ).toffoli
        == -2
    )


def test_replacing_a_shared_add_recipe_changes_subtraction_without_consumer_changes():
    n = sympy.Symbol("n", integer=True, positive=True)
    root = ArithmeticCall("controlled_subtraction", (n,))
    rows = arithmetic_call_graph(root)
    child = next(
        node
        for node in rows
        if isinstance(node, ArithmeticCall) and node.operation == "add"
    )
    baseline = query_qec_gate_cost(root, declared_calls=rows)
    rows[child] = {FixedTCall("fredkin"): 7}
    changed = query_qec_gate_cost(root, declared_calls=rows)
    assert baseline.cswap == 0 and changed.cswap == 7
    assert changed.and_bloq == n


@pytest.mark.parametrize(
    "encoding,width",
    tuple(
        product(
            ("unsigned", "montgomery", "twos_complement"),
            (3, sympy.Symbol("n", integer=True, positive=True)),
        )
    ),
)
def test_bitwise_not_composite_retains_original_on_each_x_dtype(encoding, width):
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.basic_gates.on_each import OnEach

    root = ArithmeticCall("bitwise_not", (width,), (encoding,))
    original = source(root)
    seen = []
    marker = object()

    class Builder:
        def add(self, operation, *, q):
            assert q is marker
            seen.append(operation)
            return marker

    assert original.build_composite_bloq(Builder(), x=marker) == {"x": marker}
    assert seen == [OnEach(width, XGate(), original.dtype)]
    native = arithmetic_callees(root)
    assert native == {OnEachCall(CliffordCall("x", 1), width): 1}
    assert root.encodings == (encoding,)
