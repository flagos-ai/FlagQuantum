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


@pytest.mark.parametrize("root", CASES)
def test_shared_formula_matches_unmodified_source_categories_and_peak(root):
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from flagquantum.compiler.registers.constant_add_resource_model import (
        source_constant_add_resource_formula,
    )

    for legacy in (False, True):
        actual = source_constant_add_resource_formula(
            root.widths[0],
            root.constant,
            encoding=root.encodings[0],
            adjoint=root.adjoint,
            legacy_shims=legacy,
        )
        expected = get_cost_value(source(root), QECGatesCost(legacy_shims=legacy))
        for field in fields(actual):
            value = (
                get_cost_value(source(root), QubitCount())
                if field.name == "logical_qubits"
                else getattr(expected, field.name)
            )
            assert sympy.simplify(getattr(actual, field.name) - value) == 0


@pytest.mark.parametrize(
    "root", [case for case in CASES if type(case.widths[0]) is int]
)
def test_concrete_shape_component_keeps_the_same_model_even_with_symbolic_constant(
    root,
):
    from flagquantum.compiler._component_trace.resource_model import (
        estimate_component_resources,
    )
    from flagquantum.compiler.registers.constant_add_resource_model import (
        source_constant_add_cost,
        source_constant_add_resource_formula,
    )

    options = dict(encoding=root.encodings[0], adjoint=root.adjoint)
    component = source_constant_add_cost(root.widths[0], root.constant, **options)
    actual = estimate_component_resources(component)
    expected = source_constant_add_resource_formula(
        root.widths[0], root.constant, **options
    )
    assert component.name == "AddK"
    assert actual.peak_logical_qubits == expected.logical_qubits
    for field in fields(actual.gates):
        assert getattr(actual.gates, field.name) == getattr(expected, field.name)


def test_shared_add_model_replacement_preserves_every_category_and_peak(monkeypatch):
    import flagquantum.compiler.registers.constant_add_resource_model as module
    from flagquantum.compiler.registers.symbolic_resource_model import (
        SymbolicResourceFormula,
    )

    addition = SymbolicResourceFormula(
        2, 3, 5, 29, t=7, toffoli=11, cswap=13, rotation=17
    )
    seen = []

    def child(left, right, *, legacy_shims=False):
        assert left is right is n
        seen.append(legacy_shims)
        return addition

    monkeypatch.setattr(module, "source_add_resource_formula", child)
    actual = module.source_constant_add_resource_formula(n, k, legacy_shims=True)
    assert seen == [True]
    assert actual == SymbolicResourceFormula(
        2, 3 + 2 * n, 5, sympy.Max(29, n), t=7, toffoli=11, cswap=13, rotation=17
    )


def test_shared_graph_recipe_is_replaceable_without_changing_formula_consumer(
    monkeypatch,
):
    import flagquantum.compiler.registers.constant_add_resource_model as module
    from flagquantum.compiler._component_trace.resource_generalization import FixedTCall

    def graph(root):
        return {root: {FixedTCall("fredkin"): 7 * n}}

    monkeypatch.setattr(module, "arithmetic_call_graph", graph)
    actual = module.source_constant_add_resource_formula(n, k)
    assert (
        actual.cswap == 7 * n
        and actual.and_bloq == actual.clifford == actual.measurement == 0
    )
    assert actual.logical_qubits == sympy.Max(3, 2 * n)


@pytest.mark.parametrize("width", (0, sympy.Integer(0)))
def test_original_zero_width_pair_query_failure_remains_explicit(width):
    from qualtran.resource_counting import QubitCount, get_cost_value

    from flagquantum.compiler.registers.constant_add_resource_model import (
        source_constant_add_resource_formula,
    )

    root = ArithmeticCall("constant_add", (width,), constant=0)
    assert len(arithmetic_callees(root)) == 2
    with pytest.raises(RuntimeError):
        get_cost_value(source(root), QubitCount())
    with pytest.raises(ValueError, match="zero-width AddK"):
        source_constant_add_resource_formula(width, 0)


@pytest.mark.parametrize(
    "options",
    [
        dict(width=True),
        dict(constant=True),
        dict(constant=1.5),
        dict(constant="1"),
        dict(encoding="signed"),
        dict(adjoint=1),
        dict(legacy_shims=1),
    ],
)
def test_formula_refuses_invalid_values_without_coercion(options):
    from flagquantum.compiler.registers.constant_add_resource_model import (
        source_constant_add_resource_formula,
    )

    with pytest.raises(ValueError):
        source_constant_add_resource_formula(**(dict(width=3, constant=1) | options))


def test_concrete_component_does_not_coerce_a_symbolic_width():
    from flagquantum.compiler.registers.constant_add_resource_model import (
        source_constant_add_cost,
    )

    with pytest.raises(ValueError, match="Python integer width"):
        source_constant_add_cost(sympy.Integer(3), 1)


@pytest.mark.parametrize(
    "root",
    [
        case
        for case in CASES
        if type(case.widths[0]) is int and type(case.constant) is int
    ],
)
def test_unmodified_source_primitive_stream_and_model_counts_across_encodings(root):
    from collections import Counter

    from flagquantum.compiler.registers.arithmetic import (
        add_constant_adjoint_ir,
        add_constant_ir,
    )
    from flagquantum.compiler.registers.constant_add_resource_model import (
        source_constant_add_resource_formula,
    )
    from tests.unit.test_register_constant_addition import layout, source_gates
    from tests.unit.test_register_subtraction import assert_source

    width = root.widths[0]
    target, load, carry, bits = layout(width)
    emitter = add_constant_adjoint_ir if root.adjoint else add_constant_ir
    actual = emitter(target, root.constant, load, carry, bits)
    expected, _ = source_gates(width, root.constant, source=source(root))
    assert_source(actual, expected)
    counts = Counter(g.name for g in actual)
    formula = source_constant_add_resource_formula(
        width, root.constant, encoding=root.encodings[0], adjoint=root.adjoint
    )
    assert counts["t"] + counts["tdg"] == 4 * formula.and_bloq
    assert counts["measure"] == formula.measurement == width - 1
    assert formula.logical_qubits == 3 * width - 1


@pytest.mark.parametrize(
    "width,encoding,value,adjoint",
    tuple(
        product(
            (1, 2, 3),
            ("unsigned", "twos_complement", "montgomery"),
            (0, -1),
            (False, True),
        )
    ),
)
def test_all_raw_complex_branches_match_source_values_and_clean_all_work(
    width, encoding, value, adjoint
):
    import torch

    from flagquantum.compiler.registers.arithmetic import (
        add_constant_adjoint_ir,
        add_constant_ir,
    )
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_register_constant_addition import layout

    root = ArithmeticCall("constant_add", (width,), (encoding,), constant=value)
    reference = source(root)
    limit = 1 << width

    def decoded(x):
        return x - limit if encoding == "twos_complement" and x >= limit // 2 else x

    mapping = {
        x: reference.on_classical_vals(x=decoded(x))["x"] % limit for x in range(limit)
    }
    if adjoint:
        mapping = {b: a for a, b in mapping.items()}
    target, load, carry, bits = layout(width)
    emitter = add_constant_adjoint_ir if adjoint else add_constant_ir
    gates = emitter(target, value, load, carry, bits)
    work = 2 * width - 1
    data = torch.arange(1 << (width + 1), dtype=torch.int64)
    coefficients = data.to(torch.float64)
    amplitudes = torch.complex(coefficients + 1, 3 * coefficients - 5)
    amplitudes /= amplitudes.norm()
    state = torch.zeros(1 << (3 * width), dtype=torch.complex128)
    state[((data >> 1) << (work + 1)) | (data & 1)] = amplitudes
    expected = torch.zeros_like(state)
    mapped = torch.tensor([mapping[int(x)] for x in data >> 1], dtype=torch.int64)
    expected[(mapped << (work + 1)) | (data & 1)] = amplitudes
    for history in product((0, 1), repeat=width - 1):
        branch = forced_branch(state, gates, history)
        torch.testing.assert_close(
            branch, expected / 2 ** ((width - 1) / 2), rtol=0, atol=1e-12
        )
