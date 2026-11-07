"""Native arithmetic identity contract and replaceable declared-graph consumers."""

from dataclasses import FrozenInstanceError, replace
from itertools import product

import pytest
import sympy

from flagquantum.compiler._component_trace.arithmetic_call import ArithmeticCall
from flagquantum.compiler._component_trace.call_graph import symbolic_call_graph
from flagquantum.compiler._component_trace.qec_gate_cost import (
    QECGateCost,
    query_qec_gate_cost,
)
from flagquantum.compiler._component_trace.resource_generalization import (
    FixedTCall,
    _resource_callees,
)
from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    estimate_component_resources,
)
from flagquantum.compiler._component_trace.terminal_t_cost import (
    estimate_terminal_t_cost,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "operation,arity",
    [
        ("add", 2),
        ("controlled_add", 2),
        ("bitwise_not", 1),
        ("out_of_place_add", 1),
        ("multi_controlled_x", 1),
        ("controlled_subtraction", 1),
        ("less_than", 1),
        ("controlled_half", 1),
    ],
)
def test_immutable_keys_retain_dimensions_and_unsigned_defaults(operation, arity):
    n = sympy.Symbol("n", integer=True, positive=True)
    dimensions = (n,) * arity
    key = ArithmeticCall(operation, dimensions)
    assert all(w is n for w in key.widths)
    assert key == ArithmeticCall(operation, dimensions, ("unsigned",) * arity)
    assert hash(key) == hash(ArithmeticCall(operation, dimensions))
    assert not hasattr(key, "append_to")
    with pytest.raises(FrozenInstanceError):
        key.widths = (3,) * arity


def test_operand_encodings_control_signed_and_direction_are_part_of_identity():
    root = ArithmeticCall("controlled_add", (2, 3))
    variants = [
        root,
        replace(root, encodings=("montgomery", "montgomery")),
        replace(root, encodings=("unsigned", "twos_complement")),
        replace(root, control_value=0),
        replace(root, adjoint=True),
    ]
    assert len(set(variants)) == len(variants)
    assert ArithmeticCall("less_than", (3,), signed=True) != ArithmeticCall(
        "less_than", (3,)
    )


def test_python_and_symbolic_integer_dispatch_cannot_merge_in_graph_caches():
    literal = sympy.Integer(3)
    physical = ArithmeticCall("controlled_add", (3, 3))
    symbolic = ArithmeticCall("controlled_add", (literal, literal))
    assert symbolic.widths[0] is literal
    assert physical != symbolic and len({physical, symbolic}) == 2
    root = ArithmeticCall("controlled_subtraction", (4,))
    toffoli, fredkin = FixedTCall("toffoli"), FixedTCall("fredkin")
    declarations = {
        root: {physical: 1, symbolic: 1},
        physical: {toffoli: 2},
        symbolic: {fredkin: 5},
    }
    graph = symbolic_call_graph(root, declarations)
    assert graph.leaf_counts == {toffoli: 2, fredkin: 5}
    cost = query_qec_gate_cost(root, declared_calls=declarations)
    assert (cost.toffoli, cost.cswap) == (2, 5)


@pytest.mark.parametrize("leaf", [FixedTCall("toffoli"), FixedTCall("fredkin")])
def test_existing_queries_consume_replacement_declarations_without_key_changes(leaf):
    n = sympy.Symbol("n", integer=True, positive=True)
    root = ArithmeticCall("controlled_subtraction", (n,))
    child = ArithmeticCall("controlled_add", (n, n), ("montgomery", "montgomery"))
    declarations = {root: {child: 3}, child: {leaf: n}}
    graph = symbolic_call_graph(root, declarations)
    assert graph.direct_calls[root, child] == 3
    assert graph.leaf_counts == {leaf: 3 * n}
    cost = query_qec_gate_cost(root, declared_calls=declarations)
    assert (cost.toffoli, cost.cswap) == (
        (3 * n, 0) if leaf.opcode == "toffoli" else (0, 3 * n)
    )


def test_missing_arithmetic_declarations_fail_instead_of_zero_cost():
    root = ArithmeticCall("add", (2, 3))
    with pytest.raises(NotImplementedError, match="declared callees"):
        query_qec_gate_cost(root)
    with pytest.raises(NotImplementedError, match="declared callees"):
        tuple(_resource_callees(root, None))
    with pytest.raises(NotImplementedError, match="Expand arithmetic"):
        estimate_terminal_t_cost({root: 1})
    assert query_qec_gate_cost(root, static_costs={root: QECGateCost(t=17)}).t == 17


def test_declaration_counts_do_not_weaken_physical_cost_call_validation():
    root = ArithmeticCall("multi_controlled_x", (0,))
    leaf = FixedTCall("toffoli")
    declarations = {root: {leaf: -2}}
    assert symbolic_call_graph(root, declarations).leaf_counts[leaf] == -2
    assert query_qec_gate_cost(root, declared_calls=declarations).toffoli == -2
    child = ComponentCost("physical_child", 1)
    physical = ComponentCost("physical_parent", 1, calls=(CostCall(child, -2),))
    with pytest.raises(ValueError, match="multiplicity"):
        estimate_component_resources(physical)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "encodings,cv",
    tuple(
        product(
            (
                ("unsigned", "unsigned"),
                ("montgomery", "montgomery"),
                ("unsigned", "twos_complement"),
            ),
            (0, 1),
        )
    ),
)
def test_unmodified_cadd_operand_fields_map_without_sdk_types_in_native_key(
    encodings, cv
):
    from qualtran import QInt, QMontgomeryUInt, QUInt
    from qualtran.bloqs.arithmetic.controlled_addition import CAdd

    m, n = sympy.symbols("m n", integer=True, positive=True)
    types = {"unsigned": QUInt, "twos_complement": QInt, "montgomery": QMontgomeryUInt}
    source = CAdd(types[encodings[0]](m), types[encodings[1]](n), cv=cv)
    key = ArithmeticCall(
        "controlled_add",
        (source.a_dtype.bitsize, source.b_dtype.bitsize),
        encodings,
        control_value=source.cv,
    )
    assert key.widths[0] is m and key.widths[1] is n
    assert key.encodings == encodings and key.control_value == cv


@pytest.mark.parametrize(
    "kwargs",
    [
        {"operation": "unknown"},
        {"widths": (3,)},
        {"widths": [2, 3]},
        {"widths": (True, 3)},
        {"widths": (-1, 3)},
        {"widths": ("n", 3)},
        {"encodings": ["unsigned", "unsigned"]},
        {"encodings": ("unsigned",)},
        {"encodings": ("unknown", "unsigned")},
        {"control_value": True},
        {"control_value": 2},
        {"operation": "controlled_half", "widths": (3,), "control_value": 0},
        {"operation": "less_than", "widths": (3,), "encodings": ("twos_complement",)},
        {"signed": True},
        {"signed": 1},
        {"adjoint": 1},
    ],
)
def test_identity_contract_rejects_ambiguous_or_mutable_fields(kwargs):
    args = dict(operation="controlled_add", widths=(2, 3)) | kwargs
    with pytest.raises(ValueError):
        ArithmeticCall(**args)


def test_out_of_place_output_width_is_identity_and_only_valid_on_its_operation():
    full = ArithmeticCall("out_of_place_add", (3,))
    truncated = replace(full, include_most_significant_bit=False)
    assert full != truncated
    assert full.encodings == truncated.encodings == ("unsigned",)
    assert replace(truncated, adjoint=True).include_most_significant_bit is False
    with pytest.raises(ValueError, match="truncated sum"):
        ArithmeticCall("add", (3, 3), include_most_significant_bit=False)
    with pytest.raises(ValueError, match="truncated sum"):
        ArithmeticCall("out_of_place_add", (3,), include_most_significant_bit=1)
