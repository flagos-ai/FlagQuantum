"""Locked out-of-place addition declarations, categories and width dispatch."""

from dataclasses import fields
from itertools import product

import pytest
import sympy

from flagquantum.compiler._component_trace.arithmetic_call import ArithmeticCall
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import (
    CliffordCall,
    ConjunctionCall,
)
from flagquantum.compiler.registers.arithmetic_declarations import arithmetic_call_graph

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
WIDTHS = (
    1,
    3,
    8,
    sympy.Integer(0),
    sympy.Integer(3),
    sympy.Symbol("n", integer=True, positive=True),
    sympy.Symbol("p", integer=True, nonnegative=True),
    sympy.Symbol("z", integer=True),
)


@pytest.mark.parametrize(
    "width,msb,adjoint", tuple(product(WIDTHS, (False, True), (False, True)))
)
def test_original_direct_edges_and_recursive_qec_queries(width, msb, adjoint):
    from qualtran.bloqs.arithmetic.addition import OutOfPlaceAdder
    from qualtran.bloqs.basic_gates import CNOT
    from qualtran.bloqs.mcmt import And
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    root = ArithmeticCall(
        "out_of_place_add", (width,), adjoint=adjoint, include_most_significant_bit=msb
    )
    source = OutOfPlaceAdder(
        width, is_adjoint=adjoint, include_most_significant_bit=msb
    )
    rows = arithmetic_call_graph(root)
    converted = {}
    for child, count in rows[root].items():
        if isinstance(child, ConjunctionCall):
            converted[And(uncompute=child.adjoint)] = count
        else:
            assert child == CliffordCall("cx", 2)
            converted[CNOT()] = count
    assert list(converted.items()) == list(source.build_call_graph(None).items())
    for legacy in (False, True):
        actual = query_qec_gate_cost(root, declared_calls=rows, legacy_shims=legacy)
        expected = get_cost_value(source, QECGatesCost(legacy_shims=legacy))
        for field in fields(actual):
            assert (
                sympy.simplify(
                    getattr(actual, field.name) - getattr(expected, field.name)
                )
                == 0
            )


@pytest.mark.parametrize(
    "width,msb,adjoint", tuple(product(WIDTHS, (False, True), (False, True)))
)
def test_shared_formula_matches_all_source_fields_and_width_queries(
    width, msb, adjoint
):
    from qualtran.bloqs.arithmetic.addition import OutOfPlaceAdder
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from flagquantum.compiler.registers.out_of_place_add_resource_model import (
        source_out_of_place_add_resource_formula,
    )

    source = OutOfPlaceAdder(
        width, is_adjoint=adjoint, include_most_significant_bit=msb
    )
    for legacy in (False, True):
        actual = source_out_of_place_add_resource_formula(
            width, include_msb=msb, adjoint=adjoint, legacy_shims=legacy
        )
        gates = get_cost_value(source, QECGatesCost(legacy_shims=legacy))
        for field in fields(actual):
            expected = (
                get_cost_value(source, QubitCount())
                if field.name == "logical_qubits"
                else getattr(gates, field.name)
            )
            assert sympy.simplify(getattr(actual, field.name) - expected) == 0


@pytest.mark.parametrize(
    "width,msb,adjoint,legacy",
    tuple(product((1, 2, 3, 5, 8), (False, True), (False, True), (False, True))),
)
def test_concrete_component_is_the_same_shared_model(width, msb, adjoint, legacy):
    from flagquantum.compiler._component_trace.resource_model import (
        estimate_component_resources,
    )
    from flagquantum.compiler.registers.out_of_place_add_resource_model import (
        source_out_of_place_add_cost,
        source_out_of_place_add_resource_formula,
    )

    options = dict(include_msb=msb, adjoint=adjoint, legacy_shims=legacy)
    expected = source_out_of_place_add_resource_formula(width, **options)
    actual = estimate_component_resources(
        source_out_of_place_add_cost(width, **options)
    )
    assert actual.peak_logical_qubits == expected.logical_qubits == 3 * width + int(msb)
    for field in fields(actual.gates):
        assert getattr(actual.gates, field.name) == getattr(expected, field.name)


def test_source_python_zero_failure_and_symbolic_zero_fallback_are_distinct():
    from qualtran.bloqs.arithmetic.addition import OutOfPlaceAdder
    from qualtran.resource_counting import QubitCount, get_cost_value

    from flagquantum.compiler.registers.out_of_place_add_resource_model import (
        source_out_of_place_add_resource_formula,
    )

    for msb in (False, True):
        with pytest.raises(RuntimeError):
            get_cost_value(
                OutOfPlaceAdder(0, include_most_significant_bit=msb), QubitCount()
            )
        with pytest.raises(ValueError, match="zero-width"):
            source_out_of_place_add_resource_formula(0, include_msb=msb)
        symbolic = source_out_of_place_add_resource_formula(
            sympy.Integer(0), include_msb=msb
        )
        assert symbolic.logical_qubits == 3
        assert symbolic.and_bloq == int(msb) - 1


@pytest.mark.parametrize(
    "options",
    [
        dict(width=True),
        dict(width=-1),
        dict(width="3"),
        dict(include_msb=1),
        dict(adjoint=1),
        dict(legacy_shims=1),
    ],
)
def test_resource_query_validation_does_not_coerce_bad_values(options):
    from flagquantum.compiler.registers.out_of_place_add_resource_model import (
        source_out_of_place_add_resource_formula,
    )

    with pytest.raises(ValueError):
        source_out_of_place_add_resource_formula(**(dict(width=3) | options))


def test_formula_uses_replaceable_declared_graph_without_changing_consumer(monkeypatch):
    import flagquantum.compiler.registers.out_of_place_add_resource_model as module
    from flagquantum.compiler._component_trace.resource_generalization import FixedTCall

    n = sympy.Symbol("n", integer=True, positive=True)
    seen = []

    def declarations(root):
        seen.append(root)
        return {root: {FixedTCall("fredkin"): 7 * n}}

    monkeypatch.setattr(module, "arithmetic_call_graph", declarations)
    actual = module.source_out_of_place_add_resource_formula(
        n, include_msb=False, adjoint=True
    )
    assert seen == [
        ArithmeticCall(
            "out_of_place_add", (n,), adjoint=True, include_most_significant_bit=False
        )
    ]
    assert actual.cswap == 7 * n and actual.logical_qubits == 3 * n
    assert actual.and_bloq == actual.clifford == actual.measurement == 0


@pytest.mark.parametrize(
    "width,msb,adjoint", tuple(product((1, 3, 8), (False, True), (False, True)))
)
def test_shared_model_t_and_measurements_close_against_emitted_primitives(
    width, msb, adjoint
):
    from collections import Counter

    from flagquantum.compiler.registers.out_of_place_add_resource_model import (
        source_out_of_place_add_resource_formula,
    )
    from flagquantum.compiler.registers.out_of_place_addition import out_of_place_add_ir

    left, right = tuple(range(width)), tuple(range(width, 2 * width))
    output = tuple(range(2 * width, 3 * width + int(msb)))
    slots = tuple(range(len(output) - 1)) if adjoint else ()
    counts = Counter(
        g.name
        for g in out_of_place_add_ir(
            left, right, output, include_msb=msb, adjoint=adjoint, classical_bits=slots
        )
    )
    model = source_out_of_place_add_resource_formula(
        width, include_msb=msb, adjoint=adjoint
    )
    assert counts["t"] + counts["tdg"] == 4 * model.and_bloq
    assert counts["measure"] == model.measurement
    assert model.logical_qubits == 2 * width + len(output)
