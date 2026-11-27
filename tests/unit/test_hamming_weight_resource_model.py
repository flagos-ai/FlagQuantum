"""Original count declarations, symbolic lower bounds and phasing reuse."""

from dataclasses import fields
from itertools import product
from typing import Any

import pytest
import sympy

from flagquantum.compiler._component_trace.arithmetic_call import ArithmeticCall
from flagquantum.compiler._component_trace.resource_generalization import (
    CliffordCall,
    ConjunctionCall,
)
from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    GateCost,
    estimate_component_resources,
)
from flagquantum.compiler.registers.arithmetic.arithmetic_declarations import (
    arithmetic_callees,
)
from flagquantum.compiler.registers.bits.hamming_weight import (
    hamming_weight_resource_widths,
)
from flagquantum.compiler.registers.bits.hamming_weight_resource_model import (
    source_hamming_weight_cost,
    source_hamming_weight_resource_formula,
)
from flagquantum.core.value_math import SymbolicInt

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
n = sympy.Symbol("n", positive=True, integer=True)
z = sympy.Symbol("z", integer=True)
WIDTHS = (
    2,
    3,
    4,
    8,
    16,
    sympy.Integer(0),
    sympy.Integer(1),
    sympy.Integer(3),
    sympy.Integer(8),
    n,
    z,
    n + 1,
)


@pytest.mark.parametrize(
    "width,adjoint,legacy", tuple(product(WIDTHS, (False, True), (False, True)))
)
def test_every_source_category_and_successful_peak_with_original_width_object(
    width: SymbolicInt, adjoint: bool, legacy: bool
) -> None:
    from qualtran._infra.bloq import Bloq
    from qualtran.bloqs.arithmetic.hamming_weight import HammingWeightCompute
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value
    from qualtran.resource_counting._qubit_counts import QubitCount

    source: Bloq = HammingWeightCompute(width)
    if adjoint:
        source = source.adjoint()
    model = source_hamming_weight_resource_formula(
        width, adjoint=adjoint, legacy_shims=legacy
    )
    expected = get_cost_value(source, QECGatesCost(legacy_shims=legacy))
    for field in fields(model):
        value = (
            get_cost_value(source, QubitCount())
            if field.name == "logical_qubits"
            else getattr(expected, field.name)
        )
        assert sympy.simplify(getattr(model, field.name) - value) == 0


@pytest.mark.parametrize(
    "width,adjoint",
    tuple(
        product(
            (0, 1, 2, 3, 8, sympy.Integer(0), sympy.Integer(1), sympy.Integer(3), n, z),
            (False, True),
        )
    ),
)
def test_direct_callees_keep_source_counts_and_adjoint_identity(
    width: SymbolicInt, adjoint: bool
) -> None:
    from qualtran._infra.bloq import Bloq
    from qualtran.bloqs.arithmetic.hamming_weight import HammingWeightCompute
    from qualtran.bloqs.basic_gates.cnot import CNOT
    from qualtran.bloqs.mcmt.and_bloq import And
    from qualtran.resource_counting._call_graph import get_bloq_callee_counts

    source: Bloq = HammingWeightCompute(width)
    if adjoint:
        source = source.adjoint()
    original = dict(get_bloq_callee_counts(source, ignore_decomp_failure=False))
    root = ArithmeticCall("hamming_weight", (width,), adjoint=adjoint)
    children = arithmetic_callees(root)
    translated: dict[Any, SymbolicInt] = {}
    for child, count in children.items():
        if isinstance(child, ConjunctionCall):
            leaf: Bloq = And(uncompute=child.adjoint)
        else:
            assert isinstance(child, CliffordCall) and child.opcode == "cx"
            leaf = CNOT()
        translated[leaf] = count
    assert translated == original
    assert root.widths[0] is width


@pytest.mark.parametrize("width", WIDTHS)
def test_original_logical_junk_and_output_widths(width: SymbolicInt) -> None:
    from qualtran.bloqs.arithmetic.hamming_weight import HammingWeightCompute

    source = HammingWeightCompute(width)
    junk, output = hamming_weight_resource_widths(width)
    assert sympy.simplify(junk - source.junk_bitsize) == 0
    assert sympy.simplify(output - source.out_bitsize) == 0


@pytest.mark.parametrize(
    "width,adjoint,legacy",
    tuple(product((1, 2, 3, 4, 5, 7, 8, 16), (False, True), (False, True))),
)
def test_concrete_allocation_model_retains_count_query_failure_separately(
    width: int, adjoint: bool, legacy: bool
) -> None:
    from qualtran._infra.bloq import Bloq
    from qualtran.bloqs.arithmetic.hamming_weight import HammingWeightCompute
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value
    from qualtran.resource_counting._qubit_counts import QubitCount

    source: Bloq = HammingWeightCompute(width)
    if adjoint:
        source = source.adjoint()
    report = estimate_component_resources(
        source_hamming_weight_cost(width, adjoint=adjoint, legacy_shims=legacy)
    )
    gates = get_cost_value(source, QECGatesCost(legacy_shims=legacy))
    for f in fields(report.gates):
        assert getattr(report.gates, f.name) == getattr(gates, f.name)
    if width == 1:
        assert report.peak_logical_qubits == 2
        with pytest.raises(RuntimeError):
            get_cost_value(source, QubitCount())
    else:
        assert report.peak_logical_qubits == get_cost_value(source, QubitCount())


@pytest.mark.parametrize("width,adjoint", tuple(product((0, 1), (False, True))))
def test_source_failed_qubit_queries_remain_explicit(width: int, adjoint: bool) -> None:
    from qualtran._infra.bloq import Bloq
    from qualtran.bloqs.arithmetic.hamming_weight import HammingWeightCompute
    from qualtran.resource_counting._costing import get_cost_value
    from qualtran.resource_counting._qubit_counts import QubitCount

    source: Bloq = HammingWeightCompute(width)
    if adjoint:
        source = source.adjoint()
    with pytest.raises((RuntimeError, ValueError)):
        get_cost_value(source, QubitCount())
    with pytest.raises(ValueError, match="qubit-count"):
        source_hamming_weight_resource_formula(width, adjoint=adjoint)


def test_symbolic_lower_bound_changes_counts_without_casting_literal() -> None:
    from sympy.codegen.cfunctions import log2

    concrete = source_hamming_weight_resource_formula(3)
    literal = source_hamming_weight_resource_formula(sympy.Integer(3))
    assert (concrete.and_bloq, concrete.clifford, concrete.logical_qubits) == (1, 7, 6)
    assert (literal.and_bloq, literal.clifford, literal.logical_qubits) == (2, 11, 7)
    unknown = source_hamming_weight_resource_formula(z)
    assert (
        sympy.simplify(
            unknown.logical_qubits
            - sympy.Max(3, z, 2 * z + sympy.ceiling(log2(z + 1)) - 1)
        )
        == 0
    )


def test_phasing_composes_replaceable_count_models_without_duplicate_equations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import flagquantum.compiler.registers.phase.hamming_weight_phasing as consumer

    seen: list[tuple[int, bool, bool]] = []

    def count(
        width: int, *, adjoint: bool = False, legacy_shims: bool = False
    ) -> ComponentCost:
        seen.append((width, adjoint, legacy_shims))
        return ComponentCost(
            "replacement",
            43 if adjoint else 37,
            (
                GateCost(19, 23, 29, 31, 37, 41, 43)
                if adjoint
                else GateCost(2, 3, 5, 7, 11, 13, 17)
            ),
        )

    monkeypatch.setattr(consumer, "source_hamming_weight_cost", count)
    actual = estimate_component_resources(
        consumer.source_hamming_weight_phasing_cost(4, 1.0)
    )
    assert seen == [(4, False, False), (4, True, False)]
    assert actual.gates == GateCost(21, 26, 34, 38, 51, 54, 60)
    assert actual.peak_logical_qubits == 43


@pytest.mark.parametrize(
    "kwargs", ({"width": True}, {"width": -1}, {"adjoint": 1}, {"legacy_shims": 1})
)
def test_invalid_resource_parameters_are_not_converted(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        source_hamming_weight_resource_formula(**({"width": n} | kwargs))


def test_literal_zero_negative_counts_are_diagnostics_not_physical_gate_budgets() -> (
    None
):
    literal = source_hamming_weight_resource_formula(sympy.Integer(0))
    inverse = source_hamming_weight_resource_formula(sympy.Integer(0), adjoint=True)
    assert (literal.and_bloq, literal.clifford, literal.logical_qubits) == (-1, -4, 3)
    assert (inverse.clifford, inverse.measurement, inverse.logical_qubits) == (
        -5,
        -1,
        3,
    )
    with pytest.raises(ValueError):
        source_hamming_weight_cost(0)
