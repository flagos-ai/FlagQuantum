"""CAdd resource composition follows original predicate/Add calls and lifetimes."""

from dataclasses import asdict
from itertools import product

import pytest

from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    GateCost,
    estimate_component_resources,
)
from flagquantum.compiler.registers.arithmetic_resource_model import (
    source_controlled_add_cost,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "widths,cv,legacy,dtype",
    tuple(
        product(
            ((1, 1), (1, 2), (2, 2), (2, 3), (3, 4), (4, 4), (5, 6)),
            (0, 1),
            (False, True),
            ("uint", "montgomery"),
        )
    ),
)
def test_all_original_categories_and_peak_both_directions(widths, cv, legacy, dtype):
    from qualtran import QMontgomeryUInt, QUInt
    from qualtran.bloqs.arithmetic.controlled_addition import CAdd
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    m, n = widths
    native = source_controlled_add_cost(m, n, control_value=cv, legacy_shims=legacy)
    report = estimate_component_resources(native)
    word = QUInt if dtype == "uint" else QMontgomeryUInt
    source = CAdd(word(m), word(n), cv=cv)
    for operation in (source, source.adjoint()):
        expected = get_cost_value(operation, QECGatesCost(legacy_shims=legacy))
        assert {k: v for k, v in asdict(report.gates).items() if v} == expected.asdict()
        assert report.peak_logical_qubits == get_cost_value(operation, QubitCount())
    original_calls = source.build_call_graph(None)
    assert sorted(original_calls.values()) == sorted((m, 1, m))
    assert tuple(call.repetitions for call in native.calls) == (m, 1, m)
    assert native.calls[1].component.name == "Add"
    if m == n == 1 and not legacy:
        assert report.gates.clifford == -2


@pytest.mark.parametrize("cv,legacy", tuple(product((0, 1), (False, True))))
def test_shared_model_replacement_retains_stage_order_and_live_bystanders(
    monkeypatch, cv, legacy
):
    import flagquantum.compiler.registers.arithmetic_resource_model as module

    add = ComponentCost("replacement_add", 19, GateCost(t=5))
    compute = ComponentCost("replacement_predicate", 7, GateCost(clifford=11))
    cleanup = ComponentCost("replacement_cleanup", 7, GateCost(measurement=1))
    calls = []

    def replace_add(m, n, *, legacy_shims=False):
        calls.append(("add", m, n, legacy_shims))
        return add

    def replace_predicate(values, *, uncompute=False, legacy_shims=False):
        calls.append(("predicate", values, uncompute, legacy_shims))
        return cleanup if uncompute else compute

    monkeypatch.setattr(module, "source_add_cost", replace_add)
    monkeypatch.setattr(module, "control_predicate_cost", replace_predicate)
    model = module.source_controlled_add_cost(
        2, 3, control_value=cv, legacy_shims=legacy
    )
    assert calls == [
        ("add", 2, 3, legacy),
        ("predicate", (cv, 1), False, legacy),
        ("predicate", (cv, 1), True, legacy),
    ]
    assert tuple(call.component for call in model.calls) == (compute, add, cleanup)
    assert tuple(call.bystander_qubits for call in model.calls) == (5, 3, 5)
    report = estimate_component_resources(model)
    assert (report.gates.t, report.gates.clifford, report.gates.measurement) == (
        5,
        22,
        2,
    )
    assert report.peak_logical_qubits == 22


@pytest.mark.parametrize(
    "m,n,cv,legacy",
    [
        (0, 1, 1, False),
        (2, 1, 1, False),
        (True, 2, 1, False),
        (1, 2, True, False),
        (1, 2, 2, False),
        (1, 2, 1, 0),
    ],
)
def test_model_parameters_are_explicit(m, n, cv, legacy):
    with pytest.raises(ValueError):
        source_controlled_add_cost(m, n, control_value=cv, legacy_shims=legacy)
