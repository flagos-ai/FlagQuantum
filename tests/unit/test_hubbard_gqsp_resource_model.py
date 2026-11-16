"""Check the source controlled-cost query and each direct component."""

from dataclasses import asdict

import pytest

from flagquantum.algorithms.primitives.hubbard_gqsp_resource_model import (
    source_controlled_hubbard_gqsp_cost,
    source_hubbard_gqsp_formula,
)
from flagquantum.algorithms.primitives.hubbard_prepare import PrepareHubbard
from flagquantum.compiler._component_trace.gate_budget import legacy_gate_budget
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_locked_controlled_hubbard_resource_query() -> None:
    from qualtran.bloqs.hamiltonian_simulation.hamiltonian_simulation_by_gqsp import (
        _hubbard_time_evolution_by_gqsp,
    )
    from qualtran.resource_counting import (
        QECGatesCost,
        QubitCount,
        SympySymbolAllocator,
        get_cost_value,
    )

    source = _hubbard_time_evolution_by_gqsp.make().controlled()
    native = source_controlled_hubbard_gqsp_cost(PrepareHubbard(2, 1, 1), 5, 1e-7)
    source_calls = source.build_call_graph(SympySymbolAllocator())
    for (child, count), call in zip(source_calls.items(), native.calls, strict=True):
        assert call.repetitions == count
        model = estimate_component_resources(call.component)
        assert {k: v for k, v in asdict(model.gates).items() if v} == get_cost_value(
            child, QECGatesCost()
        ).asdict()
        assert model.peak_logical_qubits == get_cost_value(child, QubitCount())

    actual = estimate_component_resources(native)
    assert {k: v for k, v in asdict(actual.gates).items() if v} == get_cost_value(
        source, QECGatesCost()
    ).asdict()
    assert actual.peak_logical_qubits == get_cost_value(source, QubitCount()) == 30


@pytest.mark.parametrize("symbolic", [False, True])
def test_plain_hubbard_peak_gate_totals_and_t_complexity(symbolic: bool) -> None:
    import sympy
    from qualtran.bloqs.hamiltonian_simulation.hamiltonian_simulation_by_gqsp import (
        _hubbard_time_evolution_by_gqsp,
        _symbolic_hamsim_by_gqsp,
    )
    from qualtran.resource_counting import (
        QECGatesCost,
        QubitCount,
        big_O,
        get_cost_value,
    )

    factory = _symbolic_hamsim_by_gqsp if symbolic else _hubbard_time_evolution_by_gqsp
    source = factory.make()
    native = source_hubbard_gqsp_formula(
        PrepareHubbard(2, 1, 1), source.alpha, source.t, source.precision
    )
    gates = asdict(native)
    peak = gates.pop("logical_qubits")
    assert peak == get_cost_value(source, QubitCount()) == 28
    original = get_cost_value(source, QECGatesCost())
    assert {name: value for name, value in gates.items() if value} == original.asdict()
    if symbolic:
        t = (
            native.t
            + 4 * (native.cswap + native.and_bloq + native.toffoli)
            + 11 * native.rotation
        )
        assert t == original.total_t_count()
        hopping, time, inv_eps = sympy.symbols(r"\tau t \epsilon^{-1}", positive=True)
        asymptotic = sympy.O(
            t, (hopping, sympy.oo), (time, sympy.oo), (inv_eps, sympy.oo)
        )
        assert asymptotic == big_O(
            hopping * time + sympy.log(2 * inv_eps) / sympy.log(sympy.log(2 * inv_eps))
        )
    else:
        legacy = source.t_complexity()
        budget = legacy_gate_budget(native)
        assert budget["t"] == legacy.t and budget["rotations"] == legacy.rotations
