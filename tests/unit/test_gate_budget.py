"""Compare gate budget conventions to locked Qualtran GateCounts."""

import pytest

from flagquantum.compiler._component_trace.gate_budget import (
    GateCategories,
    beverland_gate_budget,
    legacy_gate_budget,
    t_and_ccz_budget,
    toffoli_only_budget,
)
from flagquantum.compiler._component_trace.resource_model import GateCost
from flagquantum.compiler.registers.arithmetic.arithmetic_resource_model import (
    source_add_cost,
)
from flagquantum.compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
    source_add_resource_formula,
)

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("symbolic", [False, True])
def test_budget_conventions_and_custom_rates_match_source(symbolic: bool) -> None:
    pytest.importorskip("qualtran")
    from qualtran.resource_counting import GateCounts
    from sympy import Symbol

    n = Symbol("n", positive=True, integer=True) if symbolic else 3
    expected = GateCounts(
        t=2, toffoli=n, cswap=3, and_bloq=4, clifford=5, rotation=6, measurement=7
    )
    gates: GateCategories
    if symbolic:
        gates = SymbolicResourceFormula(
            and_bloq=4,
            clifford=5,
            measurement=7,
            logical_qubits=8,
            t=2,
            toffoli=n,
            cswap=3,
            rotation=6,
        )
    else:
        gates = GateCost(
            t=2, toffoli=3, cswap=3, and_bloq=4, clifford=5, rotation=6, measurement=7
        )
    assert t_and_ccz_budget(gates) == expected.total_t_and_ccz_count()
    assert t_and_ccz_budget(
        gates, ts_per_rotation=23
    ) == expected.total_t_and_ccz_count(23)
    assert legacy_gate_budget(gates) == expected.to_legacy_t_complexity().asdict()
    assert (
        legacy_gate_budget(
            gates,
            ts_per_toffoli=5,
            ts_per_cswap=6,
            ts_per_and=8,
            cliffords_per_and=11,
            cliffords_per_cswap=13,
        )
        == expected.to_legacy_t_complexity(5, 6, 8, 11, 13).asdict()
    )
    assert beverland_gate_budget(gates) == expected.total_beverland_count()
    assert legacy_gate_budget(gates)["t"] == 39 + 4 * n


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("symbolic", [False, True])
def test_existing_add_providers_feed_budget_conversion(symbolic: bool) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add
    from qualtran.resource_counting import QECGatesCost, get_cost_value
    from sympy import Symbol

    if symbolic:
        width = Symbol("n", positive=True, integer=True)
        gates: GateCategories = source_add_resource_formula(width, width)
    else:
        width = 5
        gates = source_add_cost(width, width).gates
    expected = get_cost_value(Add(QUInt(width), QUInt(width)), QECGatesCost())
    assert t_and_ccz_budget(gates) == expected.total_t_and_ccz_count()
    assert toffoli_only_budget(gates) == expected.total_toffoli_only() == width - 1
    assert legacy_gate_budget(gates) == expected.to_legacy_t_complexity().asdict()
    assert beverland_gate_budget(gates) == expected.total_beverland_count()


@pytest.mark.ftqc_reference
def test_toffoli_only_requires_proven_zero_t_and_rotations() -> None:
    pytest.importorskip("qualtran")
    from qualtran.resource_counting import GateCounts
    from sympy import Symbol, cos, sin

    n = Symbol("n", integer=True)
    theta = Symbol("theta", real=True)
    zero = sin(theta) ** 2 + cos(theta) ** 2 - 1
    assert zero != 0
    gates = SymbolicResourceFormula(2, 4, 6, 8, t=zero, toffoli=n, cswap=1)
    source = GateCounts(
        t=zero, toffoli=n, cswap=1, and_bloq=2, clifford=4, measurement=6
    )
    assert toffoli_only_budget(gates) == source.total_toffoli_only() == n + 3
    for t, rotation, message in ((n, 0, "T counts"), (0, n, "rotations")):
        gates = SymbolicResourceFormula(0, 0, 0, 1, t=t, rotation=rotation)
        with pytest.raises(ValueError, match=message):
            GateCounts(t=t, rotation=rotation).total_toffoli_only()
        with pytest.raises(ValueError, match=message):
            toffoli_only_budget(gates)
