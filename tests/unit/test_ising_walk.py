"""Compare periodic Hamiltonian construction and symbolic walk costs."""

import pytest

from flagquantum.algorithms.primitives.ising_walk import ising_norm_bound, ising_walk
from flagquantum.algorithms.primitives.pauli_hamiltonian_walk import (
    PauliHamiltonianWalk,
    SymbolicPauliHamiltonianWalk,
)
from tests.unit.test_first_quantization_resource_model import assert_source_model

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
def test_symbolic_ising_bound_matches_source() -> None:
    sympy = pytest.importorskip("sympy")
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.ising.hamiltonian import (
        get_1d_ising_hamiltonian_norm_upper_bound,
    )

    sites, j_zz, gamma_x = sympy.symbols("N J G", positive=True)
    assert ising_norm_bound(
        sites, j_zz, gamma_x
    ) == get_1d_ising_hamiltonian_norm_upper_bound(sites, j_zz, gamma_x)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    ("sites", "j_zz", "gamma_x"), [(1, 1, -1), (2, 1, -1), (3, -0.75, 1), (4, 1, -1)]
)
@pytest.mark.parametrize("symbolic", [False, True])
def test_ising_walk_matches_source(
    sites: int, j_zz: float, gamma_x: float, symbolic: bool
) -> None:
    sympy = pytest.importorskip("sympy")
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.ising.walk_operator import (
        get_walk_operator_for_1d_ising_model,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    eps = sympy.Symbol("e", positive=True) if symbolic else 0.3
    plan, ham = ising_walk(sites, eps, j_zz=j_zz, gamma_x=gamma_x)
    source, source_ham = get_walk_operator_for_1d_ising_model(
        sites, eps, j_zz_strength=j_zz, gamma_x_strength=gamma_x
    )
    assert [(term.coefficient, term.ops) for term in ham.terms] == [
        (
            term.coefficient.real,
            tuple(sorted((q.x, str(pauli).lower()) for q, pauli in term.items())),
        )
        for term in source_ham
    ]
    source_select = source.block_encoding.select
    assert plan.target_qubits == tuple(range(sites))
    assert plan.selector.select_unitaries == tuple(
        ("-" if term.coefficient.real < 0 else "")
        + "".join("IXYZ"[i] for i in term.pauli_mask)
        for term in source_select.select_unitaries
    )
    if symbolic:
        assert isinstance(plan, SymbolicPauliHamiltonianWalk)
        assert plan.prepare.normalization == source.sum_of_lcu_coefficients
        assert sympy.simplify(plan.prepare.probability_bits - source.prepare.mu) == 0
        gates = plan.gate_cost()
        for key, expected in get_cost_value(source, QECGatesCost()).asdict().items():
            assert sympy.simplify(getattr(gates, key) - expected) == 0
        with pytest.raises(RuntimeError, match="cannot determine truth value"):
            get_cost_value(source, QubitCount())
    else:
        assert isinstance(plan, PauliHamiltonianWalk)
        assert plan.table.mu == source.prepare.mu
        assert plan.table.alternates == tuple(source.prepare.alt)
        assert plan.table.keep == tuple(source.prepare.keep)
        assert_source_model(plan.resource_model(), source)
