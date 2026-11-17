"""Compare periodic Hamiltonian construction and symbolic walk costs."""

import pytest

from flagquantum.algorithms.primitives.ising_hamiltonian import (
    ising_hamiltonian,
    ising_lcu_coefficients,
    periodic_ising_terms,
)
from flagquantum.algorithms.primitives.ising_walk import ising_norm_bound, ising_walk
from flagquantum.algorithms.primitives.pauli_hamiltonian_walk import (
    PauliHamiltonianWalk,
    SymbolicPauliHamiltonianWalk,
)
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from tests.unit.test_first_quantization_resource_model import assert_source_model

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("labels", [(9,), (9, 2), (9, 2, 7, 1)])
@pytest.mark.parametrize("strengths", [(1.0, -0.5), (0.0, 2.0), (-2.0, 0.0)])
def test_raw_periodic_ising_terms_match_source(labels, strengths) -> None:
    pytest.importorskip("qualtran")
    import cirq
    from qualtran.bloqs.chemistry.ising.hamiltonian import (
        get_1d_ising_hamiltonian,
        get_1d_ising_pauli_terms,
    )

    def source_terms(source):
        return [
            (
                term.coefficient.real,
                tuple(
                    sorted((wire.x, str(pauli).lower()) for wire, pauli in term.items())
                ),
            )
            for term in source
        ]

    wires = tuple(map(cirq.LineQubit, labels))
    original = get_1d_ising_pauli_terms(wires, *strengths)
    for group, source in zip(
        periodic_ising_terms(labels, *strengths), original, strict=True
    ):
        assert [
            (term.coefficient, tuple(sorted(term.ops))) for term in group
        ] == source_terms(source)
    merged = ising_hamiltonian(labels, *strengths)
    assert [(term.coefficient, term.ops) for term in merged.terms] == source_terms(
        get_1d_ising_hamiltonian(wires, *strengths)
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("sites", [0, 1, 2, 4])
def test_ising_lcu_coefficients_match_merged_source_order(sites) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.ising.hamiltonian import get_1d_ising_lcu_coeffs

    assert ising_lcu_coefficients(sites) == tuple(get_1d_ising_lcu_coeffs(sites))
    for strengths in ((-0.75, 1.5), (0, 1), (1, 0), (0, 0)):
        assert ising_lcu_coefficients(sites, *strengths) == tuple(
            get_1d_ising_lcu_coeffs(sites, *strengths)
        )


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


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    ("sites", "j_zz", "gamma_x"),
    [(1, 1, -1), (2, 1, -1), (3, -0.75, 1), (4, 1, -1), (8, 1, -1), (16, 1, -1)],
)
def test_symbolic_ising_peak_specializes_to_concrete_decomposition(
    sites: int, j_zz: float, gamma_x: float
) -> None:
    sympy = pytest.importorskip("sympy")
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.ising.walk_operator import (
        get_walk_operator_for_1d_ising_model,
    )
    from qualtran.resource_counting import QubitCount, get_cost_value

    precision = sympy.Symbol("e", positive=True)
    symbolic, _ = ising_walk(sites, precision, j_zz=j_zz, gamma_x=gamma_x)
    assert isinstance(symbolic, SymbolicPauliHamiltonianWalk)
    prepare_peak = symbolic.prepare.dense_decomposition_peak()
    walk_peak = symbolic.peak_logical_qubits()
    assert prepare_peak.free_symbols == walk_peak.free_symbols == {precision}
    for epsilon in (1.0, 0.3, 0.1, 0.01):
        source, _ = get_walk_operator_for_1d_ising_model(
            sites, epsilon, j_zz_strength=j_zz, gamma_x_strength=gamma_x
        )
        concrete, _ = ising_walk(sites, epsilon, j_zz=j_zz, gamma_x=gamma_x)
        cost = estimate_component_resources(concrete.resource_model())
        assert prepare_peak.subs(precision, epsilon) == get_cost_value(
            source.prepare, QubitCount()
        )
        assert (
            walk_peak.subs(precision, epsilon)
            == cost.peak_logical_qubits
            == (get_cost_value(source, QubitCount()))
        )
