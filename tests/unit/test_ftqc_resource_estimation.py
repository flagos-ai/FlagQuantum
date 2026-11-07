"""Exercise the same resource estimate before and after circuit construction."""

import sys
from dataclasses import replace

import pytest

from examples.algorithms import ftqc_resource_estimation
from examples.algorithms.ftqc_resource_estimation import make_circuit, make_select
from flagquantum.algorithms.primitives.multiplexers import SelectTHC
from flagquantum.algorithms.primitives.multiplexers.thc_resource_model import (
    select_thc_cost,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.compiler.registers.symbolic_resource_model import (
    SymbolicResourceFormula,
    source_qrom_resource_formula,
)
from flagquantum.qec.resources.estimation import (
    estimate_ftqc_resources,
    estimate_symbolic_ftqc_resources,
)

pytestmark = pytest.mark.unit


def test_thc_cli_estimates_dimensions_without_building_ir(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def unexpected_circuit() -> None:
        pytest.fail("Dimension-only THC estimation must not build a circuit")

    monkeypatch.setattr(ftqc_resource_estimation, "make_circuit", unexpected_circuit)
    monkeypatch.setattr(
        sys,
        "argv",
        ["estimate", "--thc", "2", "4", "8", "1", "--thc-erasure-blocks", "2", "1"],
    )
    ftqc_resource_estimation.main()
    assert capsys.readouterr().out.splitlines() == [
        "Model Toffoli gates 106",
        "Model controlled swaps 11",
        "Model measurements 0",
        "Model T states 0",
        "Model CCZ states 117",
        "Model T equivalents 468",
        "Model peak logical qubits 24",
    ]


def test_model_and_emitted_counts_are_independent() -> None:
    op = make_select()
    model = op.source_resource_model(keep_bitsize=1)
    assumptions = dict(error_budget=1e-3, physical_error=1e-4)
    planned = estimate_ftqc_resources(model, **assumptions)
    report = estimate_ftqc_resources(
        model, emitted_ir=make_circuit(op).to_ir(), **assumptions
    )

    assert report.component == planned.component == "SelectTHC"
    assert planned.emitted is None
    assert report.logical == planned.logical
    assert report.logical.gates.total_t_count() == 472
    assert report.logical.peak_logical_qubits == 24
    assert report.logical.gates.measurement == 0
    assert report.surface_code == planned.surface_code
    assert report.surface_code is not None
    assert report.surface_code.data_block_physical_qubits == 6174
    assert report.emitted is not None
    assert report.emitted.t_count == 112
    assert report.emitted.declared_qubits == 39
    assert report.emitted.unconditional["measure"] == 60
    assert report.emitted.unconditional["ccx"] == 49
    assert report.emitted.unconditional["cswap"] == 11


def test_bound_thc_model_does_not_lower(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_lowering(*args: object, **kwargs: object) -> None:
        pytest.fail("The source estimate must not lower the operation")

    monkeypatch.setattr(SelectTHC, "_append_to", unexpected_lowering)
    op = replace(make_select(), workspace=(), measurement_erasure=False)
    assert op.source_resource_model(keep_bitsize=3, kr1=2, kr2=1) == (
        select_thc_cost(2, 4, 8, 3, kr1=2, kr2=1)
    )


def test_surface_code_assumptions_are_optional_and_paired() -> None:
    model = select_thc_cost(2, 4, 8, 1)
    report = estimate_ftqc_resources(model)
    assert report.surface_code is None
    assert report.emitted is None
    with pytest.raises(ValueError, match="both error_budget and physical_error"):
        estimate_ftqc_resources(model, error_budget=1e-3)
    with pytest.raises(ValueError, match="both error_budget and physical_error"):
        estimate_ftqc_resources(model, physical_error=1e-4)


def test_operation_model_can_be_used_without_lowering() -> None:
    model = select_thc_cost(2, 4, 8, 1)

    class Operation:
        def resource_model(self) -> ComponentCost:
            return model

        def to_ir(self) -> None:
            pytest.fail("A model-only estimate must not request IR")

    assert estimate_ftqc_resources(Operation()) == estimate_ftqc_resources(model)


def test_ir_alone_does_not_supply_an_algorithm_model() -> None:
    with pytest.raises(TypeError, match="concrete ComponentCost"):
        estimate_ftqc_resources(make_circuit().to_ir())  # type: ignore[arg-type]


def test_symbolic_report_uses_the_concrete_conversion_policy() -> None:
    gates = GateCost(
        t=2, toffoli=3, cswap=4, and_bloq=5, clifford=6, rotation=7, measurement=8
    )
    formula = SymbolicResourceFormula(
        and_bloq=gates.and_bloq,
        clifford=gates.clifford,
        measurement=gates.measurement,
        logical_qubits=10,
        t=gates.t,
        toffoli=gates.toffoli,
        cswap=gates.cswap,
        rotation=gates.rotation,
    )
    report = estimate_symbolic_ftqc_resources(formula)
    assert report.logical == formula
    assert report.t_states == 79
    assert report.ccz_states == 12
    assert report.t_equivalents == gates.total_t_count()


@pytest.mark.ftqc_reference
def test_symbolic_qrom_report_matches_the_pinned_source() -> None:
    pytest.importorskip("qualtran")
    sympy = pytest.importorskip("sympy")
    from qualtran.bloqs.data_loading.qrom import QROM
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    rows = sympy.Symbol("rows", integer=True, positive=True)
    report = estimate_symbolic_ftqc_resources(
        source_qrom_resource_formula((rows,), (8,), control_count=1)
    )
    source = QROM.build_from_bitsize((rows,), (8,), num_controls=1)
    gates = get_cost_value(source, QECGatesCost())
    magic = gates.total_t_and_ccz_count()
    assert sympy.simplify(report.t_equivalents - gates.total_t_count()) == 0
    assert sympy.simplify(report.t_states - magic["n_t"]) == 0
    assert sympy.simplify(report.ccz_states - magic["n_ccz"]) == 0
    assert sympy.simplify(report.logical.measurement - gates.measurement) == 0
    assert (
        sympy.simplify(
            report.logical.logical_qubits - get_cost_value(source, QubitCount())
        )
        == 0
    )
    assert report.t_equivalents.subs(rows, 17) == 64
    assert report.logical.measurement.subs(rows, 17) == 16
    assert report.logical.logical_qubits.subs(rows, 17) == 19
