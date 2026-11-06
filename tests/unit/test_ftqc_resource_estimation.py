"""Exercise the same resource estimate before and after circuit construction."""

import pytest

from examples.algorithms.ftqc_resource_estimation import make_circuit
from flagquantum.algorithms.primitives.multiplexers.thc_resource_model import (
    select_thc_cost,
)
from flagquantum.qec.resources.estimation import estimate_ftqc_resources

pytestmark = pytest.mark.unit


def test_model_and_emitted_counts_are_independent() -> None:
    model = select_thc_cost(2, 4, 8, 1)
    assumptions = dict(error_budget=1e-3, physical_error=1e-4)
    planned = estimate_ftqc_resources(model, **assumptions)
    report = estimate_ftqc_resources(
        model, emitted_ir=make_circuit().to_ir(), **assumptions
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


def test_surface_code_assumptions_are_optional_and_paired() -> None:
    model = select_thc_cost(2, 4, 8, 1)
    report = estimate_ftqc_resources(model)
    assert report.surface_code is None
    assert report.emitted is None
    with pytest.raises(ValueError, match="both error_budget and physical_error"):
        estimate_ftqc_resources(model, error_budget=1e-3)
    with pytest.raises(ValueError, match="both error_budget and physical_error"):
        estimate_ftqc_resources(model, physical_error=1e-4)
