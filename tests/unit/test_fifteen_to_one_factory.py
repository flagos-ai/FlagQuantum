"""Validate the five-qubit factory through the native density-matrix executor."""

from dataclasses import asdict

import numpy as np
import pytest
import torch

from flagquantum.compiler._component_trace.resource_model import GateCost
from flagquantum.qec.resources.fifteen_to_one import FifteenToOneFactory
from flagquantum.simulation.density_matrix import density_matrix_from_ir

pytestmark = pytest.mark.unit


def test_factory_paper_example() -> None:
    factory = FifteenToOneFactory(7, 3, 3)
    result = factory.estimate(GateCost(t=1), physical_error=1e-4, error_rate_scaler=0.1)
    assert result.physical_qubits == 810
    assert result.cycles == 19
    assert result.conditional_output_error == pytest.approx(4.4e-8, rel=0.02)
    assert result.factory_error == result.conditional_output_error


def test_ideal_factory_and_validation() -> None:
    factory = FifteenToOneFactory(9, 3, 3)
    result = factory.estimate(GateCost(), physical_error=0, error_rate_scaler=0.03)
    assert result.rejection_probability == pytest.approx(0, abs=1e-14)
    assert result.conditional_output_error == pytest.approx(0, abs=1e-14)
    assert result.cycles == result.factory_error == 0
    with pytest.raises(ValueError, match="factory distances"):
        FifteenToOneFactory(10, 3, 3)
    with pytest.raises(ValueError, match="physical error"):
        factory.estimate(GateCost(t=1), physical_error=0.01, error_rate_scaler=0.03)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    ("distances", "physical_error", "scaler"),
    [
        ((7, 3, 3), 1e-4, 0.1),
        ((9, 3, 3), 1e-4, 0.1),
        ((11, 5, 5), 1e-4, 0.1),
        ((17, 7, 7), 1e-3, 0.1),
        ((9, 3, 3), 1e-3, 0.03),
    ],
)
def test_factory_schedule_and_estimate_match_source(
    distances: tuple[int, int, int], physical_error: float, scaler: float
) -> None:
    cirq = pytest.importorskip("cirq")
    pytest.importorskip("qualtran")
    from qualtran.resource_counting import GateCounts
    from qualtran.surface_code import FifteenToOne, LogicalErrorModel, QECScheme
    from qualtran.surface_code.fifteen_to_one_factory import _build_factory

    factory = FifteenToOneFactory(*distances)
    gates = GateCost(t=3, cswap=7, toffoli=2, and_bloq=5, rotation=11, measurement=16)
    source_model = LogicalErrorModel(
        physical_error=physical_error,
        qec_scheme=QECScheme(error_rate_scaler=scaler, error_rate_threshold=0.01),
    )
    source = FifteenToOne(*distances)
    dx, dz, dm = distances
    source_ir = _build_factory(d_X=dx, d_Z=dz, d_m=dm, logical_error_model=source_model)
    ir = factory.to_ir(physical_error=physical_error, error_rate_scaler=scaler)
    assert len(ir.instructions) == 100
    for op, source_op in zip(ir.instructions, source_ir.all_operations(), strict=True):
        assert op.wires == tuple(q.x for q in source_op.qubits)
        if op.name == "h":
            assert source_op.gate == cirq.H
        else:
            # Cirq stores all three Pauli error slots, including zero weights.
            expected = [matrix for matrix in cirq.kraus(source_op) if np.any(matrix)]
            actual = [matrix.numpy() for matrix in op.matrix if bool(torch.any(matrix))]
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)

    density = density_matrix_from_ir(ir, device="cpu", dtype=torch.complex128)[0]
    np.testing.assert_allclose(
        density.numpy(), source._final_state(source_model), rtol=0, atol=2e-15
    )
    result = factory.estimate(
        gates, physical_error=physical_error, error_rate_scaler=scaler
    )
    source_gates = GateCounts(**asdict(gates))
    assert result.rejection_probability == pytest.approx(
        source.p_fail(source_model), abs=2e-14
    )
    assert result.conditional_output_error == pytest.approx(
        source.p_out(source_model), abs=2e-14
    )
    assert result.factory_error == pytest.approx(
        source.factory_error(source_gates, source_model), abs=4e-12
    )
    assert result.physical_qubits == source.n_physical_qubits()
    assert result.cycles == source.n_cycles(source_gates, source_model)
