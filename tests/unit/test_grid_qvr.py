"""Test the opaque grid QVR cost and the native exact phase convention."""

from cmath import exp

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.algorithms.primitives.grid_qvr import grid_qvr_ir, source_grid_qvr_cost
from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width", [1, 4, 12])
def test_original_qvr_cost(width):
    pytest.importorskip("qualtran")
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.chemistry.trotter.grid_ham.qvr import QuantumVariableRotation
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = QuantumVariableRotation(width)
    native = estimate_component_resources(source_grid_qvr_cost(width))
    expected = get_cost_value(source, QECGatesCost())
    assert native.gates.rotation == expected.rotation == width
    assert native.gates.total_t_count() == expected.total_t_count()
    assert native.peak_logical_qubits == get_cost_value(source, QubitCount())
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()


@pytest.mark.parametrize("coefficient", [0, -0.375, 0.125])
def test_exact_grid_qvr_absolute_phase_and_order(coefficient):
    wires = (2, 0, 1)
    gates = grid_qvr_ir(wires, radians_per_unit=coefficient)
    assert [(g.wires[0], g.params["theta"]) for g in gates] == [
        (1, coefficient),
        (0, 2 * coefficient),
        (2, 4 * coefficient),
    ]
    state = torch.tensor([1, 2j, -3, -4j, 5, 6j, -7, -8j], dtype=torch.complex128)
    state /= state.norm()
    output = (
        Circuit(3, inputs=state, dtype=torch.complex128)
        .append(FieldOperation(3, gates))
        .state()
        .flatten()
    )
    factors = torch.tensor(
        [
            exp(1j * coefficient * ((i & 1) * 4 + (i >> 2) * 2 + ((i >> 1) & 1)))
            for i in range(8)
        ],
        dtype=torch.complex128,
    )
    torch.testing.assert_close(output, factors * state, atol=1e-12, rtol=0)
