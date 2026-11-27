"""Default SDK synthesis remains executable through owned directional snapshots."""

import sys
from collections import Counter
from math import sqrt

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.algorithms.primitives.circuit_register_emitter import (
    CircuitRegisterEmitter,
)
from flagquantum.circuit import Circuit
from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", range(1, 14, 2))
def test_original_sdk_forward_and_defective_adjoint_survive_serialization_and_append(
    width: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from importlib.metadata import version

    import kickmix
    from qualtran.bloqs.phase_estimation.lp_resource_state import LPResourceState

    assert version("qualtran") == "0.7.0"
    assert getattr(kickmix, "__version__") == "0.1" and hasattr(kickmix, "Simulator")
    source = LPResourceState(width)
    expected = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    inverse = torch.as_tensor(
        source.adjoint().tensor_contract(), dtype=torch.complex128
    )
    torch.testing.assert_close(inverse, -expected.conj(), atol=2e-13, rtol=0)
    captured = import_register_emitter(source, local_ports=("qpe_reg",))
    forward, backward = (
        CircuitIR.from_json(snapshot.to_json()) for snapshot in captured.snapshots
    )
    emitter = CircuitRegisterEmitter(
        captured.layout, (forward, backward), captured.binding_swaps
    )
    work, slots = emitter.requirements()
    assert work == 3 and slots == 0
    assert any(g.name == "any" for g in forward.instructions)
    # Conversion is finished. These are owned matrices and IR, not an SDK call
    # retained by the emitter. The native canonical algorithm stays independent.
    for name in ("qualtran", "cirq", "kickmix"):
        monkeypatch.setitem(sys.modules, name, None)
    total = 1 + width + work
    phase = tuple(reversed(range(1, width + 1)))
    workspace = tuple(range(width + 1, total))
    initial = torch.zeros(1 << total, dtype=torch.complex128)
    spectator = torch.tensor((sqrt(0.3), 1j * sqrt(0.7)), dtype=torch.complex128)
    target = torch.zeros_like(initial)
    for bit, amplitude in enumerate(spectator):
        base = bit << (total - 1)
        initial[base] = amplitude
        for word, value in enumerate(expected):
            address = base | sum(
                ((word >> (width - 1 - i)) & 1) << (total - 1 - wire)
                for i, wire in enumerate(phase)
            )
            target[address] = amplitude * value
    circuit = Circuit(total, inputs=initial, dtype=torch.complex128)
    gates = emitter({"qpe_reg": phase}, workspace, (), False)
    circuit.append(FieldOperation(total, gates))
    torch.testing.assert_close(circuit.state().flatten(), target, atol=2e-12, rtol=0)
    leakage = (
        circuit.state().flatten().reshape(-1, 1 << work)[:, 1:].abs().square().sum()
    )
    assert float(leakage) < 1e-24
    gates_inverse = emitter({"qpe_reg": phase}, workspace, (), True)
    circuit.append(FieldOperation(total, gates_inverse))
    # The unchanged source generic adjoint has an extra minus sign. Capturing
    # its SDK path must preserve that defect, not substitute the exact inverse.
    torch.testing.assert_close(circuit.state().flatten(), -initial, atol=3e-12, rtol=0)
    assert Counter(g.name for g in gates)["measure"] == 0
    assert Counter(g.name for g in gates_inverse)["measure"] == 0
