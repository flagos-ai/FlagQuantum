"""Verify unsigned register addition at the gate and amplitude levels."""

from importlib.metadata import version

import pytest
import torch

from flagquantum.compiler.registers.arithmetic import (
    add_ir,
    measurement_add_or_subtract_ir,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("width_a,width_b", [(1, 1), (1, 3), (2, 3), (3, 3), (3, 8)])
def test_add_matches_qualtran_gate_order(width_a: int, width_b: int) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("qualtran")
    if version("qualtran") != "0.7.0":
        pytest.skip("Gate comparison pins Qualtran v0.7.0")
    import cirq
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add
    from qualtran.bloqs.basic_gates import CNOT
    from qualtran.cirq_interop import BloqAsCirqGate

    boundary = width_a + width_b
    wires = cirq.LineQubit.range(boundary)
    context = cirq.DecompositionContext(cirq.SimpleQubitManager())
    names = {
        cirq.H: "h",
        cirq.T: "t",
        cirq.T**-1: "tdg",
        cirq.S: "s",
        cirq.CNOT: "cx",
        BloqAsCirqGate(CNOT()): "cx",
        cirq.CZ: "cz",
    }
    upstream = cirq.decompose(
        Add(QUInt(width_a), QUInt(width_b)).decompose_from_registers(
            context=context,
            a=np.asarray(wires[:width_a]),
            b=np.asarray(wires[width_a:]),
        ),
        context=context,
        keep=lambda op: (
            op.gate in names
            or cirq.is_measurement(op)
            or isinstance(op.gate, cirq.ResetChannel)
            or isinstance(op, cirq.ClassicallyControlledOperation)
        ),
    )
    labels = {wire: i for i, wire in enumerate(wires)}
    measurements = {}
    expected = []
    for op in upstream:
        metadata = {}
        if isinstance(op, cirq.ClassicallyControlledOperation):
            (key,) = cirq.control_keys(op)
            metadata["conditions"] = ((measurements[key], 1),)
            op = op.without_classical_controls()
        operands = tuple(labels.setdefault(wire, len(labels)) for wire in op.qubits)
        if cirq.is_measurement(op):
            bit = 40 + operands[0] - boundary
            measurements[cirq.measurement_key_obj(op)] = bit
            metadata = {"is_dynamic": True, "classical_bit": bit}
            name = "measure"
        elif isinstance(op.gate, cirq.ResetChannel):
            metadata = {"is_dynamic": True}
            name = "reset"
        else:
            name = names[op.gate]
        expected.append((name, operands, metadata))
    actual = add_ir(
        tuple(range(width_a)),
        tuple(range(width_a, boundary)),
        tuple(range(boundary, boundary + width_b - 1)),
        tuple(range(40, 39 + width_b)),
    )
    assert [(op.name, op.wires, dict(op.metadata)) for op in actual] == expected
    assert sum(op.name in {"t", "tdg"} for op in actual) == 4 * (width_b - 1)


@pytest.mark.parametrize("add_when_on", [False, True])
def test_controlled_add_subtract_preserves_complex_amplitudes(
    add_when_on: bool,
) -> None:
    inputs = torch.zeros(256, dtype=torch.complex128)
    basis = torch.arange(64)
    inputs[::4] = (1j) ** (basis % 4) / 8
    expected = torch.zeros_like(inputs)
    for prefix in range(64):
        control, left, right = prefix >> 5, (prefix >> 3) & 3, prefix & 7
        result = right + left if control == add_when_on else right - left
        output = (prefix & ~7) | (result & 7)
        expected[output << 2] = inputs[prefix << 2]
    ir = CircuitIR(
        8,
        measurement_add_or_subtract_ir(
            0,
            (1, 2),
            (3, 4, 5),
            (6, 7),
            (0, 1),
            add_when_ctrl_is_on=add_when_on,
        ),
    )
    circuit = DynamicCircuit(8, dtype=torch.complex128, inputs=inputs).compose(ir)
    result = run_dynamic(circuit, shots=16, seed=29, strategy="trajectory")
    assert len(set(map(tuple, result.classical_bits.tolist()))) == 4
    torch.testing.assert_close(
        result.final_states.reshape(16, -1), expected.expand(16, -1), rtol=0, atol=1e-12
    )
