"""Exact source component pairs and their logical-output contracts."""

from collections import Counter

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.comparison_tree import (
    bi_qubits_mixer_ir,
    single_qubit_compare_ir,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("mixer", [False, True])
def test_component_pair_source_gates_complex_restore_and_cleanup(mixer: bool) -> None:
    pytest.importorskip("qualtran")
    from qualtran import BloqBuilder, QUInt, Signature
    from qualtran.bloqs.arithmetic.comparison import BiQubitsMixer, SingleQubitCompare

    if mixer:
        bb, soqs = BloqBuilder.from_signature(
            Signature.build_from_dtypes(x=QUInt(2), y=QUInt(2))
        )
        x, y, aux = bb.add(BiQubitsMixer(), **soqs)
        x, y = bb.add(BiQubitsMixer().adjoint(), x=x, y=y, ancilla=aux)
        source = bb.finalize(x=x, y=y)
        inputs = {"x": (0, 1), "y": (2, 3)}
        alloc = (4, 5, 6)
        slots = {5: 40, 6: 41}
        n_data = 4
        forward = bi_qubits_mixer_ir((0, 1), (2, 3), alloc)
        erase = bi_qubits_mixer_ir(
            (0, 1), (2, 3), alloc, adjoint=True, classical_bits=(40, 41)
        )
    else:
        bb, soqs = BloqBuilder.from_signature(Signature.build(a=1, b=1))
        a, b, less, greater = bb.add(SingleQubitCompare(), **soqs)
        a, b = bb.add(
            SingleQubitCompare().adjoint(),
            a=a,
            b=b,
            less_than=less,
            greater_than=greater,
        )
        source = bb.finalize(a=a, b=b)
        inputs = {"a": (0,), "b": (1,)}
        alloc = (2, 3)
        slots = {2: 40}
        n_data = 2
        forward = single_qubit_compare_ir(0, 1, 2, 3)
        erase = single_qubit_compare_ir(0, 1, 2, 3, adjoint=True, classical_bits=(40,))
    gates = (*forward, *erase)
    expected = reference_arithmetic_gates(source, inputs, alloc, slots)
    assert_source(gates, expected)
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    model = get_cost_value(source, QECGatesCost())
    counts = Counter(g.name for g in gates)
    assert counts["t"] + counts["tdg"] == model.total_t_count() == (8 if mixer else 4)
    assert counts["measure"] == model.measurement == (2 if mixer else 1)

    n_qubits = n_data + len(alloc)
    coeff = torch.arange(1 << n_data, dtype=torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    state = torch.zeros(1 << n_qubits, dtype=torch.complex128)
    state[:: 1 << len(alloc)] = amps
    circuit = DynamicCircuit(n_qubits, inputs=state, dtype=torch.complex128).append(
        FieldOperation(n_qubits, gates)
    )
    result = run_dynamic(circuit, shots=32, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(32, -1), state.expand(32, -1), rtol=0, atol=1e-12
    )
    # Inspect source-independent logical truth on forward outputs, not just a cancelling pair.
    for value in range(1 << n_data):
        basis = torch.zeros_like(state)
        basis[value << len(alloc)] = 1
        output = (
            Circuit(n_qubits, inputs=basis, dtype=torch.complex128)
            .append(FieldOperation(n_qubits, forward))
            .state()
            .flatten()
        )
        index = int(output.abs().argmax())
        assert output[index].item() == pytest.approx(1, abs=1e-12)
        if mixer:
            original_x, original_y = value >> 2, value & 3
            low_x = (index >> (n_qubits - 2)) & 1
            low_y = (index >> (n_qubits - 4)) & 1
            assert (original_x > original_y) - (original_x < original_y) == (
                low_x > low_y
            ) - (low_x < low_y)
        else:
            a, b = value >> 1, value & 1
            assert index == (a << 3) | ((a == b) << 2) | ((a < b) << 1) | (a > b)


def test_component_validation() -> None:
    with pytest.raises(ValueError, match="two-bit"):
        bi_qubits_mixer_ir((0,), (1, 2), (3, 4, 5))
    with pytest.raises(ValueError, match="distinct"):
        single_qubit_compare_ir(0, 0, 2, 3)
    with pytest.raises(ValueError, match="classical"):
        single_qubit_compare_ir(0, 1, 2, 3, adjoint=True)
    with pytest.raises(ValueError, match="boolean"):
        single_qubit_compare_ir(0, 1, 2, 3, adjoint=1)
