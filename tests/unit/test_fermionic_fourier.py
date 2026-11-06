"""Pinned two-qubit fermionic Fourier gate stream and exact phase."""

from collections import Counter

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.algorithms.primitives.fermionic_fourier import (
    source_two_bit_fermionic_fourier_cost,
    two_bit_fermionic_fourier_ir,
)
from flagquantum.circuit import Circuit

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("k,n", [(0, 1), (1, 8), (-1, 8), (1, 4), (2, 3)])
def test_fourier_declared_model_matches_original_resource_query(k, n):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.qft.two_bit_ffft import TwoBitFFFT
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = TwoBitFFFT(k, n)
    expected = get_cost_value(source, QECGatesCost())
    model = source_two_bit_fermionic_fourier_cost(k, n)
    assert model.qubits == get_cost_value(source, QubitCount())
    for field in (
        "t",
        "clifford",
        "rotation",
        "measurement",
        "and_bloq",
        "toffoli",
        "cswap",
    ):
        assert getattr(model.gates, field) == getattr(expected, field)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("k,n", [(0, 1), (1, 4), (2, 3), (-2, 5)])
def test_locked_fermionic_fourier_stream_resources_and_complex_state(k, n):
    pytest.importorskip("qualtran")
    import cirq
    import numpy as np
    from qualtran.bloqs.qft.two_bit_ffft import TwoBitFFFT
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source = TwoBitFFFT(k, n)
    model = get_cost_value(source, QECGatesCost())
    assert model.t == 2
    assert model.rotation == int(4 * k % n != 0)
    block = source.decompose_bloq()
    values = torch.tensor([1 + 1j, 2 - 1j, -3j, 4 + 2j], dtype=torch.complex128)
    values /= values.norm()
    forward = torch.tensor(block.tensor_contract(), dtype=torch.complex128)
    for adjoint in (False, True):
        actual = two_bit_fermionic_fourier_ir(1, 0, k, n, adjoint=adjoint)
        # Bind the source decomposition with the same physical input order.
        q = cirq.LineQubit.range(2)
        circuit, _ = block.to_cirq_circuit_and_quregs(
            qubit_manager=cirq.SimpleQubitManager(),
            x=np.array([q[1]]),
            y=np.array([q[0]]),
        )
        ops = tuple(circuit.all_operations())
        if adjoint:
            ops = tuple(cirq.inverse(ops))
        gates = {
            cirq.H: "h",
            cirq.S: "s",
            cirq.S**-1: "sdg",
            cirq.T: "t",
            cirq.T**-1: "tdg",
            cirq.CNOT: "cx",
        }
        expected = tuple(
            (
                gates[op.gate] if not isinstance(op.gate, cirq.Rz) else "rz",
                tuple(qubit.x for qubit in op.qubits),
                float(op.gate._rads) if isinstance(op.gate, cirq.Rz) else 0,
            )
            for op in ops
        )
        observed = tuple((g.name, g.wires, g.params.get("theta", 0)) for g in actual)
        assert Counter(observed) == Counter(expected)
        for wire in (0, 1):
            assert tuple(g for g in observed if wire in g[1]) == tuple(
                g for g in expected if wire in g[1]
            )
        assert len(actual) == 15
        logical_order = values.reshape(2, 2).T.flatten()
        matrix = forward.mH if adjoint else forward
        wanted = (matrix @ logical_order).reshape(2, 2).T.flatten()
        bound = Circuit(2, inputs=values, dtype=torch.complex128)
        bound.append(FieldOperation(2, actual))
        torch.testing.assert_close(bound.state().flatten(), wanted, atol=3e-12, rtol=0)


@pytest.mark.ftqc_reference
def test_locked_adjoint_flag_does_not_change_source_decomposition():
    pytest.importorskip("qualtran")
    import numpy as np
    from qualtran.bloqs.qft.two_bit_ffft import TwoBitFFFT

    original = TwoBitFFFT(2, 3).decompose_bloq().tensor_contract()
    flagged = TwoBitFFFT(2, 3, is_adjoint=True).decompose_bloq().tensor_contract()
    np.testing.assert_allclose(original, flagged, atol=1e-12)
    assert not np.allclose(flagged, original.conj().T)


def test_native_inverse_and_invalid_parameters():
    forward = two_bit_fermionic_fourier_ir(0, 1, 2, 3)
    inverse = two_bit_fermionic_fourier_ir(0, 1, 2, 3, adjoint=True)
    state = torch.tensor([1, 1j, 1j, -1], dtype=torch.complex128) / 2
    circuit = Circuit(2, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(2, (*forward, *inverse)))
    torch.testing.assert_close(circuit.state().flatten(), state, atol=3e-12, rtol=0)
    with pytest.raises(ValueError, match="nonzero n"):
        two_bit_fermionic_fourier_ir(0, 1, 2, 0)
