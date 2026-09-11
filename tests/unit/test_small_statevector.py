import pytest
import torch

import flagquantum as fq
from flagquantum.simulation import small_data_reuploading_z


@pytest.mark.parametrize(("n_qubits", "blocks"), ((2, 3), (3, 2), (4, 1)))
def test_small_data_reuploading_matches_circuit_and_gradients(n_qubits, blocks):
    torch.manual_seed(17)
    inputs = torch.randn(7, n_qubits, requires_grad=True)
    parameters = {
        "variational_rz": torch.randn(blocks + 1, n_qubits, requires_grad=True),
        "variational_ry": torch.randn(blocks + 1, n_qubits, requires_grad=True),
        "input_ry_scale": torch.randn(blocks, n_qubits, requires_grad=True),
        "input_rz_scale": torch.randn(blocks, n_qubits, requires_grad=True),
    }
    reference_inputs = inputs.detach().clone().requires_grad_(True)
    reference_parameters = {
        name: value.detach().clone().requires_grad_(True)
        for name, value in parameters.items()
    }

    def circuit(values, data):
        q = fq.Circuit(n_qubits, bsz=data.shape[0], device=data.device)
        for wire in range(n_qubits):
            q.h(wire)
        for block in range(blocks):
            for wire in range(n_qubits):
                q.rz(wire, values["variational_rz"][block, wire])
            for wire in range(n_qubits):
                q.ry(wire, values["variational_ry"][block, wire])
            edge_count = 1 if n_qubits == 2 else n_qubits
            for wire in range(edge_count):
                q.cz(wire, (wire + 1) % n_qubits)
            for wire in range(n_qubits):
                q.ry(
                    wire,
                    torch.tanh(values["input_ry_scale"][block, wire] * data[:, wire]),
                )
            for wire in range(n_qubits):
                q.rz(
                    wire,
                    torch.tanh(values["input_rz_scale"][block, wire] * data[:, wire]),
                )
        for wire in range(n_qubits):
            q.rz(wire, values["variational_rz"][blocks, wire])
        for wire in range(n_qubits):
            q.ry(wire, values["variational_ry"][blocks, wire])
        return q

    reference = circuit(reference_parameters, reference_inputs).expectation_z()
    actual = small_data_reuploading_z(inputs, **parameters)
    reference.square().sum().backward()
    actual.square().sum().backward()

    assert torch.allclose(actual, reference, atol=5e-7, rtol=5e-7)
    assert torch.allclose(inputs.grad, reference_inputs.grad, atol=2e-6, rtol=2e-6)
    for name in parameters:
        assert torch.allclose(
            parameters[name].grad,
            reference_parameters[name].grad,
            atol=2e-6,
            rtol=2e-6,
        )
