"""Tests for hybrid JAX quantum kernels exposed to PyTorch."""

import importlib.util

import pytest
import torch

import flagquantum as fq
import flagquantum.backends as fqb

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None, reason="jax is not installed"
)


def test_jax_quantum_kernel_torch_autograd_matches_statevector():
    params = torch.tensor([0.2, -0.1, 0.3], requires_grad=True)
    ref_params = params.detach().clone().requires_grad_(True)

    def build(values):
        circuit = fq.Circuit(2)
        circuit.rx(0, theta=values[0])
        circuit.ry(1, theta=values[1])
        circuit.cx(0, 1)
        circuit.rz(0, theta=values[2])
        return circuit

    kernel = fq.compile_quantum_kernel(
        build,
        params,
        backend="jax",
        interface="torch",
        mode="statevector",
        n_wires=2,
        observable="z_sum",
    )
    loss = kernel(params)
    loss.backward()

    ref_loss = build(ref_params).expectation_z().sum()
    ref_loss.backward()

    assert isinstance(kernel, fq.JAXQuantumKernel)
    assert params.grad is not None
    assert ref_params.grad is not None
    assert torch.allclose(loss.detach(), ref_loss.detach(), atol=1e-6)
    assert torch.allclose(params.grad, ref_params.grad, atol=1e-5)
    assert kernel.summary()["interface"] == "torch"


def test_jax_quantum_kernel_parameters_can_update_between_calls():
    params = torch.tensor([0.2, -0.1], requires_grad=True)

    def build(values):
        circuit = fq.Circuit(2)
        circuit.rx(0, theta=values[0]).ry(1, theta=values[1]).cx(0, 1)
        return circuit

    kernel = fq.compile_quantum_kernel(build, params, n_wires=2, observable_wires=(0,))
    loss = kernel(params)
    shifted_loss = kernel(params + 0.3)

    assert not torch.allclose(loss.detach(), shifted_loss.detach())


def test_jax_quantum_kernel_common_training_gate_set_matches_statevector():
    params = torch.tensor(
        [0.2, -0.1, 0.3, 0.17, -0.23, 0.41, -0.35, 0.13, -0.29, 0.07],
        requires_grad=True,
    )
    ref_params = params.detach().clone().requires_grad_(True)

    def build(values):
        circuit = fq.Circuit(3)
        circuit.h(0).s(1).tdg(2)
        circuit.u3(0, theta=values[0], phi=values[1], lbd=values[2])
        circuit.u2(1, phi=values[3], lbd=values[4])
        circuit.phase(2, theta=values[5])
        circuit.crx(0, 1, theta=values[6])
        circuit.cry(1, 2, theta=values[7])
        circuit.rxx(0, 1, theta=values[8])
        circuit.rzz(1, 2, theta=values[9])
        circuit.swap(0, 2)
        circuit.cz(0, 1)
        return circuit

    kernel = fq.compile_quantum_kernel(
        build,
        params,
        backend="jax",
        interface="torch",
        mode="statevector",
        n_wires=3,
        observable="z_sum",
    )
    loss = kernel(params)
    loss.backward()

    ref_loss = build(ref_params).expectation_z().sum()
    ref_loss.backward()

    assert params.grad is not None
    assert ref_params.grad is not None
    assert torch.allclose(loss.detach(), ref_loss.detach(), atol=1e-5)
    assert torch.allclose(params.grad, ref_params.grad, atol=1e-4)


def test_quantum_torch_layer_participates_in_optimizer_step():
    def build(values):
        circuit = fq.Circuit(2)
        circuit.rx(0, theta=values[0]).ry(1, theta=values[1]).cx(0, 1)
        return circuit

    layer = fq.QuantumTorchLayer(
        build,
        2,
        n_wires=2,
        init=torch.tensor([0.2, -0.1]),
    )
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.05)

    before = layer.parameters_tensor.detach().clone()
    loss = layer()
    objective = (loss - 0.5) ** 2
    objective.backward()
    optimizer.step()

    assert isinstance(layer, torch.nn.Module)
    assert layer.parameters_tensor.grad is not None
    assert not torch.allclose(layer.parameters_tensor.detach(), before)
    assert layer.summary()["module"] == "QuantumTorchLayer"


def test_jax_quantum_kernel_batched_parameters_match_per_sample_statevector():
    params = torch.tensor(
        [[0.2, -0.1, 0.3], [0.25, -0.05, 0.35], [0.4, 0.1, -0.2]],
        requires_grad=True,
    )

    def build(values):
        circuit = fq.Circuit(2)
        circuit.rx(0, theta=values[0])
        circuit.ry(1, theta=values[1])
        circuit.cx(0, 1)
        circuit.rz(0, theta=values[2])
        return circuit

    kernel = fq.compile_quantum_kernel(
        build,
        params[0].detach(),
        backend="jax",
        interface="torch",
        mode="statevector",
        n_wires=2,
        observable="z_sum",
    )
    losses = kernel(params)
    objective = losses.sum()
    objective.backward()

    ref_values = []
    ref_grads = []
    for row in params.detach():
        ref_params = row.clone().requires_grad_(True)
        ref_loss = build(ref_params).expectation_z().sum()
        ref_loss.backward()
        ref_values.append(ref_loss.detach())
        assert ref_params.grad is not None
        ref_grads.append(ref_params.grad.detach())

    assert params.grad is not None
    assert losses.shape == (3,)
    assert torch.allclose(losses.detach(), torch.stack(ref_values), atol=1e-6)
    assert torch.allclose(params.grad, torch.stack(ref_grads), atol=1e-5)


def test_jax_quantum_kernel_hamiltonian_observable_matches_native_gradient():
    params = torch.tensor([0.2, -0.1, 0.3, 0.17], requires_grad=True)
    ref_params = params.detach().clone().requires_grad_(True)
    hamiltonian = fq.Hamiltonian(
        [
            fq.pauli_term(0.7, "ZZ", (0, 1)),
            fq.pauli_term(-0.2, "X", (0,)),
            fq.pauli_term(0.13, "YY", (1, 2)),
            fq.pauli_term(0.05, "I", (0,)),
        ]
    )

    def build(values):
        circuit = fq.Circuit(3)
        circuit.rx(0, theta=values[0])
        circuit.ry(1, theta=values[1])
        circuit.rz(2, theta=values[2])
        circuit.cx(0, 1)
        circuit.rxx(1, 2, theta=values[3])
        return circuit

    kernel = fq.compile_quantum_kernel(
        build,
        params,
        n_wires=3,
        hamiltonian=hamiltonian,
    )
    loss = kernel(params)
    loss.backward()

    ref_loss = hamiltonian.expectation(build(ref_params)).sum()
    ref_loss.backward()

    assert params.grad is not None
    assert ref_params.grad is not None
    assert torch.allclose(loss.detach(), ref_loss.detach(), atol=1e-5)
    assert torch.allclose(params.grad, ref_params.grad, atol=1e-4)


def test_quantum_torch_layer_accepts_hamiltonian_observable():
    hamiltonian = fq.Hamiltonian(
        [fq.pauli_term(1.0, "Z", (0,)), fq.pauli_term(0.2, "XX", (0, 1))]
    )

    def build(values):
        circuit = fq.Circuit(2)
        circuit.rx(0, theta=values[0]).ry(1, theta=values[1]).cx(0, 1)
        return circuit

    layer = fq.QuantumTorchLayer(build, 2, n_wires=2, hamiltonian=hamiltonian)
    value = layer()
    value.backward()

    assert layer.parameters_tensor.grad is not None
    assert torch.isfinite(value)
    assert layer.summary()["observable"] == "hamiltonian"


def test_jax_mps_kernel_matches_native_mps_gradient():
    params = torch.tensor([0.17, -0.31, 0.23, -0.19], requires_grad=False)

    def build(values):
        circuit = fq.Circuit(4)
        circuit.rx(0, theta=values[0])
        circuit.ry(1, theta=values[1])
        circuit.rz(2, theta=values[2])
        circuit.cx(0, 1)
        circuit.rxx(2, 3, theta=values[3])
        circuit.cx(1, 2)
        return circuit

    kernel = fq.compile_quantum_kernel(
        build,
        params,
        backend="jax",
        interface="torch",
        mode="mps",
        n_wires=4,
        observable="z_sum",
        max_bond=8,
    )
    jax_params = params.detach().clone().requires_grad_(True)
    jax_loss = kernel(jax_params).sum()
    jax_loss.backward()

    ref_params = params.detach().clone().requires_grad_(True)
    ref_loss = fqb.run_mps(build(ref_params), max_bond=8).expectation_z_sum().sum()
    ref_loss.backward()

    assert kernel.summary()["mode"] == "mps"
    assert jax_params.grad is not None
    assert ref_params.grad is not None
    assert torch.allclose(jax_loss.detach(), ref_loss.detach(), atol=1e-5)
    assert torch.allclose(jax_params.grad, ref_params.grad, atol=1e-4)


def test_jax_mps_z_sum_does_not_materialize_statevector(monkeypatch):
    from flagquantum.runtime import compatibility as hybrid

    params = torch.tensor([0.17, -0.31, 0.23, -0.19], requires_grad=False)

    def build(values):
        circuit = fq.Circuit(4)
        circuit.rx(0, theta=values[0])
        circuit.ry(1, theta=values[1])
        circuit.rz(2, theta=values[2])
        circuit.cx(0, 1)
        circuit.rxx(2, 3, theta=values[3])
        circuit.cx(1, 2)
        return circuit

    def forbidden(*args, **kwargs):
        raise AssertionError("JAX MPS observable must not materialize a statevector")

    monkeypatch.setattr(hybrid, "_jax_mps_to_statevector", forbidden)
    kernel = fq.compile_quantum_kernel(
        build,
        params,
        backend="jax",
        interface="torch",
        mode="mps",
        n_wires=4,
        observable="z_sum",
        max_bond=8,
    )
    jax_params = params.detach().clone().requires_grad_(True)
    loss = kernel(jax_params).sum()
    loss.backward()

    ref_params = params.detach().clone().requires_grad_(True)
    ref_loss = fqb.run_mps(build(ref_params), max_bond=8).expectation_z_sum().sum()
    ref_loss.backward()

    assert jax_params.grad is not None
    assert ref_params.grad is not None
    assert torch.allclose(loss.detach(), ref_loss.detach(), atol=1e-5)
    assert torch.allclose(jax_params.grad, ref_params.grad, atol=1e-4)


def test_jax_mps_local_pauli_zz_chain_hamiltonian_uses_fastpath_and_matches_native_mps():
    params = (0.15 * torch.arange(1, 13, dtype=torch.float32)).requires_grad_(False)
    terms = []
    for wire in range(5):
        terms.append(fq.pauli_term(-1.0, "ZZ", (wire, wire + 1)))
    for wire in range(6):
        terms.append(fq.pauli_term(0.1, "Z", (wire,)))
        terms.append(fq.pauli_term(-0.2, "X", (wire,)))
    hamiltonian = fq.Hamiltonian(terms)

    def build(values):
        circuit = fq.Circuit(6)
        cursor = 0
        for _layer in range(2):
            for wire in range(6):
                circuit.ry(wire, theta=values[cursor])
                cursor += 1
            for wire in range(5):
                circuit.cx(wire, wire + 1)
        return circuit

    kernel = fq.compile_quantum_kernel(
        build,
        params,
        backend="jax",
        interface="torch",
        mode="mps",
        n_wires=6,
        hamiltonian=hamiltonian,
        max_bond=16,
    )
    jax_params = params.detach().clone().requires_grad_(True)
    jax_loss = kernel(jax_params).sum()
    jax_loss.backward()

    ref_params = params.detach().clone().requires_grad_(True)
    ref_loss = hamiltonian.expectation(
        fqb.run_mps(build(ref_params), max_bond=16)
    ).sum()
    ref_loss.backward()

    assert (
        kernel.summary()["hamiltonian_fastpath"] == "local_pauli_zz_chain_padded_scan"
    )
    assert jax_params.grad is not None
    assert ref_params.grad is not None
    assert torch.allclose(jax_loss.detach(), ref_loss.detach(), atol=1e-5)
    assert torch.allclose(jax_params.grad, ref_params.grad, atol=1e-4)


def test_jax_mps_cx_chain_scan_matches_stable_jax_path(monkeypatch):
    from flagquantum.runtime import compatibility as hybrid

    params = (0.11 * torch.arange(1, 25, dtype=torch.float32)).requires_grad_(False)
    hamiltonian = fq.zz_chain_hamiltonian(6, coupling=-1.0, field=0.1)

    def build(values):
        circuit = fq.Circuit(6)
        cursor = 0
        for _layer in range(2):
            for wire in range(6):
                circuit.ry(wire, theta=values[cursor])
                cursor += 1
                circuit.rz(wire, theta=values[cursor])
                cursor += 1
            for wire in range(5):
                circuit.cx(wire, wire + 1)
        return circuit

    stable_kernel = fq.compile_quantum_kernel(
        build,
        params,
        backend="jax",
        interface="torch",
        mode="mps",
        n_wires=6,
        hamiltonian=hamiltonian,
        max_bond=4,
        jit=False,
    )
    monkeypatch.setenv("FQ_DISABLE_JAX_MPS_CX_SCAN", "1")
    stable_params = params.detach().clone().requires_grad_(True)
    stable_loss = stable_kernel(stable_params).sum()
    stable_loss.backward()

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "experimental nearest-neighbor CX-chain scan must not use the per-gate remote path"
        )

    monkeypatch.delenv("FQ_DISABLE_JAX_MPS_CX_SCAN")
    monkeypatch.setattr(hybrid, "_jax_mps_apply_two_remote", forbidden)
    scan_kernel = fq.compile_quantum_kernel(
        build,
        params,
        backend="jax",
        interface="torch",
        mode="mps",
        n_wires=6,
        hamiltonian=hamiltonian,
        max_bond=4,
        jit=False,
    )
    scan_params = params.detach().clone().requires_grad_(True)
    scan_loss = scan_kernel(scan_params).sum()
    scan_loss.backward()

    assert stable_params.grad is not None
    assert scan_params.grad is not None
    assert torch.isfinite(scan_loss.detach())
    assert torch.all(torch.isfinite(scan_params.grad))
    assert torch.allclose(scan_loss.detach(), stable_loss.detach(), atol=1e-5)
    assert torch.allclose(scan_params.grad, stable_params.grad, atol=1e-4)


def test_jax_mps_kernel_supports_complex128_compute_dtype():
    params = torch.tensor([0.17, -0.31, 0.23, -0.19], requires_grad=False)

    def build(values):
        circuit = fq.Circuit(4)
        circuit.rx(0, theta=values[0])
        circuit.ry(1, theta=values[1])
        circuit.rz(2, theta=values[2])
        circuit.rxx(0, 3, theta=values[3])
        circuit.cx(1, 2)
        return circuit

    kernel = fq.compile_quantum_kernel(
        build,
        params,
        backend="jax",
        interface="torch",
        mode="mps",
        n_wires=4,
        observable="z_sum",
        max_bond=8,
        compute_dtype="complex128",
    )
    jax_params = params.detach().clone().requires_grad_(True)
    jax_loss = kernel(jax_params).sum()
    jax_loss.backward()

    ref_params = params.detach().clone().requires_grad_(True)
    ref_loss = fqb.run_mps(build(ref_params), max_bond=8).expectation_z_sum().sum()
    ref_loss.backward()

    assert kernel.summary()["compute_dtype"] == "complex128"
    assert jax_params.grad is not None
    assert ref_params.grad is not None
    assert torch.allclose(jax_loss.detach(), ref_loss.detach(), atol=1e-5)
    assert torch.allclose(jax_params.grad, ref_params.grad, atol=1e-4)


def test_jax_tensor_network_kernel_matches_native_tn_gradient():
    params = torch.tensor([0.17, -0.31, 0.23], requires_grad=False)

    def build(values):
        circuit = fq.Circuit(3)
        circuit.rx(0, theta=values[0])
        circuit.ry(1, theta=values[1])
        circuit.rz(2, theta=values[2])
        circuit.cx(0, 2)
        return circuit

    kernel = fq.compile_quantum_kernel(
        build,
        params,
        backend="jax",
        interface="torch",
        mode="tensor_network",
        n_wires=3,
        observable="z_sum",
    )
    jax_params = params.detach().clone().requires_grad_(True)
    jax_loss = kernel(jax_params).sum()
    jax_loss.backward()

    ref_params = params.detach().clone().requires_grad_(True)
    ref_loss = fqb.run_tensor_network(build(ref_params)).expectation_z().sum()
    ref_loss.backward()

    assert kernel.summary()["mode"] == "tensor_network"
    assert jax_params.grad is not None
    assert ref_params.grad is not None
    assert torch.allclose(jax_loss.detach(), ref_loss.detach(), atol=1e-5)
    assert torch.allclose(jax_params.grad, ref_params.grad, atol=1e-4)


def test_jax_tensor_network_z_sum_does_not_materialize_statevector(monkeypatch):
    from flagquantum.runtime import compatibility as hybrid

    params = torch.tensor([0.17, -0.31, 0.23], requires_grad=False)

    def build(values):
        circuit = fq.Circuit(3)
        circuit.rx(0, theta=values[0])
        circuit.ry(1, theta=values[1])
        circuit.rz(2, theta=values[2])
        circuit.cx(0, 2)
        return circuit

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "JAX tensor-network observable must not materialize a statevector"
        )

    monkeypatch.setattr(
        hybrid, "_jax_tensor_network_statevector_from_circuit", forbidden
    )
    kernel = fq.compile_quantum_kernel(
        build,
        params,
        backend="jax",
        interface="torch",
        mode="tensor_network",
        n_wires=3,
        observable="z_sum",
    )
    jax_params = params.detach().clone().requires_grad_(True)
    jax_loss = kernel(jax_params).sum()
    jax_loss.backward()

    ref_params = params.detach().clone().requires_grad_(True)
    ref_loss = fqb.run_tensor_network(build(ref_params)).expectation_z().sum()
    ref_loss.backward()

    assert jax_params.grad is not None
    assert ref_params.grad is not None
    assert torch.allclose(jax_loss.detach(), ref_loss.detach(), atol=1e-5)
    assert torch.allclose(jax_params.grad, ref_params.grad, atol=1e-4)


def test_jax_tensor_network_kernel_remote_hamiltonian_gradient_precision():
    params = torch.tensor(
        [
            [0.17, -0.31, 0.23],
            [-0.19, 0.11, -0.07],
            [0.29, -0.13, 0.05],
            [0.03, 0.21, -0.27],
        ],
        requires_grad=False,
    )
    hamiltonian = fq.Hamiltonian(
        [
            fq.pauli_term(0.7, "ZZ", (0, 3)),
            fq.pauli_term(-0.2, "X", (1,)),
            fq.pauli_term(0.05, "Z", (2,)),
        ]
    )

    def build(values):
        circuit = fq.Circuit(4)
        for wire in range(4):
            circuit.rx(wire, theta=values[wire, 0])
            circuit.ry(wire, theta=values[wire, 1])
            circuit.rz(wire, theta=values[wire, 2])
        circuit.cx(0, 2)
        circuit.rxx(0, 3, theta=values[0, 0] * 0.25)
        circuit.ryy(1, 3, theta=values[1, 1] * -0.5)
        return circuit

    kernel = fq.compile_quantum_kernel(
        build,
        params,
        backend="jax",
        interface="torch",
        mode="tensor_network",
        n_wires=4,
        hamiltonian=hamiltonian,
        matmul_precision="highest",
    )
    jax_params = params.detach().clone().requires_grad_(True)
    jax_loss = kernel(jax_params).sum()
    jax_loss.backward()

    ref_params = params.detach().clone().requires_grad_(True)
    ref_tn = fqb.run_tensor_network(build(ref_params))
    ref_loss = (
        0.7 * ref_tn.expectation_ps(z=[0, 3])
        - 0.2 * ref_tn.expectation_ps(x=[1])
        + 0.05 * ref_tn.expectation_ps(z=[2])
    ).sum()
    ref_loss.backward()

    assert kernel.summary()["matmul_precision"] == "highest"
    assert jax_params.grad is not None
    assert ref_params.grad is not None
    assert torch.allclose(jax_loss.detach(), ref_loss.detach(), atol=1e-5)
    assert torch.allclose(jax_params.grad, ref_params.grad, atol=1e-4)


def test_jax_tensor_network_kernel_supports_complex128_compute_dtype():
    params = torch.tensor([0.17, -0.31, 0.23, -0.19], requires_grad=False)

    def build(values):
        circuit = fq.Circuit(4)
        circuit.rx(0, theta=values[0])
        circuit.ry(1, theta=values[1])
        circuit.rz(2, theta=values[2])
        circuit.rxx(0, 3, theta=values[3])
        circuit.cx(1, 2)
        return circuit

    kernel = fq.compile_quantum_kernel(
        build,
        params,
        backend="jax",
        interface="torch",
        mode="tensor_network",
        n_wires=4,
        observable="z_sum",
        compute_dtype="complex128",
    )
    jax_params = params.detach().clone().requires_grad_(True)
    jax_loss = kernel(jax_params).sum()
    jax_loss.backward()

    ref_params = params.detach().clone().requires_grad_(True)
    ref_loss = fqb.run_tensor_network(build(ref_params)).expectation_z().sum()
    ref_loss.backward()

    assert kernel.summary()["compute_dtype"] == "complex128"
    assert jax_params.grad is not None
    assert ref_params.grad is not None
    assert torch.allclose(jax_loss.detach(), ref_loss.detach(), atol=1e-5)
    assert torch.allclose(jax_params.grad, ref_params.grad, atol=1e-4)
