"""Tests for hybrid JAX quantum kernels exposed to PyTorch."""

import importlib.util
from typing import Any

import pytest
import torch

import flagquantum as fq
import flagquantum.simulation.mps as fqmps
import flagquantum.simulation.tensor_network as fqtn
from flagquantum.algorithms import Hamiltonian, pauli_term, zz_chain_hamiltonian
from flagquantum.runtime.executors.jax import compile_quantum_kernel
from flagquantum.runtime.executors.jax.kernel import JAXQuantumKernel, QuantumTorchLayer
from flagquantum.runtime.executors.jax.mps import lowering as mps_lowering
from flagquantum.simulation.jax.mps import kernels as jax_mps

pytestmark = [
    pytest.mark.jax,
    pytest.mark.skipif(
        importlib.util.find_spec("jax") is None, reason="jax is not installed"
    ),
]


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("mode", ["statevector", "mps", "tensor_network"])
def test_derivative_demand_tracks_grad_mode_and_trainable_arguments(
    jit: bool, mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def build(parameters: Any, features: Any) -> fq.Circuit:
        return fq.Circuit(1).ry(0, parameters[0] * features[0] + parameters[1])

    parameters = torch.tensor([0.3, -0.2], dtype=torch.float64)
    features = torch.tensor([0.7], dtype=torch.float64)
    kernel = compile_quantum_kernel(
        build,
        parameters,
        n_qubits=1,
        accepts_inputs=True,
        mode=mode,
        jit=jit,
        compute_dtype="complex128",
    )
    calls: list[str] = []
    for name in (
        "_value",
        "_value_and_grad",
        "_parameter_value_and_grad",
        "_input_value_and_grad",
    ):
        original = getattr(kernel, name)

        def record(*args: Any, _name: str = name, _function: Any = original) -> Any:
            calls.append(_name)
            return _function(*args)

        monkeypatch.setattr(kernel, name, record)

    # One reusable kernel must survive changing derivative demand and values.
    for parameter_grad, input_grad in (
        (True, True),
        (False, True),
        (True, False),
        (False, False),
        (True, True),
    ):
        parameters.requires_grad_(parameter_grad)
        features.requires_grad_(input_grad)
        value = kernel(parameters, features)
        reference = (parameters[0] * features[0] + parameters[1]).cos()
        torch.testing.assert_close(value, reference, atol=1e-12, rtol=1e-12)
        selected = tuple(
            tensor for tensor in (parameters, features) if tensor.requires_grad
        )
        if selected:
            actual = torch.autograd.grad(1.7 * value, selected)
            expected = torch.autograd.grad(1.7 * reference, selected)
            for gradient, oracle in zip(actual, expected, strict=True):
                torch.testing.assert_close(gradient, oracle, atol=1e-12, rtol=1e-12)
        else:
            assert not value.requires_grad
        with torch.no_grad():
            parameters.add_(0.03)
            features.add_(0.02)
    assert calls == [
        "_value_and_grad",
        "_input_value_and_grad",
        "_parameter_value_and_grad",
        "_value",
        "_value_and_grad",
    ]

    for context in (torch.no_grad(), torch.inference_mode()):
        with context:
            value = kernel(parameters, features)
            reference = (parameters[0] * features[0] + parameters[1]).cos()
            assert not value.requires_grad
            torch.testing.assert_close(value, reference, atol=1e-12, rtol=1e-12)
    assert calls[-2:] == ["_value", "_value"]


def test_value_only_preserves_multidimensional_parameter_batches() -> None:
    def build(parameters: Any) -> fq.Circuit:
        return fq.Circuit(1).ry(0, parameters[0])

    kernel = compile_quantum_kernel(
        build, torch.zeros(1), n_qubits=1, compute_dtype="complex128"
    )
    parameters = torch.linspace(0.1, 0.6, 6, dtype=torch.float64).reshape(2, 3, 1)
    parameters.requires_grad_()
    differentiable = kernel(parameters)
    with torch.no_grad():
        inference = kernel(parameters)
    assert inference.shape == (2, 3)
    torch.testing.assert_close(inference, parameters[..., 0].cos())
    torch.testing.assert_close(inference, differentiable)
    torch.testing.assert_close(
        torch.autograd.grad(differentiable.sum(), parameters)[0], -parameters.sin()
    )
    with torch.no_grad(), pytest.raises(ValueError, match="parameter shape must end"):
        kernel(torch.zeros(2, 3, 2))


def test_cpu_bridge_keeps_cpu_placement_when_jax_defaults_to_an_accelerator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jax

    from flagquantum.runtime.executors.jax.kernel import _torch_to_jax

    # Exercise the placement decision without requiring a GPU in the CPU lane.
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    with jax.default_device(None):
        converted = _torch_to_jax(torch.tensor([0.2], dtype=torch.float32))
    assert converted.device.platform == "cpu"
    torch.testing.assert_close(
        torch.tensor(converted.__array__()), torch.tensor([0.2], dtype=torch.float32)
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_bridge_preserves_strided_input_gradients_and_parameter_updates(
    dtype: torch.dtype,
) -> None:
    """CPU buffer conversion must preserve both VJPs and changing tensor values."""
    parameters = torch.tensor([0.2, 9.0, 0.7, 8.0], dtype=dtype)[::2].requires_grad_()
    features = torch.tensor([0.4, 7.0, -0.2, 6.0], dtype=dtype)[::2].requires_grad_()

    def build(values: Any, inputs: Any) -> fq.Circuit:
        circuit = fq.Circuit(2, dtype=torch.complex128)
        circuit.ry(0, theta=values[0] + inputs[0] * values[1])
        circuit.ry(1, theta=inputs[1])
        return circuit

    kernel = compile_quantum_kernel(
        build,
        parameters,
        n_qubits=2,
        observable="z_sum",
        compute_dtype="complex128",
        accepts_inputs=True,
    )
    tolerance = 1e-5 if dtype == torch.float32 else 1e-10
    for train_input in (False, True, False):
        features.requires_grad_(train_input)
        value = kernel(parameters, features)
        gradients = torch.autograd.grad(
            2.3 * value, (parameters, features) if train_input else (parameters,)
        )
        angle = parameters[0] + features[0] * parameters[1]
        reference = angle.cos() + features[1].cos()
        expected_parameter = (
            -2.3 * angle.sin() * torch.stack((torch.ones_like(angle), features[0]))
        )
        expected_input = -2.3 * torch.stack(
            (parameters[1] * angle.sin(), features[1].sin())
        )
        torch.testing.assert_close(value, reference, atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(
            gradients[0], expected_parameter, atol=tolerance, rtol=tolerance
        )
        if train_input:
            torch.testing.assert_close(
                gradients[1], expected_input, atol=tolerance, rtol=tolerance
            )
        with torch.no_grad():
            parameters.add_(0.03)
            features.mul_(0.9)


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

    kernel = compile_quantum_kernel(
        build,
        params,
        backend="jax",
        interface="torch",
        mode="statevector",
        n_qubits=2,
        observable="z_sum",
    )
    loss = kernel(params)
    loss.backward()

    ref_loss = build(ref_params).expectation_z().sum()
    ref_loss.backward()

    assert isinstance(kernel, JAXQuantumKernel)
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

    kernel = compile_quantum_kernel(build, params, n_qubits=2, observable_qubits=(0,))
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

    kernel = compile_quantum_kernel(
        build,
        params,
        backend="jax",
        interface="torch",
        mode="statevector",
        n_qubits=3,
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

    layer = QuantumTorchLayer(
        build,
        2,
        n_qubits=2,
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

    kernel = compile_quantum_kernel(
        build,
        params[0].detach(),
        backend="jax",
        interface="torch",
        mode="statevector",
        n_qubits=2,
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


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("shape", [(3, 2), (1, 6), (2,)])
def test_jax_quantum_kernel_rejects_mismatched_parameter_shape(
    jit: bool, shape: tuple[int, ...]
) -> None:
    def build(values: Any) -> fq.Circuit:
        return fq.Circuit(2).rx(0, theta=values[0, 0]).ry(1, theta=values[1, 2])

    kernel = compile_quantum_kernel(build, torch.zeros(2, 3), n_qubits=2, jit=jit)
    with pytest.raises(ValueError, match="parameter shape must end with"):
        kernel(torch.zeros(shape, requires_grad=True))


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("sample_shape", [(), (2, 3)])
def test_jax_quantum_kernel_preserves_multiple_batch_axes(
    jit: bool, sample_shape: tuple[int, ...]
) -> None:
    def build(values: Any) -> fq.Circuit:
        angle = values[()] if not sample_shape else values[0, 0]
        return fq.Circuit(1).rx(0, theta=angle)

    kernel = compile_quantum_kernel(
        build, torch.zeros(sample_shape), n_qubits=1, jit=jit
    )
    parameters = torch.linspace(0.1, 0.8, 6).reshape(2, 3)
    if sample_shape:
        parameters = parameters[..., None, None].expand(2, 3, *sample_shape).clone()
    parameters.requires_grad_(True)
    values = kernel(parameters)
    values.sum().backward()

    angles = parameters if not sample_shape else parameters[..., 0, 0]
    expected_grad = (
        -parameters.sin() if not sample_shape else torch.zeros_like(parameters)
    )
    if sample_shape:
        expected_grad[..., 0, 0] = -angles.sin()
    assert values.shape == (2, 3)
    torch.testing.assert_close(values, angles.cos())
    torch.testing.assert_close(parameters.grad, expected_grad)


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("sample_shape", [(0,), (2, 0)])
@pytest.mark.parametrize("batch_shape", [(3,), (2, 3), (0,)])
def test_jax_quantum_kernel_batches_parameter_free_circuits(
    jit: bool, sample_shape: tuple[int, ...], batch_shape: tuple[int, ...]
) -> None:
    def build(values: Any) -> fq.Circuit:
        return fq.Circuit(1).x(0)

    kernel = compile_quantum_kernel(
        build, torch.empty(sample_shape), n_qubits=1, jit=jit
    )
    parameters = torch.empty(batch_shape + sample_shape, requires_grad=True)
    values = kernel(parameters)
    values.sum().backward()

    torch.testing.assert_close(values, -torch.ones(batch_shape))
    assert parameters.grad is not None
    assert parameters.grad.shape == parameters.shape
    assert parameters.grad.numel() == 0


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("accepts_inputs", [False, True])
def test_jax_quantum_kernel_rejects_double_backward(
    jit: bool, accepts_inputs: bool
) -> None:
    def build(values: Any, features: Any = None) -> fq.Circuit:
        angle = values[0] if features is None else values[0] + features[0]
        return fq.Circuit(1).rx(0, theta=angle)

    parameters = torch.tensor([0.3], requires_grad=True)
    inputs = torch.tensor([0.2], requires_grad=True) if accepts_inputs else None
    kernel = compile_quantum_kernel(
        build, parameters.detach(), n_qubits=1, jit=jit, accepts_inputs=accepts_inputs
    )
    loss = kernel(parameters, inputs).square()
    arguments = (parameters,) if inputs is None else (parameters, inputs)
    gradients = torch.autograd.grad(loss, arguments, create_graph=True)
    angle = parameters if inputs is None else parameters + inputs
    for gradient in gradients:
        torch.testing.assert_close(gradient, -(2 * angle).sin())
    with pytest.raises(RuntimeError, match="once_differentiable"):
        sum(gradient.sum() for gradient in gradients).backward()


def test_jax_quantum_kernel_hamiltonian_observable_matches_native_gradient():
    params = torch.tensor([0.2, -0.1, 0.3, 0.17], requires_grad=True)
    ref_params = params.detach().clone().requires_grad_(True)
    hamiltonian = Hamiltonian(
        [
            pauli_term(0.7, "ZZ", (0, 1)),
            pauli_term(-0.2, "X", (0,)),
            pauli_term(0.13, "YY", (1, 2)),
            pauli_term(0.05, "I", (0,)),
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

    kernel = compile_quantum_kernel(
        build,
        params,
        n_qubits=3,
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
    hamiltonian = Hamiltonian(
        [pauli_term(1.0, "Z", (0,)), pauli_term(0.2, "XX", (0, 1))]
    )

    def build(values):
        circuit = fq.Circuit(2)
        circuit.rx(0, theta=values[0]).ry(1, theta=values[1]).cx(0, 1)
        return circuit

    layer = QuantumTorchLayer(build, 2, n_qubits=2, hamiltonian=hamiltonian)
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

    kernel = compile_quantum_kernel(
        build,
        params,
        backend="jax",
        interface="torch",
        mode="mps",
        n_qubits=4,
        observable="z_sum",
        max_bond=8,
    )
    jax_params = params.detach().clone().requires_grad_(True)
    jax_loss = kernel(jax_params).sum()
    jax_loss.backward()

    ref_params = params.detach().clone().requires_grad_(True)
    ref_loss = fqmps.run_mps(build(ref_params), max_bond=8).expectation_z_sum().sum()
    ref_loss.backward()

    assert kernel.summary()["mode"] == "mps"
    assert jax_params.grad is not None
    assert ref_params.grad is not None
    assert torch.allclose(jax_loss.detach(), ref_loss.detach(), atol=1e-5)
    assert torch.allclose(jax_params.grad, ref_params.grad, atol=1e-4)


def test_jax_mps_z_sum_does_not_materialize_statevector(monkeypatch):
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

    monkeypatch.setattr(jax_mps, "jax_mps_to_statevector", forbidden)
    kernel = compile_quantum_kernel(
        build,
        params,
        backend="jax",
        interface="torch",
        mode="mps",
        n_qubits=4,
        observable="z_sum",
        max_bond=8,
    )
    jax_params = params.detach().clone().requires_grad_(True)
    loss = kernel(jax_params).sum()
    loss.backward()

    ref_params = params.detach().clone().requires_grad_(True)
    ref_loss = fqmps.run_mps(build(ref_params), max_bond=8).expectation_z_sum().sum()
    ref_loss.backward()

    assert jax_params.grad is not None
    assert ref_params.grad is not None
    assert torch.allclose(loss.detach(), ref_loss.detach(), atol=1e-5)
    assert torch.allclose(jax_params.grad, ref_params.grad, atol=1e-4)


def test_jax_mps_local_pauli_zz_chain_hamiltonian_uses_fastpath_and_matches_native_mps():
    params = (0.15 * torch.arange(1, 13, dtype=torch.float32)).requires_grad_(False)
    terms = []
    for wire in range(5):
        terms.append(pauli_term(-1.0, "ZZ", (wire, wire + 1)))
    for wire in range(6):
        terms.append(pauli_term(0.1, "Z", (wire,)))
        terms.append(pauli_term(-0.2, "X", (wire,)))
    hamiltonian = Hamiltonian(terms)

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

    kernel = compile_quantum_kernel(
        build,
        params,
        backend="jax",
        interface="torch",
        mode="mps",
        n_qubits=6,
        hamiltonian=hamiltonian,
        max_bond=16,
    )
    jax_params = params.detach().clone().requires_grad_(True)
    jax_loss = kernel(jax_params).sum()
    jax_loss.backward()

    ref_params = params.detach().clone().requires_grad_(True)
    ref_loss = hamiltonian.expectation(
        fqmps.run_mps(build(ref_params), max_bond=16)
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
    params = (0.11 * torch.arange(1, 25, dtype=torch.float32)).requires_grad_(False)
    hamiltonian = zz_chain_hamiltonian(6, coupling=-1.0, field=0.1)

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

    stable_kernel = compile_quantum_kernel(
        build,
        params,
        backend="jax",
        interface="torch",
        mode="mps",
        n_qubits=6,
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
    monkeypatch.setattr(mps_lowering, "_jax_mps_apply_two_remote", forbidden)
    scan_kernel = compile_quantum_kernel(
        build,
        params,
        backend="jax",
        interface="torch",
        mode="mps",
        n_qubits=6,
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

    kernel = compile_quantum_kernel(
        build,
        params,
        backend="jax",
        interface="torch",
        mode="mps",
        n_qubits=4,
        observable="z_sum",
        max_bond=8,
        compute_dtype="complex128",
    )
    jax_params = params.detach().clone().requires_grad_(True)
    jax_loss = kernel(jax_params).sum()
    jax_loss.backward()

    ref_params = params.detach().clone().requires_grad_(True)
    ref_loss = fqmps.run_mps(build(ref_params), max_bond=8).expectation_z_sum().sum()
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

    kernel = compile_quantum_kernel(
        build,
        params,
        backend="jax",
        interface="torch",
        mode="tensor_network",
        n_qubits=3,
        observable="z_sum",
    )
    jax_params = params.detach().clone().requires_grad_(True)
    jax_loss = kernel(jax_params).sum()
    jax_loss.backward()

    ref_params = params.detach().clone().requires_grad_(True)
    ref_loss = fqtn.run_tensor_network(build(ref_params)).expectation_z().sum()
    ref_loss.backward()

    assert kernel.summary()["mode"] == "tensor_network"
    assert jax_params.grad is not None
    assert ref_params.grad is not None
    assert torch.allclose(jax_loss.detach(), ref_loss.detach(), atol=1e-5)
    assert torch.allclose(jax_params.grad, ref_params.grad, atol=1e-4)


def test_jax_tensor_network_z_sum_does_not_materialize_statevector(monkeypatch):
    from flagquantum.runtime.executors.jax import kernel as jax_kernel_module

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

    monkeypatch.setattr(jax_kernel_module, "_jax_statevector_from_circuit", forbidden)
    kernel = compile_quantum_kernel(
        build,
        params,
        backend="jax",
        interface="torch",
        mode="tensor_network",
        n_qubits=3,
        observable="z_sum",
    )
    jax_params = params.detach().clone().requires_grad_(True)
    jax_loss = kernel(jax_params).sum()
    jax_loss.backward()

    ref_params = params.detach().clone().requires_grad_(True)
    ref_loss = fqtn.run_tensor_network(build(ref_params)).expectation_z().sum()
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
    hamiltonian = Hamiltonian(
        [
            pauli_term(0.7, "ZZ", (0, 3)),
            pauli_term(-0.2, "X", (1,)),
            pauli_term(0.05, "Z", (2,)),
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

    kernel = compile_quantum_kernel(
        build,
        params,
        backend="jax",
        interface="torch",
        mode="tensor_network",
        n_qubits=4,
        hamiltonian=hamiltonian,
        matmul_precision="highest",
    )
    jax_params = params.detach().clone().requires_grad_(True)
    jax_loss = kernel(jax_params).sum()
    jax_loss.backward()

    ref_params = params.detach().clone().requires_grad_(True)
    ref_tn = fqtn.run_tensor_network(build(ref_params))
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

    kernel = compile_quantum_kernel(
        build,
        params,
        backend="jax",
        interface="torch",
        mode="tensor_network",
        n_qubits=4,
        observable="z_sum",
        compute_dtype="complex128",
    )
    jax_params = params.detach().clone().requires_grad_(True)
    jax_loss = kernel(jax_params).sum()
    jax_loss.backward()

    ref_params = params.detach().clone().requires_grad_(True)
    ref_loss = fqtn.run_tensor_network(build(ref_params)).expectation_z().sum()
    ref_loss.backward()

    assert kernel.summary()["compute_dtype"] == "complex128"
    assert jax_params.grad is not None
    assert ref_params.grad is not None
    assert torch.allclose(jax_loss.detach(), ref_loss.detach(), atol=1e-5)
    assert torch.allclose(jax_params.grad, ref_params.grad, atol=1e-4)
