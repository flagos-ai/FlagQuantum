"""Ordinary Module consumers keep semantics when quantum lowering is replaced."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler.module_builder import CapturedModuleBuilder

pytestmark = pytest.mark.unit
pytest.importorskip("jax")


def build(p: torch.Tensor, x: torch.Tensor) -> fq.Circuit:
    return (
        fq.Circuit(2, bsz=x.shape[0])
        .ry(0, p[0].square() + p[1] * torch.sin(x[:, 0]))
        .rx(1, p[0].detach() * x[:, 1])
        .cx(0, 1)
    )


def analytic(p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return (p[0].square() + p[1] * x[:, 0].sin()).cos() * (
        1 + (p[0].detach() * x[:, 1]).cos()
    )


def module(
    builder=build,
    *,
    mode="statevector",
    dtype=torch.float64,
    backend="jax",
    qubits=(0, 1),
):
    return fq.Module(
        builder,
        2,
        init=torch.tensor([0.23, -0.31]),
        dtype=dtype,
        policy=fq.RuntimePolicy(
            execution_options=fq.ExecutionOptions(backend=backend, mode=mode),
            observable="z_sum",
            observable_qubits=qubits,
        ),
    )


@pytest.mark.parametrize("backend", ("pytorch", "jax"))
@pytest.mark.parametrize("mode", ("statevector", "mps", "tensor_network"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("batch", (1, 8))
def test_torch_shaped_recipe_reuses_capture_across_derivative_demands(
    mode, dtype, batch, backend
):
    calls = []

    def counted(p, x):
        calls.append(1)
        return build(p, x)

    model = module(counted, mode=mode, dtype=dtype, backend=backend)
    tolerance = 2e-5 if dtype == torch.float32 else 1e-10
    masks = ((True, True), (True, False), (False, True), (False, False), (True, True))
    for step, (parameter_grad, input_grad) in enumerate(masks):
        model.parameters_tensor.requires_grad_(parameter_grad)
        x = (
            torch.linspace(-0.3, 0.5, batch * 4, dtype=dtype).reshape(batch, 4)
            + step * 0.01
        )[:, ::2].requires_grad_(input_grad)
        assert not x.is_contiguous()
        p = model.parameters_tensor
        value = model(x)
        expected = analytic(p, x)
        torch.testing.assert_close(value, expected, atol=tolerance, rtol=tolerance)
        if parameter_grad or input_grad:
            targets = tuple(t for t in (p, x) if t.requires_grad)
            weights = torch.linspace(-0.7, 1.3, batch, dtype=dtype)
            actual_grad = torch.autograd.grad((value * weights).sum(), targets)
            expected_grad = torch.autograd.grad((expected * weights).sum(), targets)
            for actual, reference in zip(actual_grad, expected_grad, strict=True):
                torch.testing.assert_close(
                    actual, reference, atol=tolerance, rtol=tolerance
                )
        with torch.no_grad():
            p.add_(0.001)
    with torch.inference_mode():
        torch.testing.assert_close(
            model(x), analytic(p, x), atol=tolerance, rtol=tolerance
        )
    assert calls == [1]
    assert model._builder_compile_count == 1
    expected_lowering = (
        "canonical_ir_fx_bindings" if backend == "jax" else "torch_fx_bindings"
    )
    assert model.execute(x).runtime["builder_lowering"] == expected_lowering


@pytest.mark.parametrize(
    "legacy", (False, True), ids=("canonical-ir", "legacy-builder")
)
@pytest.mark.parametrize("mode", ("statevector", "mps", "tensor_network"))
def test_same_adam_and_recovery_consumer(legacy, mode, monkeypatch, tmp_path: Path):
    if legacy:
        monkeypatch.setattr(
            CapturedModuleBuilder,
            "jax_builder",
            lambda self: (None, "controlled_legacy_replacement"),
        )

    # This recipe is compatible with both historical JAX and Torch builders.
    def portable(p, x):
        return (
            fq.Circuit(2, bsz=x.shape[0])
            .ry(0, p[0] ** 2 + p[1] * x[:, 0])
            .rx(1, p[0] * x[:, 1])
            .cx(0, 1)
        )

    def oracle(p, x):
        return (p[0].square() + p[1] * x[:, 0]).cos() * (1 + (p[0] * x[:, 1]).cos())

    model = module(portable, mode=mode)
    p = model.parameters_tensor.detach().clone().requires_grad_()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    reference_optimizer = torch.optim.Adam([p], lr=0.03)
    restored = None
    for step in range(6):
        x = (
            torch.linspace(-0.3, 0.5, 16, dtype=torch.float64).reshape(8, 2)
            + step * 0.01
        )
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        target = torch.linspace(-0.2, 0.4, 8, dtype=torch.float64)
        value, reference = model(x), oracle(p, x)
        torch.testing.assert_close(value, reference, atol=1e-10, rtol=1e-10)
        (value - target).square().mean().backward()
        (reference - target).square().mean().backward()
        torch.testing.assert_close(
            model.parameters_tensor.grad, p.grad, atol=1e-10, rtol=1e-10
        )
        optimizer.step()
        reference_optimizer.step()
        torch.testing.assert_close(model.parameters_tensor, p, atol=1e-10, rtol=1e-10)
        if restored is not None:
            replay, replay_optimizer = restored
            replay_optimizer.zero_grad(set_to_none=True)
            (replay(x) - target).square().mean().backward()
            replay_optimizer.step()
            torch.testing.assert_close(
                replay.parameters_tensor,
                model.parameters_tensor,
                atol=1e-10,
                rtol=1e-10,
            )
            for key, value in optimizer.state_dict()["state"][0].items():
                torch.testing.assert_close(
                    value,
                    replay_optimizer.state_dict()["state"][0][key],
                    atol=1e-10,
                    rtol=1e-10,
                )
            assert replay._last_ir.content_hash == model._last_ir.content_hash
        if step == 2:
            path = model.save_checkpoint(
                tmp_path / "adam.pt", optimizer=optimizer, seed=11, step=3
            )
            replay = module(portable, mode=mode)
            replay_optimizer = torch.optim.Adam(replay.parameters(), lr=0.03)
            replay(x)
            replay.load_checkpoint(path, optimizer=replay_optimizer)
            assert replay._captured_builder is None
            restored = replay, replay_optimizer
    assert set(model.state_dict()) == {"parameters_tensor", "_extra_state"}
    expected_lowering = (
        "python_builder_jax_compatibility" if legacy else "canonical_ir_fx_bindings"
    )
    assert model.execute(x).runtime["builder_lowering"] == expected_lowering


def test_signature_invalidation_and_current_error_refusal():
    model = module()
    x = torch.zeros(8, 2, dtype=torch.float64)
    model(x)
    assert model._builder_compile_count == 1
    model(x[:3])
    assert model._builder_compile_count == 2
    model.set_runtime_policy(
        fq.RuntimePolicy(execution_options=fq.ExecutionOptions(backend="jax"))
    )
    model(x[:3])
    assert model._builder_compile_count == 3
    model.float()
    model(x[:3].float())
    assert model._builder_compile_count == 4
    model.circuit_builder = lambda p, x: fq.Circuit(2, bsz=x.shape[0]).ry(0, p[0] + 0.2)
    value = model(x[:3].float())
    torch.testing.assert_close(
        value,
        torch.cos(model.parameters_tensor[0] + 0.2).expand(3),
        atol=1e-6,
        rtol=1e-6,
    )
    assert model._builder_compile_count == 5
    consumed = model._last_ir.content_hash
    with torch.no_grad():
        model.parameters_tensor.fill_(float("nan"))
    with pytest.raises(fq.errors.ValidationError):
        model(x[:3].float())
    assert model._last_ir.content_hash == consumed


def test_explicit_casts_and_relu_boundary_keep_torch_vjp():
    def cast_and_relu(p, x):
        return fq.Circuit(1, bsz=x.shape[0]).ry(
            0, torch.relu(x[:, 0].float()).double() + p[0].square()
        )

    model = fq.Module(
        cast_and_relu,
        1,
        init=torch.tensor([0.2]),
        dtype=torch.float64,
        policy=fq.RuntimePolicy(execution_options=fq.ExecutionOptions(backend="jax")),
    )
    x = torch.tensor([[-0.2], [0.0], [0.3]], dtype=torch.float64, requires_grad=True)
    value = model(x)
    reference = (
        torch.relu(x[:, 0].float()).double() + model.parameters_tensor[0].square()
    ).cos()
    actual = torch.autograd.grad(value.sum(), (model.parameters_tensor, x))
    expected = torch.autograd.grad(reference.sum(), (model.parameters_tensor, x))
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-7)
    assert actual[1][1, 0] == 0


def test_unsupported_binding_retains_the_same_explicit_jax_backend():
    def product(p, x):
        return fq.Circuit(1, bsz=x.shape[0]).ry(0, p[0] + x.prod(1))

    model = module(product, qubits=(0,))
    x = torch.linspace(0.1, 0.8, 16, dtype=torch.float64).reshape(8, 2).requires_grad_()
    result = model.execute(x)
    assert result.runtime["builder_lowering"] == "python_builder_jax_compatibility"
    assert "prod" in result.runtime["builder_lowering_reason"]
    assert result.compatibility["selected_backend"] == "jax"
    assert result.compatibility["fallback_used"] is False
    expected = (model.parameters_tensor[0] + x.prod(1)).cos()
    torch.testing.assert_close(result.value, expected, atol=1e-10, rtol=1e-10)
    weights = torch.linspace(-0.7, 1.3, 8, dtype=torch.float64)
    actual_grad = torch.autograd.grad(
        (result.value * weights).sum(), (model.parameters_tensor, x)
    )
    expected_grad = torch.autograd.grad(
        (expected * weights).sum(), (model.parameters_tensor, x)
    )
    for a, b in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("inference_constant", (False, True))
def test_mutable_lifted_tensor_constant_invalidates_jax_without_recapture(
    inference_constant,
):
    with torch.inference_mode(inference_constant):
        constant = torch.tensor(0.2, dtype=torch.float64)

    def constant_builder(p, x):
        return fq.Circuit(1).ry(0, constant)

    model = module(constant_builder, qubits=(0,))
    x = torch.zeros(1, dtype=torch.float64)
    old_kernel = None
    for angle in (0.2, 0.4, -0.3):
        with torch.inference_mode(inference_constant):
            constant.fill_(angle)
        torch.testing.assert_close(
            model(x),
            torch.tensor([angle], dtype=torch.float64).cos(),
            atol=1e-10,
            rtol=1e-10,
        )
        assert model._jax_kernel is not old_kernel
        old_kernel = model._jax_kernel
    assert model._builder_compile_count == 1
    assert len(model._jax_constant_snapshots) == int(inference_constant)


@pytest.mark.parametrize("opcode", ("custom", "x"))
def test_explicit_matrix_is_refused_or_uses_the_authorized_torch_fallback(opcode):
    matrix = torch.eye(2, dtype=torch.complex128)

    def matrix_builder(p, x):
        return fq.Circuit(1).ry(0, p[0]).gate(opcode, (0,), matrix=matrix)

    model = module(matrix_builder, qubits=(0,))
    x = torch.zeros(1, dtype=torch.float64)
    with pytest.raises(fq.errors.CapabilityError, match="instruction matrices"):
        model(x)
    model.set_runtime_policy(
        fq.RuntimePolicy(
            execution_options=fq.ExecutionOptions(
                backend="jax", allow_backend_fallback=True
            ),
            observable="z_sum",
            observable_qubits=(0,),
        )
    )
    first = model.execute(x)
    assert first.compatibility["fallback_used"] is True
    assert first.compatibility["selected_backend"] == "pytorch"
    torch.testing.assert_close(
        first.value, model.parameters_tensor[:1].cos(), atol=1e-10, rtol=1e-10
    )
    matrix.copy_(torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128))
    torch.testing.assert_close(model(x), -first.value, atol=1e-10, rtol=1e-10)
    assert model._jax_kernel is None


def test_captured_trainable_tensor_is_refused_instead_of_losing_its_gradient():
    constant = torch.tensor(0.2, dtype=torch.float64)

    def constant_builder(p, x):
        return fq.Circuit(1).ry(0, constant)

    model = module(constant_builder, qubits=(0,))
    x = torch.zeros(1, dtype=torch.float64)
    model(x)
    constant.requires_grad_()
    with pytest.raises(fq.errors.CapabilityError, match="captured tensor constants"):
        model(x)


def test_hamiltonian_value_changes_rebuild_only_the_quantum_kernel_and_keep_fallback_gradients():
    from flagquantum.algorithms import Hamiltonian, HamiltonianTerm

    coefficient = torch.tensor(0.7, dtype=torch.float64)
    hamiltonian = Hamiltonian([HamiltonianTerm(coefficient, "Z", (0,))])
    model = fq.Module(
        lambda p, x: fq.Circuit(1).ry(0, p[0] + x[0]),
        1,
        init=torch.tensor([0.23]),
        dtype=torch.float64,
        hamiltonian=hamiltonian,
        policy=fq.RuntimePolicy(
            execution_options=fq.ExecutionOptions(backend="jax"),
            observable="hamiltonian",
        ),
    )
    x = torch.tensor([0.1], dtype=torch.float64)
    model(x)
    old_kernel = model._jax_kernel
    coefficient.fill_(0.4)
    torch.testing.assert_close(
        model(x),
        (coefficient * (model.parameters_tensor + x).cos()),
        atol=1e-10,
        rtol=1e-10,
    )
    assert model._jax_kernel is not old_kernel
    assert model._builder_compile_count == 1
    coefficient.requires_grad_()
    with pytest.raises(fq.errors.CapabilityError, match="Hamiltonian coefficients"):
        model(x)
    model.set_runtime_policy(
        fq.RuntimePolicy(
            execution_options=fq.ExecutionOptions(
                backend="jax", allow_backend_fallback=True
            ),
            observable="hamiltonian",
        )
    )
    result = model.execute(x)
    assert result.compatibility["selected_backend"] == "pytorch"
    assert result.compatibility["fallback_used"] is True
    assert result.runtime["builder_lowering"] == "torch_fx_bindings"
    actual = torch.autograd.grad(
        result.value.sum(), (model.parameters_tensor, coefficient)
    )
    expected = torch.autograd.grad(
        (coefficient * (model.parameters_tensor + x).cos()).sum(),
        (model.parameters_tensor, coefficient),
    )
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("mode", ("statevector", "mps", "tensor_network"))
def test_static_multiangle_and_controlled_gate_recipe_uses_the_same_public_consumer(
    mode,
):
    def family(p, x):
        return (
            fq.Circuit(2, bsz=x.shape[0], device=p.device)
            .h(0)
            .sx(1)
            .u3(0, p[0].square() + x[:, 0], p[1].sin(), p[0] * p[1])
            .cphase(0, 1, p[1] + x[:, 1])
            .crx(0, 1, p[0])
            .cry(1, 0, p[1])
            .crz(0, 1, p[0] * 0.3)
            .rxx(0, 1, p[1])
            .ryy(0, 1, p[0] + p[1])
            .rzz(0, 1, p[0] - x[:, 0])
            .sdg(1)
            .tdg(0)
            .swap(0, 1)
        )

    compiled, reference = module(family, mode=mode), module(
        family, mode=mode, backend="pytorch"
    )
    for step in range(2):
        x = (
            torch.linspace(-0.3, 0.5, 8, dtype=torch.float64).reshape(4, 2)
            + step * 0.01
        ).requires_grad_()
        reference_x = x.detach().clone().requires_grad_()
        weights = torch.tensor([-0.7, 1.3, 0.2, -0.4], dtype=torch.float64)
        actual, expected = compiled(x), reference(reference_x)
        torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)
        actual_grad = torch.autograd.grad(
            (actual * weights).sum(), (compiled.parameters_tensor, x)
        )
        reference_grad = torch.autograd.grad(
            (expected * weights).sum(), (reference.parameters_tensor, reference_x)
        )
        for a, b in zip(actual_grad, reference_grad, strict=True):
            torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)
        with torch.no_grad():
            compiled.parameters_tensor.add_(0.005)
            reference.parameters_tensor.add_(0.005)
    assert compiled._builder_compile_count == 1
    assert compiled._jax_kernel_build_count == 1


def test_subclass_build_lifecycle_is_not_bypassed_by_the_warm_binding_path():
    class Observed(fq.Module):
        def __init__(self):
            super().__init__(
                build,
                2,
                dtype=torch.float64,
                init=torch.tensor([0.23, -0.31]),
                policy=fq.RuntimePolicy(
                    execution_options=fq.ExecutionOptions(backend="jax"),
                    observable="z_sum",
                    observable_qubits=(0, 1),
                ),
            )
            self.build_calls = 0

        def _build(self, inputs, parameters):
            self.build_calls += 1
            return super()._build(inputs, parameters)

    model = Observed()
    for step in range(3):
        x = torch.tensor([[0.1, -0.2]], dtype=torch.float64) + step * 0.01
        torch.testing.assert_close(
            model(x), analytic(model.parameters_tensor, x), atol=1e-10, rtol=1e-10
        )
    assert model.build_calls == 3
    assert model._builder_compile_count == 1
