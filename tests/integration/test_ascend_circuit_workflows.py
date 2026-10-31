"""Opt-in native circuit and training acceptance on the pinned 910C stack."""

from __future__ import annotations

import copy
import importlib
import math
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

import flagquantum as fq
import flagquantum.noise as noise
from flagquantum.algorithms import transverse_field_ising
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS
from flagquantum.qec import DetectorErrorModel, get_decoder
from flagquantum.qec.decoders._ai.predecoder import Predecoder
from flagquantum.qec.decoders._ai.training import Trainer

pytestmark = [pytest.mark.integration, pytest.mark.gpu, pytest.mark.distributed_accel]
MODES = ("statevector", "mps", "tensor_network", "density_matrix")
DTYPES = (torch.complex64, torch.complex128)
GATES = tuple(
    name
    for name in OPERATOR_SCHEMAS
    if name not in {"bit_flip", "phase_flip", "depolarizing", "amplitude_damping"}
)
NOISE_FACTORIES = {
    "bit_flip_channel": (0.2,),
    "phase_flip_channel": (0.2,),
    "y_flip_channel": (0.2,),
    "depolarizing_channel": (0.2,),
    "two_qubit_depolarizing_channel": (0.2,),
    "amplitude_damping_channel": (0.2,),
    "phase_damping_channel": (0.2,),
    "reset_error_channel": (0.15, 0.1),
    "coherent_overrotation_channel": (0.2,),
    "thermal_relaxation_channel": (2.0, 1.5, 0.2),
}


@pytest.fixture(scope="module", autouse=True)
def activate_ascend() -> Iterator[None]:
    if os.environ.get("FLAGQUANTUM_TEST_ASCEND_910C") != "1":
        pytest.skip("set FLAGQUANTUM_TEST_ASCEND_910C=1 on the documented stack")
    importlib.import_module("torch_fl")
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def _options(mode: str, device: str, dtype: torch.dtype) -> fq.ExecutionOptions:
    return fq.ExecutionOptions(
        mode=mode,
        device=device,
        precision=str(dtype).split(".")[-1],
        allow_backend_fallback=False,
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
def test_probabilities(mode: str, dtype: torch.dtype) -> None:
    circuit = fq.Circuit(3, dtype=dtype).h(0).rx(1, 0.23).ry(2, -0.47)
    circuit.cx(0, 1).cz(1, 2).rz(0, 0.18)
    expected = fq.run(
        circuit, options=_options(mode, "cpu", dtype), outputs=fq.probabilities()
    ).probabilities
    execution = fq.run(
        circuit, options=_options(mode, "flagos:0", dtype), outputs=fq.probabilities()
    )
    assert execution.runtime["mode"] == mode
    assert not execution.compatibility.get("fallback_used", False)
    assert execution.probabilities.device.type == "flagos"
    tolerance = 2e-5 if dtype == torch.complex64 else 1e-11
    torch.testing.assert_close(
        execution.probabilities.cpu(), expected, atol=tolerance, rtol=tolerance
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
def test_autograd(mode: str, dtype: torch.dtype) -> None:
    real = torch.float32 if dtype == torch.complex64 else torch.float64
    theta = torch.tensor(0.37, dtype=real, device="flagos:0", requires_grad=True)
    circuit = fq.Circuit(3, dtype=dtype).h(0).ry(1, theta).cx(1, 2).cz(0, 1)
    execution = fq.run(
        circuit,
        options=_options(mode, "flagos:0", dtype),
        outputs=fq.expectation(fq.Z(1)),
    )
    value = execution.expectation()
    value.sum().backward()
    assert value.device.type == "flagos" and theta.grad.device.type == "flagos"
    tolerance = 2e-5 if dtype == torch.complex64 else 1e-11
    assert abs(float(theta.grad.cpu()) + math.sin(0.37)) < tolerance


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
def test_seeded_samples(mode: str, dtype: torch.dtype) -> None:
    circuit = fq.Circuit(3, dtype=dtype).h(0).cx(0, 1).cx(1, 2)
    options = fq.ExecutionOptions(
        mode=mode,
        device="flagos:0",
        precision=str(dtype).split(".")[-1],
        shots=64,
        seed=13,
        allow_backend_fallback=False,
    )
    first = fq.run(circuit, options=options, outputs=fq.samples()).require_samples()
    second = fq.run(circuit, options=options, outputs=fq.samples()).require_samples()
    assert first.device.type == "flagos" and first.shape == (1, 64, 3)
    bits = first.cpu()
    assert torch.equal(bits, second.cpu())
    assert bool(torch.all(bits[..., 0] == bits[..., 1]))
    assert bool(torch.all(bits[..., 1] == bits[..., 2]))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", GATES)
def test_registered_gate(name: str, dtype: torch.dtype) -> None:
    schema = OPERATOR_SCHEMAS[name]

    def execute(device: str) -> torch.Tensor:
        circuit = fq.Circuit(max(3, schema.arity), device=device, dtype=dtype)
        for wire in range(circuit.n_qubits):
            circuit.h(wire).ry(wire, 0.17 * (wire + 1))
        getattr(circuit, name)(
            *range(schema.arity),
            **{
                parameter: 0.21 + 0.13 * i
                for i, parameter in enumerate(schema.parameters)
            },
        )
        return circuit.state()

    expected, actual = execute("cpu"), execute("flagos:0")
    assert actual.device.type == "flagos"
    tolerance = 2e-5 if dtype == torch.complex64 else 1e-11
    torch.testing.assert_close(actual.cpu(), expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
def test_entropy(mode: str, dtype: torch.dtype) -> None:
    result = fq.run(
        fq.Circuit(2, dtype=dtype).h(0).cx(0, 1),
        options=_options(mode, "flagos:0", dtype),
        outputs=[fq.vn_entropy(qubits=0), fq.density_matrix(qubits=0)],
    )
    assert result.vn_entropy.device.type == "flagos"
    tolerance = 2e-5 if dtype == torch.complex64 else 1e-11
    assert abs(float(result.vn_entropy.cpu()) - math.log(2)) < tolerance


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "method", ["autograd", "parameter_shift", "finite_difference", "spsa"]
)
def test_gradient_methods(method: str, dtype: torch.dtype) -> None:
    parameters = torch.tensor([0.37], dtype=dtype, device="flagos:0")

    def build(values: torch.Tensor) -> fq.Circuit:
        complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        return (
            fq.Circuit(2, device=values.device, dtype=complex_dtype)
            .ry(0, values[0])
            .cx(0, 1)
        )

    def loss(circuit: fq.Circuit) -> torch.Tensor:
        return circuit.expectation_z(0).sum()

    result = fq.gradient(
        build,
        parameters,
        loss,
        method=method,
        **({"directions": 2, "step": 1e-3} if method == "spsa" else {}),
    )
    assert result.gradient.device.type == "flagos"
    tolerance = 3e-3 if dtype == torch.float32 else 1e-5
    assert abs(float(result.gradient.cpu()) + math.sin(0.37)) < tolerance


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_explicit_sampling_generator(dtype: torch.dtype) -> None:
    probabilities = torch.tensor([[0.1, 0.2, 0.0, 0.7]], dtype=dtype, device="flagos:0")

    def draw(seed: int) -> torch.Tensor:
        generator = torch.Generator(device="flagos:0").manual_seed(seed)
        return torch.multinomial(probabilities, 256, True, generator=generator).cpu()

    first = draw(13)
    assert torch.equal(first, draw(13))
    assert not torch.equal(first, draw(29))
    assert not bool((first == 2).any())


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", NOISE_FACTORIES)
def test_noise_channel(name: str, dtype: torch.dtype) -> None:
    def execute(device: str) -> torch.Tensor:
        channel = getattr(noise, name)(
            *NOISE_FACTORIES[name], dtype=dtype, device=device
        )
        two_wire = name == "two_qubit_depolarizing_channel"
        circuit = fq.Circuit(2, device=device, dtype=dtype).h(0).ry(1, 0.31).cx(0, 1)
        if not two_wire:
            circuit.x(1)
        model = noise.NoiseModel().add("cx" if two_wire else "x", channel)
        return fq.run(
            circuit,
            noise_model=model,
            options=_options("density_matrix", device, dtype),
            outputs=fq.probabilities(),
        ).probabilities

    expected, actual = execute("cpu"), execute("flagos:0")
    assert actual.device.type == "flagos"
    tolerance = 3e-5 if dtype == torch.complex64 else 2e-11
    torch.testing.assert_close(actual.cpu(), expected, atol=tolerance, rtol=tolerance)


def test_hybrid_module_adam_training() -> None:
    def quantum(parameters: torch.Tensor, inputs: torch.Tensor) -> fq.Circuit:
        return (
            fq.Circuit(2, bsz=inputs.shape[0], device=parameters.device)
            .ry(0, parameters[0] + inputs[:, 0])
            .rx(1, parameters[1] + inputs[:, 1])
            .cx(0, 1)
        )

    classical = torch.nn.Linear(2, 2).to("flagos:0")
    quantum_model = fq.Module(
        quantum, n_parameters=2, init=torch.tensor([0.2, -0.3]), device="flagos:0"
    )
    parameters = [*classical.parameters(), *quantum_model.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=0.02)
    inputs = torch.tensor([[0.1, 0.2], [-0.2, 0.4], [0.3, -0.1]], device="flagos:0")
    for _ in range(8):
        optimizer.zero_grad()
        loss = quantum_model(classical(inputs)).square().mean()
        loss.backward()
        assert all(
            p.grad is not None and p.grad.device.type == "flagos" for p in parameters
        )
        optimizer.step()
        assert math.isfinite(float(loss.detach().cpu()))


@pytest.mark.parametrize("dtype", DTYPES)
def test_tebd(dtype: torch.dtype) -> None:
    hamiltonian = transverse_field_ising(4, coupling=1.0, field=0.7, periodic=False)
    options = dict(
        n_qubits=4,
        total_time=0.04,
        time_step=0.02,
        max_bond=8,
        cutoff=0.0,
        initial_state="+x",
        dtype=dtype,
    )
    expected = fq.experimental.simulation.run_tebd(hamiltonian, device="cpu", **options)
    actual = fq.experimental.simulation.run_tebd(
        hamiltonian, device="flagos:0", **options
    )
    assert actual.actual_device == "flagos:0" and actual.fallback_used is False
    tolerance = 6e-5 if dtype == torch.complex64 else 3e-10
    assert abs(actual.initial_energy - expected.initial_energy) < tolerance
    assert abs(actual.final_energy - expected.final_energy) < tolerance


def test_qec_predecoder_checkpoint_and_global_decode(tmp_path: Path) -> None:
    dem = DetectorErrorModel.from_stim_text(
        "error(0.1) D0 L0\nerror(0.15) D1\ndetector D0\ndetector D1\nlogical_observable L0\n"
    )
    torch.manual_seed(11)
    reference = Predecoder(dem)
    model = copy.deepcopy(reference).to("flagos:0")
    bits = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.int8)
    labels = bits.to(torch.float32)
    cpu_optimizer = torch.optim.Adam(reference.parameters(), lr=0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(3):
        cpu_optimizer.zero_grad()
        optimizer.zero_grad()
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            reference(bits), labels
        )
        actual = torch.nn.functional.binary_cross_entropy_with_logits(
            model(bits.to("flagos:0")), labels.to("flagos:0")
        )
        expected.backward()
        actual.backward()
        for left, right in zip(model.parameters(), reference.parameters(), strict=True):
            assert left.grad.device.type == "flagos"
            torch.testing.assert_close(
                left.grad.cpu(), right.grad, atol=3e-5, rtol=3e-5
            )
        cpu_optimizer.step()
        optimizer.step()
        for left, right in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(left.cpu(), right, atol=3e-5, rtol=3e-5)
    trainer = Trainer(model, seed=7)
    trainer.train(2, batch_size=16)
    trainer.save(tmp_path / "checkpoint.pt")
    expected_losses = trainer.train(2, batch_size=16)
    resumed = Trainer(copy.deepcopy(reference).to("flagos:0"), seed=99)
    resumed.restore(tmp_path / "checkpoint.pt")
    assert resumed.train(2, batch_size=16) == expected_losses
    for left, right in zip(
        trainer.model.parameters(), resumed.model.parameters(), strict=True
    ):
        torch.testing.assert_close(left.cpu(), right.cpu(), atol=0, rtol=0)
    result = model.decode(
        bits.to("flagos:0"), get_decoder("minimum_weight_matching", dem)
    )
    assert result.device.type == "flagos"
