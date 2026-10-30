"""Public native dynamics retain device execution, gradients and plan replay."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import flagquantum as fq
import flagquantum.lindblad as fql
from flagquantum.experimental.simulation import IsingBraidProgram, RydbergProgram
from flagquantum.runtime import sample_terminal
from flagquantum.simulation.lindblad import EvolutionResult

pytestmark = [pytest.mark.integration, pytest.mark.gpu, pytest.mark.distributed_accel]


@pytest.fixture(scope="module", autouse=True)
def activate_ascend() -> None:
    if os.environ.get("FLAGQUANTUM_TEST_ASCEND_910C") != "1":
        pytest.skip("set FLAGQUANTUM_TEST_ASCEND_910C=1 on the documented stack")
    importlib.import_module("torch_fl")


def ising(sampled: bool = False, dephase: bool = False) -> dict[str, object]:
    program = IsingBraidProgram().exchange(1)
    if dephase:
        program.dephase_pair()
    if sampled:
        program.measure_pair()
    program.exchange(0, clockwise=True)
    outputs = fq.expectation(program.projector("psi"), name="psi")
    options = fq.ExecutionOptions(
        device="flagos:0", shots=1 if sampled else None, seed=17 if sampled else None
    )
    plan = fq.plan(program, outputs=outputs, options=options)
    restored = fq.ExecutionPlan.from_json(plan.to_json())
    before_cpu = torch.get_rng_state().clone()
    before_device = torch.flagos.get_rng_state(0).clone()
    actual, replay = fq.run(plan), fq.run(restored)
    assert (
        actual.state.device.type == "flagos" and actual.runtime["device"] == "flagos:0"
    )
    torch.testing.assert_close(actual.state.cpu(), replay.state.cpu(), atol=0, rtol=0)
    assert torch.equal(before_cpu, torch.get_rng_state())
    assert torch.equal(before_device, torch.flagos.get_rng_state(0))
    if sampled:
        event = actual.measurement("mid_circuit_measurement")
        index = int(event.value.cpu()[0, 0])
        expected = torch.diag(
            torch.tensor([index == 0, index == 1], dtype=torch.complex128)
        ).unsqueeze(0)
    else:
        expected = fq.run(program, outputs=outputs).state
    torch.testing.assert_close(actual.state.cpu(), expected, atol=2e-11, rtol=0)
    return {"device": str(actual.state.device), "replay": True}


def lindblad(method: str, dtype: torch.dtype) -> dict[str, object]:
    def execute(device: str) -> tuple[EvolutionResult, torch.Tensor]:
        real = torch.float64 if dtype == torch.complex128 else torch.float32
        parameter = torch.tensor(0.23, dtype=real, device=device, requires_grad=True)
        x = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
        z = torch.diag(torch.tensor([1, -1], dtype=dtype, device=device))
        result = fql.run(
            parameter * x,
            "0",
            [0.0, 0.05, 0.1],
            outputs=fq.expectation(z, name="z"),
            options=fq.ExecutionOptions(
                device=device,
                precision=str(dtype).split(".")[-1],
                require_gradients=True,
            ),
            method=method,
        )
        result.expectation("z")[-1].backward()
        return result, parameter.grad

    expected, reference_gradient = execute("cpu")
    actual, gradient = execute("flagos:0")
    tolerance = 7e-5 if dtype == torch.complex64 else 3e-10
    assert actual.populations.device.type == gradient.device.type == "flagos"
    torch.testing.assert_close(
        actual.populations.cpu(), expected.populations, atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(
        gradient.cpu(), reference_gradient, atol=tolerance, rtol=tolerance
    )
    return {"device": str(actual.populations.device), "gradient": float(gradient.cpu())}


def terminal(dtype: torch.dtype) -> dict[str, object]:
    density = torch.eye(2, dtype=dtype, device="flagos:0") / 2
    instrument = {
        str(index): (
            torch.diag(
                torch.tensor(
                    [index == 0, index == 1], dtype=dtype, device=density.device
                )
            ),
        )
        for index in range(2)
    }
    state = torch.flagos.get_rng_state(0).clone()
    actual = sample_terminal(density, instrument, shots=64, seed=31)
    repeated = sample_terminal(density, instrument, shots=64, seed=31)
    assert actual.value.device == density.device
    assert torch.equal(actual.value.cpu(), repeated.value.cpu())
    assert torch.equal(state, torch.flagos.get_rng_state(0))
    return {"device": str(actual.value.device), "shape": list(actual.value.shape)}


def rydberg(presence: bool = False, method: str = "runge-kutta") -> dict[str, object]:
    def execute(device: str) -> torch.Tensor:
        program = RydbergProgram(
            atoms=(("a", "origin"),),
            sites={"origin": (0.0, 0.0)},
            length_unit="um",
            time_unit="us",
            c6=64.0,
            loss_rate=0.1,
            minimum_separation=1.0,
            drive_channels=("laser",),
        )
        program.pulse(
            ("a",),
            start=0.0,
            duration=0.1,
            amplitude=0.8,
            detuning=0.2,
            phase=0.3,
            channel="laser",
        )
        if presence:
            program.measure_presence("a", at=0.05, efficiency=0.8)
        options = fq.ExecutionOptions(
            device=device, shots=1 if presence else None, seed=17 if presence else None
        )
        initial = torch.eye(3, dtype=torch.complex128) / 3
        plan = fql.plan(
            program,
            initial,
            [0.0, 0.05, 0.1],
            options=options,
            outputs=fq.expectation(program.survival_operator(), name="survival"),
            return_density_matrices=True,
            method=method,
        )
        result = fq.run(plan) if presence else fql.run(plan)
        restored = fql.LindbladPlan.from_json(plan.to_json())
        replay = fq.run(restored) if presence else fql.run(restored)
        tensor = result.state if presence else result.density_matrices
        repeated = replay.state if presence else replay.density_matrices
        torch.testing.assert_close(tensor.cpu(), repeated.cpu(), atol=0, rtol=0)
        return tensor

    expected = execute("cpu")
    actual = execute("flagos:0")
    assert actual.device.type == "flagos"
    torch.testing.assert_close(actual.cpu(), expected, atol=3e-10, rtol=3e-10)
    return {"device": str(actual.device), "replay": True}


@pytest.mark.parametrize(
    ("sampled", "dephase"), [(False, False), (False, True), (True, False)]
)
def test_ising_actions_sampling_and_replay(sampled: bool, dephase: bool) -> None:
    ising(sampled=sampled, dephase=dephase)


@pytest.mark.parametrize("native", [False, True])
def test_cold_plan_replay_activates_provider(native: bool, tmp_path: Path) -> None:
    if native:
        program = RydbergProgram(
            atoms=(("a", "origin"),),
            sites={"origin": (0.0, 0.0)},
            length_unit="um",
            time_unit="us",
            c6=64.0,
            loss_rate=0.1,
            minimum_separation=1.0,
            drive_channels=("laser",),
        )
        program.pulse(
            ("a",),
            start=0.0,
            duration=0.1,
            amplitude=0.8,
            detuning=0.2,
            phase=0.3,
            channel="laser",
        )
        plan = fql.plan(
            program,
            program.basis_state(("g",)),
            [0.0, 0.1],
            options=fq.ExecutionOptions(device="flagos:0"),
        )
    else:
        plan = fql.plan(
            torch.tensor([[0.0, 0.2], [0.2, 0.0]], dtype=torch.complex128),
            "0",
            [0.0, 0.1],
            options=fq.ExecutionOptions(device="flagos:0"),
        )
    plan_path, output_path = tmp_path / "plan.json", tmp_path / "populations.pt"
    plan_path.write_text(plan.to_json())
    expected = fql.run(plan).populations.cpu()
    code = """
import sys
from pathlib import Path
import torch
import flagquantum.lindblad as fql
assert 'torch_fl' not in sys.modules
plan = fql.LindbladPlan.from_json(Path(sys.argv[1]).read_text())
result = fql.run(plan)
assert result.populations.device.type == 'flagos'
torch.save(result.populations.cpu(), sys.argv[2])
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(plan_path), str(output_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "[flagos cpu_fallback]" not in completed.stdout + completed.stderr
    torch.testing.assert_close(
        torch.load(output_path, weights_only=True), expected, atol=3e-10, rtol=3e-10
    )


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("method", fql.INTEGRATOR_NAMES)
def test_evolution_gradient_matches_cpu(dtype: torch.dtype, method: str) -> None:
    lindblad(method, dtype)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_seeded_terminal_instruments_preserve_rng(dtype: torch.dtype) -> None:
    terminal(dtype)


@pytest.mark.parametrize("method", fql.INTEGRATOR_NAMES)
@pytest.mark.parametrize("presence", [False, True])
def test_native_rydberg_replay_matches_cpu(presence: bool, method: str) -> None:
    rydberg(presence=presence, method=method)


@pytest.mark.parametrize("method", fql.INTEGRATOR_NAMES)
def test_native_control_optimizer_matches_cpu(method: str) -> None:
    def trajectory(device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        parameter = torch.tensor(
            0.8, dtype=torch.float64, device=device, requires_grad=True
        )
        optimizer = torch.optim.SGD([parameter], lr=0.02)
        losses = []
        gradients = []
        for _ in range(3):
            program = RydbergProgram(
                atoms=(("a", "origin"),),
                sites={"origin": (0.0, 0.0)},
                length_unit="um",
                time_unit="us",
                c6=64.0,
                loss_rate=0.1,
                minimum_separation=1.0,
                drive_channels=("laser",),
            )
            program.pulse(
                ("a",),
                start=0.0,
                duration=0.1,
                amplitude=fq.Parameter("omega"),
                detuning=0.2,
                phase=0.3,
                channel="laser",
            )
            optimizer.zero_grad()
            result = fql.run(
                program,
                program.basis_state(("g",)),
                [0.0, 0.05, 0.1],
                parameter_values=[{"omega": parameter}] * 2,
                outputs=fq.expectation(program.projector({"a": "r"}), name="r"),
                options=fq.ExecutionOptions(device=device, require_gradients=True),
                method=method,
            )
            loss = (result.expectation("r")[-1] - 0.2).square()
            loss.backward()
            assert parameter.grad.device == parameter.device
            gradients.append(parameter.grad.detach().cpu().clone())
            losses.append(loss.detach().cpu())
            optimizer.step()
        return torch.stack(losses), torch.stack(gradients), parameter.detach().cpu()

    expected = trajectory("cpu")
    actual = trajectory("flagos:0")
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, atol=3e-10, rtol=3e-10)


@pytest.mark.parametrize("method", fql.INTEGRATOR_NAMES)
def test_adjoint_initial_cotangent_matches_cpu(method: str) -> None:
    hamiltonian = torch.tensor([[0.2, 0.1], [0.1, -0.2]], dtype=torch.complex128)
    initial = torch.eye(2, dtype=torch.complex128) / 2

    def cost(states: torch.Tensor) -> torch.Tensor:
        return states[-1, 1, 1].real

    expected = fql.adjoint_gradient(
        hamiltonian, initial, [0.0, 0.05, 0.1], cost, method=method
    )
    actual = fql.adjoint_gradient(
        hamiltonian,
        initial,
        [0.0, 0.05, 0.1],
        cost,
        method=method,
        options=fq.ExecutionOptions(device="flagos:0"),
    )
    assert actual.device.type == "flagos"
    torch.testing.assert_close(actual.cpu(), expected, atol=3e-10, rtol=3e-10)
