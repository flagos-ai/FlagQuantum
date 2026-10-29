"""Train a split-FP32 quantum circuit with explicit parameter-shift SGD.

Run with ``--device flagos:0`` in the documented Ascend 910C environment,
or use ``--device cpu`` to inspect the same workflow locally.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RotationTrainingResult:
    device: str
    steps: int
    initial_expectation: float
    final_expectation: float
    final_angle: float
    maximum_gradient_absolute_error: float
    state_device: str
    gradient_device: str
    optimizer: str = "explicit_sgd"


def train_rotation(
    device: str = "cpu", *, steps: int = 64, learning_rate: float = 0.2
) -> RotationTrainingResult:
    """Minimize the analytically known expectation cos(theta) toward -1."""
    if steps < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("steps and learning_rate must be positive")
    if device.startswith("flagos"):
        importlib.import_module("torch_fl")
    import torch

    import flagquantum as fq
    from flagquantum.algorithms import pauli_term

    torch.set_float32_matmul_precision("highest")
    circuit = fq.Circuit(2).ry(0, fq.Parameter("theta")).cx(0, 1)
    observable = pauli_term(1.0, "Z", (0,))
    theta = torch.tensor(0.25, dtype=torch.float32, device=device)
    maximum_error = 0.0
    initial_expectation = math.cos(0.25)
    numerics = fq.experimental.numerics
    for _ in range(steps):
        result = numerics.parameter_shift_split_real_imag_gradient(
            circuit,
            observable,
            parameter_bindings={"theta": theta},
            device=device,
        )
        gradient = result.gradient[0]
        if gradient.device != theta.device:
            raise RuntimeError("quantum gradient left the parameter device")
        # This scalar transfer is an explicit analytic diagnostic. The state,
        # gradient and SGD arithmetic stay on the selected execution device.
        error = float((gradient + torch.sin(theta)).abs().detach().cpu())
        maximum_error = max(maximum_error, error)
        theta = theta - learning_rate * gradient
    final = numerics.execute_split_real_imag_expectation(
        circuit,
        observable,
        parameter_bindings={"theta": theta},
        device=device,
    )
    return RotationTrainingResult(
        device=str(theta.device),
        steps=steps,
        initial_expectation=initial_expectation,
        final_expectation=float(final.value.detach().cpu()),
        final_angle=float(theta.detach().cpu()),
        maximum_gradient_absolute_error=maximum_error,
        state_device=str(final.state.real.device),
        gradient_device=str(gradient.device),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "flagos:0"), default="cpu")
    parser.add_argument("--steps", type=int, default=64)
    args = parser.parse_args()
    result = train_rotation(args.device, steps=args.steps)
    if result.maximum_gradient_absolute_error > 2e-5:
        raise RuntimeError("parameter-shift gradient failed its analytic reference")
    print(json.dumps(asdict(result), sort_keys=True))


if __name__ == "__main__":
    main()
