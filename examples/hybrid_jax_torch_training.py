"""PyTorch training loop with a FlagQuantum JAX quantum kernel.

This example keeps PyTorch as the user-facing training interface while
FlagQuantum executes the quantum kernel through its optional JAX backend.
"""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import flagquantum as fq  # noqa: E402


def circuit_builder(theta: torch.Tensor) -> fq.Circuit:
    circuit = fq.Circuit(3)
    circuit.rx(0, theta=theta[0])
    circuit.ry(1, theta=theta[1])
    circuit.rz(2, theta=theta[2])
    circuit.cx(0, 1)
    circuit.rxx(1, 2, theta=theta[3])
    circuit.rzz(0, 2, theta=theta[4])
    return circuit


def main() -> None:
    hamiltonian = fq.Hamiltonian(
        [
            fq.pauli_term(0.7, "ZZ", (0, 1)),
            fq.pauli_term(-0.3, "X", (0,)),
            fq.pauli_term(0.2, "YY", (1, 2)),
        ]
    )
    layer = fq.Module(
        circuit_builder,
        n_parameters=5,
        init=torch.linspace(-0.2, 0.2, steps=5),
        policy=fq.RuntimePolicy(
            execution_options=fq.ExecutionOptions(
                backend="jax", allow_backend_fallback=False
            ),
            observable="hamiltonian",
            observable_wires=(0, 1, 2),
        ),
        hamiltonian=hamiltonian,
    )
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.05)

    target = torch.tensor(-0.25)
    for step in range(10):
        optimizer.zero_grad()
        prediction = layer()
        loss = (prediction - target) ** 2
        loss.backward()
        optimizer.step()
        print(
            {
                "step": step,
                "prediction": float(prediction.detach()),
                "loss": float(loss.detach()),
            }
        )

    batched_theta = torch.stack(
        (
            layer.parameters_tensor.detach(),
            layer.parameters_tensor.detach() + 0.05,
            layer.parameters_tensor.detach() - 0.05,
        )
    )
    batched_predictions = torch.stack(
        tuple(layer.execute(parameters=theta).value for theta in batched_theta)
    )
    print({"batched_predictions": batched_predictions.detach().tolist()})
    print(layer.execute().summary())


if __name__ == "__main__":
    main()
