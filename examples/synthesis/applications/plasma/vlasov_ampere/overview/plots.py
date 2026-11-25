"""Plot the actual reconstructed operator and both original electric fields."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .workflow import Result


def save(result: Result, destination: Path) -> tuple[str, ...]:
    destination.mkdir(parents=True, exist_ok=True)
    names = (
        "encoded-operator.png",
        "quantum-electric-field.png",
        "classical-electric-field.png",
    )
    figure, axis = plt.subplots(figsize=(6.4, 5.5), layout="constrained")
    axis.spy(result.encoded_matrix.abs().numpy() > 1e-14, markersize=2)
    axis.set_title("Actual projected 128 by 128 native block operator")
    axis.set_xlabel("Input data basis index")
    axis.set_ylabel("Output data basis index")
    figure.savefig(destination / names[0], dpi=140)
    plt.close(figure)
    x = torch.linspace(0.0, 100.0, 8, dtype=torch.float64).numpy()
    for name, field, title in (
        (names[1], result.quantum_field, "Original degree-251 quantum inverse"),
        (names[2], result.classical_field, "Original classical Vlasov-Ampere solution"),
    ):
        figure, axis = plt.subplots(figsize=(8, 4.5), layout="constrained")
        axis.plot(x, field.abs().numpy(), "o-", color="#285fa4")
        axis.set_xlabel("x")
        axis.set_ylabel("|E(x)|")
        axis.set_title(title)
        figure.savefig(destination / name, dpi=140)
        plt.close(figure)
    return names
