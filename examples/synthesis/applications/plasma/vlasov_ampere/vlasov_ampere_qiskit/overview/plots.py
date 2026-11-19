"""Compare the complete original-grid expected and projected amplitudes."""

from pathlib import Path

import matplotlib.pyplot as plt

from .workflow import Result


def save(result: Result, destination: Path) -> tuple[str, ...]:
    destination.mkdir(parents=True, exist_ok=True)
    name = "block-projection.png"
    figure, axis = plt.subplots(figsize=(10, 4.8), layout="constrained")
    axis.plot(
        range(128),
        [abs(value) for value in result.expected],
        "o",
        markersize=4,
        label="Fresh classical operator applied to input",
    )
    axis.plot(
        range(128),
        [abs(value) for value in result.projected],
        ".",
        markersize=4,
        label="Native projected block times normalization",
    )
    axis.set_xlabel("Original data basis index, v + 8x + 64E")
    axis.set_ylabel("Absolute amplitude")
    axis.set_title("Vlasov-Ampere block projection on the full 8 by 8 grid")
    axis.legend()
    figure.savefig(destination / name, dpi=140)
    plt.close(figure)
    return (name,)
