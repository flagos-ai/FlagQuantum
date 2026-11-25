"""Build and sample the original one-step alanylalanine walk."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import flagquantum as fq
from examples.synthesis.applications.chemistry.protein_folding.protein_folding_with_quantum_walk.qfold.main import (
    build_program,
)
from flagquantum.synth import Program

SHOTS = 2048
SEED = 7


@dataclass(frozen=True)
class Dataset:
    protein: str
    rotation_bits: int
    phi_angles: tuple[float, ...]
    psi_angles: tuple[float, ...]
    energy_differences: dict[str, float]
    psi4_min_energy: float

    @property
    def move_bits(self) -> int:
        return (len(self.phi_angles) + len(self.psi_angles) - 1).bit_length()

    @property
    def encoding_width(self) -> int:
        return (
            self.rotation_bits * (len(self.phi_angles) + len(self.psi_angles))
            + self.move_bits
            + 1
        )


def dataset() -> Dataset:
    """Load the original notebook's precomputed energy data."""
    raw = json.loads(Path(__file__).with_name("alanylalanine.json").read_text())
    initial = raw["initialization_stats"]
    return Dataset(
        raw["protein"],
        raw["numberBitsRotation"],
        tuple(initial["phi_angles_psi4"]),
        tuple(initial["psi_angles_psi4"]),
        raw["deltas"],
        raw["psi4_min_energy"],
    )


def coin_angles(data: Dataset, beta: float = 1.0) -> tuple[float, float]:
    """Return the angles actually used in move-zero and move-one order."""
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")

    def averaged_rotation(keys: tuple[str, str]) -> float:
        acceptance = sum(
            math.exp(-beta * max(0.0, data.energy_differences[key])) for key in keys
        ) / len(keys)
        return 2 * math.asin(math.sqrt(acceptance))

    # The original cell assigns the first computed angle to data_psi.
    return averaged_rotation(("00101", "10101")), averaged_rotation(("00001", "01001"))


def build(angles: tuple[float, float]) -> Program:
    """Synthesize the original one-step program using freshly calculated angles."""
    program: Program = build_program(*angles)
    return program


def sample(program: Program) -> dict[tuple[int, int], int]:
    registers = (*program.registers["phi"], *program.registers["psi"])
    result = fq.run(
        program.circuit,
        options=fq.ExecutionOptions(mode="statevector", shots=SHOTS, seed=SEED),
        outputs=fq.counts(registers),
    )
    return {
        (int(str(word)[0]), int(str(word)[1])): count
        for word, count in cast(dict[str | int, int], result.counts[0]).items()
    }


def energy_lookup(data: Dataset) -> dict[str, float]:
    """Reproduce the source's final lookup for the 01 prefix."""
    return {
        key: value
        for key, value in data.energy_differences.items()
        if key.startswith("01")
    }
