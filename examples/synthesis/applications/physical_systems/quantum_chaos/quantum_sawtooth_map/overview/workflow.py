"""Signed kicked-map sampling, exact target costs and historical observations."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

import flagquantum as fq
from examples.synthesis.applications.physical_systems.quantum_chaos.quantum_sawtooth_map.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.applications.physical_systems.quantum_chaos.quantum_sawtooth_map.main import (
    build_program,
)
from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.qsvt.tket_qsvt.overview.exact_basis import (
    exact_rules,
    preserve_rotation_phases,
)
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate, estimate_resources
from flagquantum.compiler.basis_conversion import convert_basis
from flagquantum.core.ir import CircuitIR

if TYPE_CHECKING:
    from matplotlib.figure import Figure

TIMESTEPS = (2, 4, 6, 8)
DISPLAY_MOMENTA = tuple(range(-4, 5))


@dataclass(frozen=True)
class Observation:
    momentum: int
    count: int
    probability: float


@dataclass(frozen=True)
class Evolution:
    kicks: int
    program: synth.Program
    executable: CircuitIR
    resources: ResourceEstimate
    maximum_native_error: float
    maximum_target_error: float
    dirty_workspace_probability: float


@dataclass(frozen=True)
class SampledEvolution:
    evolution: Evolution
    observations: tuple[Observation, ...]
    exact_probabilities: tuple[float, ...]


def localization_parameters() -> tuple[float, float]:
    """Return the original kick bound and localization length without resizing."""
    return math.pi * 8 ** (-2 / 5), 3.3 * 0.1 ** (5 / 2) * (8 / 2 * math.pi) ** 2


def verify(kicks: int, *, max_added_operations: int = 10_000) -> Evolution:
    """Count a real CX/U3 conversion and compare absolute Circuit amplitudes."""
    program = build_program(kicks)
    converted = convert_basis(
        preserve_rotation_phases(program.circuit),
        gates=("cx", "u3"),
        max_added_operations=max_added_operations,
        rules=exact_rules(),
    )
    expected = fq.run(reference_circuit(kicks)).statevector().flatten()
    native = fq.run(program).statevector().flatten()
    target = fq.run(converted.program).statevector().flatten()
    torch.testing.assert_close(native, expected, atol=3e-11, rtol=0)
    torch.testing.assert_close(target, expected, atol=3e-11, rtol=0)
    resources = estimate_resources(converted.program)
    assert set(resources.operation_counts) <= {"cx", "u3"}
    assert resources.n_qubits == 3 and program.report["workspace_qubits"] == 0
    return Evolution(
        kicks,
        program,
        converted.program,
        resources,
        float(torch.max(torch.abs(native - expected))),
        float(torch.max(torch.abs(target - expected))),
        0.0,  # The complete three-wire state has no hidden workspace.
    )


def probabilities(program: synth.Program) -> tuple[float, ...]:
    """Read the eight signed momenta through the public named-register API."""
    result = fq.run(program)
    named = dict(program.numeric_probabilities(result, registers=("p",)))
    values = {int(key[0]): value for key, value in named.items()}
    return tuple(values.get(p, 0.0) for p in range(-4, 4))


def sample(
    values: tuple[float, ...], *, shots: int = 1000, seed: int = 1970
) -> tuple[Observation, ...]:
    if (
        len(values) != 8
        or any(isinstance(p, bool) or not math.isfinite(p) or p < 0 for p in values)
        or not math.isclose(sum(values), 1.0, abs_tol=1e-10)
    ):
        raise ValueError(
            "eight normalized probabilities for momenta -4..3 are required"
        )
    if type(shots) is not int or not 1 <= shots <= 1_000_000:
        raise ValueError("shots must be an integer in 1..1000000")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    draws = torch.multinomial(
        torch.tensor(values, dtype=torch.float64),
        shots,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    counts = torch.bincount(draws, minlength=8)
    return tuple(
        Observation(p, int(counts[i]), int(counts[i]) / shots)
        for i, p in enumerate(range(-4, 4))
    )


def run_workflow(
    *, shots: int = 1000, seed: int = 1970
) -> tuple[SampledEvolution, ...]:
    """Execute all four original times; each has its own actual sample batch."""
    rows = []
    for kicks in TIMESTEPS:
        evolution = verify(kicks)
        values = probabilities(evolution.program)
        rows.append(
            SampledEvolution(
                evolution, sample(values, shots=shots, seed=seed + kicks), values
            )
        )
    return tuple(rows)


def historical_observations(kicks: int) -> tuple[Observation, ...]:
    """Load original upstream measurements, never report them as fresh execution."""
    if type(kicks) is not int or kicks not in TIMESTEPS:
        raise ValueError("historical kicks must be one of 2, 4, 6, 8")
    directory = Path(__file__).resolve().parent
    proof = json.loads((directory / "provenance.json").read_text())
    relative = f"historical_ionq/df_{kicks}kicks.csv"
    asset = next(row for row in proof["historical_assets"] if row["path"] == relative)
    source = directory / relative
    if hashlib.sha256(source.read_bytes()).hexdigest() != asset["sha256"]:
        raise ValueError("historical measurement asset hash mismatch")
    with source.open(newline="") as handle:
        records = tuple(
            Observation(int(row["p"]), int(row["count"]), float(row["probability"]))
            for row in csv.DictReader(handle)
        )
    if (
        len(records) != 8
        or {row.momentum for row in records} != set(range(-4, 4))
        or sum(row.count for row in records) != 1000
        or any(
            row.count < 0 or abs(row.probability - row.count / 1000) > 1e-12
            for row in records
        )
    ):
        raise ValueError("invalid original eight-state, 1000-shot historical table")
    return tuple(sorted(records, key=lambda row: row.momentum))


def display_probabilities(rows: tuple[Observation, ...]) -> tuple[float, ...]:
    """Preserve the source plot's extra +4 padding point as zero, outside QNum."""
    values = {row.momentum: row.probability for row in rows}
    return tuple(values.get(p, 0.0) for p in DISPLAY_MOMENTA)


def plot_comparison(rows: tuple[SampledEvolution, ...]) -> Figure:
    """Compare actual native samples with explicitly historical IonQ measurements."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(10, 6))
    for row in rows:
        kicks = row.evolution.kicks
        axes.plot(
            DISPLAY_MOMENTA,
            display_probabilities(row.observations),
            ":o",
            color="gray",
            alpha=0.6,
        )
        axes.plot(
            DISPLAY_MOMENTA,
            display_probabilities(historical_observations(kicks)),
            "-o",
            label=f"Historical upstream IonQ, T={kicks}",
        )
    axes.plot(
        [], [], ":o", color="gray", label="FlagQuantum CPU, T=2/4/6/8; 1000 draws each"
    )
    axes.set_title("Signed sawtooth map: local samples and historical qpu.forte-1 data")
    axes.set_xlabel("Momentum p (+4 is display padding)")
    axes.set_ylabel("Observed probability")
    axes.legend(fontsize=9)
    figure.tight_layout()
    return figure
