"""Evaluate all eleven source times and preserve signed postselection amplitudes."""

import math
from dataclasses import dataclass

from flagquantum.compiler import ResourceEstimate

from .model import GRID, exact_solution, taylor_encoding
from .quantum import build_program, logical_state


@dataclass(frozen=True)
class Observation:
    """The quantum and analytic values for one time in the source grid."""

    time: float
    position: float
    velocity: float
    exact_position: float
    exact_velocity: float
    postselection_probability: float
    resources: ResourceEstimate

    @property
    def potential(self) -> float:
        """Potential energy for unit frequency and mass."""
        return self.position**2 / 2

    @property
    def kinetic(self) -> float:
        """Kinetic energy for unit mass."""
        return self.velocity**2 / 2


def run_workflow() -> tuple[Observation, ...]:
    """Read the zero-selector branch at each source Taylor encoding."""
    rows: list[Observation] = []
    for time in GRID:
        encoding = taylor_encoding(time)
        program = build_program(encoding)
        selected = logical_state(program).reshape(2, 8, 2)[:, 0, 0]
        if float(selected.imag.abs().max()) > 1e-11:
            raise ValueError("The oscillator solution should have real amplitudes")
        position, velocity = map(float, selected.real * encoding.scale)
        exact_position, exact_velocity = exact_solution(time)
        resources = program.report["resources"]
        if not isinstance(resources, ResourceEstimate):
            raise TypeError("Expected the native resource estimate")
        rows.append(
            Observation(
                time,
                position,
                velocity,
                exact_position,
                exact_velocity,
                float(selected.abs().square().sum()),
                resources,
            )
        )
    return tuple(rows)


def show_observations(rows: tuple[Observation, ...]) -> None:
    """Report signed readout, source magnitudes and correctly named energies."""
    for row in rows:
        print(
            f"t={row.time:.1f}, position={row.position:.12f}, velocity={row.velocity:.12f}, "
            f"source magnitudes=({abs(row.position):.12f}, {abs(row.velocity):.12f}), "
            f"postselection={row.postselection_probability:.12f}"
        )
    print("t   Potential (analytic / native)    Kinetic (analytic / native)")
    for row in rows:
        print(
            f"{row.time:.1f}   {row.exact_position**2 / 2:.9f} / {row.potential:.9f}    "
            f"{row.exact_velocity**2 / 2:.9f} / {row.kinetic:.9f}"
        )
    potential_error = math.fsum(
        abs(row.potential - row.exact_position**2 / 2) / (row.exact_position**2 / 2)
        for row in rows
    )
    kinetic_error = math.fsum(
        abs(row.kinetic - row.exact_velocity**2 / 2) / (row.exact_velocity**2 / 2)
        for row in rows
    )
    print(
        f"Mean relative energy error (%): potential={100 * potential_error / len(rows):.9f}, kinetic={100 * kinetic_error / len(rows):.9f}"
    )
