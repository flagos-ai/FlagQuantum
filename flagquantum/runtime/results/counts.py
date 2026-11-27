"""Measured counts and grouped Hamiltonian result extraction."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any
from typing import get_type_hints as _class_type_hints

import torch

from flagquantum.algorithms.primitives.hamiltonian import Hamiltonian
from flagquantum.compute.remote.packages import DeploymentPackage
from flagquantum.runtime.parallel import ObservableGroup


@dataclass(frozen=True)
class PauliMeasurementPlan:
    """Grouped, basis-rotated deployment packages for one Hamiltonian."""

    hamiltonian: Hamiltonian
    groups: tuple[ObservableGroup, ...]
    packages: tuple[DeploymentPackage, ...]

    def __post_init__(self) -> None:
        if not self.groups or len(self.groups) != len(self.packages):
            raise ValueError("Pauli measurement groups and packages must align")

    @property
    def shots_per_group(self) -> tuple[int, ...]:
        return tuple(package.shots for package in self.packages)

    def expectation(
        self,
        counts_by_group: Sequence[Mapping[str, int]],
    ) -> torch.Tensor:
        return hamiltonian_expectation_from_grouped_counts(counts_by_group, self)

    def standard_error(
        self,
        counts_by_group: Sequence[Mapping[str, int]],
    ) -> torch.Tensor:
        """Return the standard error of the grouped expectation estimator."""

        _, standard_error = _grouped_hamiltonian_statistics(counts_by_group, self)
        return torch.tensor([standard_error], dtype=torch.float32)

    def summary(self) -> dict[str, Any]:
        return {
            "group_count": len(self.groups),
            "term_count": self.hamiltonian.n_terms,
            "shots_per_group": self.shots_per_group,
            "groups": tuple(
                {
                    "term_indices": group.term_indices,
                    "basis": group.basis,
                    "package_name": package.name,
                }
                for group, package in zip(self.groups, self.packages, strict=True)
            ),
        }


def _validate_count_histogram(counts: Mapping[str, int]) -> tuple[int, int]:
    """Validate an ungrouped histogram and return its qubit and shot counts."""

    if not counts:
        raise ValueError("Counts cannot be empty.")
    n_qubits = len(str(next(iter(counts))))
    if n_qubits == 0:
        raise ValueError("Count bitstrings cannot be empty.")
    shots = 0
    for bitstring, count in counts.items():
        bits = str(bitstring)
        if len(bits) != n_qubits:
            raise ValueError("Count bitstrings must have the same width.")
        if any(bit not in "01" for bit in bits):
            raise ValueError("Count bitstrings must be binary.")
        if isinstance(count, bool) or not isinstance(count, Integral):
            raise TypeError("Counts must be non-negative integers.")
        if count < 0:
            raise ValueError("Counts must be non-negative integers.")
        shots += int(count)
    if shots == 0:
        raise ValueError("Counts must contain at least one shot.")
    return n_qubits, shots


def expectation_z_from_counts(
    counts: Mapping[str, int],
    qubits: int | Sequence[int] | None = None,
) -> torch.Tensor:
    """Estimate Z expectations from cloud measurement counts."""

    n_qubits, shot_count = _validate_count_histogram(counts)
    if qubits is None:
        qubit_tuple = tuple(range(n_qubits))
    elif isinstance(qubits, int):
        qubit_tuple = (qubits,)
    else:
        qubit_tuple = tuple(int(qubit) for qubit in qubits)

    if any(qubit < 0 or qubit >= n_qubits for qubit in qubit_tuple):
        raise ValueError("Requested qubits are outside the measured register.")
    shots = float(shot_count)
    values = []
    for qubit in qubit_tuple:
        total = 0.0
        for bitstring, count in counts.items():
            bit = int(str(bitstring)[qubit])
            total += (1.0 if bit == 0 else -1.0) * int(count)
        values.append(total / shots)
    return torch.tensor([values], dtype=torch.float32)


def _measured_coefficient(value: float | complex | torch.Tensor) -> float:
    """Return one finite real coefficient for count-based measurement statistics."""

    tensor = torch.as_tensor(value).detach()
    if tensor.numel() != 1:
        raise ValueError("measured Hamiltonian coefficients must be scalars")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("measured Hamiltonian coefficients must be finite")
    if tensor.is_complex() and bool(torch.abs(tensor.imag) > 1e-12):
        raise ValueError("measured Hamiltonian coefficients must be real")
    return float(tensor.real)


def hamiltonian_expectation_from_counts(
    counts: Mapping[str, int],
    hamiltonian: Hamiltonian,
) -> torch.Tensor:
    """Estimate an I/Z-only Hamiltonian from computational-basis counts."""

    n_qubits, shot_count = _validate_count_histogram(counts)
    if hamiltonian.n_qubits > n_qubits:
        raise ValueError("Hamiltonian references qubits outside the measured register.")
    shots = float(shot_count)
    total = 0.0
    for term in hamiltonian.terms:
        coeff = _measured_coefficient(term.coefficient)
        term_total = 0.0
        for bitstring, count in counts.items():
            parity = 1.0
            for qubit, name in term.ops:
                if name != "z":
                    raise ValueError(
                        "Counts-based Hamiltonian estimation currently supports only I/Z terms. "
                        "Use basis-rotated deployment packages for X/Y observables."
                    )
                parity *= 1.0 if str(bitstring)[qubit] == "0" else -1.0
            term_total += parity * int(count)
        total += coeff * term_total / shots
    return torch.tensor([total], dtype=torch.float32)


def hamiltonian_expectation_from_grouped_counts(
    counts_by_group: Sequence[Mapping[str, int]],
    plan: PauliMeasurementPlan,
) -> torch.Tensor:
    """Aggregate basis-rotated counts from a ``PauliMeasurementPlan``."""

    expectation, _ = _grouped_hamiltonian_statistics(counts_by_group, plan)
    return torch.tensor([expectation], dtype=torch.float32)


def _grouped_hamiltonian_statistics(
    counts_by_group: Sequence[Mapping[str, int]],
    plan: PauliMeasurementPlan,
) -> tuple[float, float]:
    """Return the grouped estimator mean and standard error."""

    if len(counts_by_group) != len(plan.groups):
        raise ValueError(
            "counts_by_group must contain one result per measurement group"
        )
    total = 0.0
    total_variance = 0.0
    for group_index, (counts, group, package) in enumerate(
        zip(counts_by_group, plan.groups, plan.packages, strict=True)
    ):
        if not counts:
            raise ValueError(f"measurement group {group_index} counts cannot be empty")
        for count in counts.values():
            if isinstance(count, bool) or not isinstance(count, Integral):
                raise TypeError(
                    f"measurement group {group_index} counts must be non-negative integers"
                )
            if count < 0:
                raise ValueError(
                    f"measurement group {group_index} counts must be non-negative integers"
                )
        shots = sum(int(count) for count in counts.values())
        if shots != package.shots:
            raise ValueError(
                f"measurement group {group_index} returned {shots} shots; "
                f"expected {package.shots}"
            )
        for bitstring in counts:
            if len(str(bitstring)) != package.n_qubits:
                raise ValueError(
                    f"measurement group {group_index} bitstring width does not "
                    "match its deployment package"
                )
            if any(bit not in "01" for bit in str(bitstring)):
                raise ValueError(
                    f"measurement group {group_index} bitstrings must be binary"
                )
        coefficients: list[tuple[float, tuple[tuple[int, str], ...]]] = []
        for term_index in group.term_indices:
            term = plan.hamiltonian.terms[term_index]
            coefficients.append((_measured_coefficient(term.coefficient), term.ops))

        mean = 0.0
        second_moment = 0.0
        for bitstring, count in counts.items():
            sample_value = 0.0
            for coefficient, ops in coefficients:
                parity = 1.0
                for qubit, _name in ops:
                    parity *= 1.0 if str(bitstring)[qubit] == "0" else -1.0
                sample_value += coefficient * parity
            probability = int(count) / shots
            mean += probability * sample_value
            second_moment += probability * sample_value * sample_value
        total += mean
        total_variance += max(second_moment - mean * mean, 0.0) / shots
    return total, math.sqrt(total_variance)


# Resolve field types before preserving historical public class module names.
# Dataclass constructor/field contracts stay unchanged; introspection needs no facade import.
for _public_hint_class in (PauliMeasurementPlan,):
    _public_field_hints = _class_type_hints(_public_hint_class, include_extras=True)
    _public_hint_class.__annotations__ = {
        name: _public_field_hints[name] for name in _public_hint_class.__annotations__
    }
del _public_hint_class, _public_field_hints

for _identity in (
    PauliMeasurementPlan,
    expectation_z_from_counts,
    hamiltonian_expectation_from_counts,
    hamiltonian_expectation_from_grouped_counts,
):
    _identity.__module__ = "flagquantum.deployment.cloud"
del _identity
