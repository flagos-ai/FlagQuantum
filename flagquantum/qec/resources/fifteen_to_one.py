# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 FifteenToOne and _build_factory.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/surface_code/fifteen_to_one_factory.py

"""Five-qubit noisy circuit and postselection for 15-to-1 distillation."""

from dataclasses import dataclass
from math import ceil, pow, sqrt

import torch

from ...compiler._component_trace.resource_model import GateCost
from ...core.ir import CircuitIR, Instruction
from ...simulation.density_matrix import density_matrix_from_ir
from .distillation_channels import noisy_pauli_rotation, storage_error

_WIRES = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class FifteenToOneEstimate:
    """Factory footprint, expected cycles and first-order output error."""

    physical_qubits: int
    cycles: int
    rejection_probability: float
    conditional_output_error: float
    factory_error: float


@dataclass(frozen=True)
class FifteenToOneFactory:
    """Code distances for the X, Z, and measurement dimensions."""

    x_distance: int
    z_distance: int
    measurement_distance: int

    def __post_init__(self) -> None:
        if (
            any(
                type(distance) is not int or distance < 1
                for distance in (
                    self.x_distance,
                    self.z_distance,
                    self.measurement_distance,
                )
            )
            or self.x_distance > 3 * self.measurement_distance
        ):
            raise ValueError(
                "Use positive factory distances with X at most three times measurement"
            )

    def to_ir(self, *, physical_error: float, error_rate_scaler: float) -> CircuitIR:
        """Emit the source schedule on output wire 0 and check wires 1 to 4."""
        if not 0 <= physical_error < 0.01 or not 0 < error_rate_scaler <= 1:
            raise ValueError(
                "Invalid distillation physical error or logical error scaler"
            )
        dx, dz, dm = self.x_distance, self.z_distance, self.measurement_distance
        px, pz, pm = (
            error_rate_scaler * pow(physical_error / 0.01, (distance + 1) / 2)
            for distance in (dx, dz, dm)
        )
        injection = physical_error / 3
        storage_x = 0.5 * (dz / dx) * px * dm
        storage_z = 0.5 * (dx / dz) * pz * dm
        output_storage = 0.5 * px * dm

        def single(word: str) -> Instruction:
            return noisy_pauli_rotation(
                word,
                injection + 0.5 * (dm / dz) * pz * dm,
                injection + 0.5 * dz * pm,
                injection,
                _WIRES,
            )

        def joint(word: str, length: int) -> Instruction:
            return noisy_pauli_rotation(
                word,
                injection + 0.5 * pm * dm,
                injection + 0.5 * pm * dm + 0.5 * length * dx / dm * pm,
                injection,
                _WIRES,
            )

        def store(output: float, *, last: bool) -> tuple[Instruction, ...]:
            return (
                *storage_error(
                    "X",
                    (output, storage_x, storage_x, storage_x, storage_x if last else 0),
                    _WIRES,
                ),
                *storage_error(
                    "Z",
                    (output, storage_z, storage_z, storage_z, storage_z if last else 0),
                    _WIRES,
                ),
            )

        wide = dx + 4 * dz
        output_long_storage = 0.5 * px * (dm + 2 * dx)
        instructions = (
            *(Instruction("h", (wire,)) for wire in _WIRES),
            single("IZIII"),
            single("IIZII"),
            single("IIIZI"),
            joint("IZZZI", 3 * dz),
            *store(0, last=False),
            joint("ZZZII", dx + 2 * dz),
            joint("ZZIZI", dx + 3 * dz),
            *storage_error(
                "Z",
                (0.5 * ((dx + 2 * dz) + (dx + 3 * dz)) / dx * px * dm, 0, 0, 0, 0),
                _WIRES,
            ),
            *store(output_storage, last=False),
            joint("ZIZZI", dx + 3 * dz),
            joint("ZIIZZ", wide),
            single("IIIIZ"),
            *storage_error(
                "Z", (0.5 * ((dx + 3 * dz) + wide) / dx * px * dm, 0, 0, 0, 0), _WIRES
            ),
            *store(output_storage, last=True),
            joint("ZZIIZ", wide),
            joint("ZIZIZ", wide),
            *storage_error(
                "Z", (0.5 * (wide + wide) / dx * px * dm, 0, 0, 0, 0), _WIRES
            ),
            *store(output_storage, last=True),
            joint("ZZZZZ", wide),
            joint("IIZZZ", 3 * dz),
            *storage_error("Z", (0.5 * wide / dx * px * dm, 0, 0, 0, 0), _WIRES),
            *store(output_long_storage, last=True),
            joint("IZIZZ", 4 * dz),
            joint("IZZIZ", 4 * dz),
            *store(0, last=True),
        )
        return CircuitIR(5, instructions).validate()

    def estimate(
        self, gates: GateCost, *, physical_error: float, error_rate_scaler: float
    ) -> FifteenToOneEstimate:
        """Postselect four X=+1 checks and estimate the cost of accepted states."""
        demand = gates.total_t_count()
        if demand < 0:
            raise ValueError("T-state demand must be nonnegative")
        ir = self.to_ir(
            physical_error=physical_error, error_rate_scaler=error_rate_scaler
        )
        density = density_matrix_from_ir(ir, device="cpu", dtype=torch.complex128)[0]
        check = torch.ones((16, 16), dtype=torch.complex128, device="cpu") / 16
        projector = torch.kron(torch.eye(2, dtype=check.dtype, device="cpu"), check)
        acceptance = float(torch.trace(projector @ density).real)
        if acceptance <= 0:
            raise ValueError("Distillation postselection has zero acceptance")
        postselected = (projector @ density @ projector.mH) / acceptance
        target = torch.tensor(
            (1, complex(1, 1) / sqrt(2)), dtype=check.dtype, device="cpu"
        )
        target = target / sqrt(2)
        target_density = torch.kron(target[:, None] @ target.conj()[None, :], check)
        output_error = 1 - float(torch.trace(postselected @ target_density).real)
        dx, dz, dm = self.x_distance, self.z_distance, self.measurement_distance
        return FifteenToOneEstimate(
            physical_qubits=2 * (dx + 4 * dz) * 3 * dx + 4 * dm,
            cycles=ceil(demand * 6 * dm / acceptance),
            rejection_probability=1 - acceptance,
            conditional_output_error=output_error,
            factory_error=output_error * demand,
        )
