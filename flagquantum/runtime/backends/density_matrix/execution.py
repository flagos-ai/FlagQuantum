"""Execution entrypoints for the exact density-matrix backend."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch

from ....compilation.noise import lower_noise_model
from ....core.ir import CircuitIR
from ....noise import NoiseModel
from ....ops.gate_matrix import gate_matrix
from ....ops.matrices import get_global_precision
from .kernels import apply_kraus_density, apply_unitary_density, density_matrix

if TYPE_CHECKING:
    from ....compilation.noise import NoisyExecutionPlan


def density_matrix_from_ir(
    circuit_or_ir: Any,
    *,
    bsz: int = 1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Execute unitary and channel IR instructions as a density matrix."""

    if hasattr(circuit_or_ir, "to_ir"):
        circuit = circuit_or_ir
        ir = circuit.to_ir()
        state = circuit.initial_state()
    elif isinstance(circuit_or_ir, CircuitIR):
        ir = circuit_or_ir
        out_dtype = dtype or get_global_precision()
        state = torch.zeros(bsz, 2**ir.n_wires, dtype=out_dtype, device=device)
        state[:, 0] = 1
    else:
        raise TypeError("density_matrix_from_ir expects a Circuit or CircuitIR.")

    rho = density_matrix(state)
    for instruction in ir:
        if instruction.metadata.get("is_channel"):
            rho = apply_kraus_density(
                rho,
                instruction.matrix,
                instruction.wires,
                ir.n_wires,
            )
            continue
        matrix = gate_matrix(
            instruction,
            bsz=rho.shape[0],
            device=rho.device,
            dtype=rho.dtype,
        )
        rho = apply_unitary_density(rho, matrix, instruction.wires, ir.n_wires)
    return rho


def noisy_density_matrix(
    circuit_or_ir: Any,
    noise_model: NoiseModel | None = None,
    *,
    bsz: int = 1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Run a circuit or IR with optional per-instruction native noise channels."""

    lowered = lower_noise_model(circuit_or_ir, noise_model)
    return density_matrix_from_ir(lowered, bsz=bsz, device=device, dtype=dtype)


def execute_density_plan(
    ir: CircuitIR,
    plan: NoisyExecutionPlan,
    options: Mapping[str, Any],
) -> torch.Tensor:
    """Execute an exact-channel plan through the density-matrix backend."""

    if plan.representation != "density_matrix" or plan.evolution != "exact_channel":
        raise ValueError("density executor requires density_matrix exact_channel plan")
    supported = {"bsz", "device", "dtype"}
    density_options = {key: value for key, value in options.items() if key in supported}
    return density_matrix_from_ir(ir, **density_options)


__all__ = ("density_matrix_from_ir", "execute_density_plan", "noisy_density_matrix")
