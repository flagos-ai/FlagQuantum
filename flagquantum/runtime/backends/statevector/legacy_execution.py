"""Compatibility execution for the historical DTensor statevector device."""

from __future__ import annotations

from typing import Any

import torch

from ....core.ir import CircuitIR, Instruction
from ....core.parameters import value_to_tensor
from ....measurement import measure_allZ
from ....ops import functional
from ...distributed.backend_policy import DistributedBackendPolicy
from ...execution_plan import ExecutionPlan
from .legacy_device import DistributedQuantumDevice

_GATE_ALIASES = {
    "cnot": "cx",
    "ccnot": "ccx",
    "toffoli": "ccx",
    "fredkin": "cswap",
    "sd": "sdg",
    "td": "tdg",
    "p": "phase",
}

_PARAM_ALIASES = {
    "rx": ("theta",),
    "ry": ("theta",),
    "rz": ("theta",),
    "phase": ("theta",),
    "p": ("theta",),
    "u1": ("theta",),
    "u2": ("phi", "lbd"),
    "u3": ("theta", "phi", "lbd"),
    "crx": ("theta",),
    "cry": ("theta",),
    "crz": ("theta",),
    "cphase": ("theta",),
    "rxx": ("theta",),
    "ryy": ("theta",),
    "rzz": ("theta",),
}


def _parameter_tensor(
    instruction: Instruction,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    names = _PARAM_ALIASES.get(instruction.name)
    if not names:
        return None
    values = [instruction.params.get(name) for name in names]
    if any(value is None for value in values):
        return None
    tensors = [value_to_tensor(value) for value in values]
    params = torch.stack(
        [tensor.reshape(()) if tensor.ndim == 0 else tensor for tensor in tensors],
        dim=-1,
    )
    return params.to(device=device, dtype=dtype)


def _matrix_tensor(
    matrix: Any,
    device: torch.device | str,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = getattr(matrix, "tensor", matrix)
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    tensor = torch.as_tensor(tensor, dtype=dtype, device=device).reshape(-1)
    width = int(tensor.numel() ** 0.5)
    return tensor.reshape(width, width)


class DistributedExecutor:
    """Execute IR on the historical DTensor device compatibility surface."""

    def __init__(self, device: DistributedQuantumDevice):
        self.device = device

    def apply(self, instruction: Instruction) -> None:
        if instruction.metadata.get("is_channel"):
            raise NotImplementedError(
                "Distributed execution currently supports unitary instructions."
            )
        if instruction.metadata.get("mpo") or instruction.metadata.get("diagonal"):
            raise NotImplementedError(
                "MPO and diagonal tensor-network instructions require tensor-network execution."
            )

        name = _GATE_ALIASES.get(instruction.name, instruction.name)
        params = _parameter_tensor(
            instruction,
            device=self.device.device,
            dtype=self.device.real_dtype,
        )
        if hasattr(self.device, name):
            getattr(self.device, name)(wires=list(instruction.wires), params=params)
            return

        if instruction.matrix is None:
            raise KeyError(f"No native distributed implementation for gate {name!r}.")
        matrix = _matrix_tensor(
            instruction.matrix,
            self.device.device,
            dtype=self.device.complex_dtype,
        )
        functional.gate(
            matrix, self.device, wires=list(instruction.wires), params=params
        )

    def run(self, ir: CircuitIR) -> DistributedQuantumDevice:
        for instruction in ir:
            self.apply(instruction)
        return self.device


def is_legacy_distributed_device(value: Any) -> bool:
    """Return whether a caller explicitly supplied the historical device."""

    return isinstance(value, DistributedQuantumDevice)


def run_legacy_distributed(
    ir: CircuitIR,
    *,
    device: Any,
    device_options: dict[str, Any],
    precision: torch.dtype,
    execution_plan: ExecutionPlan,
    statevector_plan: Any,
    backend_policy: DistributedBackendPolicy,
    jax_distributed_plan: dict[str, Any],
    measure: bool,
) -> Any:
    """Run the protected backend-native compatibility path."""

    if isinstance(device, DistributedQuantumDevice):
        qdev = device
        if qdev.complex_dtype != precision:
            raise ValueError(
                "existing distributed device precision conflicts with execution: "
                f"{qdev.complex_dtype} != {precision}"
            )
    else:
        options = dict(device_options)
        if device is not None:
            options["device"] = str(device)
        options["precision"] = precision
        qdev = DistributedQuantumDevice(n_wires=ir.n_wires, **options)

    DistributedExecutor(qdev).run(ir)
    qdev.execution_ir = ir
    qdev.execution_plan = execution_plan
    qdev.distributed_statevector_plan = statevector_plan
    qdev.distributed_statevector_summary = statevector_plan.summary()
    qdev.distributed_backend_policy = backend_policy
    qdev.distributed_statevector_summary["distributed_backend_policy"] = (
        backend_policy.summary()
    )
    qdev.distributed_statevector_summary["jax_distributed_plan"] = jax_distributed_plan
    return measure_allZ(qdev) if measure else qdev


__all__ = [
    "DistributedExecutor",
    "is_legacy_distributed_device",
    "run_legacy_distributed",
]
