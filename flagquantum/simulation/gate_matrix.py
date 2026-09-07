"""Backend-neutral conversion from an IR instruction to a gate matrix."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from ..core.ir import Instruction
from ..core.operator_schema import canonical_opcode, get_operator_schema
from ..core.parameters import value_to_tensor
from ..core.runtime_config import get_runtime_config
from ..ops.matrices import GATE_MAT_DICT

_FIXED_GATE_CACHE: dict[tuple[str, str, torch.dtype], torch.Tensor] = {}


def parameter_tensor(
    name: str,
    params: Mapping[str, Any],
    *,
    bsz: int,
    device: torch.device | str,
    complex_dtype: torch.dtype | None = None,
    direct_values: tuple[Any, ...] | None = None,
) -> torch.Tensor | None:
    """Resolve an instruction parameter mapping into a real batched tensor."""

    schema = get_operator_schema(name)
    names = schema.parameters if schema is not None else ()
    if not names:
        return None

    if direct_values is None:
        values = []
        for parameter_name in names:
            if parameter_name not in params:
                return None
            values.append(params[parameter_name])
    else:
        values = list(direct_values)

    complex_dtype = complex_dtype or getattr(torch, get_runtime_config().complex_dtype)
    real_dtype = torch.float64 if complex_dtype == torch.complex128 else torch.float32
    tensors = [
        value_to_tensor(value, device=device, dtype=real_dtype) for value in values
    ]
    stacked = torch.stack(
        [tensor.reshape(()) if tensor.ndim == 0 else tensor for tensor in tensors],
        dim=-1,
    )
    stacked = stacked.to(device=device, dtype=real_dtype)
    if stacked.ndim == 1:
        stacked = stacked.reshape(1, -1)
    if stacked.shape[0] == 1 and bsz > 1:
        stacked = stacked.expand(bsz, -1)
    return stacked


def gate_matrix(
    instruction: Instruction,
    *,
    bsz: int,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
    parameter_bindings: tuple[torch.Tensor, ...] | None = None,
) -> torch.Tensor:
    """Return the concrete matrix for one backend-neutral IR instruction."""

    dtype = dtype or getattr(torch, get_runtime_config().complex_dtype)
    if instruction.matrix is not None:
        matrix = getattr(instruction.matrix, "tensor", instruction.matrix)
        return torch.as_tensor(matrix, dtype=dtype, device=device).reshape(
            2 ** len(instruction.wires), 2 ** len(instruction.wires)
        )

    name = canonical_opcode(instruction.name)
    gate = GATE_MAT_DICT[name]
    slots = getattr(instruction, "parameter_slots", ())
    constants = getattr(instruction, "parameter_constants", ())
    direct_values = None
    if parameter_bindings is not None and slots:
        direct_values = tuple(
            parameter_bindings[slot] if slot >= 0 else constants[index]
            for index, slot in enumerate(slots)
        )
    parameters = parameter_tensor(
        name,
        instruction.params,
        bsz=bsz,
        device=device,
        complex_dtype=dtype,
        direct_values=direct_values,
    )
    if callable(gate):
        if parameters is None:
            raise ValueError(f"Gate {name!r} requires parameters.")
        return gate(parameters).to(device=device, dtype=dtype)

    key = (name, str(torch.device(device)), dtype)
    cached = _FIXED_GATE_CACHE.get(key)
    if cached is None:
        cached = gate.to(device=device, dtype=dtype)
        _FIXED_GATE_CACHE[key] = cached
    return cached


__all__ = ("gate_matrix", "parameter_tensor")
