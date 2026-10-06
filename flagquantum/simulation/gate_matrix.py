"""Backend-neutral conversion from an IR instruction to a gate matrix."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import torch

from ..core.ir import Instruction
from ..core.operator_schema import canonical_opcode, get_operator_schema
from ..core.parameters import value_to_tensor
from ..core.runtime_config import get_runtime_config
from ..errors import CapabilityError, ValidationError
from .matrices import GATE_MAT_DICT

_FIXED_GATE_CACHE: dict[tuple[str, str, torch.dtype], torch.Tensor] = {}
_PARAMETER_BATCH_WINDOW: ContextVar[tuple[int, int, int] | None] = ContextVar(
    "flagquantum_parameter_batch_window", default=None
)


@contextmanager
def _parameter_batch_window(
    start: int, stop: int, full_batch_size: int
) -> Iterator[None]:
    """Select one execution window from otherwise full-batch gate parameters."""

    token = _PARAMETER_BATCH_WINDOW.set((start, stop, full_batch_size))
    try:
        yield
    finally:
        _PARAMETER_BATCH_WINDOW.reset(token)


def _parameter_rows(
    name: str,
    parameter_names: tuple[str, ...],
    tensors: list[torch.Tensor],
    *,
    bsz: int,
) -> int:
    """Return the single row count shared by every parameter of one instruction.

    A value is either a scalar broadcast over the batch, a vector of length one, or a
    vector of exactly ``bsz`` entries. Anything else cannot become one row per batch
    entry, and the shape is refused here with the gate and parameter names attached,
    rather than left to fail inside ``torch.bmm`` or ``expand``.
    """

    rows: int | None = None
    for parameter_name, tensor in zip(parameter_names, tensors, strict=True):
        if tensor.ndim == 0:
            continue
        if tensor.ndim > 1:
            raise ValidationError(
                f"Gate {name!r} parameter {parameter_name!r} has shape "
                f"{tuple(tensor.shape)}; expected a scalar, a vector of length 1, or a "
                f"vector of length {bsz} for a batch of {bsz}"
            )
        length = int(tensor.shape[0])
        if length not in (1, bsz):
            raise ValidationError(
                f"Gate {name!r} parameter {parameter_name!r} has length {length}; "
                f"expected 1 or the batch size {bsz}"
            )
        if length > 1 and rows is not None and rows != length:
            raise ValidationError(
                f"Gate {name!r} parameters mix lengths {rows} and {length}; "
                f"expected one length for the whole gate"
            )
        rows = length if length > 1 else rows
    return rows or 1


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
    window = _PARAMETER_BATCH_WINDOW.get()
    if window is not None:
        start, stop, full_batch_size = window
        # Validate against the public circuit batch before slicing. Otherwise a
        # malformed vector whose length merely equals the internal chunk could
        # be mistaken for one valid parameter row per circuit entry.
        _parameter_rows(name, names, tensors, bsz=full_batch_size)
        tensors = [
            (
                tensor[start:stop]
                if tensor.ndim == 1 and int(tensor.shape[0]) == full_batch_size
                else tensor
            )
            for tensor in tensors
        ]
    rows = _parameter_rows(name, names, tensors, bsz=bsz)
    target = bsz if rows == 1 and bsz > 1 else rows
    columns = [
        (tensor.reshape(-1) if tensor.ndim else tensor.reshape(1)).expand(target)
        for tensor in tensors
    ]
    stacked = torch.stack(columns, dim=-1)
    return stacked.to(device=device, dtype=real_dtype)


def _refuse_channel(
    name: str, instruction: Instruction, operators: Any
) -> CapabilityError:
    """Explain that a channel is not a dense operator, and what applies it instead.

    A channel reaches here for one of two reasons, and each has its own repair, so
    the refusal names the one that happened. An instruction that carries operators
    is a complete channel: the density-matrix, MPS, tensor-network and dynamic
    channel paths apply it, and this function builds a dense unitary for one
    instruction and cannot. An instruction that carries none is a materialization
    gap instead -- the declared parameters were never turned into Kraus operators
    -- and the caller can still do that with the same constructor the public gate
    path uses.
    """

    operands = tuple(instruction.wires)
    if instruction.matrix is None:
        declared = ", ".join(
            repr(parameter) for parameter in (instruction.params or {})
        )
        return CapabilityError(
            f"instruction {name!r} on operands {operands} is a channel carrying no "
            f"Kraus operators, so no engine can build its matrix; the parameters it "
            f"was given ({declared or 'none'}) have to be turned into operators "
            f"first, which is what the public gate path does for a declared channel"
        )
    return CapabilityError(
        f"instruction {name!r} on operands {operands} is a channel carrying "
        f"{operators} Kraus operator(s), and gate_matrix builds the dense operator "
        f"of one unitary instruction; apply a channel through an engine that executes "
        f"channels, such as the density-matrix or MPS channel path, or lower it into "
        f"a noise model this engine already applies"
    )


def _refuse_unregistered_opcode(name: str, instruction: Instruction) -> CapabilityError:
    """Explain that no operator exists for an opcode the registry does not declare.

    A misspelled gate and a directive are both absent from the operator registry
    and both used to arrive at a dictionary lookup, so the report has to say which
    question was asked. It also has to say that this is not a decision made here:
    the registry is the one definition of what an opcode means, and a directive is
    only executed by a path that interprets it by name, which operator
    materialization is not.
    """

    return CapabilityError(
        f"the operator registry declares no operator for opcode {name!r}, so no "
        f"engine can build a dense operator for instruction {name!r} on operands "
        f"{tuple(instruction.wires)}; a misspelled gate and a directive such as "
        f"'measure', 'reset' or 'barrier' both arrive here, and this path interprets "
        f"no directive by name"
    )


def _dense_operator(
    instruction: Instruction, *, dtype: torch.dtype, device: torch.device | str
) -> torch.Tensor:
    """Return the dense operator one instruction carries, or refuse it by value.

    ``Instruction.matrix`` is ``Any`` because a matrix may be a carrier object that
    exposes its tensor as ``.tensor``, so the value is converted before it is
    checked. A type that is not a matrix at all is a ``TypeError``, as the errors
    module reserves for a wrong Python type; a matrix of the wrong size is a
    ``ValidationError``. The entry count decides the size, not the shape, so
    ``(1, 2, 2)`` is the same operator as ``(2, 2)``.
    """

    arity = len(instruction.wires)
    entries = 4**arity
    carried = getattr(instruction.matrix, "tensor", instruction.matrix)
    if isinstance(carried, (str, bytes, bytearray)):
        raise TypeError(
            f"instruction {instruction.name!r} on operands "
            f"{tuple(instruction.wires)} carries a {type(carried).__name__} where a "
            f"{entries}-entry dense operator is required"
        )
    try:
        tensor = torch.as_tensor(carried, dtype=dtype, device=device)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValidationError(
            f"instruction {instruction.name!r} on operands "
            f"{tuple(instruction.wires)} carries a {type(carried).__name__} that is "
            f"not a dense operator; gate_matrix requires {entries} entries for "
            f"{arity} operand(s)"
        ) from error
    if tensor.numel() != entries:
        raise ValidationError(
            f"instruction {instruction.name!r} on operands "
            f"{tuple(instruction.wires)} carries a dense operator of shape "
            f"{tuple(tensor.shape)} with {tensor.numel()} entries; gate_matrix "
            f"requires {entries} entries for {arity} operand(s)"
        )
    return tensor.reshape(2**arity, 2**arity)


def _carried_operators(instruction: Instruction) -> int:
    """How many Kraus operators the instruction carries, for the refusal report."""

    if isinstance(instruction.matrix, (list, tuple)):
        return len(instruction.matrix)
    return 1


def gate_matrix(
    instruction: Instruction,
    *,
    bsz: int,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
    parameter_bindings: tuple[torch.Tensor, ...] | None = None,
) -> torch.Tensor:
    """Return the dense operator for one backend-neutral IR instruction.

    One instruction has a dense operator only when it is a unitary gate, and every
    other shape is refused by name here rather than escaping as a dictionary key or
    as a tensor-conversion failure that names neither the instruction nor the
    repair. A channel is refused for carrying operators, or for having declared
    none yet; an opcode the operator registry does not declare is refused as either
    a misspelled gate or a directive this path does not interpret; a carried matrix
    is refused when it is not a dense operator of the instruction's size.
    """

    dtype = dtype or getattr(torch, get_runtime_config().complex_dtype)
    name = canonical_opcode(instruction.name)
    schema = get_operator_schema(name)
    if bool(instruction.metadata.get("is_channel")) or (
        schema is not None and schema.channel
    ):
        operators = _carried_operators(instruction)
        raise _refuse_channel(name, instruction, operators)
    if instruction.matrix is not None:
        return _dense_operator(instruction, dtype=dtype, device=device)

    gate = GATE_MAT_DICT.get(name)
    if gate is None:
        raise _refuse_unregistered_opcode(name, instruction)
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
