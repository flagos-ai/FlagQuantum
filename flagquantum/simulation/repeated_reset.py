"""Exact reduced-channel evolution for a proved stationary reset cycle."""

from __future__ import annotations

from dataclasses import replace

import torch

from ..core.ir import CircuitIR, Instruction
from ..core.operator_schema import get_operator_schema
from ..noise import amplitude_damping_channel


def _matches_reset(gate: Instruction, kraus: tuple[torch.Tensor, ...]) -> bool:
    return (
        gate.name == "amplitude_damping"
        and len(gate.wires) == 1
        and gate.params == {"gamma": 1.0}
        and gate.metadata == {"is_channel": True}
        and isinstance(gate.matrix, tuple)
        and len(gate.matrix) == len(kraus)
        and all(
            isinstance(actual, torch.Tensor) and torch.equal(actual, expected)
            for actual, expected in zip(gate.matrix, kraus, strict=True)
        )
    )


def repeated_reset_density(
    ir: CircuitIR,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    count = ir.metadata.get("construction_reset_repetitions")
    if (
        type(count) is not int
        or not 2 <= count <= 256
        or not 2 <= ir.n_wires <= 12
        or device.type != "cpu"
        or len(ir.instructions) % count
    ):
        return None
    width = ir.n_wires
    stride = len(ir.instructions) // count
    first = ir.instructions[:stride]
    expected = amplitude_damping_channel(1.0, dtype=dtype, device=device).kraus
    discarded: set[int] = set()
    unitary: list[Instruction] = []
    for gate in first:
        if discarded.intersection(gate.wires):
            return None
        if _matches_reset(gate, expected):
            discarded.add(gate.wires[0])
            continue
        schema = get_operator_schema(gate.name)
        if (
            schema is None
            or schema.semantic_kind != "unitary"
            or gate.matrix is not None
            or gate.metadata.get("is_dynamic")
            or "condition_clauses" in gate.metadata
            or "conditions" in gate.metadata
            or any(type(value) not in {int, float} for value in gate.params.values())
        ):
            return None
        unitary.append(gate)
    carried = tuple(wire for wire in range(width) if wire not in discarded)
    if not discarded or not 1 <= len(carried) <= 4:
        return None

    for index, gate in enumerate(ir.instructions[stride:]):
        original = first[index % stride]
        if original.metadata.get("is_channel"):
            if gate.wires != original.wires or not _matches_reset(gate, expected):
                return None
        elif gate.matrix is not None or gate != original:
            return None

    from ..circuit import Circuit

    size = 1 << len(carried)
    dimension = 1 << width
    labels = torch.arange(size, device=device)
    basis = torch.zeros_like(labels)
    for index, wire in enumerate(carried):
        basis |= ((labels >> (len(carried) - 1 - index)) & 1) << (width - 1 - wire)
    inputs = torch.zeros((size, dimension), dtype=dtype, device=device)
    inputs[labels, basis] = 1
    body = replace(
        ir, instructions=tuple(unitary), measurements=(), observables=(), metadata={}
    )
    evolved = Circuit.from_ir(
        body, bsz=size, dtype=dtype, device=device, inputs=inputs
    ).state()
    environment = tuple(sorted(discarded))
    # Each batch row is one carried input basis state. Tracing the reset bits
    # gives one Kraus operator per environment basis word.
    kraus = (
        evolved.reshape((size,) + (2,) * width)
        .permute(
            *(wire + 1 for wire in environment), *(wire + 1 for wire in carried), 0
        )
        .reshape(-1, size, size)
    )
    channel = torch.einsum("kij,kml->imjl", kraus, kraus.conj()).reshape(
        size**2, size**2
    )
    # Only the first column is needed because CircuitIR starts at all-zero.
    result = torch.linalg.matrix_power(channel, count)[:, 0].reshape(size, size)
    full = torch.zeros((1, dimension, dimension), dtype=dtype, device=device)
    full[0, basis[:, None], basis] = result
    return full
