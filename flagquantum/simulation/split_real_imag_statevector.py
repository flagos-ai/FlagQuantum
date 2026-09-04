"""Pure FP32 split-real/imag gate matrices and state updates."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from ..core.ir import CircuitIR, Instruction
from ..core.parameters import value_to_tensor

SPLIT_REAL_IMAG_SUPPORTED_GATES = frozenset(
    {
        "i",
        "x",
        "y",
        "z",
        "h",
        "s",
        "sdg",
        "t",
        "tdg",
        "sx",
        "sxdg",
        "rx",
        "ry",
        "rz",
        "phase",
        "u1",
        "u2",
        "u3",
        "cx",
        "cy",
        "cz",
        "swap",
        "crx",
        "cry",
        "crz",
        "cphase",
        "rxx",
        "ryy",
        "rzz",
    }
)


def _scalar(value: Any, *, device: torch.device) -> torch.Tensor:
    tensor = value_to_tensor(value)
    if tensor.numel() != 1:
        raise ValueError("split real/imag P0 requires scalar gate parameters")
    if tensor.requires_grad:
        raise NotImplementedError(
            "split real/imag P0 is forward-only and rejects trainable parameters"
        )
    if tensor.is_complex():
        if bool(torch.any(tensor.imag != 0)):
            raise ValueError("gate angles must be real")
        tensor = tensor.real
    return tensor.detach().to(device=device, dtype=torch.float32).reshape(())


def _matrix(
    entries: Sequence[Sequence[tuple[Any, Any]]], *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    width = len(entries)
    if width == 0 or any(len(row) != width for row in entries):
        raise ValueError("gate matrix entries must be non-empty and square")
    real = torch.stack(
        [_scalar(item[0], device=device) for row in entries for item in row]
    ).reshape(width, width)
    imag = torch.stack(
        [_scalar(item[1], device=device) for row in entries for item in row]
    ).reshape(width, width)
    return real, imag


def _zero_entries(width: int) -> list[list[tuple[Any, Any]]]:
    return [[(0.0, 0.0) for _ in range(width)] for _ in range(width)]


def fixed_matrix_pair(
    name: str, *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    q = 1.0 / math.sqrt(2.0)
    fixed: Mapping[str, Sequence[Sequence[tuple[Any, Any]]]] = {
        "i": (((1, 0), (0, 0)), ((0, 0), (1, 0))),
        "x": (((0, 0), (1, 0)), ((1, 0), (0, 0))),
        "y": (((0, 0), (0, -1)), ((0, 1), (0, 0))),
        "z": (((1, 0), (0, 0)), ((0, 0), (-1, 0))),
        "h": (((q, 0), (q, 0)), ((q, 0), (-q, 0))),
        "s": (((1, 0), (0, 0)), ((0, 0), (0, 1))),
        "sdg": (((1, 0), (0, 0)), ((0, 0), (0, -1))),
        "t": (((1, 0), (0, 0)), ((0, 0), (q, q))),
        "tdg": (((1, 0), (0, 0)), ((0, 0), (q, -q))),
        "sx": (((0.5, 0.5), (0.5, -0.5)), ((0.5, -0.5), (0.5, 0.5))),
        "sxdg": (((0.5, -0.5), (0.5, 0.5)), ((0.5, 0.5), (0.5, -0.5))),
    }
    if name in fixed:
        return _matrix(fixed[name], device=device)
    entries = _zero_entries(4)
    if name == "cx":
        for row, column in enumerate((0, 1, 3, 2)):
            entries[row][column] = (1, 0)
    elif name == "cy":
        entries[0][0] = entries[1][1] = (1, 0)
        entries[2][3] = (0, -1)
        entries[3][2] = (0, 1)
    elif name == "cz":
        for index, value in enumerate((1, 1, 1, -1)):
            entries[index][index] = (value, 0)
    elif name == "swap":
        for row, column in enumerate((0, 2, 1, 3)):
            entries[row][column] = (1, 0)
    else:
        raise KeyError(name)
    return _matrix(entries, device=device)


def _rotation_pair(angle: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    half = angle * 0.5
    return torch.cos(half), torch.sin(half)


def _parameter_matrix(
    instruction: Instruction, *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    name = instruction.name
    params = instruction.params
    if name in {"rx", "ry", "rz", "crx", "cry", "crz", "rxx", "ryy", "rzz"}:
        theta = _scalar(params["theta"], device=device)
        c, s = _rotation_pair(theta)
    if name == "rx":
        return _matrix((((c, 0), (0, -s)), ((0, -s), (c, 0))), device=device)
    if name == "ry":
        return _matrix((((c, 0), (-s, 0)), ((s, 0), (c, 0))), device=device)
    if name == "rz":
        return _matrix((((c, -s), (0, 0)), ((0, 0), (c, s))), device=device)
    if name in {"phase", "u1"}:
        theta = _scalar(params["theta"], device=device)
        return _matrix(
            (((1, 0), (0, 0)), ((0, 0), (torch.cos(theta), torch.sin(theta)))),
            device=device,
        )
    if name in {"u2", "u3"}:
        phi = _scalar(params["phi"], device=device)
        lbd = _scalar(params["lbd"], device=device)
        if name == "u2":
            c = s = _scalar(1.0 / math.sqrt(2.0), device=device)
        else:
            theta = _scalar(params["theta"], device=device)
            c, s = _rotation_pair(theta)
        return _matrix(
            (
                ((c, 0), (-s * torch.cos(lbd), -s * torch.sin(lbd))),
                (
                    (s * torch.cos(phi), s * torch.sin(phi)),
                    (c * torch.cos(phi + lbd), c * torch.sin(phi + lbd)),
                ),
            ),
            device=device,
        )
    if name in {"crx", "cry", "crz"}:
        base = Instruction(name=name[1:], wires=(0,), params={"theta": theta})
        block_real, block_imag = _parameter_matrix(base, device=device)
        entries = _zero_entries(4)
        entries[0][0] = entries[1][1] = (1, 0)
        for row in range(2):
            for column in range(2):
                entries[row + 2][column + 2] = (
                    block_real[row, column],
                    block_imag[row, column],
                )
        return _matrix(entries, device=device)
    if name == "cphase":
        theta = _scalar(params["theta"], device=device)
        entries = _zero_entries(4)
        for index in range(3):
            entries[index][index] = (1, 0)
        entries[3][3] = (torch.cos(theta), torch.sin(theta))
        return _matrix(entries, device=device)
    if name in {"rxx", "ryy", "rzz"}:
        entries = _zero_entries(4)
        for index in range(4):
            entries[index][index] = (c, 0)
        if name == "rxx":
            for row, column in ((0, 3), (1, 2), (2, 1), (3, 0)):
                entries[row][column] = (0, -s)
        elif name == "ryy":
            for row, column, sign in ((0, 3, 1), (1, 2, -1), (2, 1, -1), (3, 0, 1)):
                entries[row][column] = (0, sign * s)
        else:
            for index, sign in enumerate((-1, 1, 1, -1)):
                entries[index][index] = (c, sign * s)
        return _matrix(entries, device=device)
    raise KeyError(name)


def instruction_matrix_pair(
    instruction: Instruction, *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one supported gate as separate FP32 real and imaginary matrices."""

    if instruction.matrix is not None:
        raise NotImplementedError(
            "split real/imag P0 rejects custom matrices until their basis and "
            "device-resident conversion contract is certified"
        )
    if instruction.name not in SPLIT_REAL_IMAG_SUPPORTED_GATES:
        raise NotImplementedError(
            f"split real/imag P0 does not support gate {instruction.name!r}"
        )
    try:
        return fixed_matrix_pair(instruction.name, device=device)
    except KeyError:
        return _parameter_matrix(instruction, device=device)


def apply_gate_pair(
    real: torch.Tensor,
    imag: torch.Tensor,
    matrix_real: torch.Tensor,
    matrix_imag: torch.Tensor,
    wires: Sequence[int],
    *,
    n_wires: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one split-complex gate matrix to a flat statevector."""

    wires = tuple(int(wire) for wire in wires)
    remaining = tuple(wire for wire in range(n_wires) if wire not in wires)
    permutation = remaining + wires
    inverse = tuple(permutation.index(wire) for wire in range(n_wires))
    gate_dimension = 2 ** len(wires)
    logical_shape = (2,) * n_wires
    values_real = (
        real.reshape(logical_shape).permute(permutation).reshape(-1, gate_dimension)
    )
    values_imag = (
        imag.reshape(logical_shape).permute(permutation).reshape(-1, gate_dimension)
    )
    updated_real = values_real @ matrix_real.transpose(
        0, 1
    ) - values_imag @ matrix_imag.transpose(0, 1)
    updated_imag = values_real @ matrix_imag.transpose(
        0, 1
    ) + values_imag @ matrix_real.transpose(0, 1)
    return (
        updated_real.reshape(logical_shape).permute(inverse).reshape(-1),
        updated_imag.reshape(logical_shape).permute(inverse).reshape(-1),
    )


def run_split_real_imag_statevector(
    ir: CircuitIR, *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evolve a zero state with the split-real/imag numerical kernel."""

    amplitude_count = 2**ir.n_wires
    real = torch.zeros(amplitude_count, dtype=torch.float32, device=device)
    imag = torch.zeros_like(real)
    real[0] = 1.0
    for instruction in ir.instructions:
        matrix_real, matrix_imag = instruction_matrix_pair(instruction, device=device)
        real, imag = apply_gate_pair(
            real,
            imag,
            matrix_real,
            matrix_imag,
            instruction.wires,
            n_wires=ir.n_wires,
        )
    return real, imag


__all__ = (
    "SPLIT_REAL_IMAG_SUPPORTED_GATES",
    "apply_gate_pair",
    "fixed_matrix_pair",
    "instruction_matrix_pair",
    "run_split_real_imag_statevector",
)
