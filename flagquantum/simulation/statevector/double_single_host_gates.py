"""Explicit CPU gate encoding for the P3 Double-Single statevector path."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from ...core.ir import Instruction
from ...numerics.double_single import DoubleSingleComplexTensor
from ...ops.matrices import GATE_MAT_DICT

_PARAMETER_ORDER: dict[str, tuple[str, ...]] = {
    "rx": ("theta",),
    "ry": ("theta",),
    "rz": ("theta",),
    "phase": ("theta",),
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


def _fixed_matrix(name: str) -> torch.Tensor:
    q = 1.0 / math.sqrt(2.0)
    fixed: dict[str, Sequence[Sequence[complex]]] = {
        "i": ((1, 0), (0, 1)),
        "x": ((0, 1), (1, 0)),
        "y": ((0, -1j), (1j, 0)),
        "z": ((1, 0), (0, -1)),
        "h": ((q, q), (q, -q)),
        "s": ((1, 0), (0, 1j)),
        "sdg": ((1, 0), (0, -1j)),
        "t": ((1, 0), (0, complex(q, q))),
        "tdg": ((1, 0), (0, complex(q, -q))),
        "sx": (
            (complex(0.5, 0.5), complex(0.5, -0.5)),
            (complex(0.5, -0.5), complex(0.5, 0.5)),
        ),
        "sxdg": (
            (complex(0.5, -0.5), complex(0.5, 0.5)),
            (complex(0.5, 0.5), complex(0.5, -0.5)),
        ),
        "cx": ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0)),
        "cy": ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, -1j), (0, 0, 1j, 0)),
        "cz": ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, -1)),
        "swap": ((1, 0, 0, 0), (0, 0, 1, 0), (0, 1, 0, 0), (0, 0, 0, 1)),
    }
    try:
        return torch.tensor(fixed[name], dtype=torch.complex128, device="cpu")
    except KeyError as exc:
        raise KeyError(name) from exc


def host_matrix_complex128(instruction: Instruction) -> torch.Tensor:
    """Build the CPU complex128 reference matrix used by P3."""

    if instruction.matrix is not None:
        raise NotImplementedError("split real/imag P3 rejects custom matrices")
    try:
        return _fixed_matrix(instruction.name)
    except KeyError:
        pass
    try:
        order = _PARAMETER_ORDER[instruction.name]
    except KeyError as exc:
        raise NotImplementedError(
            f"split real/imag P3 does not support gate {instruction.name!r}"
        ) from exc
    values = []
    for name in order:
        raw_value = instruction.params[name]
        tensor = (
            raw_value.detach()
            if isinstance(raw_value, torch.Tensor)
            else torch.as_tensor(raw_value, dtype=torch.float64, device="cpu")
        )
        if tensor.numel() != 1 or tensor.requires_grad:
            raise ValueError("split real/imag P3 gate parameters must be fixed scalars")
        if tensor.is_complex():
            if bool(torch.any(tensor.imag != 0).item()):
                raise ValueError("split real/imag P3 gate parameters must be real")
            tensor = tensor.real
        values.append(tensor.detach().cpu().to(torch.float64).reshape(()))
    params = torch.stack(values)
    generator = GATE_MAT_DICT[instruction.name]
    if not callable(generator):
        raise RuntimeError(
            f"missing parameterized matrix generator for {instruction.name}"
        )
    matrix = generator(params)
    if matrix.ndim == 3 and matrix.shape[0] == 1:
        matrix = matrix[0]
    return matrix.detach().cpu().to(torch.complex128)


def encode_host_double_single_matrix(
    instruction: Instruction, *, device: torch.device
) -> DoubleSingleComplexTensor:
    """Encode a P3 gate on CPU before moving its four FP32 words to a device."""

    return DoubleSingleComplexTensor.from_complex128(
        host_matrix_complex128(instruction)
    ).to(device)


__all__ = ("encode_host_double_single_matrix", "host_matrix_complex128")
