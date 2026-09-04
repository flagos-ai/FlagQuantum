"""Device-native FP32 Double-Single matrices for the P4 statevector path."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from ..core.ir import Instruction
from ..core.parameters import Parameter, ParameterExpression
from ..numerics.double_single import (
    DoubleSingleComplexTensor,
    DoubleSingleTensor,
    double_single_sin_cos,
)

P4_PARAMETER_GATES = frozenset({"rx", "ry", "rz", "rxx", "ryy", "rzz"})
P4_MAX_ABS_ANGLE = 1024.0

_ZERO = (0.0, 0.0)
_ONE = (1.0, 0.0)
_NEG_ONE = (-1.0, 0.0)
_HALF = (0.5, 0.0)
_NEG_HALF = (-0.5, 0.0)
_SQRT_HALF = (0.7071067690849304, 1.2101617485882343e-8)
_NEG_SQRT_HALF = (-0.7071067690849304, -1.2101617485882343e-8)
_PI_OVER_TWO = (1.5707963705062866, -4.371138828673793e-8)

_Pair = tuple[float, float]
_ComplexPair = tuple[_Pair, _Pair]


def _scalar(value: _Pair, *, device: torch.device) -> DoubleSingleTensor:
    return DoubleSingleTensor(
        torch.tensor(value[0], dtype=torch.float32, device=device),
        torch.tensor(value[1], dtype=torch.float32, device=device),
    )


def _complex_scalar(
    real: _Pair = _ZERO,
    imag: _Pair = _ZERO,
    *,
    device: torch.device,
) -> DoubleSingleComplexTensor:
    return DoubleSingleComplexTensor(
        _scalar(real, device=device), _scalar(imag, device=device)
    )


def _matrix(
    rows: Sequence[Sequence[DoubleSingleComplexTensor]],
) -> DoubleSingleComplexTensor:
    width = len(rows[0])
    if not rows or any(len(row) != width for row in rows):
        raise ValueError("Double-Single gate matrix rows must be rectangular")

    def word(component: str, part: str) -> torch.Tensor:
        return torch.stack(
            tuple(
                getattr(getattr(entry, component), part)
                for row in rows
                for entry in row
            )
        ).reshape(len(rows), width)

    return DoubleSingleComplexTensor(
        DoubleSingleTensor(word("real", "high"), word("real", "low")),
        DoubleSingleTensor(word("imag", "high"), word("imag", "low")),
    )


def _fixed_entries(name: str) -> tuple[tuple[_ComplexPair, ...], ...]:
    z = (_ZERO, _ZERO)
    o = (_ONE, _ZERO)
    n = (_NEG_ONE, _ZERO)
    i = (_ZERO, _ONE)
    ni = (_ZERO, _NEG_ONE)
    q = (_SQRT_HALF, _ZERO)
    nq = (_NEG_SQRT_HALF, _ZERO)
    hq = (_HALF, _HALF)
    hnq = (_HALF, _NEG_HALF)
    fixed: dict[str, tuple[tuple[_ComplexPair, ...], ...]] = {
        "i": ((o, z), (z, o)),
        "x": ((z, o), (o, z)),
        "y": ((z, ni), (i, z)),
        "z": ((o, z), (z, n)),
        "h": ((q, q), (q, nq)),
        "s": ((o, z), (z, i)),
        "sdg": ((o, z), (z, ni)),
        "t": ((o, z), (z, (_SQRT_HALF, _SQRT_HALF))),
        "tdg": ((o, z), (z, (_SQRT_HALF, _NEG_SQRT_HALF))),
        "sx": ((hq, hnq), (hnq, hq)),
        "sxdg": ((hnq, hq), (hq, hnq)),
        "cx": ((o, z, z, z), (z, o, z, z), (z, z, z, o), (z, z, o, z)),
        "cy": ((o, z, z, z), (z, o, z, z), (z, z, z, ni), (z, z, i, z)),
        "cz": ((o, z, z, z), (z, o, z, z), (z, z, o, z), (z, z, z, n)),
        "swap": ((o, z, z, z), (z, z, o, z), (z, o, z, z), (z, z, z, o)),
    }
    try:
        return fixed[name]
    except KeyError as exc:
        raise KeyError(name) from exc


def _fixed_matrix(name: str, *, device: torch.device) -> DoubleSingleComplexTensor:
    return _matrix(
        tuple(
            tuple(_complex_scalar(real, imag, device=device) for real, imag in row)
            for row in _fixed_entries(name)
        )
    )


def _parameter_pair(
    value: Any,
    *,
    bindings: Mapping[str, Any],
    device: torch.device,
) -> tuple[DoubleSingleTensor, bool]:
    if isinstance(value, Parameter):
        try:
            value = bindings[value.name]
        except KeyError as exc:
            raise ValueError(f"missing P4 parameter binding {value.name!r}") from exc
    elif isinstance(value, ParameterExpression):
        raise NotImplementedError("P4 does not yet support parameter expressions")

    if isinstance(value, DoubleSingleTensor):
        if value.high.numel() != 1 or value.high.device != device:
            raise ValueError(
                "P4 Double-Single parameters must be scalar and already reside "
                "on the execution device"
            )
        pair = DoubleSingleTensor(value.high.reshape(()), value.low.reshape(()))
        host_ingestion = False
    elif isinstance(value, torch.Tensor):
        if value.numel() != 1 or value.requires_grad or value.is_complex():
            raise ValueError("P4 gate parameters must be fixed real scalars")
        if value.dtype != torch.float32:
            raise TypeError(
                "P4 tensor parameters must use float32 or DoubleSingleTensor; "
                "double-precision tensors would hide host precision encoding"
            )
        host_ingestion = value.device != device
        pair = DoubleSingleTensor.from_float32(value.detach().to(device).reshape(()))
    elif isinstance(value, (float, int)):
        pair = DoubleSingleTensor.from_float32(
            torch.tensor(value, dtype=torch.float32, device=device)
        )
        host_ingestion = True
    else:
        raise TypeError(f"unsupported P4 parameter type {type(value).__name__}")

    if bool(torch.any(torch.abs(pair.to_float32()) > P4_MAX_ABS_ANGLE).item()):
        raise ValueError(
            f"P4 device trigonometry certifies |angle| <= {P4_MAX_ABS_ANGLE}"
        )
    return pair, host_ingestion


def _shifted(
    value: DoubleSingleTensor,
    direction: int,
) -> DoubleSingleTensor:
    if direction not in {-1, 0, 1}:
        raise ValueError("P4 parameter-shift direction must be -1, 0, or 1")
    if direction == 0:
        return value
    shift = _scalar(_PI_OVER_TWO, device=value.high.device)
    return value.add(shift) if direction > 0 else value.subtract(shift)


def _rotation_matrix(
    name: str,
    angle: DoubleSingleTensor,
) -> DoubleSingleComplexTensor:
    half = angle.multiply(_scalar(_HALF, device=angle.high.device))
    sine, cosine = double_single_sin_cos(half)
    zero = DoubleSingleTensor.zeros_like(angle.high)

    def entry(
        real: DoubleSingleTensor | None = None,
        imag: DoubleSingleTensor | None = None,
    ) -> DoubleSingleComplexTensor:
        return DoubleSingleComplexTensor(
            real if real is not None else zero,
            imag if imag is not None else zero,
        )

    c = entry(real=cosine)
    s = entry(real=sine)
    ns = entry(real=sine.negate())
    positive_i_s = entry(imag=sine)
    negative_i_s = entry(imag=sine.negate())
    z = entry()
    if name == "rx":
        return _matrix(((c, negative_i_s), (negative_i_s, c)))
    if name == "ry":
        return _matrix(((c, ns), (s, c)))
    if name == "rz":
        return _matrix(
            (
                (entry(real=cosine, imag=sine.negate()), z),
                (z, entry(real=cosine, imag=sine)),
            )
        )
    if name == "rxx":
        return _matrix(
            (
                (c, z, z, negative_i_s),
                (z, c, negative_i_s, z),
                (z, negative_i_s, c, z),
                (negative_i_s, z, z, c),
            )
        )
    if name == "ryy":
        return _matrix(
            (
                (c, z, z, positive_i_s),
                (z, c, negative_i_s, z),
                (z, negative_i_s, c, z),
                (positive_i_s, z, z, c),
            )
        )
    if name == "rzz":
        negative_phase = entry(real=cosine, imag=sine.negate())
        positive_phase = entry(real=cosine, imag=sine)
        return _matrix(
            (
                (negative_phase, z, z, z),
                (z, positive_phase, z, z),
                (z, z, positive_phase, z),
                (z, z, z, negative_phase),
            )
        )
    raise KeyError(name)


def encode_device_double_single_matrix(
    instruction: Instruction,
    *,
    bindings: Mapping[str, Any],
    device: torch.device,
    shift_parameter: str | None = None,
    shift_direction: int = 0,
) -> tuple[DoubleSingleComplexTensor, bool]:
    """Encode a supported gate using only real FP32 accelerator tensors."""

    if instruction.matrix is not None:
        raise NotImplementedError("P4 rejects custom matrices")
    try:
        return _fixed_matrix(instruction.name, device=device), False
    except KeyError:
        pass
    if instruction.name not in P4_PARAMETER_GATES:
        raise NotImplementedError(
            f"P4 device gate generation does not support {instruction.name!r}"
        )
    raw_angle = instruction.params.get("theta")
    angle, host_ingestion = _parameter_pair(raw_angle, bindings=bindings, device=device)
    if shift_parameter == "theta":
        angle = _shifted(angle, shift_direction)
    return _rotation_matrix(instruction.name, angle), host_ingestion


__all__ = (
    "P4_MAX_ABS_ANGLE",
    "P4_PARAMETER_GATES",
    "encode_device_double_single_matrix",
)
