"""Pure-PyTorch double-single FP32 arithmetic primitives.

A value is stored as the unevaluated sum ``high + low`` of two float32
tensors. All arithmetic kernels in this module execute with float32 tensor
operations; float64 conversion helpers are intentionally diagnostic-only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_SPLITTER = 4097.0  # 2**ceil(24 / 2) + 1 for IEEE-754 binary32.


def _require_float32(value: torch.Tensor, *, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must use torch.float32, got {value.dtype}")


def _require_finite(value: torch.Tensor, *, name: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite, non-overflowing values")


def _require_pair(left: torch.Tensor, right: torch.Tensor) -> None:
    _require_float32(left, name="left")
    _require_float32(right, name="right")
    if left.shape != right.shape:
        raise ValueError(f"tensor shapes differ: {left.shape} != {right.shape}")
    if left.device != right.device:
        raise ValueError(f"tensor devices differ: {left.device} != {right.device}")


def two_sum(
    left: torch.Tensor, right: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a rounded sum and its exact binary32 residual."""

    _require_pair(left, right)
    total = left + right
    right_virtual = total - left
    residual = (left - (total - right_virtual)) + (right - right_virtual)
    return total, residual


def quick_two_sum(
    left: torch.Tensor, right: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Renormalize a leading value and a smaller correction."""

    _require_pair(left, right)
    total = left + right
    residual = right - (total - left)
    return total, residual


def split_float32(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split finite, non-overflowing binary32 values into high and low words."""

    _require_float32(value, name="value")
    _require_finite(value, name="value")
    scaled = value * _SPLITTER
    _require_finite(scaled, name="scaled value")
    high = scaled - (scaled - value)
    low = value - high
    return high, low


def two_prod(
    left: torch.Tensor, right: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a binary32 product and Dekker residual without float64 compute."""

    _require_pair(left, right)
    product = left * right
    left_high, left_low = split_float32(left)
    right_high, right_low = split_float32(right)
    residual = (
        ((left_high * right_high - product) + left_high * right_low)
        + left_low * right_high
    ) + left_low * right_low
    return product, residual


@dataclass(frozen=True)
class DoubleSingleTensor:
    """One real tensor represented by normalized float32 high/low words."""

    high: torch.Tensor
    low: torch.Tensor

    def __post_init__(self) -> None:
        _require_pair(self.high, self.low)

    @classmethod
    def from_float32(cls, value: torch.Tensor) -> "DoubleSingleTensor":
        _require_float32(value, name="value")
        return cls(value, torch.zeros_like(value))

    @classmethod
    def from_float64(cls, value: torch.Tensor) -> "DoubleSingleTensor":
        """Create a pair from a float64 reference; not an accelerator kernel."""

        if not isinstance(value, torch.Tensor) or value.dtype != torch.float64:
            raise TypeError("value must be a torch.float64 tensor")
        high = value.to(torch.float32)
        low = (value - high.to(torch.float64)).to(torch.float32)
        return cls(high, low)

    @classmethod
    def zeros_like(cls, value: torch.Tensor) -> "DoubleSingleTensor":
        _require_float32(value, name="value")
        zero = torch.zeros_like(value)
        return cls(zero, zero)

    def renormalized(self) -> "DoubleSingleTensor":
        high, low = two_sum(self.high, self.low)
        return DoubleSingleTensor(high, low)

    def add(self, other: "DoubleSingleTensor") -> "DoubleSingleTensor":
        _require_pair(self.high, other.high)
        high, residual = two_sum(self.high, other.high)
        correction = residual + self.low + other.low
        normalized_high, normalized_low = two_sum(high, correction)
        return DoubleSingleTensor(normalized_high, normalized_low)

    def multiply(self, other: "DoubleSingleTensor") -> "DoubleSingleTensor":
        _require_pair(self.high, other.high)
        high, residual = two_prod(self.high, other.high)
        correction = (
            residual
            + self.high * other.low
            + self.low * other.high
            + self.low * other.low
        )
        normalized_high, normalized_low = two_sum(high, correction)
        return DoubleSingleTensor(normalized_high, normalized_low)

    def square(self) -> "DoubleSingleTensor":
        return self.multiply(self)

    def negate(self) -> "DoubleSingleTensor":
        return DoubleSingleTensor(-self.high, -self.low)

    def subtract(self, other: "DoubleSingleTensor") -> "DoubleSingleTensor":
        return self.add(other.negate())

    def to_float64(self) -> torch.Tensor:
        """Reconstruct in float64 for certification and diagnostics only."""

        return self.high.to(torch.float64) + self.low.to(torch.float64)

    def to_float32(self) -> torch.Tensor:
        return self.high + self.low

    def to(self, device: str | torch.device) -> "DoubleSingleTensor":
        """Move both words without permitting an accidental dtype conversion."""

        return DoubleSingleTensor(self.high.to(device), self.low.to(device))


@dataclass(frozen=True)
class DoubleSingleComplexTensor:
    """Split real/imaginary complex values with double-single components."""

    real: DoubleSingleTensor
    imag: DoubleSingleTensor

    def __post_init__(self) -> None:
        _require_pair(self.real.high, self.imag.high)

    @classmethod
    def from_complex64(cls, value: torch.Tensor) -> "DoubleSingleComplexTensor":
        if not isinstance(value, torch.Tensor) or value.dtype != torch.complex64:
            raise TypeError("value must be a torch.complex64 tensor")
        return cls(
            DoubleSingleTensor.from_float32(value.real),
            DoubleSingleTensor.from_float32(value.imag),
        )

    @classmethod
    def from_complex128(cls, value: torch.Tensor) -> "DoubleSingleComplexTensor":
        """Create from a complex128 reference; not an accelerator kernel."""

        if not isinstance(value, torch.Tensor) or value.dtype != torch.complex128:
            raise TypeError("value must be a torch.complex128 tensor")
        return cls(
            DoubleSingleTensor.from_float64(value.real),
            DoubleSingleTensor.from_float64(value.imag),
        )

    def add(self, other: "DoubleSingleComplexTensor") -> "DoubleSingleComplexTensor":
        return DoubleSingleComplexTensor(
            self.real.add(other.real), self.imag.add(other.imag)
        )

    def multiply(
        self, other: "DoubleSingleComplexTensor"
    ) -> "DoubleSingleComplexTensor":
        real = self.real.multiply(other.real).subtract(self.imag.multiply(other.imag))
        imag = self.real.multiply(other.imag).add(self.imag.multiply(other.real))
        return DoubleSingleComplexTensor(real, imag)

    def abs_squared(self) -> DoubleSingleTensor:
        return self.real.square().add(self.imag.square())

    def to_complex128(self) -> torch.Tensor:
        return torch.complex(self.real.to_float64(), self.imag.to_float64())

    def to_complex64(self) -> torch.Tensor:
        return torch.complex(self.real.to_float32(), self.imag.to_float32())

    def to(self, device: str | torch.device) -> "DoubleSingleComplexTensor":
        return DoubleSingleComplexTensor(self.real.to(device), self.imag.to(device))


def double_single_sum(
    value: torch.Tensor | DoubleSingleTensor, dim: int | None = None
) -> DoubleSingleTensor:
    """Deterministically accumulate a tensor using double-single addition."""

    pair = (
        value
        if isinstance(value, DoubleSingleTensor)
        else DoubleSingleTensor.from_float32(value)
    )
    high = pair.high.reshape(-1) if dim is None else pair.high.movedim(dim, -1)
    low = pair.low.reshape(-1) if dim is None else pair.low.movedim(dim, -1)
    if high.shape[-1] == 0:
        raise ValueError("double_single_sum requires a non-empty reduction dimension")
    accumulator = DoubleSingleTensor.zeros_like(high[..., 0])
    for index in range(high.shape[-1]):
        accumulator = accumulator.add(
            DoubleSingleTensor(high[..., index], low[..., index])
        )
    return accumulator


def double_single_dot(left: torch.Tensor, right: torch.Tensor) -> DoubleSingleTensor:
    """Return a deterministic double-single dot product of float32 tensors."""

    _require_pair(left, right)
    products = DoubleSingleTensor.from_float32(left).multiply(
        DoubleSingleTensor.from_float32(right)
    )
    return double_single_sum(products)


__all__ = (
    "DoubleSingleComplexTensor",
    "DoubleSingleTensor",
    "double_single_dot",
    "double_single_sum",
    "quick_two_sum",
    "split_float32",
    "two_prod",
    "two_sum",
)
