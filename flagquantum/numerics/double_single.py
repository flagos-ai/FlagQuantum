"""Pure-PyTorch double-single FP32 arithmetic primitives.

A value is stored as the unevaluated sum ``high + low`` of two float32
tensors. All arithmetic kernels in this module execute with float32 tensor
operations; float64 conversion helpers are intentionally diagnostic-only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_SPLITTER = 4097.0  # 2**ceil(24 / 2) + 1 for IEEE-754 binary32.
_TWO_OVER_PI = (0.6366197466850281, 2.5682552973194106e-8)
_PI_OVER_TWO = (1.5707963705062866, -4.371138828673793e-8)
_SIN_COEFFICIENTS = (
    (-0.1666666716337204, 4.967053879312289e-9),
    (0.008333333767950535, -4.34617203337595e-10),
    (-0.00019841270113829523, 2.725596874933456e-12),
    (2.7557318844628753e-6, 3.793571224297229e-14),
    (-2.5052107943679403e-8, -4.4176230446483665e-16),
    (1.6059044372074283e-10, -5.352526511562726e-18),
    (-7.647163609812713e-13, -1.2200710471178288e-20),
    (2.8114573589663704e-15, -1.0462084739763658e-22),
)
_COS_COEFFICIENTS = (
    (-0.5, 0.0),
    (0.0416666679084301, -1.2417634698280722e-9),
    (-0.0013888889225199819, 3.3631094437103215e-11),
    (2.4801587642286904e-5, -3.40699609366682e-13),
    (-2.755731998149713e-7, 7.575112209051195e-15),
    (2.0876755879584152e-9, 1.1082839809204342e-16),
    (-1.147074536050896e-11, -2.372207689231238e-19),
    (4.7794772561329454e-14, 7.62544404448643e-22),
    (-1.5619206814541513e-16, -1.5404471465941993e-24),
)


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

    def reciprocal(self, *, iterations: int = 2) -> "DoubleSingleTensor":
        """Return a Newton-refined reciprocal using FP32 tensor operations."""

        if iterations < 1:
            raise ValueError("reciprocal requires at least one refinement iteration")
        estimate = DoubleSingleTensor.from_float32(torch.reciprocal(self.to_float32()))
        one = DoubleSingleTensor.from_float32(torch.ones_like(self.high))
        for _ in range(iterations):
            estimate = estimate.multiply(one.subtract(self.multiply(estimate))).add(
                estimate
            )
        return estimate.renormalized()

    def reciprocal_sqrt(self, *, iterations: int = 2) -> "DoubleSingleTensor":
        """Return a Newton-refined reciprocal square root in Double-Single."""

        if iterations < 1:
            raise ValueError(
                "reciprocal_sqrt requires at least one refinement iteration"
            )
        if bool(torch.any(self.to_float32() <= 0).item()):
            raise ValueError("reciprocal_sqrt requires strictly positive values")
        estimate = DoubleSingleTensor.from_float32(torch.rsqrt(self.to_float32()))
        half = DoubleSingleTensor.from_float32(torch.full_like(self.high, 0.5))
        three_halves = DoubleSingleTensor.from_float32(torch.full_like(self.high, 1.5))
        for _ in range(iterations):
            correction = three_halves.subtract(
                half.multiply(self).multiply(estimate.square())
            )
            estimate = estimate.multiply(correction)
        return estimate.renormalized()

    def sqrt(self, *, iterations: int = 2) -> "DoubleSingleTensor":
        """Return a Double-Single square root using reciprocal-sqrt refinement."""

        return self.multiply(self.reciprocal_sqrt(iterations=iterations))

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


def _constant_like(
    value: tuple[float, float], like: torch.Tensor
) -> DoubleSingleTensor:
    return DoubleSingleTensor(
        torch.full_like(like, value[0]), torch.full_like(like, value[1])
    )


def _polynomial(
    squared: DoubleSingleTensor,
    coefficients: tuple[tuple[float, float], ...],
) -> DoubleSingleTensor:
    result = _constant_like(coefficients[-1], squared.high)
    for coefficient in reversed(coefficients[:-1]):
        result = result.multiply(squared).add(_constant_like(coefficient, squared.high))
    return result


def _where(
    condition: torch.Tensor,
    left: DoubleSingleTensor,
    right: DoubleSingleTensor,
) -> DoubleSingleTensor:
    return DoubleSingleTensor(
        torch.where(condition, left.high, right.high),
        torch.where(condition, left.low, right.low),
    )


def double_single_sin_cos(
    value: DoubleSingleTensor,
) -> tuple[DoubleSingleTensor, DoubleSingleTensor]:
    """Evaluate sine and cosine with FP32-only Double-Single arithmetic.

    Range reduction uses the nearest multiple of pi/2. The quadrant decision is
    made from the FP32 leading approximation, while the subtraction and both
    Taylor polynomials retain high/low residuals. The certified P4 envelope
    limits absolute inputs to 1024 radians; callers enforce that boundary.
    """

    quadrant = torch.round(value.to_float32() * _TWO_OVER_PI[0])
    reduced = value.subtract(
        DoubleSingleTensor.from_float32(quadrant).multiply(
            _constant_like(_PI_OVER_TWO, value.high)
        )
    )
    squared = reduced.square()
    one = DoubleSingleTensor.from_float32(torch.ones_like(value.high))
    sine = reduced.add(
        reduced.multiply(squared).multiply(_polynomial(squared, _SIN_COEFFICIENTS))
    )
    cosine = one.add(squared.multiply(_polynomial(squared, _COS_COEFFICIENTS)))

    phase = torch.remainder(quadrant, 4.0)
    phase_one = phase == 1.0
    phase_two = phase == 2.0
    phase_three = phase == 3.0
    resolved_sine = _where(
        phase_one,
        cosine,
        _where(phase_two, sine.negate(), _where(phase_three, cosine.negate(), sine)),
    )
    resolved_cosine = _where(
        phase_one,
        sine.negate(),
        _where(phase_two, cosine.negate(), _where(phase_three, sine, cosine)),
    )
    return resolved_sine.renormalized(), resolved_cosine.renormalized()


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
    "double_single_sin_cos",
    "double_single_sum",
    "quick_two_sum",
    "split_float32",
    "two_prod",
    "two_sum",
)
