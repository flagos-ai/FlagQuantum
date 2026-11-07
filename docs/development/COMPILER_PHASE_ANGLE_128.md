# Exact compiler phase angles

`flagquantum.compiler.artifacts.phase_angle.PhaseAngle128` ports the value
semantics of Kickmix `FixedPrecisionAngle128` from commit
`9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`. The original implementation blob is
`492b3d63d9bc6678283c974c99cc3bcaf91d9d3c`.

## Ownership and interface

This is a compiler-owned exact value, not an execution kernel or a stable Core
parameter type. Two unsigned 64-bit words describe
`2*pi*(low/2**128 + high/2**64)`. Their arithmetic wraps modulo a full turn.
Python values are immutable: source in-place operators correspond to assignment
of the resulting value, preserving word results without exposing C++ references.
There is no new stable export, instruction schema or public serialization field.

The port includes the source's power-of-two and double factories, rounded
decimal parsing, exact decimal rendering, hierarchy classification, Boolean
value, addition, negation, unsigned multiplication, integer half-turn update,
half-turn rotation, nearest-half-turn predicate and 45/90/180-degree predicates.
The source FILE and ostream writers share the decimal renderer; Python string
rendering retains the same decimal value and pi suffix.

The decimal parser retains the original algorithm: ASCII decimal grammar,
integer parity, trailing-zero removal, a 128-digit fractional prefix and sticky
suffix, eight-limb fractional long division, and ties-to-even rounding. A Python
integer packs the same eight 64-bit limbs; each digit performs the original
high-to-low division by ten. This does not route rounding through a float or
Decimal implementation. Exact decimal output repeatedly multiplies the source
128-bit remainder by ten until it vanishes.

## Source and portable evidence

A C++ oracle linked against the unchanged locked object files agrees on 4,755
records, including 701 decimal strings extracted from the original tests.
Factories, both word carries, wrapping operations, hierarchy, all predicates and
exact decimal text agree. All 28 rejected inputs agree as well. The portable
fixture keeps 194 native records, source commit/blob and Apache-2.0 attribution,
so regular tests need no C++ compiler or installed Kickmix to reproduce them.

Additional tests cover exact midpoint rounding, positive/negative sticky digits
past the 128th place, full-turn wrap at a rounding boundary, the ASCII grammar,
unsigned word/multiplier bounds, and the representable power grid. The source
reference remains unmodified.

The source double factory halves its input before checking the low bits. The
smallest positive and negative subnormal double therefore underflow to zero
instead of rejecting a nonrepresentable angle. This behavior is retained and
recorded as a source boundary; the factory name does not strengthen that source
guarantee. Ordinary values below the grid still raise as the source does.

## Numerical conversion boundary

`half_turns()` returns the stored half-turn value as an exact Fraction.
`signed_half_turns_float()` explicitly converts the centered representative to
binary64. It retains the stored words and does not feed the rounded value back
into the source arithmetic or decimal codec. Its half-turn rounding error is
at most `2**-54` over the representative interval.

Centering matters: the negative of the smallest stored angle has a positive
half-turn representative extremely close to two, which would round to two as a
float. The centered representative is exactly `-2**-127`, preserving that tiny
phase at the numerical boundary. Tests apply both signs of this phase through
FlagQuantum's existing `phase` instruction on a complex128 state and verify the
imaginary amplitude down to `1e-53` absolute tolerance.

```python
from fractions import Fraction
from flagquantum.compiler.artifacts.phase_angle import PhaseAngle128

angle = -PhaseAngle128(1, 0)
assert angle.half_turns() == 2 - Fraction(1, 1 << 127)
assert angle.signed_half_turns_float() == -(2.0 ** -127)
assert -angle == PhaseAngle128(1, 0)
```

The numerical bridge uses ordinary simulation precision. It does not claim
128-bit complex-state execution. This value-only module emits no gates, owns no
quantum work and supplies no gate-cost estimate. The next consumer is the locked
QFT demolition measurement. Its exact conditional phase operations require an
internal artifact/codec contract before they can be emitted and bound; that
consumer is not certified by the value tests alone.
