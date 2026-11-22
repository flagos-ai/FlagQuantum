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
128-bit complex-state execution. The value module's physical primitive, quantum workspace and measurement counts
are zero. The accepted exact Z_POW artifact, KMX codec and conditional phase
binding now connect it to QFT demolition measurement. Those execution consumers
have their own source-stream, complex-branch and clean-work evidence in
`test_reversible_phase_angles.py`, `test_kickmix_qft_measurement.py` and
`test_kickmix_qft_measurement_file.py`. Their measured physical primitives and
analytical models have separate owners.

## Complete source-file disposition

The complete header K-0296 has blob
`96eec0ecc72d02dbf31098d14fa85e2d8671fdc8`.
The test parent K-0297 has blob
`247d8da592df92737bf09bab0278b548e5708093`. All three angle source files and the
shared `word_ops.h` dependency were rechecked against the fixed inventory during
the complete-file intake.

| Complete original definition | Native target and disposition |
| --- | --- |
| Raw words, zero/copy construction, equality and Boolean conversion | Frozen low/high words, dataclass value equality and `__bool__` |
| `clifford_hierarchy_level` | Same named native method using the exact least-significant set bit |
| `from_power_of_2_half_turns` | `from_power_of_two_half_turns` with the same exponent grid and zero wrap |
| `from_half_turns_exact_double` | Same named factory, original half-first conversion and representability refusal |
| `from_rounded_decimal_half_turns` | Same named factory, source 128-digit sticky prefix, limb-equivalent long division and ties-to-even |
| `to_half_turns_decimal_helper` | `to_decimal_half_turns` uses the same base-ten digit recurrence over the exact ring remainder |
| `to_decimal_half_turns` | Same named native renderer |
| `write_decimal_half_turns_to(FILE*)` | Shared exact decimal text is handed to caller-owned output storage. Native value objects have no FILE ownership. |
| `str` and ostream insertion | Native `__str__` retains exact text and the pi suffix |
| Angle `operator+` and `operator+=` | `__add__` and assignment of the resulting immutable value |
| Unary negation | `__neg__`, two-word modular negation |
| `rotated180` | `rotated_half_turn`, the same high-word bit toggle |
| `is_closer_to_half_turn_than_no_turn` | `closer_to_half_turn_than_zero`, the same two high-bit test |
| 45/90/180-degree predicates | Same named native predicates and exact ring masks |
| Integer assignment and addition | `from_integer_half_turns` and `plus_integer_half_turns` retain integer parity |
| `operator*` and `operator*=` | `__mul__` and immutable assignment retain unsigned 64-bit factor semantics |

All 23 unchanged original C++ tests pass, including actual temporary FILE
output, stream output, the complete decimal tables, all factories, predicates,
word arithmetic and in-place results. A freshly compiled original oracle again
matches all 4,755 accepted comparison requests, including the 701 source decimal
strings and 28 matching refusals. Existing portable native tests reuse the same
full definition mapping. Native exact-word, KMX, QFT feedback and branch consumers
are rerun together for this intake.

K-0296 and K-0297 are complete. The implementation parent K-0295 stays open for
the already recorded smallest-subnormal underflow boundary. The native factory
retains that original result, and ordinary below-grid double values retain the
original refusal. The source implementation is preserved in its locked form.
The wider `word_ops.h` parent remains independently open even though this
angle module's limb multiply/divide dependencies are represented by exact Python
integer arithmetic. Pending Proposals 116 through 119 stay outside this scope.

The complete-file intake passes 342 concentrated exact-angle/QFT checks and
29 local integration checks. The documentation-only closure passes Integration
team scope and whitespace validation. Production typing, formatting, API and
architecture evidence is reused from the accepted unchanged implementations.
