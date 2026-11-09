# Native exact signed and binary-scaled values

Standing native-library implementation authorization covers nine remaining
standalone arithmetic originals. Extend the pre-freeze QNum annotation to
`QNum[width, signed, fractional_bits]` with a literal Boolean signedness and
0..width binary fractional bits. The type-checkable spelling is
`Annotated[QNum, (width, signed, fractional_bits)]`; existing unsigned width-only
forms retain their meaning. QBit and array elements remain unscaled bits.

Reuse Compiler-owned exact word inference and qint emission from the numeric
foundation. Existing unsigned expressions retain their established capture path.
Fresh numeric results may infer signedness and exact binary scale for subtraction,
negation, min/max and unsigned multiplication. Decimal values must be exactly
binary representable; there is no implicit eight-bit quantization or truncation.
Arithmetic reads of scaled inputs and general signed operators remain refused
until their alignment semantics are admitted. Inferred formats cannot overwrite
a conflicting declared format or silently discard significant bits.

Owned registers can allocate from their numeric annotation. Literal `^=` changes
the raw encoded word of an initialized, writable scalar, with a fitting
nonnegative integer mask; it does not numerically add or change the declared
scale. Fresh `|=` initializes a numeric value or computes into fresh output.
Signed initialization requires range checks and exact sign extension. Mutable
helper borrowing must preserve compatible numeric formats; numeric arithmetic
helpers, arithmetic locals with signed/fractional results and numeric use under
unsupported controls refuse. Existing coherent inverse/ownership checks remain.

A `numeric_format` tuple `(signed, fractional_bits)` on existing qreg declarations
records output interpretation in the canonical checked program, without a new IR
or storage type. Compiler validates shape, exact types, width bounds and disallows
record-field combinations. Services derives native numeric-format report data
only from this verified program. Existing Qmod vendor-prefixed report fields and
raw `Program.probabilities` keys retain their established interpretation. Numeric
readout is explicit through `Program.numeric_probabilities`, with the same
register selection, threshold and resource limits as `Program.probabilities`.
It returns an immutable `Mapping[tuple[Fraction, ...], float]`. Exact Fraction
keys express two's-complement signed values divided by the binary scale; Python
integer keys compare equally for integral values. Raw probabilities retain
their existing signature, diagnostics and integer keys. The decoded method
requires verified native numeric declarations; it refuses older programs or
Qmod reports rather than guessing numeric formats. Formats are sealed in the
private checked Services product, not trusted from editable report metadata.

This is an additive pre-freeze namespace admission under the user's standing
authorization to implement the complete native example library. It changes no
Stable Core signature, export, result or serialization schema; no release or
human trial approval is claimed. Compiler owns checked declaration semantics,
Services owns named value interpretation and Runtime remains unchanged. The
alternative of changing `probabilities` keys implicitly is rejected because it
would break existing raw-word consumers. Unsupported signed/scaled phase reads,
numeric arithmetic locals and numeric helper calls fail early in this slice.

For example, a native root can declare `a: Output[QNum[3]]` and
`result: Output[QNum]`, initialize `a |= 4`, then compute
`result |= a - 3.5`. After `result = fq.run(program)`, the explicit decoded
reader reports `{(Fraction(1, 2),): 1.0}` for the result register; the existing
raw reader reports `{(1,): 1.0}`. No manual bit-string decoding is required.

Acceptance preserves all nine exact original programs, including both maximum
files whose source is `max(3.5,a)`. Independent Circuit/full-state comparisons,
negative and extreme signed inputs, exact Fraction oracles, relative/global
phase, nonzero target XOR, scratch cleanup, initialization/alias/format errors,
nonbinary decimals and resource budgets are required. Each original counts as
native only after its complete high-level workflow and readout acceptance.
