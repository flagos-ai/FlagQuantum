# Complete declared-width integer intake

## Frozen scope and implementation interfaces

The source group is Kickmix `util/fixed_width_int.cc`, `.h`, `.test.cc` and
`.perf.cc`, with direct `word_ops.h`, `word_ops.test.cc` and `bit.h` dependencies,
at `9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`. All source blobs are verified
against the immutable inventory before implementation. The complete original
38 integer tests pass unchanged. Native value/header and test correspondence
is recorded below. The original
implementation defects and unmet performance goals remain separate open scope.

Core owns private `FixedWidthInteger` values with declared `width` and unsigned
little-endian storage `value`. Allocated storage has `ceil(width/64)` words.
Source bit updates can set hanging bits within that storage, so raw storage may
exceed declared width until an explicit clipping operation. Factory construction
clips to declared width. This distinction is part of the interface.

Core text helpers own source integer grammar and digit conversion. Compiler
format entry points retain their existing unsigned input and report formatting.
Core product helpers retain source word traversal and alias reads. Simulation
owns random state and consumes the accepted standard mt19937_64 stream.
Core values have no register allocation, backend or numerical execution owner.

Immutable methods return successors. Carry/borrow operations return successor
and flag. Widening operators preserve source output widths, while named
in-place counterparts retain the left width. `other is self` identifies source
sequential alias reads inside one operation. Declared-bit iteration is LSB-first.
Quantum register words retain their established MSB-first boundary.

The source's scalar carry, short-AND high-word retention and self-multiplication
counterexamples are retained as comparison targets. Native invalid-memory and
nonterminating inputs require explicit refusal and independent source evidence.
The source files remain unchanged. All quantum primitive, measurement and work
counts for scalar value operations are zero. Their execution consumers retain
separate observed primitive and analytical resource accounts.

## Complete definition disposition

| Frozen definition group | Authoritative native owner and behavior |
| --- | --- |
| Constructors, copy/move and width/word ownership | FixedWidthInteger, from_integer and from_text. Immutable value construction replaces C++ transfer/allocation. Signed scalar construction retains sign extension. Raw construction retains allocated hanging bits. |
| Generic/binary/hex/decimal parsing and output | _unsigned_integer_text and value formatting. Explicit decimal widths truncate. Binary/hex widths include leading digits. Generic parsing trims ASCII spaces. Hanging-state decimal conversion preserves the source first-division clipping order. |
| Iterator, size, front/back/bit references and bit mutation | width, iteration, bit/reference_bit, set_bit and flip_bit. Immutable successor values replace writable pointer references. Source references admit declared bits, while set/reset/flip admit allocated hanging positions. |
| clear/resize/write/clip | resize_clear, resize_shrink, write_from and clip. Self write_from clears its own operand, matching the original sequential alias. |
| bool, uint64, double, mantissa and comparison overloads | Native conversions/comparisons retain source numeric equality independent of width, 64-bit scalar bounds and truncated leading-53-bit float conversion. Negative signed scalars compare below unsigned values. |
| Increment/decrement, word/scalar addition and subtraction | Value methods return successors and flags. Scalar carry preserves original increment-at-zero behavior. Signed addition preserves its strict comparison edge. Subtraction clips even after an allocated-boundary borrow. |
| XOR/AND/OR, negate and shifts | Value methods and Python assignment operators. Short AND retains unmatched target words. Scalar XOR uses the signed-32 mask contract. Shift traversal follows original word update order and signed offset conversion, including extreme uint64 wrapping. |
| Shifted arithmetic and all product overloads | _fixed_width_products and value methods preserve original increasing/decreasing limb traversal, low/high carry and target aliases. subtract_shifted returns the allocated-boundary borrow separately from clipping. |
| Widening arithmetic, reverse scalar operators and shifts | Python value operators retain original declared output widths. Assignment operators retain the left width. Self in-place multiplication reads the changing target. Scalar non-assignment arithmetic uses the source 64-bit size_t word. |
| Modular accumulation, double/half/negation, remainder, coprimality and inverse | _fixed_width_modular owns binary-cofactor and approximate quotient algorithms. Nonunit inverse returns the original cofactor. Modulus aliases are reread after mutation where the source does so. Nonmutating remainder copies the target. |
| random/random_mod/randomize/randomize_mod/randomize_coprime_mod | Simulation fixed_width_sampling consumes caller-owned uint64 draws. Existing mt19937_64 supplies source seeded state. Width masking, retries and the next consumed draw match original C++. |
| All four original benchmark bodies | The unchanged .perf.cc is executed with its original seed_seq and goals. A source harness records all four complete stateful bodies after 128 iterations and native results agree. Native timings and original goals are reported independently below. |

## Executable comparison and retained defects

The checked-in frozen-source harness, generator and JSON-line corpus compare
74,043 original observations. They cover every definition group above, widths
0/1/7/63/64/65/127/128/129/256/512/1024, raw hanging states, cross-width operands,
small modular sweeps, nonunits, both carry/borrow states, sequential target
aliases, all four performance bodies, unsigned shift wrapping, negative signed
shifts, malformed literals, explicit widths and a 5,000-digit decimal input.
All 38 unchanged original C++ tests pass. The complete source header/test parents
can reuse these observations and native operator/lifetime assertions.

The original implementation parent was held open for independently demonstrated
source defects; the [2026-11-28 review](G2_FUNCTIONAL_MIGRATION_DISPOSITIONS.md)
accepts complete defined-domain migration with those witnesses retained.
AddressSanitizer reports a heap-buffer-overflow for both unsigned
assignment and scalar carry on zero-width storage. The unsanitized program can
return normally, so its exit status alone supplies no memory-safety evidence.
Native methods refuse these two cases. An isolated self-alias `%=` call exceeds
a two-second subprocess bound. The source decrements the aliased modulus to zero
and keeps comparing zero against zero. Native mutating remainder refuses this
case. Ordinary `value % value` copies its target and returns zero in both owners.
The original source files remain unchanged.

Scalar carry on UINT64_MAX plus one returns one with no carry at widths64/128.
Self in-place multiplication at width8 maps2/3/5/7 to0/4/8/76. Wide zero plus
signed negative one retains only the low UINT64_MAX word. Short AND preserves
higher unmatched words. These observed source semantics remain comparison targets.
The new private value interface uses immutable snapshots and successor values.
It introduces no public type, schema or stable entry. Original raw pointer
addresses, move-from object inspection, implicit C++ numeric coercion and CPU
intrinsic selection retain their language/platform ownership.

## Resources and performance

All scalar operations emit zero quantum primitives, measurements and work bits.
Accepted field, lookup and fixed-point quantum consumers retain their own complete
complex-state, measured feedback and work-zero evidence. Source circuit model
estimates are independent from actual emitted primitive counts.

| Stateful workload | Observed native seconds per iteration | Original source goal seconds |
| --- | --- | --- |
| 1024-bit scalar multiplication | 7.45e-7 | 1.4e-8 |
| 1024-bit word product accumulation | 8.48e-5 | 1.9e-7 |
| 1024-bit scalar product accumulation | 1.00e-5 | 1.5e-8 |
| 2048/1024-bit repeated remainder | 3.94e-4 | 2.4e-6 |

These one-run local observations include Python immutable updates and source
limb order. All four final values match original C++. The native implementation
does not meet the original C++ benchmark goals. K-0300's functional migration
is now accepted by that review, while performance disposition remains open.
Original C++ observed timings are stored separately
from both its goals and these native observations. No throughput parity is claimed.

Regeneration uses `tests/unit/fixtures/generate_fixed_width_integer.py` with
`--source-root`, `--corpus` and `--output`. It verifies six frozen dependency
blobs, compiles the unchanged implementation and streams the same record schema.
All validation receipts use `ftqc-reference-env/g2-fixed-width-` outside the repo.
Compiler unsigned report formatting now consumes the shared Core parser/digits
and preserves its existing refusal wording and consumers.

## Batch verification

The final concentrated regression passes285checks with no skips. Local register
integration passes92checks, including actual append execution, phase branches
and work cleanup. Strict mypy covers12Python files. Black, Ruff, Integration
scope, architecture, multi-level IR and public API checks pass. All74,043scalar
records regenerate byte-for-byte from the frozen source after the final oracle
readability pass. The word dependency independently compares3,472records in
both source product routes. Qualtran0.7.0 and Kickmix0.1 with the actual Simulator
are verified in the environment receipt.

Whole-repository dependency policy reports720existing findings. No finding
names a touched path. This batch supplies no whole-repository dependency-policy
or full-default-suite pass. An initial direct IR command resolved another
checkout's editable installation. Repeating it through the explicit local
reference wrapper passes against this worktree. Oracle fixture setup was
corrected for empty raw words, and its regenerated outputs remain exact.
