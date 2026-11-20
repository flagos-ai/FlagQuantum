# Unsigned fixed-point physical storage

## Approved contract and ownership

Proposal 115 was explicitly approved on November 20, 2026. Core now admits
canonical unsigned quantum fixed-point words of every positive built-in integer
width, with a built-in integer fractional width at most that word width.
Negative fractional widths retain their original metadata. Logical validation
continues checking canonical parameter order, exact False signedness and zero
classical width before physical admission. Nominal type and MSB-first bit
storage survive the existing Split/Join and Cast paths.

Compiler separately checks that fixed-point control ports have width one.
The locked CtrlSpecAnd refuses wider QFxp predicates as noninteger types.
The storage extension therefore requires an explicit predicate guard. Existing
one-bit control polarities and negative fractional metadata reuse Proposals
113 and 114. Signed, symbolic, zero-width, classical and noncanonical fixed-point
storage keep their physical refusals.

## Executable acceptance

`tests/unit/test_unsigned_fixed_point_storage.py` verifies raw source bits and
zero-gate Cast identity for scalar and row-major shaped ports, including an
empty shape. Empty identities use one unused clean spectator because the
existing physical binder requires at least one physical wire. RIGHT allocation
and matching LEFT zero-image consumption retain owned output lifetimes.

Fixed-point words cast to the existing unsigned Add component and cast back.
Forward and exact inverse preserve both nominal ports. Primitive multisets,
per-wire ordering, operands and measurement feedback match the accepted flat
Add owners after fresh classical-slot renaming. Renaming checks that each
conditional consumes its actual immediately preceding measurement. Existing
immutable source Add/And comparisons are reused. Coherent widths one, two and
three include an entangled spectator, complete complex states and clean work.
All forward/inverse measurement history pairs are checked, including sixteen
pairs at width three. Normal circuit.append(op) round trips use the existing
runtime entry. Positive, zero and negative fractional formats are covered.

## Resources and source boundaries

| Operation | Observed native primitives and clean work | Original analytical account |
| --- | --- | --- |
| Cast identity | Zero primitives and zero allocated work | Zero QEC gates |
| Two-bit shared Add | 4 T/Tdg, 9 CX, 1 measurement/reset, 1 work bit | 1 AND, 4 Clifford, 1 measurement, 4 T-equivalent |
| Three-bit shared Add | 8 T/Tdg, 21 CX, 2 measurements/resets, 2 work bits | 2 AND, 11 Clifford, 2 measurements, 8 T-equivalent |
| Eight-bit shared Add | 28 T/Tdg, 81 CX, 7 measurements/resets, 7 work bits | 7 AND, 46 Clifford, 7 measurements, 28 T-equivalent |

The original width-one Add model reports negative three Cliffords although its
actual circuit emits one CX. The unchanged model is retained in the receipt.
The multiplication source roots have no declared decomposition. Admitting their
fixed-point port storage supplies a direct dependency. Original root
primitive-order alignment remains unavailable. SquareRealNumber's unused second input and
InvertRealNumber's mixed THRU result versus native pure-fraction output remain
specific open executable contracts in INTEGER_MULTIPLICATION_FILE_AUDIT.md.

Evidence receipts use `ftqc-reference-env/g2-wide-fixed-point-storage-` outside
the repository. Five frozen source blobs are checked against the locked
ledger, with Qualtran 0.7.0 and Kickmix 0.1 Simulator presence required. Parent
Q-0208 and Q-0209 remain open until their remaining full-file scope is disposed.

## Batch verification

Concentrated regression passes 10,830 checks with 32 existing warnings and zero
skips. Local integration passes 17 checks. Final focused acceptance passes 204
checks. Strict typing covers six Python files. Black, Ruff, Integration scope,
architecture, IR and public API checks pass. Dependency policy reports 687
existing whole-repository findings and zero touched-path findings after adding
the required NumPy optional-dependency guard. The final focused run includes
explicit primitive-count assertions added after broad-suite collection.

The known default docstring coverage failure is reused after checking all ten
implicated files against base f98f00dc2. The retained receipt names the original
failure and its earlier run summary. Default-suite completion remains open.
Earlier focused failures from empty physical binding and mismatched fresh
classical slots were corrected in the test setup. Both comparison streams now
validate feedback before slot normalization, and the final focused run passes.
