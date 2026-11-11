# Binary field codecs in the default classical tensor path

The existing metadata-to-codec composition now handles admitted binary quantum
finite fields through `resolve_binary_field_encoding`. This is ten lines of
composition in Simulation: no field arithmetic, codec, graph representation,
registry or Core admission rule is duplicated. The reference remains unchanged
Qualtran v0.7.0 at `bcb17bb25360ed68ce3bc084645db3e929799389`.

## Ownership and exact behavior

The existing binary-field resolver restores the complete polynomial and its
coefficient-field identity. Scalar input calls `FiniteFieldEncoding.from_bits`;
shaped input calls its original array codec. Both return the source galois class,
not ordinary integers. Output calls `to_bits_array` on the supplied value without
casting it. The existing integer/bit branches never request optional field
resolution. Unsupported odd, classical, symbolic or malformed descriptors retain
the authoritative Core admission refusal before optional reconstruction.

The default dense tensor entry already supplies these callbacks. This closes
its missing binary-field dispatch, including successful input-only effects;
it does not make every field-valued action contractible. Two source boundaries
are explicitly retained:

- Some multidimensional bit arrays fail in the original UInt array decoder or
  subsequent field construction. The same words and shapes are tested against
  the actual source methods, retaining both successes and errors.
- A direct field identity through `bloq_to_dense_via_classical_action` calls
  `np.asarray` on its field output. That strips the field subclass, and the
  original field scalar validator then rejects the integer. The native dense
  entry now reaches the same error and cause, rather than rejecting metadata
  earlier or coercing the value back into a field. Typed field arrays passed
  directly to the codec preserve their original successful encoding behavior.

These behaviors are not repaired in the reference or claimed as successful dense
field identity evaluation. Explicit callbacks remain available for other domains;
Core executable type admission, physical lowering and protected APIs are unchanged.

## Source and physical verification

50 new cases cover six field declarations: GF2, GF4, GF8 with default polynomial,
GF8 with explicit None, and GF8 polynomials 11 and 13.

- 24 scalar/vector/matrix/empty cases enumerate every small binary word, checking
  numerical values, exact field class, typed output bits and source array/type
  failures. Encoding compares both typed arrays and their stripped NumPy forms.
- Six complete dense identity negatives match the original error text and cause.
- Six input-only dense effects succeed and match all source coefficients.
- Six Add consumers build full matrices through the new typed codec path. All
  inputs, full complex amplitudes and the unchanged physical primitive stream
  agree with `GF2Addition`. Ordinary `circuit.append(op)` forward and inverse
  execution restores the complete state and keeps the unused clean wire zero.
- Seven malformed/non-admitted descriptors reject in both directions. One integer
  consumer proves that default integer tables do not request field resolution.

Codec dispatch emits zero primitives and owns zero quantum work. Per Add consumer,
measured CX counts are 1, 2 or 3 according to field width. Independently, the
original QEC model reports the same number of Clifford gates and no measurements.
These consumers have no measurement branches; complex state comparison is exact
up to the stated numerical tolerance and does not normalize away global phase.
The clean spectator is separate from the two logical field registers.

Q-0895 stays open for broader domains and source classical-axis/empty/phase-free
boundaries. Q-0882/Q-0883 source-wide admission and foreign decomposition errors
remain open. No parent/child denominator changes and G3 remains unstarted.

## Local checks

50 new cases, 3,394 concentrated regressions (207 existing numerical warnings)
and 84 CPU/tensor-network integration checks (two existing warnings) passed.
The shared codec owner has 100% statement/branch coverage (62 statements and
36 branches). Strict typing of both changed Python files, Black, Ruff, Simulation
scope, architecture, internal IR and protected API checks passed. Seven frozen
reference Git blobs match the inventory. Three captured before/source/after
cases verify the new field dispatch, retaining field class and full polynomial
metadata. Evidence is retained in
`ftqc-reference-env/g2-classical-field-codec-source.json` and verification logs.

Default smoke/unit result: `6 failed, 22103 passed, 300 skipped, 1037 warnings in 130.20s (0:02:10)`. All observed failure IDs belong to the seven known baseline IDs; 1 baseline IDs were not observed before the max-failure stop. This is not a green full-suite result. The comparison is retained in `ftqc-reference-env/g2-classical-field-codec-default-comparison.json`.

## Numerical domain extension

[Shared field metadata reconstruction](FINITE_FIELD_METADATA.md) now restores
odd-characteristic and classical descriptors for numerical codec use. The earlier
restriction in this batch applied to the former binary-only adapter. Its two
domain refusal tests are replaced by successful codec parity and independent
Core execution refusal tests. Malformed projections still reject. All recorded
source dense type-loss and array failures remain unchanged.
