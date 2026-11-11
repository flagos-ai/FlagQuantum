# Shared numerical finite-field metadata reconstruction

Simulation now owns `finite_field_metadata.resolve_finite_field_encoding`, the
single inverse of the existing `FiniteFieldEncoding.register_type` projection.
It supports concrete quantum and classical field declarations, including odd
characteristic fields. The old binary resolver remains the executable boundary:
it calls the unchanged Core verifier first, then delegates reconstruction to
this numerical owner. No executable IR admission is expanded.

The source remains unmodified Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Qualtran retains a numerical dtype
object directly; this inverse is required by FlagQuantum's neutral metadata
representation. The reconstructed object uses the existing original-derived
finite-field codecs, not a new field implementation or vendor object in Core.

## Responsibility and policy

Before optional imports, validate the exact projection layout: quantum/classical
storage widths, concrete integer characteristic/degree/order, positive degree,
order consistency and complete polynomial metadata. Preserve explicit None and
all coefficient-field properties. Reconstruct the same polynomial and require a
complete projection roundtrip, including classical storage identity. Numerical
library validation stays at its existing lazy boundary; no prime testing,
polynomial repair or coercion is introduced.

The register codec adapter now delegates every field descriptor to this shared
numerical resolver. Quantum and classical scalar field values use the same
underlying source class. Non-power-of-two domains retain invalid binary words:
those words raise the original field construction error, not a modulo reduction.

Original tensor boundaries remain explicit. Default field identity output loses
its type through np.asarray and is rejected. Binary input-only effects succeed;
odd-field effects eventually encounter binary words outside the field domain and
fail. Shaped classical input still follows the original zero-quantum-width
reshape policy. These source failures are retained rather than repaired.

The earlier binary-codec test's quantum/classical/odd refusals represented that
adapter's former numerical limitation. Those two domain refusals are replaced
by successful numerical codec tests; malformed projections still reject. The
separate executable binary entry continues to reject both domains, verified for
every new descriptor. This is numerical coverage, not a new physical gate family.

## Acceptance evidence

109 new cases cover eight field declarations, each with quantum and classical
storage: GF2, GF8 default/None/polynomial13, GF3, GF9, GF5 and GF7.

- 16 projection cases compare exact source field class, complete element domains,
  all-pairs multiplication tables and the unchanged executable admission boundary.
- 48 scalar/vector/empty cases enumerate every binary word, comparing field
  values, exact types, typed encodings, stripped-array failures and out-of-domain
  errors with actual source methods.
- 32 dense identity/effect cases compare successful coefficients or exact original
  error and cause, retaining field type loss and odd-field invalid bit patterns.
- 12 malformed projections reject before optional imports. One replacement test
  proves the executable binary entry delegates to this single numerical owner.

The concentrated regression retains six binary Add consumers: their full
matrices, primitive gate order/operands, complete complex append execution,
forward/inverse restoration and clean spectator zero continue to pass. This
batch adds no odd-field or classical physical decomposition. Numerical metadata
reconstruction emits zero primitives and owns no quantum work bits. Physical
binary consumers emit CX1/2/3; the source Clifford1/2/3 model is counted
independently. There are no measurements or measurement branches in those
consumers.

Q-0882/Q-0883 and Q-0895 remain open for source-wide execution, foreign exception
semantics and original tensor limitations. The frozen parent/child inventory does
not grow and G3 remains unstarted.

## Local verification

109 new cases, 3,503 concentrated regressions (207 existing numerical warnings)
and 84 CPU/tensor-network integration checks (two existing warnings) passed.
The metadata owner has 100% statement/branch coverage (35 statements, 12 branches);
the binary entry (9 statements, 2 branches) and shared codec (62 statements,
36 branches) also have 100%. Strict typing of six Python files, Black, Ruff,
Simulation scope, architecture, internal IR and protected API checks passed.
Seven frozen source blobs match the inventory. Three before/source/after domain
comparisons and retained physical consumer evidence are recorded in
`ftqc-reference-env/g2-field-metadata-source.json` and verification logs.

Default smoke/unit result: `7 failed, 22234 passed, 300 skipped, 1037 warnings in 133.59s (0:02:13)`. All observed failure IDs belong to the seven known baseline IDs; 0 baseline IDs were not observed before the max-failure stop. This is not a green full-suite result. Evidence is in `ftqc-reference-env/g2-field-metadata-default-comparison.json`.
