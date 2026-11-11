# Persistent measurement IDs in recursive evaluation

`simulate_phased_component_calls` now accepts an optional per-function
`instance_ids` mapping. Compiler already returns persistent occurrence labels
from `expand_component_calls`; the existing state already accepts those labels
for individual functions. This change connects that existing boundary through
recursive evaluation without introducing another identity representation.

Call-order tuples still contain stored operation positions. Instance-ID tuples
align with stored positions, independently of visitation. For example, two
remaining calls at positions 0 and 1 may retain source IDs 8 and 9 after several
flattening passes. Fixed outcomes keyed by 8 and 9 must not be looked up as
0 and 1. Omitted functions retain their previous position-based default.

The evaluator snapshots the outer mapping, then passes each reached function's
tuple to the existing state validation and random-value binder. IDs remain
local to each function: two invocations of the same stored body reuse its
source instance IDs. RNG selection still takes precedence over fixed values;
visitation order determines draw order, not numeric ID order. No global call
path or instance registry is created.

## Source comparison and explicit boundary

The reference is unmodified Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`: original composite flattening,
`ClassicalSimState` output assignment and the fixed/random handlers in
`simulation/classical_sim.py`. Compiler-selected labels are compared directly
with actual source BloqInstance IDs after flattening; native schedules are
supplied separately. Reversing the native storage order and its aligned labels
tests this distinction without rewriting the source graph.

The source `CompositeBloq.on_classical_vals` calls the pure classical helper.
Putting a raw CompositeBloq containing MeasureX directly in a phased graph
therefore still raises a pure-phase rejection. A dedicated negative test keeps
that source failure. Nested source comparisons use a test-local declared Bloq
whose decomposition is the unchanged stored source body, so the real source
phased recursion protocol is exercised. This fixture is not presented as a fix
to raw CompositeBloq dispatch or as whole-file equivalence.

## Verification

- 24 root-expansion cases cover depths 0/1/3, all four pairs of fixed outcomes
  and both storage directions. Omitting IDs reproduces the missing-key failure;
  supplying Compiler labels matches source values and phases.
- Every root-expansion case reconstructs the complete unnormalized branch
  amplitude from all four complex input coefficients. Both measured physical
  outcomes match that amplitude; the unused clean spectator remains zero.
- Six nested-body cases verify that the mapping reaches recursive states and
  preserves source local-ID reuse across two calls to the same body.
- Six seed/storage cases compare 16 pairs of RNG samples, complete generator
  state and precedence. An explicit reversed visitation restores source draw
  order independently of reversed storage.
- Six invalid-ID cases reuse the existing state's rejection for wrong arity,
  duplicate, negative and boolean IDs. One mutation case confirms the mapping
  is snapshotted before numerical actions. One source-only negative retains the
  raw CompositeBloq pure-dispatch limitation described above.

## Separate resource accounts

The mapping and numerical evaluation emit zero primitives and own zero quantum
work bits. The two-MeasureX physical consumer emits **H2, measurement2**. The
original source QEC model reports **measurement2**; it does not include the
physical Hadamards. Two input wires are measured and an unused clean spectator
stays zero. Each of the four branches retains its 1/2 projection factor; no
normalization hides phase or amplitude differences.

This is a Simulation `single_device_fast_path` numerical-boundary extension.
Compiler label creation, Core records and physical gate generation are unchanged.
Users continue `circuit.append(op)`. Q-0882/Q-0883, pure recursive policy and
foreign decomposition boundaries remain open; the fixed completion denominator
and checked-child counts do not increase.

## Local verification results

- 44 new tests, 3,150 concentrated regressions and 84 CPU/tensor-network checks passed. Existing warning counts remain 207 concentrated and two integration.
- Strict typing of both changed Python files, Black, Ruff, Simulation scope, architecture, internal multi-level IR and protected API checks passed. State coverage is 97% (142 statements, 54 branches), with three existing refusal sites uncovered.
- Seven frozen reference blobs match the inventory. Two captured root/nested before/source/after cases show old KeyError(0) versus matching native/source outputs with persistent IDs.
- Default smoke/unit: `5 failed, 21904 passed, 300 skipped, 1037 warnings in 149.21s (0:02:29)`. All observed failure IDs belong to the seven known baseline IDs; 2 baseline IDs were not observed before the max-failure stop. This is not a green full-suite result.
- Workspace evidence: `ftqc-reference-env/g2-component-phased-instance-ids-{source,default-comparison}.json` and corresponding verification logs.
