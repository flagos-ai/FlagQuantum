# Reusing dependency lookahead within a routing frontier

Compiler owns this `single_device_fast_path` optimization. It reduces repeated
dependency traversal while preserving the existing routing candidate choices,
scoring arithmetic, trial count, restoration and public contracts.

## Evidence and upstream review

The completed [routing-policy control study](https://github.com/sdkbench-org/sdkbench/blob/9c888cad90211c9a6fb7a17e4d54729e362320db/docs/analysis/COMPILER_ROUTING_POLICY_RESULTS_20261120.md)
records default-policy latency costs even where delivered gate counts and depth
match an explicit policy. That study does not identify a particular internal
phase as the cause, and its host is not publication-qualified.

A separate local development profile of deterministic CX circuits at widths
8, 16 and 32 found repeated extended-frontier traversal among the planning
costs. Reusing static trial preparation was considered first; the profile
instead motivated removing repeated breadth-first traversal within a frontier.
This is a source-level hypothesis checked by the experiment below, not a causal
explanation of all SDKBench latency differences.

Qiskit 2.6.2's pinned
[`State::begin`, `apply_swap` and `update_route`](https://github.com/Qiskit/qiskit/blob/0c26a5a158c4d62b2b303a8c2f3c6eb2b14653b4/crates/transpiler/src/passes/sabre/route.rs)
separate topological frontier updates from physical layout updates. A SWAP
updates placement; executing a source operation advances topological layers.
This change applies that lifecycle distinction to FlagQuantum's existing BFS
lookahead. It does not port Qiskit's layer container, decay heuristic or Rust
implementation. Existing IBM/Cirq notices remain with the native planner.

## Invariant and lifetime

`extended_layer` uses the current front, its blocked subset, successor edges and
the instruction blocking classification. None changes during speculative SWAPs
or their rollback. Physical placement changes, so each candidate still recomputes
distance scores. The BFS result can therefore be retained until any source
instruction executes. Execution invalidates it before changing the frontier.

The retained list belongs to one `_plan_sabre_trial` invocation. Every trial,
program, layout and compilation starts without a retained result. Empty
lookahead is retained too. No global cache, new type, dependency, configuration,
public API, serialization field or cross-domain state is introduced. The list
has the same existing `LOOKAHEAD_EXTENDED_SIZE` bound and consumers only read it.

Tests exercise repeated SWAP scoring, changing frontiers, empty lookahead,
separate trials and immutable retained contents. Existing complete-operator,
symbolic-gradient, classical-dependency, progress-recovery, directed-device,
restoration and deployment tests remain applicable.

## Reproducible development probe

Reuse the existing repository probe against integrated main
`ea1118f56af0b8e9c26d10a24b0fe72efb30f850`, with this change as the working tree:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python - <<'PY'
from benchmarks import compiler_sabre_scoring as probe
probe.BASELINE = "ea1118f56af0b8e9c26d10a24b0fe72efb30f850"
probe.main()
PY
```

The existing output label `relative` denotes the working-tree implementation;
the baseline already includes relative scoring. Neither side invokes another
SDK. Both use the same fixed CX circuits, line topology, native planner and
restored-layout contract. One warmup precedes five alternating timed samples.
The [raw result](evidence/compiler-frontier-lookahead-development-20261120.json)
retains every sample, source hash and host description. All 18 complete plans
match, including operation sequence, placements, SWAPs, anchors and layouts.

Seventeen per-case median paired time ratios are below one and one is above
one. Ratios range from 0.92575 to 1.01060; the latter regression is retained.
These short shared-host measurements have no confidence or publication claim.
They measure planning only, not compilation, runtime, hardware or memory.
This does not change the earlier SDKBench evidence or establish cross-SDK
superiority. A future qualified study must use the unchanged neutral protocol.

## Existing user workflow and checks

```python
import flagquantum as fq
from flagquantum.compiler import CouplingMap, compile

source = fq.Circuit(8).h(0).cx(0, 7).ry(3, theta=0.2).cx(3, 6)
compiled = compile(source, coupling_map=CouplingMap.grid(2, 4))
assert compiled.metadata["routing"]["mapping_restored"] is True
```

The result preserves the logical state and delivers restored output placement.
Run the focused lifetime, scoring and recovery checks with:

```bash
python -m pytest tests/unit/test_compilation_sabre_scoring.py \
  tests/unit/test_compilation_sabre_progress.py \
  tests/unit/test_compilation_sabre_trials.py \
  tests/unit/test_compilation_layer_lookahead.py -q
```

Local validation passed 1,498 tests across 47 compiler and artifact files, plus
strict typing of all 1,142 package source files. Architecture boundaries,
formatting and lint checks are recorded separately. Full default/runtime tiers
and exact-head CI remain required before integration; focused results are not
a substitute for those gates.

## Evidence field normalization

The first CI run rejected the undeclared `baseline_revision` field in the
new development record. It was renamed to the existing `base_commit` field;
the commit value, source hashes, host description and every measurement remain
byte-for-byte unchanged. The original artifact SHA-256 was
`90139597e4fa011a6eb6a06685325a5b98b70c36e7f976642e51003ce0f7cf64`; the normalized
artifact SHA-256 is
`fadaaa10a8b303dfc06888f4b5e583c1aeaf4bb078cf3ac952d094756729d5b0`.
No evidence vocabulary or acceptance check was expanded.
