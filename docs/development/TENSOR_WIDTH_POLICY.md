# Source-default width planning and explicit ADCR selection

```bash
python examples/algorithms/tensor_width_policy.py
# Width 12: ADCR not selected; 4096 complex outputs
# Width 13: ADCR selected; 8192 complex outputs
# Optional Cotengra plans shapes; native PyTorch executes. Added physical gates: 0.
```

The existing internal pure and doubled register consumers now accept
`simplification_atol`. `None` preserves the unsimplified execution path. A finite
nonnegative value explicitly enables source-default width planning and ADCR
when its reported width is strictly greater than 12. `0` requests exact support
detection; `1e-12` reproduces the locked upstream default tolerance. Zero tolerance
still does not retain ambient derivatives of discarded zero entries. Existing
component dense entries forward this keyword to the same numerical consumers.
No stable public export, physical append API or default training path changes.

## Native ownership and existing dependency reuse

`contraction_width.py` owns a small shape-only boundary to existing optional
Cotengra. It passes ordered input labels, shapes, inferred outputs and explicit
`optimize="greedy"` to `array_contract_tree`, returning only its reported numeric
width. It never constructs a Quimb graph, copies amplitudes, imports Qualtran or
returns an external tree. Scalar, repeated-index and hyperedge path handling are
performed by the same existing planner. Meta tensors work because only shapes
are read. Empty networks, zero extents, inconsistent dimensions and invalid
planner values fail explicitly. Missing Cotengra raises an actionable ImportError;
there is no native-greedy fallback hidden behind a source-equivalence claim.

`full_simplification.simplify_wide_tensor_network` owns the strict threshold and
reuses the already accepted ADCR fixed-point owner. Register grouping is captured
before planning/preprocessing; protected outputs can become hyperedges after D
and are passed directly to the same native contraction consumer. The optional
keyword is also forwarded by both component dense entries. Reuse of Cotengra is
already admitted for optional contraction/slicing; no dependency is added.

No new planner registry, graph representation, executor or backend fallback is
introduced. Default `None` consumers do not import or invoke Cotengra. The width
is a planner result, not a measured peak process RAM or capacity guarantee.

## Locked call chain and singleton source defect

The comparison executes unchanged Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`, `_dense.py` blob
`13aa81f50654ccaaa4293e8a63991686723022f5`. Qualtran does not pin its planning
dependencies. This evidence explicitly locks restored **Quimb 1.15.0** and
**Cotengra 0.8.3**:

`quimb_to_dense` → `TensorNetwork.contraction_width` → `contraction_tree` →
Quimb's `array_contract_tree` wrapper → Cotengra `array_contract_tree` →
`ContractionTree.contraction_width` → `max_size(log=2)`.

Quimb's restored process default is **greedy**, not auto/auto-hq. The native
boundary explicitly implements this locked default. Quimb's mutable process or
thread-local strategy overrides and custom optimizer objects are not mirrored.
Planner behavior on a different optional dependency version has not been certified.

For multiple leaves, width is log2 of the largest **produced** tensor size. It
is not the raw largest input rank or native total-memory proxy: two rank-13
leaves that contract directly to a scalar report width zero. However,
Cotengra0.8.3 `max_size` returns `get_size(root)` immediately for a single leaf,
before applying its log argument. Thus a single rank-12 tensor reports **4096**,
not 12, and triggers simplification. Native code preserves that original result
and selection, with explicit tests and this defect disclosure. It does not
silently repair the source and then claim identical triggers.

| Original method | SHA256 |
| --- | --- |
| Quimb width | `2c8252fa297cae3644361df74a6097c31549a7eca29d9d49f57f01f76dccc0bb` |
| Quimb tree | `0fc56992da6eaa029b2630d4f85715d6226267467462bbabe9b11610c846acca` |
| Quimb actual delegate body | `32c5ff93ec3cfa2b14d1757b1ce46c9999fff6e8555cd94395f50dd32647cb6e` |
| Cotengra array tree | `a55d170a34cbd751506cead0964588e6100f47b3949fd7a346bcca1e33748aeb` |
| Cotengra max_size | `fb2cfd0b6ecc8ca083196ac64eb1f478d0c3a84832924fa9d1e31fd580dccd0a` |
| Cotengra width | `530b975ce4e9a23f4d2ca0bdd451764100eb054bf381e3636026d8746e309b1e` |

The delegate is decorated with functools.wraps, so tests hash its actual code
body rather than accidentally hashing the wrapped Cotengra function twice.

## Evidence and remaining boundaries

Tests compare original/native exact planner inputs, arguments, dynamic paths
and reported widths across CPU/meta, scalar, single-leaf, repeated indices,
nonbinary dimensions, disconnected graphs and hyperedges. Actual 12/13-output
networks compare source `quimb_to_dense` trigger count, full complex outputs and
fixed-support phase gradients; the source default call is exactly
`full_simplify(inplace=True)`. Doubled-state and unitary singleton cases retain
the source singleton defect and compare correctly ordered dense axes. A real
width-13 Bell graph demonstrates D-created protected output hyperedges arriving
at the unchanged native numerical consumer. Component pure/mixed Bell entries
prove keyword forwarding through existing wiring and supplier boundaries.

All [ADCR source differences](TENSOR_FIXED_POINT.md), repeated-index A correction,
scalar distribution/NaN differences, positive-tolerance error amplification and
ambient-gradient limitations remain explicit. Planning does not justify enabling
support-based transformations for arbitrary trainable entries. Default consumers
remain unsimplified; changing public/default training policy requires its own
acceptance. Consequently Q-0887 is still open for complete default-entry policy
and all remaining parent-file dispositions; this batch extends Q-0887.03 without
a duplicate capability ID or parent closure.

Measured added physical gates, measurements, work bits and feedback slots are
all zero. Quantum resource modeling is not applicable to shape planning. The
example returns 4096/8192 complex128 entries (65536/131072 output bytes), not a
measured process memory peak. No GPU or distributed-performance claim is made.
Existing physical all-measurement-branch/workspace/resource tests remain in
concentrated regression. G3 and integration PR merge remain outside this batch.

Local acceptance: **43 new scenarios**, **2,395 related tests in 89.91 s**
and **84 CPU/TN tests in 5.44 s** pass. Strict mypy covers all seven changed
Python files; Black/Ruff/scope/architecture/IR/protected API and the executable
example pass. Width adapter plus full loop/policy and example cover **86
statements and 24 branches at 100%**, not all existing consumers. Implementation
`73816a0d`. Existing source warnings and default-suite timeout remain open; this
is not whole-repository certification.
