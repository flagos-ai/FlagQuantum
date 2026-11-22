# Prospective native-CZ routing quality results

The frozen experimental selector passes this study's quality rejection rule:
60 complete target-artifact workflow pairs give 120 independent acceptances,
zero exceptions, zero unsupported attempts and zero CZ/depth regressions. Two
pairs improve and 58 are unchanged. Keep the production selector unchanged:
the earlier measured compilation-time overhead remains unresolved, and formal
resource-qualified SDKBench is still user-deferred.

## Frozen source and actual new inputs

The [protocol](COMPILER_SABRE_NATIVE_COST_PROSPECTIVE.md) was committed at
`fa2dfce21e2c8c0a055a90bc5f005de6d6b2a2bd` before drawing one fresh nonce.
Compiler and benchmark bytes match integrated main
`11745ab11070765974b90f5a8f512bf4c30576b1`; only the protocol document differs.
The unchanged SDKBench revision is
`9aa9260920984e2ca05bcb293abb885fccd9c144`. The manifest records protocol,
runner and selector hashes, the nonce and the existing adapter's 20,000-operation
native growth budget. No nonce redraw, candidate change, evaluator change or
post-draw tuning occurred. The runner completed once with exit status zero under
a separate 600-second process-group supervisor and one numerical thread.

The [complete archive](evidence/sabre-native-cost-prospective.json) retains the
pre-execution manifest, every pair and evaluator metric, adapter audit counters,
all source/target and probe fingerprints, invocation, log and decision. It also
records the exact original report hash and formatting needed to reconstruct its
bytes from the nested report. The initial manifest preparation failed on an
incorrect historical field name before drawing a nonce. A subsequent result
audit initially expected a different probe label. Those setup/audit errors were
corrected without rerunning or changing any workload result.

Fingerprint comparison against both the original inspected nonce and the prior
depth-selector prospective nonce finds 51 new source-operation/target-edge
combinations and nine repeated QFT controls. New means a new source/target pair,
not necessarily a new source program: sparse-target edges can change while QFT
operations remain identical. Probe-only differences do not count as new inputs.
After this study all these inputs are inspected development cases.

| Input group | Improved CZ/depth pairs | Unchanged | Regressed |
| --- | ---: | ---: | ---: |
| New source/target pairs | 1 | 50 | 0 |
| Repeated controls | 1 | 8 | 0 |
| Complete study | 2 | 58 | 0 |

## Delivered improvements and semantic scope

| Input | Source/target group | Production CZ/depth | Experiment CZ/depth |
| --- | --- | ---: | ---: |
| 6-wire QFT, grid | Repeated control | 65 / 48 | 63 / 38 |
| 8-wire QFT, sparse | New topology combination | 150 / 95 | 138 / 89 |

The second result removes 12 delivered CZ operations and six two-body dependency
levels on a new source/target combination. The first reproduces a previously
inspected improvement. These observations do not establish optimality, general
superiority, or an SDK ranking. The candidate selects a bounded shortlist using
the existing Qiskit-derived Weyl/KAK machinery and actual native synthesis; the
full compiler still independently verifies routing, rewrites and serialization.

All 40 width-six executions pass complete-operator acceptance up to global phase.
The remaining 80 executions pass three independent complete-output probes only;
they do not prove full-operator equivalence at widths eight and twelve. The
maximum reported semantic residual is `5.251875114378881e-14`, below the unchanged
`1e-8` evaluator threshold. Both variants must restore identity endpoint layouts;
the existing runner refuses otherwise. The full frozen case set is retained.

## Remaining admission requirements

Passing this prospective quality check does not satisfy the separate resource
requirement. The [contemporaneous production control](COMPILER_SABRE_NATIVE_COST_EVALUATION.md)
still records approximately 115% and 80% more compilation time on two earlier
six-wire controls and higher traced Python peaks. This study's baseline-first
elapsed fields are neither repeated alternating latency trials nor process-RSS
measurements and must not be used to erase that evidence. More affordable
selection, supported-input/failure boundaries and qualified resource evidence
remain necessary before any production admission decision. No public API,
default, numerical tolerance or production capability claim changes here.

The protocol includes the existing user workflow and exact runner command.
Before the draw, 176 focused routing-preview, native-block and physical-evidence
tests passed. The prospective execution supplies the distinct quality evidence
reported above; passing those tests alone would not have established it.

After recording the results, the sparse checkout's complete seeded selections
pass 134 default-tier unit tests and 42 runtime-tier integration tests. Language,
local-link, evidence-origin, revision-vocabulary and diff checks pass for these
added files. These local selections are not whole-repository certification;
full-source PR CI remains required before integration.
