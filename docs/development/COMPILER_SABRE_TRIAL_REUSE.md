# Reuse completed SABRE trials for native-cost evaluation

Production selection now consumes a private stream of completed trials rather
than discarding each alternative after comparing its cost. Its objective,
deterministic tie order, pruning limits, four additional trials and local fast
path are unchanged. The stream permits the opt-in native-CZ experiment to inspect
those same plans without a second search. This is a `single_device_fast_path`
implementation change, not default admission of native-cost routing.

## Implementation and ownership

`topology/sabre.py` owns `_sabre_trial_costs`. It scores each completed plan once,
tightens the existing incumbent bound before continuing, and prepares one
call-local dependency graph for the additional trials. The ordinary selector
retains the first minimum. Already local programs return without constructing
the stream or invoking a caller's cost function. Directed/custom objectives
retain their existing selection path.

`benchmarks/compiler_sabre_trial_reuse.py` is an experimental consumer. It deduplicates
completed plans, keeps alternatives no more expensive in restored SWAPs than
the ordinary incumbent, and calls the existing native evaluation helper. Both
experiments share the same native lowering, Weyl shortlist, actual bounded
synthesis, strict delivered CZ reduction and non-increasing two-body depth guard.
The Weyl estimate cannot override residual or operation-growth rejection.

The stream does not recover routes pruned by the original search. In particular,
an equal-cost route previously found by rerunning with `swap_limit + 1` can be
absent. Therefore the new experiment is not universally equivalent to the old
one. This narrower search must retain explicit quality comparisons. No mutable
state survives a selection call, no executable is reused between compiler calls,
and independent routing/native verification and artifact emission still run.

The numerical and routing mechanisms retain their documented Qiskit/Cirq
attribution in [upstream reuse](COMPILER_UPSTREAM_REUSE.md). This change reuses
FlagQuantum's existing implementations; it adds no external dependency, public
API, default, result field or serialized schema.

## Complete inspected quality screen

The [archive](evidence/sabre-trial-reuse-screen.json) records integrated base
`1dc11c0a5607f32a89032ca779f8c37d1b79e07e`, measured source hashes, the exact
driver and all attempts. SDKBench remains unchanged at
`9aa9260920984e2ca05bcb293abb885fccd9c144`. Three already inspected nonces each
use all widths 6/8/12, five application families and four topologies: 180 task
pairs, with production, previous experiment and reused-trial experiment each
executed, giving 540 accepted workflows and no exceptions or unsupported cases.
The adapter's existing 20,000-operation native growth budget and seed 11 are
unchanged. No fresh or held-out input claim applies.

Both experiments improve eight task pairs and leave 172 unchanged relative to
production, with no CZ/depth regression or tradeoff. They produce identical
operation sequences in all 180 comparisons. The eight pairs include repeated
QFT controls and are not eight independent experiments.

| Nonce index | Task | Production CZ / depth | Both experiments CZ / depth |
| --- | --- | ---: | ---: |
| 0, 1, 2 | 6-wire QFT, grid | 65 / 48 | 63 / 38 |
| 0 | 6-wire repetition, ring | 82 / 57 | 76 / 55 |
| 1 | 8-wire QFT, sparse | 150 / 95 | 138 / 89 |
| 2 | 6-wire QFT, sparse | 99 / 75 | 89 / 63 |
| 2 | 6-wire repetition, sparse | 60 / 57 | 56 / 51 |
| 2 | 12-wire repetition, grid | 64 / 61 | 62 / 59 |

The 12-wire sparse QFT gap remains: nonce 2 retains 328 CZ operations and depth
156. Removing repeated search does not resolve that quality deficit.

The maximum semantic residual is `6.072657351995283e-14`. Width-six executions
check complete operators; widths eight and twelve check three probes only.
Identity endpoint layouts and coupling-edge legality are required throughout.
Operation sequences are deduplicated by exact JSON SHA-256 into bounded payload
files; the index retains every original row and digest. The
[independent replay](evidence/sabre-trial-reuse-replay.json) reconstructs original
report bytes and rechecks all 540 outputs in a fresh process without importing
FlagQuantum, Qiskit or Cirq. This proves the recorded checks, not universal
operator equivalence for larger widths.

## Verification and admission

The [plan comparison](evidence/sabre-trial-reuse-plan-parity.json) loads the exact
integrated selector and compares 80 constructed CX programs across widths
6/8/12/16, line/grid topologies and ten seeds. Every complete `SabrePlan` is
identical; each restored circuit also passes an exact GF(2) full-operator check.
This is regression evidence for the default selector, not a quality improvement.

The existing user workflow remains:

```python
import flagquantum as fq
from flagquantum.compiler import CouplingMap, compile

circuit = fq.Circuit(4)
circuit.h(0)
circuit.cx(0, 3)
compiled = compile(circuit, coupling_map=CouplingMap.line(4))
assert compiled.metadata["routing"]["mapping_restored"] is True
```

Expected: a topology-legal circuit with restored logical output order. Focused
trial, native-budget and physical-evidence tests plus seven compiler integration
files pass 252 tests. Run the focused tests with:

```sh
python -m pytest tests/unit/test_compilation_sabre_trials.py \
  tests/unit/test_compilation_sabre_dependencies.py \
  tests/unit/test_compiler_sabre_native_cost.py \
  tests/unit/test_native_block_optimization.py \
  tests/integration/test_native_block_physical_evidence.py -q
```

The initial new test incorrectly tried to realize a routed CX circuit with a
budget below its native translation cost; both parametrizations failed for that
reason. Positive workflow checks now explicitly provide sufficient budgets.
The existing zero-growth rejection/counterexample tests remain unchanged. The
first parity driver failed on missing postponed annotation evaluation before
executing cases; the corrected driver and complete output are retained.

The production default stays unchanged. Formal resource-qualified SDKBench
remains deferred; no kernel upgrade, reboot or service restart is needed or
authorized. A successful quality screen alone cannot admit the experimental
selector or establish SDK superiority.

The [resource observations](evidence/sabre-trial-reuse-resources.json) retain two
warmups, seven samples in rotating three-variant order, separate traced Python
peaks and planner call counts. The first diagnostic overlapped default-tier
tests; it is preserved but excluded from latency interpretation. The separate
retained run started after that supervisor completed. The host remains shared
and unqualified; these values are development observations, not certified
latency, process RSS, significance or rankings.

| Inspected control | Production median | Previous experiment | Reused trials | Production / reused Python peak bytes |
| --- | ---: | ---: | ---: | ---: |
| 6-wire QFT, grid | 90.041 ms | 195.718 ms | 192.438 ms | 2,199,494 / 2,271,158 |
| 6-wire repetition, ring | 116.079 ms | 205.337 ms | 203.687 ms | 2,567,989 / 2,602,643 |
| 12-wire QFT, grid | 563.180 ms | 580.163 ms | 561.589 ms | 10,119,735 / 10,121,285 |

All three workflows reduce planner calls from 27 to 15, matching production's
15. Nevertheless the two six-wire cases remain substantially slower than
production. Native preview and synthesis remain the principal unresolved work;
removing repeated routing alone does not satisfy resource admission. The
12-wire case has no quality gain and its close timing is not a demonstrated
speedup. Keep the native-cost consumer experimental.

The required full `python tools/ci_tier.py pr-default` attempt reached roughly
41 percent before its explicit 300-second deadline. Its process group was
terminated; it is not a passing full-tier result. Focused checks above are the
completed local evidence. Ruff and mypy pass the changed source modules; full
repository CI is still required before integration.
