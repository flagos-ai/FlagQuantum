# Private direct-call retirement

Parent: CI-qualified PR #1340, `4c77fa9b1313ea53ea5a7c522ae74ea6d9741f17`.
Status: complete source and installed-artifact acceptance passed.
The published child must pass its own exact-head CI.

## Ownership and complete consumer closure

Eight private functions only forwarded their arguments to existing authoritative
functions. Delete the wrappers and bind callers directly to those same functions.
This removes one call frame, without changing input normalization, numerical
arithmetic, transform ordering, refusals, RNG consumption or public signatures.
Sixteen product files include fourteen same-module bindings and eight cross-module
import bindings. Circuit still normalizes observable qubits before calling the
unchanged statevector measurement implementation. CDR imports noise lowering from
its defining Compiler module rather than its historical public facade.

| Former defining module | Retired private function | Existing authority |
| --- | --- | --- |
| `flagquantum/algorithms/error_mitigation/cdr.py` | `_lowered` | `flagquantum/compiler/lowering/noise.py` |
| `flagquantum/compiler/optimization/pass_manager.py` | `_validated_program` | `flagquantum/core/ir/__init__.py` |
| `flagquantum/compiler/optimization/pipeline.py` | `_as_ir` | `flagquantum/core/ir/__init__.py` |
| `flagquantum/runtime/executors/jax/planning_core.py` | `_as_ir` | `flagquantum/core/ir/__init__.py` |
| `flagquantum/runtime/executors/mps/execution.py` | `_as_ir` | `flagquantum/core/ir/__init__.py` |
| `flagquantum/simulation/statevector/local.py` | `_expectation_z` | `flagquantum/simulation/statevector/measurement_ops.py` |
| `flagquantum/simulation/tensor_network/entrypoints.py` | `_ensure_tensor_network_plan` | `flagquantum/simulation/tensor_network/local.py` |
| `flagquantum/simulation/tensor_network/path_search.py` | `_as_ir` | `flagquantum/core/ir/__init__.py` |

The cumulative ledger has 580 rows; retirement-only rows claim no moved
implementation definitions. The former CDR relocation row no longer claims that
its deleted wrapper is an implementation definition. The existing 102-entry
public facade inventory is unchanged. No public compatibility is added.

## Deliberate retained adapters

An exact-argument census found eleven candidates. The two capability-blocker
constructors remain typed domain record constructors. The existing
`compiler/optimization/pass_registry.py::_commutation_blocks` callback remains
because the public `AnalysisSpec` returned by the public default registry
serializes that exact qualified function global. A real parent-wheel pickle
(SHA-256 `2d153a1904a4b691fde710165bc292cb04002d960381ca2f107f401118549928`)
restores in both installations, re-pickles identically and executes three
programs with identical complete fields and mapping order. The restoration
snapshot SHA-256 is
`d0fb789c8c55527b1cb0817ace7b77c505971098410f0be73bd41eeb87d66264`.

Compiler and Integration own that existing serialized callback binding. Its
scope is the pre-existing public AnalysisSpec payload, with no new root export
or general private-helper fallback. Removing it requires a versioned public
payload migration and an old-payload reader. The full registry itself refuses
pickle; that existing refusal is not confused with the standalone public spec.
This one Compiler payload binding is separately itemized; it does not add a
historical facade file or change the existing 102-entry facade inventory.
The registry conformance test also retains its callable-identity check.

Other one-return functions adapt parameters, choose policy/defaults, construct
domain values or define independent references. Their shape alone does not prove
compatibility debt. This bounded top-level scan is not an exhaustive review of
every method, nested definition or module.

## Private measurement bookkeeping

The vocabulary contract distinguishes private implementation measurements from
public migration ledgers and zero-mismatch invariants. Removing the actual
`local.py::_expectation_z(wires)` wrapper removes precisely one private parameter
site (251 to 250) and two tokens in that removed private docstring (42 to 40).
Only those measurements, their explanation and the two now-stale exemptions are
reconciled. Both source scans agree on the exact removed identifiers; no new
private site appears. Every public parameter, alias, attribute, definition,
serialized exclusion, schema, rule and zero-call-mismatch invariant stays equal.
This does not rebaseline public API or relax a scientific threshold. Other frozen
contracts, fixtures and original scientific evidence must remain byte-identical.

## Source evidence and incoming boundary

The sixteen complete module ASTs preserve all 266 remaining definitions after
exact name-binding and import changes are normalized. The only existing test change is the
private census equality constant, 251 to 250. All other test AST and every
public/scientific assertion stay unchanged. The eight numerical/coercion authorities remain real
implementations; no replacement wrapper or duplicate implementation is introduced.
The initial incomplete consumer run, import-order failure and first complete
source run with a stale private-census test count are retained. The latter had
19428 passed, 958 skipped and one exact-counter failure. The corrected one-site
measurement equality passes its focused test and the complete source rerun.

Focused original tests: 692 passed, 107 existing optional/hardware skips. Eleven
quality gates pass after consumer closure and import ordering, including strict
mypy over 1262 product sources, public API, vocabulary and architecture checks.
The complete source rerun passes all three phases:

| Phase | Passed | Conditional skips | Pytest seconds |
| --- | ---: | ---: | ---: |
| unit-api-team-final | 19429 | 958 | 455.35 |
| qec-full | 2801 | 329 | 73.76 |
| root-local-integration | 1592 | 223 | 83.63 |

The separate actual-checkout optional-JAX phase passes 120 with six existing
hardware skips. The unchanged base environment retains its optional skips;
those skips are not counted as numerical execution. Four supplemental gates and
both supported Black bounds pass. Pinned hooks, final document checks and
exact-commit artifact binding are required before publication.

The native wheel SHA-256 is
`9dc6e4f9ce24c479d61b94d42d8b3569e2466d51481ce623337e877d716fcda6`.
All 1262 product Python files match source, wheel and clean installation byte for
byte. Parent and child each pass the same 673 original installed tests with
seven existing skips using actual task-local JAX. All 488 loaded product modules
belong to their own installation. One repository-source AST test remains in the
full source suite; no fake product source is added to the installed test root.
Two source-tool test files also remain source-only. The initial isolated-harness
missing-path failure is retained, and the required approval document is copied
byte for byte. Third-party dependencies are borrowed from the same parent path;
this is not an independent dependency-resolution certificate. Cold catalog,
both native extensions, benchmark CLI and eight retired-attribute probes pass.

All 11908 other parent tracked blobs, 3046 other test blobs and 1173 of 1174
contract/fixture blobs remain byte-identical. The sole contract-file difference
is the private measurement bookkeeping described above; every public value and
invariant stays equal. No native ABI, production dependency, packaging policy,
scientific fixture, oracle, workload, sample or numerical threshold changes.

At audited draft/conflicting #1084 head
`ad44551e4f0aa3c42eb1fc2274f04968d1a6e52d`, all eight actual donor definitions
match the qualified parent's wrapper ASTs. The full product-tree blobs are
SHA-verified; the latest two product changes affect legacy component resolution
and the Cirq register graph, with no donor/current-consumer intersection. The changed
incoming source set has 3278 Python files; the 580-row ledger still reports the
same seventeen unfiltered old-import diagnostics, with no new or removed
reports. That changed-source scan is not a full-branch consumer certification.
G0 integration must carry the current direct-caller closure rather than restore
these wrappers. Construction/IR, related integrated families, native/development
distribution and Rust retain their existing closeout gates. This batch remains
unmerged and does not certify whole-architecture completion or scalability.


## One-GPU CI device-admission followup

The complete source counts above belong to migration commit `f05f9bafe`.
Its one-GPU job passed all 21 correctness tests but failed the unchanged
performance-variance gate: CV 0.5677354127477401 and robust CV
0.1123136768104932, against 0.10. The original artifact records
`correctness_passed=true` and a contended measurement environment. Physical GPU0
held 33143 MiB before the measurement while six other devices were idle. All
30 original samples and ten warmups remain retained; artifact ZIP SHA-256 is
`6e3c46297a98c78c562bce267ddca0f6005d8d990dabe3b4ed4946235d8d6efd`.
Its tested merge tree equals the exact published f05 source tree. Occupancy
establishes contention, not the exclusive cause of the measured variance.

The normal one-GPU workflow hardcoded physical GPU0. Reuse the existing two-GPU
admission rule for one device: memory below 1024 MiB and zero measured GPU
utilization. Refuse before measurement if none is idle. Correctness, benchmark
and manifest steps inherit the selected device; the manifest records its actual
physical ID. The two frozen manual campaigns retain their original device
configuration, and all other jobs are unchanged.

The benchmark, ten warmups, thirty samples, twenty inner loops, workload,
performance evaluator, thresholds and watchdog limits remain unchanged. This
admission check is not an exclusive hardware lease; measurement validity and
variance checks still evaluate the actual run. The old failed job is not rerun.
The followup requires a new exact-head CI, including both GPU jobs.

Six regression cases cover the original busy-device inventory, utilization,
the memory boundary, idle GPU0, no-idle refusal, and selected-device propagation
without changing frozen manual campaigns. All 48 related hardware/workflow
checks pass. Original tests are unchanged and the new cases are appended.
All 1566 product blobs remain identical to the fully regressed migration
commit; all 1262 product Python blobs still match the same qualified native
wheel and installation. No new wheel or repeated numerical benchmark baseline
is substituted for the retained original evidence.
