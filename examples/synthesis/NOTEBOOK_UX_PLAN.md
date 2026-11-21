# Notebook user-experience delivery plan

## Outcome and independent measures

A reader should express the quantum problem in FlagQuantum, understand what the
compiler supplies, execute the program and interpret its result. Migration
acceptance alone does not establish that teaching outcome.

Retain the 212 accepted upstream workflows, 216 original-source replay rows and
six approved alternative paths with their existing semantics and evidence.
Do not relabel alternative paths as original passes. The deferred original
calibration workflow remains outside the 212 approved denominator.

Track three independent measures:

- Migration: 212/212 approved workflows accepted in the historical snapshot;
  current revisions require their applicable regression tests.
- Teaching: six new learning-path Notebooks passed their initial numerical checks;
  212/212 migration Notebooks have been reviewed and replayed. See the
  [per-case editorial ledger](NOTEBOOK_UX_REVIEW.md). Compiler capability gaps
  remain separate from this completed editorial denominator.
- Capability gaps: inspect actual code and missing public behavior. A hidden
  helper alone is a presentation issue, not proof that synth lacks a feature.

## Delivery sequence

| Batch | Deliverable | Acceptance | Status |
| --- | --- | --- | --- |
| UX-01 | Static screen of all 212 accepted workflows; separate navigation | Reproducible report tied to existing inventory; no evidence rewrite | Complete; all 212 cases reviewed |
| UX-02 | Six learning-path Notebooks | All cells execute; independent amplitudes, cleanup, topology and optimizer checks | Initial independent checks passed |
| UX-03 | Review delegated construction by shared helper family | Expose useful public program at the point of use; preserve upstream workflow and numerical scope | Complete; useful central programs exposed and helper responsibilities explained |
| UX-04 | Review direct Circuit uses and capability gaps | Identify reference versus application code; reuse existing high-level APIs first | Complete editorial inspection; Circuit/hybrid routes and remaining capabilities explicitly separated |
| UX-05 | Review remaining visible-qfunc workflows and five-category navigation | Explain problem, quantum semantics, compilation, result and limits; changed workflows pass tests | Functions 31/31, Algorithms 45/45, Tutorials 49/49, Applications 69/69 and Community 18/18 replayed |
| UX-06 | Final evidence and handoff | All 212 editorial reviews resolved; applicable corpus regression and default checks recorded | 212/212 complete; 216 original paths plus six alternatives freshly replayed; raw default failures and environment rechecks recorded |

Work proceeds in reviewable local batches. Remote CI is reserved for integration;
there is no need for one remote run per Notebook. Normal repository API-change
rules still apply to any discovered capability extension.

## Editorial standard

Each user-facing Notebook must explain the problem and why a quantum program is
being built, identify classical versus quantum values, and expose the central
program or an appropriate documented public algorithm call. A private workflow
wrapper with no explanation of its quantum construction is insufficient.

Show the route from program to synthesis constraints, execution, named results
and interpretation. Explain the specific work performed by FlagQuantum (for
example arithmetic construction, inverse cleanup or connectivity routing),
without claiming that simple gate syntax itself is automatic optimization.

Keep loading, plotting and reusable numerical algorithms in helpers when that
improves readability. References, exhaustive assertions and evidence internals
belong in tests or a clearly separated appendix. Do not inline whole algorithm
implementations just to increase the visible-qfunc count.

Declare simulation method, approximation, original-scale substitutions and
unsupported behavior. An exact probability is not a finite-shot measurement;
an optimization objective is not proof of a global optimum. Each lesson ends
with a small modification and a meaningful expected effect.

## Initial evidence-backed findings

The initial static screen identified 129 workflows without literal
qfunc definitions, 76 with visible qfunc definitions and seven with direct
Circuit constructors. These mutually exclusive buckets are review signals,
not quality scores or capability counts. The [current generated screen](NOTEBOOK_UX_AUDIT.md)
reflects subsequent source edits.

- Image-edge detection hides its central quantum composition behind
  `workflow.run()`. Its `quantum.py` also explicitly implements amplitude
  preparation with Circuit gates. Investigate current public preparation
  support and original-size budgets before calling this a missing capability.
- The small VQE workflow already uses actual synth executions and the local
  optimizer. It needs explanations of parameter binding and expectation
  evaluation more than another algorithm implementation.
- Hardware-aware MCX already exposes constraints, topology and reports. Retain
  this useful content while separating teaching from exhaustive evidence.

## Product follow-up after the editorial pass

Editorial completion does not close compiler capability gaps. Prioritize the
following vertical slices using the original Notebook requirements and separate
API admission when the protected surface would change.

| Priority | Native user outcome | Existing code to reuse | Evidence required before calling it complete |
| --- | --- | --- | --- |
| 1 | Full-image real amplitude preparation without manual gates | Existing state-preparation lowering and its rotation ladders | Measure construction time and peak memory; replace quadratic Walsh work with a checked bounded implementation; retain the 16384-amplitude source image, full-state error, phase convention and budget refusal tests. Raising the protected 256-entry profile requires its own approved contract. |
| 2 | Reuse one declared variational program across angle evaluations | Circuit parameters and runtime binding, existing qfunc specialization | Compare all nonzero bindings, amplitudes, gradients where supported and resources; measure capture, lowering, binding, execution and readout separately. Do not call fresh synthesis runtime binding. |
| 3 | Compose block encoding and QSVT without copying physical wire arithmetic | Existing algorithms/primitives and verified native control/inverse machinery | One original CFD or plasma vertical path with explicit callback borrowing, aliases, inverse, selection, normalization, success amplitude, phase and budgets; compare a full independent state. Packed matrices and compile-only large programs keep separate resource claims. |
| 4 | Obtain finite-shot named results through the existing execution boundary | Current result and sampling adapters | Retain named logical bit order, seed, shots, marginalization and conditional-denominator semantics; inspect current runtime before introducing another sampling API. |
| 5 | Keep domain examples readable without hiding numerical substitutions | Existing fermion, state-family and signed arithmetic implementations | Preserve source-size chemistry/physics models; distinguish classical model generation, explicit Circuit kernels, native qfunc kernels, local references and hardware claims. |

The two coloring workflows also expose an expensive repeated-specialization
profile. Their original full searches remain acceptance work; timing limits are
not resolved by reducing iterations, changing the objective or weakening checks.
A local stage profile records actual synthesis, execution and named readout
costs before any performance change is proposed.

The diagnostic used four Torch threads while other acceptance jobs were active,
so its wall times are not an isolated performance benchmark. The two declared
programs contain 19975 and 34185 instructions. A follow-up read-only profile
measured `Program._validate()` at about 1.7 and 2.7 seconds respectively, within
public named-readout totals of about 4.0 and 6.3 seconds. Investigate conversion,
identity checking and probability materialization separately in the existing
Services/Runtime boundary before designing another API. Any reuse must retain
circuit, layout, constraint and result-identity tamper refusal; bypassing those
checks is not a performance solution. Raw profiles are retained with the local
acceptance evidence.

## Capability follow-up checkpoint

The editorial denominator and these implementation steps are independent.

| Work | Delivered locally | Remaining acceptance |
| --- | --- | --- |
| Amplitude construction | `b7d7d4383`: replace the Compiler's quadratic Walsh loop with an O(N log N), O(N)-storage butterfly; 69 preparation/conformance tests passed | The [bounded capacity proposal](../../docs/api-changes/FQ-SYNTH-NATIVE-PREPARATION-CAPACITY-20261121.md) is pending explicit approval. The public 256-entry cap remains. The full-image native workflow is not yet claimed. |
| Repeated inspection and parameter use | `4346ad249`: reuse a shared plan's inspection within one result-verification call; 106 contract and 94 execution-mode/numeric tests passed. Lesson six now verifies existing Circuit runtime-plan inputs and PyTorch gradients at three nonzero angles | The existing runtime route does not make qfunc symbolic. Native qfunc parameter reuse still requires its own measured design and admission. |
| Block/QSVT composition | Original CFD Pauli block now also composes existing preparation, controlled helpers and within/apply in native qfunc; all eight columns and two signed inputs match complete independent states | Native uses 12 qubits (four clean work bits), versus eight for Circuit; zero-work native budget refuses. The original degree-33 Hamiltonian QSVT now composes both full streams and its Pauli signal in native qfunc; banded construction and general reusable QSVT APIs remain follow-ups. |
| Named sampling | Bell and QSVT lessons use existing `fq.counts`/`fq.samples` on synthesized Circuit IR, selecting wires from named ports; the original joint draw is preserved for conditioning | Sealed Program output overrides still refuse. A direct named finite-shot convenience surface requires separate API admission; no competing sampler was introduced. |
| Signed numeric casts | `65e1243c8`: arithmetic candidates borrow immutable root input bits for exact masks and copy-only sign extension; 2022 Compiler/integration tests and six final focused tests passed | The original workshop still needs 39 native qubits versus 14 Circuit qubits. Retain the Circuit route until a reusable lower-cost implementation satisfies its original budget. |
| Gaussian state family | Winning iQuHack executable stage now uses logical qfunc ports, a controlled borrowed helper and pairwise phases; full 17-cell replay and 18 related tests passed | The 127-qubit extension retains explicit Circuit compilation only. It is not a native executable capacity claim. |

The signed workshop preflight retains all 256 input pairs and all 64 arbitrary
XOR targets, checked with exact reversible simulation under both schedules.
Two smaller full-state witnesses independently check absolute complex phases
and clean work bits; they are not original-size statevector acceptance.
Retained scheduling changed from 36 work bits / 286 instructions to 25 / 264;
recomputation changed from 25 / 304 to 25 / 264. These are logical resource counts,
not runtime speed claims. Sign-bit aliases are allowed only for copy-based
consumers; arithmetic kernels requiring injective operands retain owned copies.

For the two measured coloring programs, per-result verification now inspects the
identical sealed plan once rather than twice, and complete named reading performs
two such inspections rather than three. Local diagnostic read times changed from
3.22 to 2.66 seconds and from 6.09 to 4.25 seconds; the same resources, key counts
and probability mass were retained. Other acceptance jobs were running, so these
are not controlled benchmark ratios. Every later call still validates mutable
controls; a distinct, serialized result plan still receives its own inspection.
Raw before/after measurements remain separate from the original resource baseline.

The QSVM feature-map diagnostic compares four nonzero angle triples with five
trials after warmup. Concrete qfunc specialization, symbolic Circuit binding
and accepted-plan runtime inputs produce the same complete complex state within
1.3e-16, retaining two qubits and ten logical operations. Local median synthesis,
execution and named reading were approximately 3.48, 1.12 and 1.19 ms;
existing runtime-plan execution was 1.16 ms. Other checks were active, so these
are diagnostics rather than a throughput claim. Python-traced compilation peaks
were about 106--108 kB; that is not total process or tensor peak memory. The
intrusive C1--C6 profile is retained separately from primary wall-time samples.
The variational tutorial demonstrates the existing runtime route directly,
including the need to admit a gradient-capable plan on its first execution.


## Validation policy

Execute changed Notebook cells in order with the existing source-aware executor.
Use independent mathematical/Circuit references where quantum construction
changes. Keep resource limits and original-case numerical checks unchanged.
Run the catalog checks after navigation changes, then repository default checks
at each integration checkpoint. Record raw failures and distinguish environmental
rechecks from a passing complete run.

## Native QSVT composition checkpoint

The original Hamiltonian-evolution Notebook now visibly executes the 33/34-angle
native sequence. Its signal uses the original four Pauli terms, within/apply,
controlled helpers and adjoints, reusing existing public semantics. All 16
signal columns and three complete evolution states (seeds 132/247 and a signed
coherent vector) pass without phase alignment. Four work bits return to zero
within 1e-24 probability. Time 22, block scale 0.75, response scale 0.9999 and
the original 1e-7 evolution threshold remain unchanged.

Actual seed-132 native resources are ten qubits, 3077 logical instructions and
depth 2729. The bounded native route allows 3500 operations and 1 MiB; it refuses
zero workspace, six total qubits or a 3000-operation limit. A preliminary dense
signal admitted 25958 operations on nine qubits; that diagnostic remains outside
the shipped implementation. The packed six-qubit Circuit remains an independent
route with a different instruction vocabulary, not an efficiency comparator.
The 26 scenario tests and all 12 Notebook stages pass. An initial column-probe
used unsupported classical-name basis assignment; it was changed to the existing
explicit amplitude-preparation API without changing compiler semantics.

## Finite-shot execution checkpoint

The Bell lesson requests 4096 real local draws with the existing Runtime API,
using its named port mapping and a named measurement result. Tests retain exact
Bell amplitudes separately, check same-seed reproducibility, changed-seed draws,
shot totals and the Program override/result-identity refusals.

The native QSVT lesson samples all response flags and data in one joint request.
It reports unconditional flag acceptance and conditional data frequencies with
the accepted-shot denominator. Independent checks compare the empirical histogram
with the matrix-exponential distribution and verify asymmetric requested-wire
order. Zero accepted shots refuse conditional estimation rather than inventing
a distribution. This creates a separate measurement plan on the same IR; it is
not runtime mutation of the sealed synthesis plan or a direct named shot reader.
The 98 focused lesson, QSVT and existing observable-output tests pass.
