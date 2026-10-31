# FlagQuantum Release Notes

Release notes describe user-visible behavior and support-boundary changes.
Benchmark claims require audited artifacts and are not inferred from this file.

## 0.2.0

- Established FlagQuantum IR, `fq.Circuit`, `fq.Module`, `fq.run`, runtime
  planning, explicit sharded statevector/MPS execution, distributed training,
  deployment packaging, and versioned result contracts as the maintained
  product architecture.
- Removed the pre-release v0.1 `DistributedQuantumDevice`, `GeneralEncoder`,
  `InvertibleUnitary`, DTensor interchange helpers, device-oriented gates, and
  device-oriented measurement path without a compatibility layer. They are not
  part of the v0.2 product or its capability evidence.
- Fixed the package release identity at `0.2.0`.

## Unreleased

- Extend the unreleased synthesize candidate with bounded local .qmod files
  (PathLike) and explicit source text (source_format="qmod"). Both use the
  existing checked Program, runtime and named reader. Source diagnostics and
  whole-source/actual resource reports are retained; native inputs default to
  a 512 MiB planned-state buffer ceiling. Three unchanged originals are verified
  through both forms. This is a pinned subset, with independent trials and
  release review still pending; it does not admit the full Qmod standard library.

- Add the maintained `flagquantum.synth` implementation candidate and
  `fq.run(program)` for its factory-created Program. The normal high-level path
  uses named preparation and `program.probabilities(result)`; the root facade
  validates and executes the existing bound plan without replanning. Execution
  overrides are rejected. The initial exact CPU arithmetic/line-profile scope
  is unchanged; independent user trials and final release review remain pending.
  No Program/.fq serialization, general Qmod or hardware capability is implied.

- Add `fq.save` and `fq.load` for portable `.fqx` experiment archives containing
  CircuitIR and optional recorded numeric results. I/O never executes the
  program; loading restores detached CPU data and historical execution metadata.
  Version 1.0 supports retained static execution plans, bounded data-only ZIP
  archives, corruption checks, and atomic writes with explicit overwrite. See
  [experiment files](EXPERIMENT_FILES.md) for limits and excluded workflows.

- Complete API Change 098 with typed `ch`, `csx`, `csxdg`, `dcx`, and
  `ecr` methods on both `Circuit` and `DynamicCircuit`. Each gate has a
  canonical matrix, closed adjoint behavior, exact basis lowering, static
  exporter behavior, IR round-trip coverage, and an explicit ecosystem support
  boundary. This does not claim native provider or QPU support.

- Add typed `iswap` and `iswapdg` methods to both `Circuit` and
  `DynamicCircuit`. The pair has exact dense semantics, closes under
  `adjoint()`, lowers through the existing fixed basis for OpenQASM, QIR, and
  QCIS, participates in compilation and optimization, and fails closed in
  ecosystem adapters that do not yet declare a mapping. This does not claim
  native provider or QPU support.

- Make every schema-installed fluent gate visible to static type checkers on both
  `Circuit` and `DynamicCircuit`. Gate chains now retain the concrete builder
  type, including aliases and uppercase compatibility spellings, without
  changing runtime installation, signatures, validation, or execution support.
  Distributions include the declaration stub and PEP 561 marker.

- Admit a full-register terminal-measurement subset of `DynamicCircuit` through
  the existing static Quafu `fq.run(...)` route. Preserve declared
  qubit-to-classical-bit permutations in returned counts, retain inspectable
  normalization identity, and keep partial, repeated, mid-circuit, reset,
  conditional, samples, expectation, and Qsteed physical paths fail closed.
  Add a local CPU benchmark for normalization and counts projection; it performs
  no provider, credential, network, task, quota, or QPU work.

- Add caller-configured, offline QPU Twin evidence-freshness assessment. Age,
  calibration drift, and verified TV-error thresholds return explicit current,
  refresh-due, insufficient-evidence, or incompatible-history outcomes without
  provider I/O, model mutation, promotion, or routing authority.

- Make experimental dynamic backend assessment consume the same fail-closed
  requirement/snapshot matcher as stable remote `fq.run(...)`. A coarse
  `supports_dynamic_circuits=True` profile can still describe experimental
  packaging constraints but cannot report execution admission without detailed
  device-scoped facts and evidence.
- Guard stable remote `DynamicCircuit` calls at the `fq.run(...)` facade before
  compiler or provider dispatch. Remote targets now fail closed with the exact
  target and ordered capability blockers until a detailed device-scoped
  snapshot satisfies the existing provider-neutral matcher. This does not
  activate any remote dynamic provider or perform discovery, credential,
  network, package, or task work; local dynamic execution and static remote
  dispatch remain unchanged.
- Target QEC retirement for **0.4.0**: remove the 23 approved historical flat
  module imports and their pickle lookup paths. Keep all 108 ordered
  `flagquantum.qec` exports, signatures, numerical results and registration
  behavior. Use root imports or the canonical responsibility packages listed in
  the [migration guide](../guides/QEC_PACKAGE_LAYOUT.md).
- Keep only the original registered `RotatedSurfaceCode`; root, `qec.codes`,
  `codes.surface` and the factory use that same class. Its existing
  `flagquantum.qec.codes` pickle identifier survives. The historical
  `qec.surface` class and private duplicate are removed without a replacement
  frame class. Artifacts referencing retired paths must be loaded/exported
  using the pre-retirement version before upgrading; no compatibility
  unpickler is provided. Original frozen restructuring fixtures remain unchanged.

- Promote the completed local DynamicCircuit execution contract to stable.
  `fq.plan(...)` and `fq.run(...)` remain the single planning and execution
  entries, return `fq.ExecutionResult`, and preserve plan identity. This is a
  contract-status change only: no signature, root export, schema, or runtime
  behavior changes, and `run_dynamic` remains experimental.
- Add a reproducible local performance runner for the stable DynamicCircuit
  `fq.plan` / `fq.run` path. It measures planning, explicit-plan execution,
  automatic planning and execution, the native numerical oracle, memory-bound
  trajectory selection, and an unchanged static sampling sentinel. Its budgets
  and recorded results are single-process CPU evidence only.
- Execute supported noiseless `DynamicCircuit` programs through the stable
  `fq.plan(...)` and `fq.run(...)` workflow on local CPU. Direct `shots=` and
  restored `ExecutionPlan` execution now return the stable `ExecutionResult`
  while preserving plan identity, batch size, precision, and seeded sampling.
  Noise, custom initial states, remote execution, and native dynamic result
  fields remain outside this stable slice and fail closed.
- Refuse collisions between explicit and generated Lindblad expectation output
  names with `duplicate_output_name`, instead of silently overwriting a result.
  Supply distinct names to keep both outputs. The existing planner now captures
  numeric output matrices without detaching gradients, preparing the approved
  matrix request contract. Existing plan formats remain unchanged.
- Accept full-space dense numeric tensors in the existing `expectation` factory
  and `OutputRequest.observable` on CPU Lindblad workflows. Reuse existing
  result accessors, live gradients, numeric validation and plan 1.3/1.4/1.5/1.6.
  Tensor requests compare/hash by object identity and metadata; Pauli request
  behavior remains unchanged. Invalid matrix representations, non-expectation
  tensor requests and local selectors are refused at construction; unsupported
  circuit/remote paths raise `CapabilityError` before execution or submission.
  No new root export, output kind, solver or native hardware capability is added.

- Add ordered basis_labels to the existing CPU Lindblad plan/run and
  initial-density adjoint entries for non-factorized matrix coordinates.
  Static and piecewise labelled requests use plan 1.6 with retained literal
  labels and no qubit count; 1.3/1.4/1.5 payloads and identities remain unchanged.
  Existing Core parameters, Liouvillian, integrators, results and PyTorch paths
  are reused. Direct boson conversion still requires product dimensions, and
  labels alone do not certify a physical sector or implement native braiding.

- Preserve live scalar tensor coefficients in static full-space Hamiltonian
  term mappings. Numeric matrices and existing boson operators share the same
  normalization and dynamics path; named values bind with the existing
  `Parameter.bind` / `ParameterExpression.bind`. Include a PyTorch Adam example.
  Existing numeric Pauli descriptors remain matrix-free; plan schemas, canonical
  boson coefficients, and Stable Core are unchanged.
- Accept existing boson operators directly in static Lindblad Hamiltonians,
  collapse pairs, and the low-level observable mapping with explicit finite
  `dimensions=`. Reuse dense conversion, its memory ceiling, existing plans,
  solvers, batches, and initial-state gradients. The two-mode interference/loss
  example is a bounded CPU reference; algebra coefficients remain numeric and
  `fq.expectation` remains qubit-only.
- Extend the existing static CPU Lindblad workflow with explicit ordered
  `dimensions=` for finite product spaces. Three-level and spin-mode models
  reuse the existing numerical core, sealed plans, result type, and gradients.
  Non-qubit plans use schema 1.4; existing qubit plans retain their 1.3 content
  and identities. CircuitIR 1.0 and Stable Core exports are unchanged. Pauli
  descriptors remain qubit-only; this is not a photonic or hardware backend.
- Use compensated summation for contracted Gaussian overlaps and their
  independent closed-form test references. Small contributions survive
  cancellation and the existing integral accuracy and rotation checks pass on
  Python 3.10 through 3.12 without relaxed assertions or a public API change.
- Correct the Lindblad capability description to include discretized autograd,
  initial-state batching under a shared generator, and the adjoint gradient
  with respect to the normalized initial density matrix. CPU and static-generator
  restrictions remain; this is a description correction, not
  a maturity promotion or general pulse-gradient claim.

## 0.3.1 (release pending)

- Includes the approved local research-capability services added after the
  0.3.0 distribution was built. The public `flagquantum.services` surface
  discovers and describes three bounded operations, validates and estimates
  requests, executes supported local CPU work, and validates structured
  evidence. No MCP transport or agent runtime is included in the core package.
- The supported ground-state VQE scope remains a supplied two-qubit Pauli sum
  with an explicit ansatz and optimizer. Molecular Hamiltonian construction,
  remote execution, and unsupported scientific scopes are not implied.

## 0.3.0 (2026-10-16)

- Publishes the 0.3 release line as a stable package for Python 3.10 through
  3.12, with the cross-platform Stable ABI distribution pipeline.
- Adds recoverable remote job submissions for Quafu and Jiuding through
  opt-in receipts, plus ordered multi-circuit Quafu collection results.
- Includes English and Chinese editions of the six-lesson FlagOS 2026
  workshop.

- Added opt-in `fq.submit(..., receipt=path)` recovery for Quafu and native
  Jiuding jobs. Matching requests restore automatically; configuration changes,
  damaged records and uncertain attempts fail without automatic resubmission.
  Existing calls without a receipt and standalone `fq.restore_job` are unchanged.

## 0.3.0rc3 (2026-10-10)

- Corrected the cross-platform Stable ABI release build after rc2 validation:
  Linux x86_64, macOS arm64, and Windows amd64 now each produce and test one
  `cp310-abi3` wheel for the supported Python 3.10 through 3.12 range.
- Made the native CPU extension portable to MSVC, including Stable ABI flag
  propagation, OpenMP SIMD compilation, and portable bit lookup, without
  changing the public Python API or the supported PyTorch runtime range.
- Kept `0.3.0rc2` immutable. rc3 is the release candidate that includes the
  verified macOS arm64 and Windows amd64 wheel pipeline corrections.

## 0.3.0rc2 (2026-10-10)

- Migrated the native CPU extension to CPython's limited API and PyTorch's
  Stable ABI. Each supported platform now ships one `cp310-abi3` wheel for
  Python 3.10 through 3.12 instead of rebuilding the extension for every
  interpreter minor.
- Compiled release artifacts against the PyTorch 2.10 Stable ABI baseline and
  validated them on the supported PyTorch `>=2.13,<2.15` runtime range. PyTorch
  2.10 is a build baseline, not the user runtime dependency.
- Updated development images to install the self-contained
  `flagquantum-compiler-qsteed==0.2.0` package. The plugin vendors its verified
  compiler-only closure and no longer requires cloning QSteed during image
  builds.
- Preserved the existing `v0.3.0rc1` tag and incomplete GitHub prerelease as an
  immutable record. Its wheel build did not complete and no `0.3.0rc1`
  distribution was uploaded to PyPI; rc2 is the corrected release candidate.

## 0.3.0rc1 (2026-10-10)

This release candidate packages the user-visible work completed since 0.2.0 for
installation and provider-workflow validation before the final 0.3.0 release.

These entries were written while the release that shipped as `0.2.0` was still
called the first alpha, which is why some of them say the "first-alpha API
freeze" remains pending. That freeze landed as the protected Stable Core
recorded in `docs/public_api_v1.json` and
[Public API Protection](../development/PUBLIC_API_PROTECTION.md); the entries
are otherwise unchanged.

- Added `Circuit.power(k)` to the construction-time composition family. It returns
  the circuit that applies the receiver's program `k` times, for an integer `k`,
  and it is exact for every instruction: the whole program is appended in order,
  so a gate with no angle, a custom `Circuit.unitary(...)` operation, and a noise
  channel all work. `power(1)` copies the program rather than multiplying its
  angles by one, so a symbolic parameter or a trainable angle stays the same
  object, and `power(0)` is the empty program of the same shape. A negative
  exponent is `adjoint().power(-k)`, which keeps the inverse rule in one place, so
  a channel under a negative exponent raises the same `CapabilityError` that
  `adjoint` raises. One rewrite shortens the common case: a receiver that is
  exactly one instruction declaring exactly one parameter has that parameter
  multiplied by the exponent instead of the gate being repeated, and the rule is
  declared per opcode in `flagquantum/core/operator_schema.py` and measured
  through the dense operator by the composition contract gate, so 12 of the 35
  registered opcodes take the closed form and the other 23 repeat. A fractional
  exponent is refused with `TypeError` rather than approximated, because the IR
  has no instruction form for a matrix root. See
  [the power API change](../api-changes/FQ-CIRCUIT-POWER-20261021.md).

- Added the stable `fq.density_matrix(qubits=None, *, name=None)` output request and
  the `ExecutionResult.density_matrix` property, so an execution can be asked for the
  state on a named subset of qubits instead of only for outcome statistics.
  `qubits=None` means every qubit, a subset answers as a `(batch, 2 ** k, 2 ** k)`
  tensor whose index order is the order the caller named, and `Tr(O rho)` read off
  the matrix agrees with `fq.expectation(O)` on the same qubits. Every execution mode
  answers the same matrix; a statevector route refuses above ten qubits and names
  `options=fq.ExecutionOptions(mode='density_matrix')` as the cheaper route, and a
  noise model carrying a readout rule is refused by name because readout confusion is
  applied to outcomes after the state. `Circuit.noisy_density_matrix` now reads the
  circuit's dtype the way `Circuit.density_matrix` already did, so the two no longer
  disagree by the `complex64` rounding floor on a `complex128` circuit. See
  [the density-matrix output change](../api-changes/FQ-DENSITY-MATRIX-OUTPUT-20261006.md).
- Added the stable `fq.vn_entropy(qubits, *, log_base=None, name=None)` output
  request and the `ExecutionResult.vn_entropy` property, so an execution can be asked
  for the von Neumann entropy of a named subsystem instead of the caller having to
  diagonalise a reduced state by hand. At least one qubit is required and the
  selection may be any subset rather than a contiguous run; `log_base=None` answers in
  nats and `log_base=2` in bits, under the same `log_base` keyword PennyLane's
  `qml.vn_entropy` accepts.
  Every execution mode that holds a state answers the same number: dense modes reduce
  and diagonalise, while `mps` and `tensor_network` modes contract the named sites out
  of the state they already hold, and the `mps` route reads the canonical form's
  Schmidt values directly when the selection is a leading run, so `fq.vn_entropy([0])`
  answers where `fq.density_matrix([0])` is refused by the amplitude ceiling. Entropy
  is not a sampled quantity, so `shots=` is refused and `mode='stabilizer'` is refused
  by name; a noise model carrying a readout rule is refused because readout confusion
  acts on outcomes after the state. A base with no logarithm is refused rather than
  answered by accident: `0`, a negative base, `1`, and a non-finite base each raise a
  sentence naming the reason, where PennyLane 0.45.1 returns the natural logarithm,
  `nan`, `inf`, and `nan` respectively. See
  [the von Neumann entropy output change](../api-changes/FQ-VN-ENTROPY-OUTPUT-20261101.md).
- Added `optimization_level` to `flagquantum.compiler.optimize` and
  `flagquantum.compiler.compile`. It is keyword-only and defaults to `2`, which is
  the pass composition those entry points already ran, so a caller that does not
  name it receives the program it received before. Level `0` returns the submitted
  program unchanged, level `1` cancels and merges what is adjacent, and level `2`
  also removes diagonal gates before a measurement and merges rotations across a
  proven commuting gap. Level `3` is declared and reserved and raises
  `CompilationError`, because the unitary-synthesis stage it would add needs a
  target basis and `optimize` is target-independent; a value that is not an integer
  is refused the same way. The level that ran is recorded in `metadata["optimization"]`. See
  [the optimization-level API change](../api-changes/FQ-COMPILER-OPTIMIZATION-LEVEL-20261006.md).

- Added the stable `fq.gradient(program, parameters, loss=None, *, method="auto",
  step=None, directions=1, generator=None)` entry point, which returns the
  detached derivative together with the method that produced it, whether that
  method is exact, and the displacement an approximation used. `method="auto"`
  probes the program instead of declaring a route, so it reports `autograd` when
  the program carries a graph, `parameter_shift` when the circuit behind the loss
  is available, and `finite_difference` otherwise. `method="adjoint"` is refused
  with a `CapabilityError` because FlagQuantum has no standalone adjoint entry
  point; the reversible sweep is the backward pass behind PyTorch autograd and no
  result reports whether backward replayed it. See
  [the gradient API change](../api-changes/FQ-GRADIENT-API-20261002.md) and the
  [gradient methods example](../../examples/gradient_methods/README.md).
- Added the stable `fq.jacobian(program, parameters)`,
  `fq.jvp(program, parameters, tangents)`, and
  `fq.vjp(program, parameters, cotangents)` entry points for a program that
  returns several values, so that no single scalar gradient is defined.
  `fq.jacobian` returns every partial derivative shaped
  `(*program_output.shape, *parameters.shape)`, `fq.jvp` applies it forward as
  `J @ tangents`, and `fq.vjp` applies it in reverse as `cotangents @ J`. None
  of the three takes a `method`: there is no shift rule for a vector output and
  no displacement to report, so all three are autograd-only and a program whose
  output carries no PyTorch graph is refused rather than answered with a zero
  derivative. A tangent must be shaped exactly like `parameters` and a cotangent
  exactly like the program output, because a broadcast direction would
  differentiate along a different parameterization. All three serve
  `statevector`, `mps`, and `tensor_network` mode and return the parameter dtype
  and device. See
  [the vector-derivative API change](../api-changes/FQ-VECTOR-DERIVATIVES-20261026.md).
- Added the stable `fq.from_openqasm(source)` entry point, which reads back the
  OpenQASM 2 and OpenQASM 3 text
  `flagquantum.compiler.openqasm.emit_openqasm` writes. It reports the imported
  program, the version it declared, and the classical-bit to qubit mapping its
  terminal measurement block records, and exposes `to_ir()`, `to_circuit()`, and
  `to_openqasm()`. Text outside the canonical subset is refused with an
  `OpenQASMImportError` whose `issue_code` names the reason, so nothing is
  imported approximately. See
  [OpenQASM import](../api-changes/FQ-OPENQASM-IMPORT-20261002.md).

- Added the experimental
  `flagquantum.experimental.distributed.train_distributed_tensor_network`
  workflow, which slices one logical tensor-network contraction across ranks and
  keeps gradient, optimizer, and checkpoint semantics on a declared cut.
  Distributed summaries report rank ownership, memory and communication, and
  keep `scalability_claim_allowed` and `release_gate_allowed` false with their
  blockers attached. The automatic slicer now excludes labels carried only by
  state-copy nodes, which would otherwise yield ranks whose partial is exactly
  zero while the run reported sharded execution. See the
  [two-node runbook](../guides/MULTINODE_RUNBOOK.md).

- Promoted two-node statevector, MPS, and tensor-network training to
  `production_supported` on recorded two-host A800 evidence. Each entry names
  its artifact, its integration tests, and its runbook, and each recorded run
  reports `scalability_claim_allowed` and `release_gate_allowed` false with six
  blockers attached. The recorded shape is two hosts at one device per host;
  wider topologies run on the lane but are not recorded and no wider claim is
  made.

- Added `TwinRegionRelease.assess_support(...)` to distinguish exact released
  circuits from unseen circuits that only fit a regional release's frozen
  mapping, topology, operation, instruction-count, and depth envelope. Only an
  exact released circuit returns a prediction; structurally compatible but
  unvalidated circuits and out-of-envelope circuits return no prediction,
  confidence, or error bound. The check is provider-neutral, offline, and does
  not route or submit work. See
  [QPU digital twins](../guides/QPU_DIGITAL_TWIN.md).

- Added prospective regional Twin holdout studies. Applications can freeze
  disjoint reference and holdout circuit suites before hardware work, persist
  that split, and validate both groups under one simultaneous confidence
  statement. The result reports the holdout Twin-QPU TV-error increase without
  claiming arbitrary-circuit accuracy; cross-group circuit or task reuse fails
  closed. See [QPU digital twins](../guides/QPU_DIGITAL_TWIN.md).

- Added immutable regional Twin validation histories. Applications can align
  the same fixed regional circuit suite across chronological calibration
  snapshots and persist chart-ready Twin-QPU agreement, ideal-QPU agreement,
  QPU repeatability, uncertainty, task, and shot sequences. Changed targets,
  mappings, topologies, suites, or reused reports fail closed; the history
  performs no provider operation and defines no trust or promotion policy. See
  [QPU digital twins](../guides/QPU_DIGITAL_TWIN.md).

- Added fixed regional Twin validation suites. Applications can predeclare
  distinct connected-region circuits and repeated task counts, checkpoint each
  explicit submission, and validate the complete result collection under one
  simultaneous confidence level. The resulting support verifies only the exact
  suite circuits and exercised directed couplers; missing, duplicate, reordered,
  or identity-mismatched tasks fail closed. See
  [QPU digital twins](../guides/QPU_DIGITAL_TWIN.md).

- Added prospective validation for a composed `TwinRegionModel`. Applications
  can freeze a covered regional circuit before explicit Quafu submission, build
  a repeated validation series from at least two identity-bound tasks, and
  produce exact-circuit `TwinCircuitSupport`. The support includes only couplers
  exercised by that circuit and does not claim arbitrary regional accuracy. See
  [QPU digital twins](../guides/QPU_DIGITAL_TWIN.md).

- Added `fq.twin.compose_region_twin()` and immutable `TwinRegionModel` for
  offline measurement-distribution prediction across a connected union of
  compatible local Twin cells. Composition remaps local calibration and noise
  semantics into one regional qubit order and fails closed on overlap conflicts;
  it does not infer cross-cell correlated noise or turn local evidence into
  regional accuracy. See [QPU digital twins](../guides/QPU_DIGITAL_TWIN.md).

- Added `fq.twin.compose_connected_region()` and immutable connected-region
  coverage reports. Applications can combine overlapping, contemporaneous
  `TwinCircuitSupport` cells from one QPU and check an explicit circuit mapping
  against their directed couplers and conservative shared limits. Composition
  is structural only: local TV error bounds and confidence levels are never
  combined into a region-level accuracy claim. See
  [QPU digital twins](../guides/QPU_DIGITAL_TWIN.md).

- Added `fq.twin.TwinCircuitSupport` so an existing frozen evidence envelope can
  be narrowed to prospectively validated directed physical couplers and circuit
  depth. Offline support checks fail closed without an error bound for untested
  interactions, reversed directions, excessive depth, or operations of arity
  greater than two. Canonical persistence preserves the boundary without
  changing the frozen evidence-envelope v1 schema. See
  [QPU digital twins](../guides/QPU_DIGITAL_TWIN.md).

- Added fixed Twin candidate circuit suites. Applications can freeze distinct
  circuit predictions for one incumbent and candidate, explicitly submit and
  checkpoint each existing single-task trial, then obtain a simultaneous
  finite-shot comparison across the suite. The result reports mean incumbent,
  candidate, and ideal-SV agreement without claiming arbitrary-circuit support
  or automatically promoting a model. See
  [QPU digital twins](../guides/QPU_DIGITAL_TWIN.md).

- Added resumable Twin candidate submissions. A complete frozen incumbent-versus-
  candidate trial can be bound to one existing Quafu receipt, written once to a
  private file, restored in a later process, and validated without submitting a
  second hardware task. See
  [QPU digital twins](../guides/QPU_DIGITAL_TWIN.md).

- Added `fq.twin.prepare_candidate_trial()` and immutable candidate-evaluation
  records for comparing an incumbent and later Twin against the exact same
  prospectively collected QPU counts. The result reports a conservative
  finite-shot interval and returns `improved`, `degraded`, or `inconclusive`;
  only the trial's explicit experiment submission performs provider I/O, and no
  model is automatically promoted. See
  [QPU digital twins](../guides/QPU_DIGITAL_TWIN.md).

- Added `fq.twin.align_histories()` and immutable `TwinEvolutionHistory` to
  align calibration drift with Twin↔QPU, Ideal↔QPU, QPU-repeatability and
  verified-bound changes over the exact same snapshots. The offline comparison
  reports synchronized observations without claiming causality or choosing an
  update/promotion policy. See [QPU digital twins](../guides/QPU_DIGITAL_TWIN.md).

- Added immutable `TwinValidationHistory.append()` for incorporating one later
  snapshot-bound validation series without rebuilding or mutating earlier
  evidence. It preserves target, mapping, circuit, chronology and globally
  unique hardware-report invariants; collection and scheduling remain explicit.
  See [QPU digital twins](../guides/QPU_DIGITAL_TWIN.md).

- Added `fq.twin.dump_validation_history()` and
  `fq.twin.load_validation_history()` for canonical offline persistence of
  longitudinal Twin accuracy evidence. Loading reconstructs nested validation
  series and rejects altered derived metrics; writing uses private permissions,
  is idempotent for identical content, and never replaces different or invalid
  files. See [QPU digital twins](../guides/QPU_DIGITAL_TWIN.md).

- Added `fq.twin.build_validation_history()` and immutable
  `TwinValidationHistory` for one fixed circuit across chronological frozen
  Twins. It keeps Twin↔QPU agreement, noiseless Ideal↔QPU agreement, QPU
  repeatability, and verified TV bounds separate while enforcing snapshot,
  target, mapping, circuit, chronology, and hardware-report identity. See
  [QPU digital twins](../guides/QPU_DIGITAL_TWIN.md).

- Added `fq.twin.dump_calibration_history()` and
  `fq.twin.load_calibration_history()` for canonical, credential-free offline
  persistence of cumulative and interval calibration-drift series. Loading
  validates nested records and summaries; writing is private, idempotent for
  identical content, and refuses destructive replacement. See
  [QPU digital twins](../guides/QPU_DIGITAL_TWIN.md).

- Added `fq.twin.build_calibration_history()` and immutable
  `TwinCalibrationHistory`. Applications can now turn strictly chronological,
  comparable frozen Twins into cumulative-from-baseline and adjacent-interval
  drift series suitable for charts. The operation is offline and does not
  collect calibration, persist a history database, choose trust thresholds, or
  update a model. See [QPU digital twins](../guides/QPU_DIGITAL_TWIN.md).

- Added `fq.twin.compare_calibrations()` and immutable per-qubit, per-gate, and
  summary drift records. Comparable saved Twins now expose time-normalized T1,
  T2, readout, gate-duration, and channel-model changes using physical QPU
  identifiers. The comparison is offline and reports calibration facts without
  inventing a significance threshold, accuracy claim, automatic model update,
  or trust decision. See [QPU digital twins](../guides/QPU_DIGITAL_TWIN.md).

- Added `fq.twin.dump_twin()` and `fq.twin.load_twin()` so a calibrated QPU
  digital Twin can be reproduced after a process restart without fetching a
  newer calibration. The strict `flagquantum.qpu_digital_twin.v1` artifact
  binds the complete provider-neutral noise model to its frozen snapshot,
  contains no credential or task receipt, and performs no provider operation.
  See [QPU digital twins](../guides/QPU_DIGITAL_TWIN.md).

- Added `fq.twin.TwinSubmission`, `fq.twin.dump_submission()`, and
  `fq.twin.load_submission()` so a frozen Twin experiment and its credential-free
  Quafu receipt can survive a process restart. Loading is offline and never
  submits, polls, retries, cancels, or fetches a task; those provider actions
  remain explicit. The strict `flagquantum.twin_submission.v1` file is private,
  identity-bound, and refuses destructive replacement. See
  [QPU digital twins](../guides/QPU_DIGITAL_TWIN.md).

- Added `fq.submit()` and `fq.restore_job()` for detached Quafu counts and
  Jiuding managed program jobs. Jobs support one-shot status/result queries,
  explicit waits, cancellation requests, and credential-free JSON receipts.
  Existing synchronous `fq.run()` behavior is unchanged. See
  [remote jobs](../guides/REMOTE_JOBS.md).

- Added direct Quafu execution through `fq.run(..., target="quafu:Baihua", shots=1024)`
  without a local compiler dependency. The service uses QuarkCircuit compilation;
  explicit `compiler="qsteed"` preserves local precompilation. Counts and grouped
  Pauli expectations share the existing result contract. This path has mocked
  transport coverage and a limited [Baihua hardware check](../development/evidence/remote_jobs_quafu_20260911.json).

- Renamed the VQE and ADAPT-VQE `optimizer_cls` keyword to
  `optimizer_factory` and published the `flagquantum.algorithms.OptimizerFactory`
  protocol. The old keyword is intentionally unsupported; Adam remains the
  default, and compatible optimizer classes or factory functions are accepted.

- Completed Circuit noise and inspection return annotations and specified that
  `StrictExecutionScope` does not suppress exceptions. CPU probe metadata is
  now read-only through `CPUCapabilityProbe`, supporting both frozen and mutable
  provider implementations. Update mutable metadata through the concrete
  provider type before passing it to the adapter; assignments through the
  protocol are rejected by static type checkers. Runtime call syntax, results,
  and serialized schemas are unchanged.

- Extended `fq.run` with the explicit
  `compiler="qsteed", target="quafu:<backend>", shots=...` remote journey. It
  performs compilation, sealed packaging, submission, and result retrieval
  without changing the `ExecutionResult` return type or silently selecting a
  compiler, provider, or fallback.
- Added `fq.compile(circuit, compiler="qsteed", target="quafu:<backend>")` as
  the direct compiler-selection journey. QSteed results retain logical qubit
numbers and carry an ordered physical `target_qubits` mapping through
  deployment; Quafu submission now uses `compiler=None` and no deprecated
  top-level compile flag.
- Added explicit discovery for independently installed circuit-compiler plugins
  through the existing Ecosystem extension registry. Entry-point identity,
  capability negotiation, CircuitIR ownership, determinism, lifecycle cleanup,
  and fail-closed errors are covered by reusable conformance tests; pulse-level
  artifacts remain outside this circuit-level contract.
- Validated the declared Qiskit 2.0.x/2.5.x, Aer 0.17.x, and PennyLane
  0.44.1/0.45.1 interoperability window with real dependencies. Corrected the
  Qiskit golden path to use the diagnostic `export_qiskit()` /
  `import_qiskit()` pair rather than treating the concise conversion return as
  a report wrapper.
- Promoted the isolated `flagquantum.ecosystem.extensions` protocol to a first-alpha
  candidate with a machine-readable contract and root-surface guard. Public
  conformance entry points and a third-party-style backend example now prove
  that extensions need no runtime or core internals. Individual extensions are
  still experimental by default. Before public release, the protocol moved to
  `flagquantum.ecosystem.extensions` without a compatibility namespace; its
  freeze and the overall first-alpha API freeze remain pending.
  The audit also retains `noise_model` and keeps local `ExecutionPlan` separate
  from provider-facing `DeploymentPackage`.
- Added the stable `flagquantum.errors` lifecycle categories while preserving
  compatibility with Python's `ValueError`, `RuntimeError`, and
  `NotImplementedError`. Existing IR, plan, result, capability, compilation,
  and training-state failures now map to backend-neutral categories. Removed
  the unused `deployment_binding` dictionary from `fq.Module`; application and
  deployment models own this state and preserve it through checkpoints. Old
  Module extra-state containing the field remains readable and is ignored.
- Fixed the first-alpha candidate boundary for Module and training:
  `Module.forward()` returns an autograd Tensor, `Module.execute()` returns
  `ExecutionResult`, and `fq.run()` does not accept Module. `fq.train()` remains
  a minimal caller-owned PyTorch optimizer loop; checkpoint/resume stays on
  Module, while distributed training remains experimental. Added the stable
  `flagquantum.training` lifecycle namespace plus versioned result diagnostics,
  training summaries, and checkpoint-restore metadata. The Module/training
  contract is now frozen; the overall first-alpha API freeze remains pending.
- Unified stable measurement and noise execution through `ExecutionPlan`.
  Explicit measurements are embedded before planning and can no longer silently
  replace IR requests; versioned noise models survive verified plan JSON round
  trips. Added explicit result accessors and removed implicit backend-native
  attribute forwarding. Result summaries now carry schema and version fields.
- Added executable plans with deterministic program/options/environment/compiler
  fingerprints, SHA-256 identity, strict JSON round trips, tamper and stale
  environment detection, and exact `fq.run(plan)` execution without replanning,
  recompilation, or silent fallback. `fq.ExecutionPlan` is now available from
  the curated root API; the overall first-alpha API freeze remains pending.
- Added fail-closed single-node MPS development evidence on 8×A800:
  2/4/8-GPU rank-owned forward, 2/8-GPU packed boundary transport and fault
  cleanup, 2-GPU accelerator backward, and a 100-step 8-GPU SGD soak with
  stable memory and exact checkpoint/restart equivalence. This does not promote
  multi-node or release-certified support.
- Removed the pre-release `flagquantum.algorithms_stack` namespace. Import
  algorithm helpers from `flagquantum.algorithms` and optimization contracts
  from `flagquantum.algorithms.optimization`.
- Removed the pre-release `flagquantum.runtime_stack` compatibility namespace.
  Runtime integrations, benchmarks, plugins, and serialized references must use
  the canonical `flagquantum.runtime` packages.
- Removed the unused pre-release `flagquantum.core.circuit` compatibility shim.
  Import `Circuit` from `flagquantum`, or from `flagquantum.circuit` for an
  explicit implementation-module dependency.
- `fq.Circuit(n_qubits=...)` is now the preferred public spelling for circuit
  size, with `n_qubits` and Qiskit-compatible `num_qubits` properties.
  Positional construction and the `n_wires`/`nqubits` keyword aliases remain
  compatible, while conflicting counts now fail during construction. Serialized
  IR and compiler payload keys are unchanged: renaming a key that a saved plan is
  read back through is a schema version bump, not a vocabulary change.
- Generated gate methods now accept optional semantic qubit keywords without
  removing concise positional calls: single-qubit gates use `qubit=`,
  controlled gates use `control=`/`target=`, and symmetric two-qubit gates use
  `qubit1=`/`qubit2=`. Duplicate, conflicting, and missing qubit arguments fail
  before instructions are added.
- Added staged hybrid optimization for named classical and quantum parameter
  groups. Local VQE convergence workflows can combine Adam/AdamW/SGD/L-BFGS,
  exact full or block QNG, and Rotosolve, with per-step gradient/update/evaluation
  diagnostics. Heisenberg helpers now provide phase-augmented bond-resolved HVA,
  dimer-singlet initialization, and exact small-system energy references.
  The convergence example preserves requested float64 parameters, supports
  identity-layer continuation, applies bounded-memory L-BFGS, and emits an
  explicit exact-energy convergence decision instead of treating loss decrease
  as convergence.
- Distributed MPS training supports an experimental `adam_lbfgs` schedule with
  owner-local limited-memory histories and collective scalar products. The N=8
  Heisenberg benchmark records aligned optimizer-step energy/error traces.
  Exact gradient policy now automatically retains factorization graphs for
  statevector-parity gradients; the recomputation pullback remains restricted
  to explicitly approximate-gradient work.
- `flagquantum.experimental.distributed.train_distributed_statevector(...)` now
  provides multi-step native
  PyTorch training with owner-sharded SGD/Adam state, checkpoint/resume,
  cancellation, memory preflight/measurement and structured lifecycle progress.
- Experimental statevector execution now preserves amplitude sharding through
  PyTorch backward for multi-layer shared-parameter circuits using rank-local
  rematerialization, explicit gate adjoints, and cross-rank VJP reduction.
- An experimental PyTorch-native distributed statevector forward executor now
  runs validated unitary IR with rank-local amplitude ownership across local and
  multi-sharded-qubit gates on Gloo or NCCL. Full-state materialization is
  forbidden and forward-only training/release blockers remain explicit.
- Correctness infrastructure now generates deterministic circuit properties,
  covers every registered operator/lowering pair, versions numerical
  tolerances, retains minimal failure artifacts, fuzzes IR parsing, and
  classifies distributed no-progress with cleanup requirements.
- An experimental extension SDK now supports isolated backend, kernel,
  operator, compiler-pass, device, provider, measurement-collector, and planner
  plugins with version checks, capability negotiation, lifecycle containment,
  credential-safe configuration, and reusable conformance tests.
- The stable root API is bounded by a checked manifest. Experimental and
  compatibility imports no longer silently expand the supported surface.
- Circuit operations and backend lowering availability derive from one typed
  operator registry, with generated capability documentation.
- Runtime configuration is immutable, task-scoped, serializable, and propagated
  explicitly through compile, plan, and execution boundaries.
- Planner requests, executor observations, measurements, failures, provenance,
  and audit decisions use strict versioned contracts with deterministic JSON.
- Performance and memory baselines record uncertainty, correctness, progress,
  and measured hardware provenance without promoting development runs to
  release scalability evidence.
- Packaging enforces Python 3.10–3.12, bounded dependency families,
  reproducible source/wheel contents, lazy imports, and repository hygiene.
- Documentation derives stable API, operator capability, and current issue
  state tables from authoritative sources and fails CI when claims drift.

No release build or distributed scalability release artifact is declared by
these unreleased changes.
