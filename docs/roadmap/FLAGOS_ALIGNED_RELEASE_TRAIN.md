# FlagQuantum Roadmap

> **Status:** Future intent
> **Direction:** A general quantum computing framework with shared program,
> simulation, and execution capabilities, PyTorch-native training, and the
> FlagOS unified multi-chip backend.

This roadmap describes what FlagQuantum plans to deliver next. It is not a
statement of current support. For capabilities available today, see the
[capability catalog](../generated/CAPABILITIES.md) and
[known limitations](../reference/KNOWN_LIMITATIONS.md).

## The destination

FlagQuantum is moving toward one durable quantum computing workflow:

```text
express a circuit or physical model
  → declare outputs, constraints, and accuracy requirements
  → plan a supported representation and execution method
  → simulate or execute and inspect results
  → optionally optimize with PyTorch, scale, or deploy
```

Backend choice should not require users to rewrite their model. Planning,
execution, gradients, and deployment should remain visible through stable,
typed interfaces.

Domain-native programs may express physical controls, optical modes, or anyon
fusion and braiding instead of gates. Shared semantics must preserve those
distinctions until a supported lowering is selected. Arbitrary backend switching
is not promised: the planner must reject unsupported operations, state spaces,
outputs, or gradients before numerical execution.

## Shared programming and M0-M4 delivery

This is the delivery sequence for the general-framework scope in the
[vision](FLAGQUANTUM_VISION.md). The existing [architecture](../../ARCHITECTURE.md),
[implementation map](../development/IR_IMPLEMENTATION_STATUS.md), and
[ARCH-012](../architecture/decisions/ARCH_012_CUDAQ_PARITY_CONTROL_SEQUENCE.md)
remain authoritative. No new public API, IR schema, top-level package, numerical
core, or compatibility exception is approved by this roadmap.

The baseline reviewed for M0 is public main commit
`0f389c9d8f30179557eb14920bbc9e9dd49b1bd0` (2026-10-19). Supported behavior
continues to come from `capability-maturity.toml`; the CUDA-Q comparison continues
to come from `contracts/cudaq-parity-matrix.toml`. Update their generated views
with the existing tools. Do not create another capability registry or scoreboard.

### Four independent dimensions

| Dimension | Examples |
| --- | --- |
| Physical system | Superconducting circuits, neutral atoms, trapped ions, optical systems, Majorana systems |
| Computing model | Gate circuits, continuous evolution, measurement feedback, anyon braiding and fusion |
| Numerical representation | Statevector, density matrix, MPS/TN, stabilizer, truncated Fock space, fusion space |
| Execution target | CPU, accelerator, distributed resources, external simulator, real device |

Only evidenced combinations are supported. In particular, local simulation,
device discovery, cloud submission, and a physics-native SDK are separate
capabilities. Topological computation is not synonymous with a topological
error-correcting code or a provider backend.

### Milestones and acceptance

The ordering below is binding for this workstream; durations are planning
estimates for a four-to-six-person team with physics support, not release dates.
Milestones advance by evidence, not elapsed time. M0 is a baseline repair and
documentation change. The bounded M1 mathematical slices below have merged;
their consolidated acceptance is proposed in the
[M1 acceptance review](../development/M1_SHARED_EXECUTION_ACCEPTANCE.md).
M2-M4 retain their own exit gates.

| Milestone | Estimate | Owning domains | Deliverable and exit gate |
| --- | --- | --- | --- |
| M0: reliable baseline | 1-2 weeks | Integration; Algorithms for the numerical repair | Reproduce and repair the Python 3.10/3.11 CPU failures without weakening numerical assertions; reconcile capability descriptions and generated views; keep public APIs and architecture unchanged; run supported-version regressions and required repository checks. |
| M1: shared mathematical execution | 4-6 weeks | Core, Operators/Observables, Simulation | Extend existing state-space and dynamics boundaries for mixed dimensions and bounded time-dependent controls; connect the existing boson algebra to execution; two different physical models share a solver and result path, with independent references, ordering, truncation, convergence, and gradient checks. |
| M2: shared developer workflow | 3-4 weeks | Integration, Runtime, Core | Reuse ExecutionOptions, output requests, parameter binding, planning, and PyTorch lifecycle; demonstrate two models using the same workflow without a second trainer or backend registry; explicitly validate result axes, gradient availability, serialization, and exact-plan execution. |
| M3: domain-native complete scenarios | 6-10 weeks | Domain implementation owners under existing boundaries | Deliver the reference scenarios below with input, validation, planning, execution, outputs, diagnostics, and examples; domain additions reuse the common path and carry their own support limits. |
| M4: performance and device validation | Continuous after correctness gates | Simulation, Compute, Runtime, Remote, Twin | Measure acceleration, batching, approximation, calibration, or hardware execution against fixed accuracy requirements; distinguish throughput from capacity and simulation from hardware evidence. |

The M0 baseline's failing Python 3.10/3.11 jobs reproduced two failures in
`tests/unit/test_algorithms_chemistry_integrals.py`: ordinary float summation
lost precision in a contracted overlap and in its independent closed-form
reference. The repair uses `math.fsum` in those accumulations, retains the
existing accuracy and rotation assertions, and adds an analytic cancellation
regression over every primitive ordering. Reproduce the focused acceptance
with `python -m pytest tests/unit/test_algorithms_chemistry_integrals.py -q`
on each supported Python minor; the broader testing tiers still apply. A local
pass does not relabel the original failed CI run as successful.

M1 starts with static mixed-dimension CPU references before time-dependent
controls. Preserve CircuitIR 1.0 and the working qubit fast path. Reuse
`core/ir/`, `compiler/pass_manager.py`, `lindblad/`, and `simulation/lindblad*`;
do not introduce a parallel IR hierarchy, pass manager, or dynamics engine.
The existing boson algebra is a starting point, not a complete Fock backend.
Trainable coefficients and live tensor bindings require explicit validation;
serializing parameter values does not serialize an autograd graph.

The first M1 slice implements approved
[proposal 078](../development/API_CHANGE_PROPOSAL_078_MIXED_DIMENSION_STATE_SPACES.md):
internal product-space semantics under `core/ir/`, explicit `dimensions=` on
the existing Lindblad namespace, backward-readable unchanged qubit plans, and
static non-qubit plan schema 1.4. `examples/mixed_dimension_dynamics.py` runs
three-level leakage and spin-mode exchange through the same existing generator,
integrators, result, and PyTorch gradient path. The focused acceptance is
`tests/unit/test_lindblad_mixed_dimensions.py`, including analytic decay and
initial-density adjoints, basis order, live/JSON gradients, truncation, and
time-grid convergence. This records the static finite-product step; the later
coefficient, temporal and explicit-basis slices have their own evidence below.
Proposal 073 states its separate temporal contract against this baseline; the
owner approved it on 2026-10-20. The bounded piecewise implementation is described
below. Continuously varying controls remain outside that approved contract.

The next M1 slice connects existing `BosonOperator` values directly to the
same static CPU dynamics input normalization. Explicit `dimensions=` supplies
mode truncations; Hamiltonians, collapse operators, and the low-level observable
mapping reuse `to_matrix` and its existing byte ceiling. No algebra, numerical
engine, public export, or plan schema is added. Run
`python -m examples.boson_interference`; the focused acceptance is
`tests/unit/test_lindblad_boson_consumer.py`: independent two-photon interference
and loss laws, basis order and spectator modes, the exact two-excitation cutoff,
unchanged dense-plan identities, batching, gradients/adjoints, and early refusals.
Static operator term lists now carry live scalar coefficients outside the
unchanged canonical boson algebra. Existing `Parameter.bind` and
`ParameterExpression.bind` resolve named values; the existing normalization,
plans, generators, and gradients consume the bound request. No second binder,
new plan schema, or widened algebra coefficient field is required. Run
`python -m examples.train_boson_coupling`; acceptance in
`tests/unit/test_lindblad_operator_coefficients.py` includes two physical models,
analytic gradients, actual optimizer updates, complex conjugate terms, batching,
initial-density adjoints, unchanged dense-plan identities, and retained Pauli
matrix-free execution. This completes the static operator-coefficient consumer,
not symbolic coefficients inside the algebra. The next M1 time-control contract
is the updated [proposal 073](../development/API_CHANGE_PROPOSAL_073_EVOLUTION_SCHEDULE.md).
It specifies one binding per interval, existing full-space operator terms and Core
parameter binding, all three current integrators, and scheduled-only plan 1.5
while preserving static 1.3/1.4 plans. Live gradients, initial-density adjoints,
exact replay, early refusals, and independent numerical checks are implementation
gates. The owner approved the contract on 2026-10-20. The contract-first
`tests/unit/test_piecewise_control_contract.py` now runs unchanged physical
assertions against both a static-composition fake and the real scheduled public
entry point. `tests/unit/test_lindblad_piecewise_controls.py` checks numerical
references, gradients and actual optimizer updates, reverse adjoints, early
refusals, plan 1.5 replay and unchanged static identities. Run
`python -m examples.train_piecewise_drive` for the user workflow. The implementation
reuses one trajectory loop and the existing Core binder, Liouvillian and three
integrators. It adds no Schedule class, route package or numerical engine. This
closes the piecewise-constant control slice; arbitrary callables/interpolation,
time-dependent collapse, time-boundary gradients and hardware pulses remain
unsupported. Local Krylov defects and global numerical error have separate
acceptance bounds. Implementation [#647](https://github.com/flagos-ai/FlagQuantum/pull/647)
merged as `4f9d5a6c010b4b351579bfe1e95053ba58aa09b6` after all 36 checks
and both full workflows succeeded on its latest head.

The next bounded M1 slice implements
[proposal 079](../development/API_CHANGE_PROPOSAL_079_EXPLICIT_BASIS_SPACES.md):
ordered explicit bases on the existing Lindblad path. The owner approved it on
2026-10-20. #649 merged its review; #651 merged its replaceable integration
baseline as `22b70b654eb973e7c59f39a13a993be50e91b8a2` after all 36 checks
and both full workflows passed. Core's internal ExplicitBasisSpace and the real
labelled public runner now replace their test-only factories through unchanged
consumer assertions. ProductStateSpace and CircuitIR 1.0 retain their behavior.

Existing entries accept basis_labels, exclusive with dimensions/n_qubits.
Labelled static and piecewise plans use 1.6 and keep ordered labels without
inventing tensor factors or a wire count; product plans 1.3/1.4/1.5 remain
unchanged. The request resolver, numeric validation, Core binding, Liouvillian,
three integrators, shared trajectory, result and initial-density adjoint are
reused. No state-space registry, solver, namespace or physical-model builder is
introduced. Run `python -m examples.constrained_basis_reference` for photon-sector
gradients and a declared fixed-charge Ising convention with retained labels.

`tests/unit/test_explicit_basis_contract.py` runs fake/Core and encoded/labelled
replacement consumers, including scheduled batches and a real optimizer update.
`tests/unit/test_constrained_basis_reference.py` retains
independent product-space physical references and the leakage projection
counterexample. `tests/unit/test_lindblad_explicit_basis.py` adds strict plan
replay/refusals, backward compatibility, a vacuum-plus-one-photon loss model,
control gradients and independent discrete-adjoint checks.
Default Krylov graph conditioning limits remain disclosed; labels do not certify
orthonormality, charges, fusion trees or invariance. This completes the labelled
matrix-space implementation slice subject to repository/CI acceptance, and does
not close the M3 native anyon-program gate.

An internal state-space proposal must be tested against both tensor-product
spaces and an anyon fusion-space example with fixed total charge. Product
dimensions alone do not describe all constrained spaces. Introduce only the
representation required by a verified consumer; do not prebuild a universal
type system. Domain-only model construction can start in its current owning
domain. A new `physics/` package requires a separate boundary decision and
architecture-policy change justified by multiple consumers; this roadmap
creates neither it nor empty packages for each physical route.

### M2 matrix expectation workflow

[Proposal 080](../development/API_CHANGE_PROPOSAL_080_MATRIX_EXPECTATION_OUTPUTS.md)
was approved on 2026-10-21. Its replaceable contract baseline landed in #669;
#671 supplies Simulation-owned matrix snapshots and output-name validation.
The Core implementation reuses `fq.expectation`, `OutputRequest`, and the existing
CPU Lindblad plan/result path. A full-space numeric tensor follows the declared
product or explicit basis; the Simulation boundary owns dimension, finiteness,
and Hermiticity checks. CircuitIR 1.0 and Pauli outputs remain unchanged.

Acceptance for this slice requires unchanged contract consumers to run against
the real implementation, tensor-identity request equality/hash behavior, matrix
snapshots on plans, exact-plan replay, named results and control gradients.
Circuit, receipt, resident and QPU front doors must refuse dense outputs before
execution or submission; Remote-owned preparation is reviewed separately in
#678. Run `python -m examples.matrix_expectation_dynamics` for a three-level
leakage expectation and its coupling derivative. The example is a CPU dynamics
workflow, not circuit/GPU/QPU dense-output support or completion of all M2 gates.
Each implementation PR still requires its own merge authorization and complete
applicable verification.

The next bounded M2 acceptance composes these existing APIs across a three-level
model and a spin-mode model. Run `python -m examples.shared_dynamics_workflow`
for named matrix outputs, retained time/batch axes, PyTorch optimizer updates and
detached numeric JSON replay. The [shared workflow review](../development/M2_SHARED_WORKFLOW_ACCEPTANCE.md)
records its independent references and refusal/compatibility checks. This is a
CPU workflow acceptance slice; domain-native motion and loss retain their M3
contract and implementation gates.

Proposal 081 was explicitly approved on 2026-10-22 for the native Rydberg
transport profile. The [authorization and baseline record](../api-changes/FQ-RYDBERG-TRANSPORT-PROGRAMS-20261022.md)
tracks the replaceable fixed-reference consumers and domain deliveries. Core
construction and Deployment refusal have landed; Simulation #707 merged native
program consumption and source-aware plan 1.7 after full applicable CI/GPU
verification. It replaced the three numerical fixtures with real program inputs.
The fixed-reference baseline alone was not
native transport support. Event instruments, feedback, motional heating, hardware and
the complete M3 scenario remain open gates. Existing plan compatibility fixtures
and scientific budgets remain unchanged; each implementation PR has separate
merge and verification requirements.

[Proposal 082](../development/API_CHANGE_PROPOSAL_082_REMOTE_PROVIDER_BOUNDARY.md)
records the requested service-owned Remote boundary correction and exact generic
NeutralAtomProvider retirement. Its four decisions, including replacement of 081's Remote-consumer gate, were
explicitly approved on 2026-10-22. Remote retirement #705 has landed; Core #700
and Deployment #701 are also on main. The merged Simulation delivery is bounded
to one/two-actor CPU dynamics, live control gradients and strict native replay.
Native control training #714 and commanded-path/schedule visualization #715 also
merged after their separate authorizations and full CI/GPU success. They reuse
the same native program, planner, integrator and PyTorch workflow; the view does
not infer atom presence or physical trap activity. The
[delivery record](../api-changes/FQ-RYDBERG-TRANSPORT-PROGRAMS-20261022.md#merged-bounded-deliveries)
links their final-head and post-merge evidence. No concrete cloud service,
detector events, feedback, motional heating or full M3 completion follows.

[Proposal 083](../development/API_CHANGE_PROPOSAL_083_TERMINAL_INSTRUMENT_SAMPLING.md)
was explicitly approved on 2026-10-23 for staged terminal instrument sampling.
Its document merge #716 was not that approval. The replaceable baseline #726
and separate Runtime export/metadata admission #730 are on main with their
full applicable CI/GPU and post-merge evidence. The Runtime delivery binds the
unchanged two-model consumers to sample_terminal, reusing MeasurementResult,
SuperOperator, Lindblad density checks, Core encoding and the shared private
probability draw. It retains existing plan/IR versions and qubit sampling
behavior. This CPU terminal slice does not provide a timestamped detector
trajectory, feedback or erasure-decoder record. The
[delivery record](../api-changes/FQ-TERMINAL-INSTRUMENT-SAMPLING-20261023.md)
retains independent merge and validation gates for the implementation.

The native control-training example now composes the delivered primitives:
analytic PyTorch optimization, detached native plan 1.7 replay, then explicit
terminal a/b presence-instrument samples. Run
`python -m examples.train_rydberg_controls --steps 40 --intervals 128 --shots 4096`.
Its observed probabilities are checked against independent exponential loss;
shot-frequency uncertainty is separate from solver error. Physical survival,
observed no-flag selection and unconditional RR remain distinct. This closes a
user-workflow composition gap without a new API or an event/feedback claim.

The existing two-mode boson-interference example also feeds its replayed density
into the same Runtime number-instrument sampler. Run
`python -m examples.boson_interference` for balanced two-photon bunching, uniform
loss and unconditional QND count outcomes including vacuum and one photon.
Independent closed-form density matrices and a three-/four-level embedding
check coherence, loss and truncation closure; finite-shot uncertainty remains a
separate check. This supplies the bounded photonic reference scenario below
through existing shared capabilities, without promoting a Gaussian backend or
native optical-program frontend.

[Proposal 085](../development/API_CHANGE_PROPOSAL_085_ISING_BRAID_PROGRAMS.md)
was explicitly approved on 2026-10-23 for baseline-first staged implementation.
It specifies a fixed-charge Ising braid frontend on the same
Core ProgramRecord, exact existing operator actions, shared root planning and
results, and explicit terminal QND charge instruments. Its native-only plan
admission must preserve old circuit and Lindblad plans without reporting fusion
coordinates as qubits or braid ordinals as physical time. The runnable
current-API matrix probe establishes reuse only, not native support. The
replaceable baseline must precede domain implementations, and each later PR
retains its own merge and complete-verification gates.

The replaceable baseline landed in #761. The Core source delivery adds the
experimental IsingBraidProgram, immutable four-sigma context/exchanges and
native-source encoding 1.1 through existing ProgramRecord authorities. It
preserves Rydberg source 1.0 and old plan bytes. Run
`python -m examples.ising_braid_source` for source snapshots, fusion coordinates
and shared matrix outputs. The subsequent Simulation and Runtime deliveries add
finite F/R actions through existing SuperOperator algebra and the same root
plan/run, ExecutionPlan and ExecutionResult. Native-only plan 1.1 seals Core
source 1.1 and ordered matrix outputs, with null circuit analysis and unchanged
old circuit/Lindblad payloads. The four baseline factories now use the real
source/planner/executor/reader while their independent physical assertions stay
fixed. `python -m examples.ising_braid_execution` exercises exact JSON replay and
explicit terminal pair-charge QND observations. Full applicable remote gates
remain required before each authorized merge. This is only the fixed four-sigma
vacuum CPU slice; it does not certify full M3, other anyon sectors, physical
hardware, gradients or universal anyon computation.

[Proposal 089](../development/API_CHANGE_PROPOSAL_089_ISING_UNREAD_OBSERVATION.md)
now adds one ideal unread observation of the current first pair between native
Ising exchanges. Baseline #798, source #804, Simulation #807 and Runtime #809
reuse ProgramRecord, existing Kraus actions and root plan/run/results. Only the
new operation uses native source/plan 1.2; braid-only 1.1 and all old captures
remain unchanged. The four replaceable factories now use the real path while
independent physical assertions and 1e-12/rtol=0 budgets stay fixed. The
[delivery record](../api-changes/FQ-ISING-UNREAD-OBSERVATION-20261024.md) runs an
exchange/channel/inverse program with P(psi)=0.5 and exact JSON replay. It
produces a mixed density without selecting an intermediate outcome or creating
an event/feedback record. The separately approved sampled slice below selects one branch; intermediate
atom-presence instruments and hardware feedback retain separate contract gates.

[Proposal 091](../development/API_CHANGE_PROPOSAL_091_ISING_SAMPLED_OBSERVATION.md)
was approved before baseline #821, Core #824, Simulation #827 and Runtime #838.
One `measure_pair()` reuses the native IR, existing Kraus continuation, terminal
probability/private-draw authority and canonical results. Explicit `shots=1`
and seed admit a CPU conditional trajectory with native source/plan 1.3 and a
PyTorch-version-bound RNG environment; braid-only 1.1, unread 1.2 and other old
artifacts retain exact bytes. Four baseline factories now use the real path,
with unchanged independent F/R consumers. The
[M3 acceptance review](../development/M3_SCENARIO_ACCEPTANCE.md#current-api-user-workflow)
contains runnable JSON replay and the expected `1.3 / True / True / True` output.
This fixed four-sigma vacuum profile adds no multi-shot native batch, classical
register, feedback, detector event time, other sector or hardware support.

### Reference scenarios

The [M3 scenario acceptance review](../development/M3_SCENARIO_ACCEPTANCE.md)
links executable witnesses, shared boundaries and separate numerical,
truncation, gradient and sampling budgets. It includes the merged driven
three-level (#787), thermal spin-mode (#782) and native fusion-observation
(#778), unread continuation (#809) and single sampled continuation (#838)
slices. These are bounded CPU references, not full M3/M4 or hardware
certification. The separately approved proposal 092 now adds one intermediate
atom-presence instrument on the existing native Rydberg schedule and a single
seeded CPU conditional continuation. Core #869, Simulation #875 and Runtime
[#879](https://github.com/flagos-ai/FlagQuantum/pull/879) reuse existing IR,
full-grid binding, Liouvillian, integrators, private sampling and canonical result
authorities. The [accepted trajectory evidence and public workflow](../api-changes/FQ-RYDBERG-PRESENCE-TRAJECTORY-20261025.md)
retain atom identity, distinguish physical loss from no-flag detection, and keep
post-observation and forward time rows explicit. Multi-event trajectories,
feedback, heating, calibration and device execution remain separate decisions;
this inventory approves no additional protected interface or M4 claim.

The user approved the bounded
[driven three-level relaxation contract](../development/M3_NEXT_SCENARIO_CONTRACT_REVIEW.md)
on 2026-10-26. Implementation [#886](https://github.com/flagos-ai/FlagQuantum/pull/886)
merged as `05bca74d08ac905f4ccdbb03235f2a478b2f89b7` under its independent human
grant, after complete CI `40269085812`, GPU `40269085833` and the remote Python
3.10/3.11/3.12 matrix succeeded at head
`2bb50f2fd54a5315bf80772569b0527bcef91488`. The example and independent
integration acceptance reuse existing numeric Lindblad planning, fixed collapse
rates, interval bindings, autograd and results; public APIs and library
production code are unchanged. Local default verification passed 16,017 cases
(224 skipped) and integration passed 2,515 (177 skipped), both with zero XML
failures/errors. Main smoke `40271009760` succeeded; this is only a smoke check.
The accepted slice covers a finite phenomenological cascade, not calibrated
hardware, gate fidelity, higher-level truncation or all M3/M4 gates. The existing
closed-model reference remains unchanged.

The separately approved
[fixed local thermal bath for the spin-mode CPU model](../development/M3_SPIN_MODE_THERMAL_BATH_CONTRACT_REVIEW.md)
is accepted through [#897](https://github.com/flagos-ai/FlagQuantum/pull/897).
It composes existing boson operators, mixed-dimension Lindblad planning, fixed
upward/downward jumps, gradients and detached numeric replay. Independent
acceptance was frozen before the example; the original lossless reference,
library production code and public APIs remain unchanged. Latest complete CI
`40277142842`, GPU `40277142859` and the remote Python 3.10/3.11/3.12 matrix
succeeded at `bf9afadc8cca81b5f0d7ae48462f32e2df7e71a1`. The actual external
merge is `5ee74364cdbab17d1f0bebbee253bf400b467800`; Main smoke `40279655585`
succeeded, as a smoke check only. Local default verification passed 16,049
cases/224 skipped and integration passed 2,539/177 skipped with zero XML
failures/errors at code-identical `ed557965c5d1dae93a862b77495f999503240257`.
The finite blocked-top model, bath-only analytic bound and coupled cutoff
stability remain explicit. This does not establish calibrated ion-trap heating,
a global Gibbs state, infinite-dimensional accuracy or full M3/M4 completion.

The user-approved
[absorptive terminal photon-counting slice](../development/M3_ABSORPTIVE_PHOTON_COUNTING_CONTRACT_REVIEW.md)
is accepted through [#921](https://github.com/flagos-ai/FlagQuantum/pull/921).
It composes the existing finite-Fock evolution, square vacuum-reset Kraus actions
and terminal sampler; the QND reference and library/public API remain unchanged.
At head `84f15a6a99e7a36ab993989d1d6ae94e8c10ec74`, complete CI `40295642672`
(attempt 3, all 54 jobs), GPU `40295642686` (attempt 2) and the remote three-Python
matrix succeeded. All 60 checks completed with 57 success and three original
manual-pool skips. Its independent human grant preceded normal merge
`d15d681b88d6062e02b97317e53fbea5bafc7105`; exact-merge main smoke `40316383759`
succeeded, as smoke only. Local default passed 16,148/224 skipped and integration
2,620/177 skipped, with zero XML failures/errors and identical tested code blobs.
Network interruptions and the separately authorized stalled-checkout recovery
are recorded in [M3 acceptance](../development/M3_SCENARIO_ACCEPTANCE.md#accepted-absorptive-terminal-counting-scenario).
No physics budget, audit or CI protection was weakened.

The [M3 exit-gate audit](../development/M3_EXIT_GATE_AUDIT.md) now maps the six
bounded reference profiles to input, admission, planning, execution, outputs,
diagnostics and examples. It is an evidence inventory for review, not a milestone
promotion. Candidate extensions and M4 device/performance requirements are listed
separately; no new scenario implementation or protected contract is approved.

| Scenario | Shared capability under test | Additional acceptance |
| --- | --- | --- |
| Existing variational circuit training | CircuitIR, Module, planning, gradients | Preserve current APIs and the local CPU fast path. |
| Driven three-level superconducting model | Mixed dimensions, controls, dynamics | Validate populations, leakage, and time-grid convergence against an independent reference. |
| Moving neutral-atom pair | Geometry, time dependence, noise and events | Separate atom/trap identity and physical loss/detected loss; report unconditional outcomes, survival, and postselection. |
| Spin coupled to a motional mode | Mixed dimensions and boson operators | Validate basis ordering, truncation convergence, and shared evolution with the superconducting example. |
| Two-mode photonic interference | Fock states, optical operations, counting and loss | Validate interference and number statistics; do not imply Gaussian-backend support from a truncated Fock example. |
| Small anyon braiding and fusion measurement | Constrained spaces, sectors, basis changes, sampling | Check braid relations, unitarity, total charge, and measurement probabilities; Ising braiding alone is not a universal gate-set claim. |

The topological scenario participates in M1 design review even when its full
execution arrives in M3. Majorana parity measurement and conditional feedback
are a subsequent bounded scenario using fermionic and measurement primitives;
they are not evidence of a physical Majorana device or a complete microscopic
model. QEC owns codes and decoders, not all topological-program semantics.

### Work packaging and verification

1. Repair the reproduced CPU regression; retain the independent physics oracle
   and its accuracy bound. Run the focused tests on Python 3.10, 3.11, and 3.12.
2. Reconcile current implementation, capability fields, parity gaps, and their
   generated views without promoting maturity or claiming unmeasured support.
3. Review a narrow state-space and parameter-binding contract change before
   dependent implementations. Include migration/refusal behavior and a
   contract fake or replacement test under the existing boundary.
4. Deliver static mixed-dimension evolution, then the boson consumer, then
   time-dependent controls. Each change extends a complete CPU path.
5. Prove reuse with two physical models before expanding domain packages.
6. Add neutral-atom motion/loss, then the remaining bounded scenarios, each with
   a local ten-minute example and independent scientific checks.

Every implementation task names its user outcome, owning domain, existing
components reused, unsupported scope, and reproducible validation. Integration
owns shared/protected contract changes; the owning domain implements after that
baseline lands. Parallel work must not share an unproven contract.

Use the existing [testing tiers](../development/TESTING.md), architecture and
dependency checks, public-API protection, capability checker, and generated-doc
checks. Report solver error, Hilbert-space truncation, statistical uncertainty,
and gradient error separately. No single global tolerance establishes them all.
Performance claims require matched accuracy and environment evidence.

The FlagOS, distributed-training, and deployment sections below remain existing
workstreams. They do not require each new physical model to invent its own
execution or resource-management layer.

## Next: a unified quantum AI framework

The next release focuses on making the current product model coherent and easy
to adopt.

Planned capabilities:

- stable `fq.Circuit`, `fq.Module`, `fq.run`, `fq.train`, and `fq.plan`
  interfaces;
- one versioned FlagQuantum IR shared by execution, compilation, and
  deployment;
- statevector, MPS, and tensor-network execution selected from the same
  program;
- PyTorch-native gradients and optimizer workflows across supported modes;
- explainable runtime planning with explicit blockers and no silent fallback;
- reproducible installation, examples, tests, and migration guidance.

The result should be a reliable local development experience on CPU or one GPU,
with advanced execution modes clearly labeled by maturity.

## Then: the FlagOS unified multi-chip backend

The following stage connects FlagQuantum to a unified FlagOS execution layer
instead of exposing accelerator-specific behavior to users.

Planned capabilities:

- a stable, Torch-FL-backed FlagOS provider integration with no core-package
  hard dependency;
- unified device, stream, memory, kernel, and collective contracts;
- optimized contraction, complex linear algebra, QR, and SVD kernels for MPS
  training;
- consistent execution across NVIDIA and FlagOS-supported accelerators;
- automatic detection of unsupported operations and hidden CPU fallback;
- reproducible cross-accelerator correctness and performance evidence.

Users should be able to move from local development to a supported accelerator
without changing the quantum AI model.

## Scale: train workloads that do not fit on one device

FlagQuantum will extend the same programming model to genuinely sharded
training.

Planned capabilities:

- sharded statevector forward and backward execution;
- rank-owned MPS forward, backward, and optimizer state;
- tensor-network slicing and reduction with differentiable reverse
  contraction;
- topology-aware placement and communication through FlagCX;
- distributed checkpointing, recovery, and failure diagnostics;
- runtime evidence that distinguishes sharded capacity from replicated
  throughput.

This stage is complete only when one logical workload exceeds a single-device
capacity limit and trains successfully across devices. Replicated execution
will not be presented as capacity scaling.

## Deploy: connect training to quantum hardware

The final product loop carries a trained parameterized program from simulation
to a provider or QPU without leaving FlagQuantum.

Planned capabilities:

- bind trained parameters into a versioned deployment artifact;
- compile and validate programs against target-device constraints;
- submit, monitor, and retrieve provider jobs through a common interface;
- preserve shots, calibration data, routing, and result provenance;
- compare ideal, noisy, and hardware results through one result model;
- support hardware-in-the-loop evaluation and future training workflows.

Provider and QPU support will remain target-specific until it has been
validated on real hardware.

## What users can expect

| Stage | User-visible outcome |
| --- | --- |
| Unified framework | One API for circuits, training, planning, and execution |
| FlagOS backend | The same model runs across supported accelerator stacks |
| Distributed training | A single oversized workload trains across chips |
| Quantum deployment | A trained program reaches a QPU with traceable results |

## Release standard

A planned capability moves into supported documentation only when:

- its public interface and compatibility boundary are defined;
- forward execution, gradients, and optimizer behavior pass the relevant
  correctness tests;
- supported hardware paths have reproducible evidence;
- fallback, limitations, and distribution semantics are explicit;
- installation and an end-to-end example are verified.

Until then, the capability remains experimental, development evidence, or
future intent.
