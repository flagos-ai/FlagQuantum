# M3 bounded-scenario exit-gate audit

Status: **documentation and evidence review; no new implementation or milestone promotion**.
The user authorized reconciliation of #921 and an audit of the six existing M3
reference scenarios. This review inspects main
`65fca3f7891848eaf4b97260c0b11d0ed9a92f12` on 2026-10-26.
It changes no API, schema, capability maturity, scientific assertion or CI gate.

The governing [M3 exit gate](../roadmap/FLAGOS_ALIGNED_RELEASE_TRAIN.md#milestones-and-acceptance)
requires domain scenarios with input, validation, planning, execution, outputs,
diagnostics and examples, through existing owners with explicit support limits.
The [scenario acceptance record](M3_SCENARIO_ACCEPTANCE.md) supplies the accepted
PRs, measured budgets and merge provenance. A mapped witness below is a source
and accepted-evidence finding, not a new test execution or certification of every
input, device or route on current main. Main smoke remains smoke only.

## Findings

- All six named bounded CPU profiles have executable input-to-result examples
  and independent physical or workflow acceptance witnesses.
- #921's accepted absorptive reference was still marked pending in the roadmap,
  scenario inventory and original contract. This change reconciles those three
  records, including its own merge authorization and exact-merge main smoke.
- Admission evidence has two authorities: native source verifiers and shared
  numeric/circuit/runtime admission. The existing evidence does not justify a
  blanket claim that every example helper's invalid-input path has been tested.
- A consolidated, explicitly scoped M3 exit decision is still a review item.
  Hardware, calibration and acceleration evidence belongs to M4; absence of it
  does not invalidate an accepted bounded CPU scenario.
- Majorana parity readout remains an extension candidate. The existing fermion
  algebra and Ising measurement routes must be compared before proposing it as
  another implementation. This audit approves neither that contract nor code.

## Seven-stage coverage within the named profiles

| M3 stage | Mapped evidence | Limit of this finding |
| --- | --- | --- |
| Input | Circuit templates, ordered numeric control matrices, native atom rosters/paths, finite oscillator/Fock initial states and fixed-sector Ising programs in the examples below. | Only the stated dimensions, sectors and physical models; no arbitrary technology frontend. |
| Validation | Core/circuit admission, numeric Lindblad admission, terminal-instrument checks and native Rydberg/Ising verifiers. The qutrit relaxation consumer explicitly rejects invalid rates. | Shared admission is reusable evidence, not an exhaustive per-recipe negative-path proof; see E1. |
| Planning | Existing `fq.plan`/ExecutionPlan for circuits and native Ising; `fql.plan`/LindbladPlan for numeric dynamics. Rydberg presence uses its accepted sealed 1.8 route. | Entry points and source/plan versions differ; this is not universal root admission. |
| Execution | Existing CPU circuit, dynamics and native conditional-continuation owners, with accepted local/full PR verification linked by the scenario record. | Historical successful heads are identified there; this documentation audit does not rerun all scientific layers on current main. |
| Outputs | Expectations, complete complex densities, populations, explicit axes and sampling indices checked by the named consumers. | Terminal sampling returns indices; example-side conditional analysis does not become a new result field. |
| Diagnostics | Independent density/coherence, gradient, refinement, truncation, statistical, selection-denominator and replay witnesses. Native refusal/compatibility tests retain old bytes. | Each budget certifies its own quantity. No global tolerance, physical calibration or performance conclusion follows. |
| Examples | The six runnable commands below, with existing public/domain interfaces and no credentials required. | A numerical reference is not a submitted provider job, device frontend or hardware experiment. |

## Evidence by scenario

### Circuit training and deployment packaging

Run `python -m examples.train_parameterized_circuit_then_deploy`.
The existing template/optimizer, `fq.plan/run`, exported package and in-memory
provider form one user workflow. The
[golden-path consumers](../../tests/integration/test_train_then_deploy_golden_path.py)
check training, physical ground-state agreement, exported angles and agreement
between the trained and packaged circuit. The
[error-metric regressions](../../tests/unit/test_train_deploy_error_metrics.py)
keep variational error separate from finite-shot uncertainty.
[Execution options](../../tests/unit/test_execution_options.py) and the existing
Core/planner checks own shared admission. The provider is local contract
execution; an external submitted task or QPU measurement remains M4 evidence.

### Driven three-level model

Run `python -m examples.driven_three_level_relaxation`.
Numeric initial states, four ordered control intervals and fixed `2 -> 1 -> 0`
jumps reach the existing `fql.plan/run` path and EvolutionResult. The
[closed-model consumers](../../tests/integration/test_driven_three_level.py)
and [relaxation consumers](../../tests/integration/test_driven_three_level_relaxation.py)
cover full complex densities, independent generator/analytic cascade checks,
all eight control derivatives, pulse order, refinement, optimizer updates and
numeric replay. `test_existing_rate_admission` checks invalid collapse rates;
[mixed-dimension admission](../../tests/unit/test_lindblad_mixed_dimensions.py)
and [piecewise controls](../../tests/unit/test_lindblad_piecewise_controls.py)
are shared authorities. Heating, calibrated pulses, higher-level physical
truncation and device gate fidelity are outside this accepted profile.

### Moving neutral-atom pair

Run `python -m examples.train_rydberg_controls --steps 40 --intervals 128 --shots 4096`.
The native roster, commanded transport, controls and declared loss reuse source
validation and full-schedule/grid binding. The
[native dynamics](../../tests/unit/test_native_rydberg_dynamics.py),
[training/selection consumers](../../tests/integration/test_rydberg_control_training.py),
[presence consumers](../../tests/unit/test_rydberg_presence_contract.py) and
[sealed trajectory workflow](../../tests/integration/test_rydberg_presence_execution.py)
cover deterministic evolution, conditional densities, physical loss versus
no-flag detection, gradients where admitted, source-boundary mapping and replay.
The sampled presence profile admits one CPU complex128 trajectory with explicit
shots/seed and its exact RNG environment; it does not admit feedback, multiple
observations, gradients through sampling or inferred physical event times.
Calibration and actual transport/heating measurements remain M4 work.

### Spin coupled to a mode

Run `python -m examples.spin_mode_thermal_bath`.
Conditioned finite thermal inputs, exchange coupling and fixed upward/downward
jumps use existing boson algebra and numeric Lindblad planning. The
[lossless reference](../../tests/integration/test_spin_mode_convergence.py) and
[fixed-bath consumers](../../tests/integration/test_spin_mode_thermal_bath.py)
cover full scalar-reference densities, coherences, finite moment corrections,
bath-only analytic bounds, cutoff stability, gradients, fitting and replay.
[Shared boson admission](../../tests/unit/test_lindblad_boson_consumer.py) and
[parameter/gradient admission](../../tests/unit/test_lindblad_batching_and_gradients.py)
retain their own scope. The finite blocked top, conditioned initial thermal
state, integration error and coupled cutoff stability are separate statements.
No infinite interacting Gibbs state or calibrated heating model is certified.

### Two-mode photonic interference and counting

Run `python -m examples.absorptive_photon_counting`.
The accepted QND reference and absorptive instrument share finite evolution,
terminal probability/private-draw authority and SuperOperator actions. The
[QND/HOM consumers](../../tests/integration/test_boson_interference.py) and
[absorptive consumers](../../tests/integration/test_absorptive_photon_counting.py)
check full complex branches and coherences, count laws, original-trace
normalization, unread recovery, mode ordering, statistics, private RNG and replay.
[Terminal admission](../../tests/unit/test_terminal_instrument_admission.py)
owns dtype, shape, completeness and execution restrictions. Cutoff 3/4 embedding
is sector-closed evidence for the declared input, not general infinite-Fock
convergence. Vacuum reset retains the original joint coordinates, even for
missed counts. Detector timing, calibration, feedback and Gaussian execution
remain outside the profile. #921's full CI/GPU and exact main smoke are recorded
in [its accepted delivery](M3_SCENARIO_ACCEPTANCE.md#accepted-absorptive-terminal-counting-scenario).

### Small anyon braiding and fusion

Run `python -m examples.ising_fusion_measurement`.
The existing native fixed four-sigma/vacuum source has discrete exchanges and
zero/one observation; it does not need an invented Hamiltonian or clock.
[Independent F/R consumers](../../tests/unit/test_ising_braid_contract.py),
[fusion acceptance](../../tests/integration/test_ising_fusion_measurement.py),
[unread consumers](../../tests/unit/test_ising_unread_contract.py) and
[sampled consumers](../../tests/unit/test_ising_sampled_contract.py)
cover full complex states, noncommuting continuation, weighted/unread recovery,
physical probabilities and explicit selection. Native
[admission](../../tests/unit/test_ising_sampled_runtime.py),
[old-byte compatibility](../../tests/unit/test_ising_unread_compatibility.py)
and [sampled replay](../../tests/integration/test_ising_sampled_execution.py)
retain their respective versions and refusal boundaries. Other sectors,
multi-shot native batches, continuous braid gradients, universal braiding and
Majorana-device claims are not supplied by this finite fusion-space example.

## Remaining work, with its actual gate

| ID | Remaining work | Classification and next evidence | Owner |
| --- | --- | --- | --- |
| D1 | Reconcile #921's accepted status in the three living records. | Included in this documentation change; its own review/delivery still applies. No scientific rerun is needed to rewrite an already verified merge result. | Integration |
| E1 | Make the scenario-level refusal/diagnostic boundary review explicit. | M3 evidence follow-up: map a small set of physically invalid user requests to existing shared/native refusals, identify any uncovered golden-path negative case, and freeze expected behavior before adding tests. Shared admission already exists; this is not a known production failure or a demand to test every helper branch. | Simulation/Runtime with Integration review |
| C1 | Decide the scope of a consolidated M3 exit claim. | Review the six named bounded CPU profiles and E1; explicitly distinguish accepted support from later extensions. This audit does not promote a milestone or change capability maturity. | Maintainer/Integration |
| X1 | Evaluate finite Majorana-pair parity readout. | Separate contract candidate: establish a distinct user outcome beyond existing Ising/QND sampling, then review representation, Hamiltonian/map, allowed states, independent oracle and refusals. No implementation is authorized here. | Observables/Simulation/Runtime |
| X2 | Feedback, multiple intermediate observations, more native sectors or new frontends. | Independently reviewed support extensions; do not silently make all of them prerequisites for the already accepted bounded references. | Existing domain owners |
| M4 | Calibrated devices/heating, external provider submissions, acceleration/performance or distributed capacity. | Fixed-accuracy and actual resource/device evidence under M4, with explicit execution semantics. CPU correctness and in-memory providers cannot certify these claims. | Simulation/Compute/Remote/Twin |

## Next smallest review package

Prepare E1's compact refusal/diagnostic evidence list and the bounded M3 closure
scope for human review. Examples of questions to resolve are invalid qutrit rates
or control shapes, illegal spin-mode dimensions/time subdivisions, and malformed
or unsupported terminal instruments. Reuse the existing verifier and exception
authorities; do not change protected errors or snapshots to make a new test pass.
Only add an integration witness when it proves a missing user-visible failure
path, rather than mirroring a helper's implementation. No such tests or library
changes are made by this audit.

Majorana parity readout stays a later candidate until its distinction is clear.
The [existing fermion algebra](../../flagquantum/observables/fermion.py) already
provides Jordan-Wigner and parity-encoding images. Its
[independent consumers](../../tests/unit/test_observables_fermion.py) compare
canonical anticommutation, finite Fock matrices and expectations in both bases.
Those algebra witnesses do not alone establish a complete measurement workflow.
A future review must decide mode order, total-parity sector and allowed initial
coherences, distinguish Fock-space parity from the existing fixed fusion sector,
and state which existing planning/measurement route actually accepts it.
It must include equations and an independent oracle before implementation.

## Verification of this audit

This change edits documentation only. Check ownership, local links/anchors,
repository language, Markdown hygiene and the diff; retain the historical
scientific results at their actual tested heads. Do not manufacture a new full
scientific acceptance claim by quoting main smoke or rerunning tests merely to
change pending prose. Any documentation PR's own CI/GPU and merge decision remain
separate from #921's already accepted implementation evidence.
