# M3 scenario acceptance and remaining boundaries

This review inventories the bounded CPU paths required by the
[shared roadmap](../roadmap/FLAGOS_ALIGNED_RELEASE_TRAIN.md#reference-scenarios).
It changes no API, schema, capability declaration, maturity level or approval.
An accepted reference scenario is not certification of a hardware technology or
of all M3/M4 exit gates. The original inventory inspected main
`4be2375eff47c112ce8d8e24f8206ed5592130d8`. This update inspects main
`65fca3f7891848eaf4b97260c0b11d0ed9a92f12` on 2026-10-26, including
approved 089/091 implementations and the accepted 092 Core, Simulation and
Runtime stages and the independently accepted three-level relaxation and
fixed thermal-bath spin-mode and absorptive photon-counting scenarios.
Historical design reviews and approval records remain historical.

## Executable evidence by user scenario

| Scenario | Runnable example (`python -m examples.` prefix) | Independent acceptance and merge evidence | Remaining boundary |
| --- | --- | --- | --- |
| Circuit training and deployment packaging | `train_parameterized_circuit_then_deploy` | [Golden path](../../tests/integration/test_train_then_deploy_golden_path.py), [metrics/RNG regressions](../../tests/unit/test_train_deploy_error_metrics.py); [#771](https://github.com/flagos-ai/FlagQuantum/pull/771) separates variational error from finite-shot error and retains small exported angles. | The in-memory provider is local contract execution, not evidence of a submitted cloud job or a QPU measurement. |
| Driven three-level superconducting model | `driven_three_level`; `driven_three_level_relaxation` | [Full density, all eight control derivatives, pulse order and refinement](../../tests/integration/test_driven_three_level.py); [#787](https://github.com/flagos-ai/FlagQuantum/pull/787); [fixed cascade, analytic coherence, independent full density and training/replay](../../tests/integration/test_driven_three_level_relaxation.py); [#886](https://github.com/flagos-ai/FlagQuantum/pull/886). | Closed or fixed zero-temperature two-channel relaxation in a finite rotating-frame qutrit; no calibrated device, pulse emission, heating, gate-fidelity or higher-level truncation certification. |
| Moving neutral-atom pair | `train_rydberg_controls --steps 40 --intervals 128 --shots 4096`; `visualize_rydberg_transport` | [Native source/dynamics](../../tests/unit/test_native_rydberg_dynamics.py), [training, loss, explicit observed flags and conditioning](../../tests/integration/test_rydberg_control_training.py), [commanded-path view](../../tests/integration/test_rydberg_transport_visualization.py), [intermediate presence consumers](../../tests/unit/test_rydberg_presence_contract.py), [sealed trajectory workflow](../../tests/integration/test_rydberg_presence_execution.py); [#879](https://github.com/flagos-ai/FlagQuantum/pull/879); [#707](https://github.com/flagos-ai/FlagQuantum/pull/707)/[#714](https://github.com/flagos-ai/FlagQuantum/pull/714)/[#715](https://github.com/flagos-ai/FlagQuantum/pull/715)/[#738](https://github.com/flagos-ai/FlagQuantum/pull/738). | One/two-actor imposed paths and declared Markov loss; one seeded CPU presence observation; no feedback, inferred trap activity, motional heating, species calibration or hardware execution. |
| Spin coupled to a motional mode | `spin_mode_convergence`; `spin_mode_thermal_bath` | [Complex Rabi blocks, total excitation, finite-mode dark top, analytic gradient and replay](../../tests/integration/test_spin_mode_convergence.py); [#782](https://github.com/flagos-ai/FlagQuantum/pull/782); [fixed bath, scalar finite densities, analytic bounds, gradients and replay](../../tests/integration/test_spin_mode_thermal_bath.py); [#897](https://github.com/flagos-ai/FlagQuantum/pull/897). | Resonant rotating-wave exchange, lossless or with a fixed local oscillator bath and explicit finite thermal conditioning/top boundary; no ion-trap pulse frontend, calibrated heating, global Gibbs or infinite coupled-density certification. |
| Two-mode photonic interference | `boson_interference`; `absorptive_photon_counting` | [Closed attenuation/HOM density, cutoff embedding and QND counts](../../tests/integration/test_boson_interference.py); [#741](https://github.com/flagos-ai/FlagQuantum/pull/741); [absorptive full complex branches, missed counts, conditional/unread states and replay](../../tests/integration/test_absorptive_photon_counting.py); [#921](https://github.com/flagos-ai/FlagQuantum/pull/921). | Finite Fock QND or vacuum-reset counting reference; no Gaussian backend, optical-program frontend, detector timing/calibration or photonic hardware submission. |
| Small anyon braiding and fusion | `ising_braid_execution`; `ising_fusion_measurement` | [Unchanged physical contract consumers](../../tests/unit/test_ising_braid_contract.py), [native execution/replay](../../tests/integration/test_ising_braid_execution.py), [fusion coherence and QND workflow](../../tests/integration/test_ising_fusion_measurement.py); [#769](https://github.com/flagos-ai/FlagQuantum/pull/769)/[#778](https://github.com/flagos-ai/FlagQuantum/pull/778); [unread continuation/replay](../../tests/unit/test_ising_unread_contract.py), [native 1.2 admission](../../tests/unit/test_ising_unread_runtime.py), [old-byte compatibility](../../tests/unit/test_ising_unread_compatibility.py); [#809](https://github.com/flagos-ai/FlagQuantum/pull/809); [sampled conditional consumers](../../tests/unit/test_ising_sampled_contract.py), [native 1.3 admission/refusals](../../tests/unit/test_ising_sampled_runtime.py), [saved trajectory workflow](../../tests/integration/test_ising_sampled_execution.py); [#838](https://github.com/flagos-ai/FlagQuantum/pull/838). | Exactly four sigma anyons, total vacuum and fixed gauge; CPU ideal braids with zero/one unread channel or one explicitly seeded sampled first-pair QND observation. No multi-shot native batch, feedback, other sectors, continuous gradients, universal braid set or Majorana-device claim. |

The neutral-atom row now includes one declared intermediate presence observation
and conditional continuation in its bounded CPU profile. It does not establish
all roadmap event requirements, feedback or an observed hardware-loss record.
The other rows likewise establish their named references rather than unrestricted
route support. None is a distributed capacity or performance acceptance result.

The [M3 exit-gate audit](M3_EXIT_GATE_AUDIT.md) maps each named bounded profile
to the milestone stages and separates evidence gaps from later extensions/M4.

## What is actually shared

- Circuit programs and the native Ising program use `fq.plan/run`; analog and
  mixed-dimension models use `flagquantum.lindblad.plan/run`. The Rydberg
  presence profile is planned by `fql.plan` and executes only sealed 1.8 plans
  through `fq.run(plan)`. Do not claim that
  every source is accepted by every root entry point. Existing refusals matter.
- Controls reuse Core parameters and interval binding. Numeric matrices and
  physical sources reach their existing domain owners rather than a new
  technology registry. Ising exchange words are discrete and do not acquire
  control gradients merely because another route trains through PyTorch.
- Matrix requests reuse `fq.expectation` and named outputs. Analog evolution
  retains its existing evolution result; native Ising and the sampled Rydberg
  presence profile use the existing ExecutionResult/MeasurementResult.
  Shared vocabulary does not mean identical result shapes or result classes.
- Analog models reuse Liouvillian assembly and existing integrators. Ideal
  Ising braids reuse finite SuperOperator actions, not an invented Hamiltonian
  or an artificial time coordinate. Circuit execution keeps its native path.
- Terminal observations reuse `sample_terminal` and MeasurementResult. The
  native Ising unread operation reuses `SuperOperator.from_kraus` inside ordered
  evolution. It sums both charge branches without sampling or normalization;
  the terminal sampler remains a separate call returning indices, not
  conditional states, detector trajectories or feedback. Native sampled continuation
  instead uses the same probability/draw authorities and existing conditional
  ExecutionResult/MeasurementResult under the separately approved 091 contract.
- Saved numeric plans capture values and their supported semantics. Rebuild
  live plans after optimizer updates; JSON is not a saved training graph.
  Circuit, Lindblad and native plan versions remain their respective contracts.

No new framework, solver, binder, result envelope, registry, evidence schema or
capability catalog is needed to express this inventory.

## Accepted absorptive terminal counting scenario

The user approved all five decisions in the
[absorptive photon-counting contract](M3_ABSORPTIVE_PHOTON_COUNTING_CONTRACT_REVIEW.md).
The new `python -m examples.absorptive_photon_counting` composition uses existing
numeric evolution, terminal sampling and finite SuperOperator actions. It resets
the measured mode to vacuum in the same joint coordinates, retains all outcomes
including zero and missed counts, and sums unresolved incoming-number alternatives
incoherently. Independent scalar blocks distinguish this from the accepted QND
reference. No library production code, public API or result type changes.

Six scalar references and consumer assertions were captured before the example.
Five reference ASTs remain unchanged; the sixth received an explicit tensor
wrapper for the same complex witness assignment, verified bit-for-bit against
the frozen inputs. Consumers received only import sorting and a typed reduction
for the deliberate incorrect-map witness. No physical assertion or threshold
changed; the scenario table above now includes this accepted finite reference.

Complete local default verification passed 16,148 cases/224 skipped
(465.96s); integration passed 2,620/177 skipped (204.75s).
Both XML reports contain zero failures/errors: default at implementation commit
`3d2785ca8cd94ed9c91882b7ab13872b78604087`, integration at README-repair commit
`635afbe4dd0ba7b9aafe0e83abfafd9830e4705a` with identical example/test/library blobs.
The first integration run retained 2,619 passes/177 skips and one failure because
the new example was missing from the README inventory. Adding its entry fixed
the unchanged inventory test; the complete integration rerun passed. Its original
negative log/XML remains retained. Subsequent progress prose retains identical
example/test/library blobs. Python 3.10 passed 41 focused cases; Python 3.12
passed the same 41 plus the existing Markdown hygiene check. Package strict
checking passed 842 files; focused strict checking passed the two new files.
The initial expanded new-file invocation also exposed two existing diagnostics
in imported `terminal_instrument_reference.py`; those original diagnostics are
retained, and no unrelated example was changed. Ruff, Black 24/26, ownership,
repository gates and local links/language passed.

Maximum full evolved/branch density error was 7.50e-16, conditional density
error 2.22e-15, unread error 7.50e-16 and completeness error 2.22e-16.
The smallest independent positive branch weight was 0.1348608243; analytic zero
branches were checked unnormalized. All original budgets remain unchanged.
[#921](https://github.com/flagos-ai/FlagQuantum/pull/921) passed complete CI
`40295642672` on attempt 3 (all 54 jobs), GPU `40295642686` on attempt 2,
and the remote Python 3.10/3.11/3.12 matrix at head
`84f15a6a99e7a36ab993989d1d6ae94e8c10ec74`. All 60 applicable checks completed:
57 success and only the original three manual-pool skips. Its own explicit human
grant preceded a normal fixed-head merge as
`d15d681b88d6062e02b97317e53fbea5bafc7105`. Main smoke `40316383759` succeeded on
that exact merge; it is smoke only, not full main CI, a release or device evidence.

Original remote failures were Action archive/PyPI timeouts. One quality checkout
stalled in `git fetch --unshallow` for about 27 minutes; the user separately
authorized its termination and recovery of unsuccessful jobs. The targeted retry
retained successful tests and GPU, and changed no code, physical assertion,
security audit, workflow limit or repository protection. Original logs and
annotations remain retained. No local Python 3.11, physical detector or full
M3/M4 certification is claimed.

## Accepted fixed thermal-bath spin-mode scenario

The user approved all five decisions in the
[thermal-bath contract](M3_SPIN_MODE_THERMAL_BATH_CONTRACT_REVIEW.md) before
independent acceptance and implementation. [#897](https://github.com/flagos-ai/FlagQuantum/pull/897)
adds the example and 24 integration cases using existing numeric Lindblad
planning, boson operators, fixed collapse rates, gradients and numeric replay.
The original lossless example/tests and all library production/API/schema
surfaces are unchanged. Six scalar reference-function ASTs were captured before
the example and remain unchanged; six consumers received only import sorting.

Latest head `bf9afadc8cca81b5f0d7ae48462f32e2df7e71a1` passed complete CI
`40277142842` and GPU `40277142859`, both on attempt 1. All 54 standard CI jobs,
including the actual remote Python 3.10/3.11/3.12 matrix, succeeded. All 60
applicable checks completed: 57 success and the original three manual-pool
skips under their existing rules. The PR was externally merged by FlagQuantum
as `5ee74364cdbab17d1f0bebbee253bf400b467800` on 2026-10-26. Main smoke `40279655585`
then succeeded on that merge commit; it is only smoke, not full main CI or a
release/device validation.

Complete local default verification passed 16,049 cases/224 skipped (412.16s)
and integration passed 2,539/177 skipped (193.38s), both XML reports with zero
failures/errors, at `ed557965c5d1dae93a862b77495f999503240257`. The final commit
changed only progress prose, retaining identical example/test/library blobs.
A first full default attempt had one Markdown-equation link failure alongside
16,048 passed/224 skipped; equivalent D_L notation corrected only the document,
and a full rerun passed without changing physics, assertions or thresholds.
Original negative logs/XML are retained. Local Python 3.10/3.12 each passed 29
focused cases; no local Python 3.11 installation is claimed.

Maximum complete-density error was 2.48e-14 and gradient error 4.12e-12, within
unchanged 1e-10 and 1e-9 budgets. The N=9/11 spin and occupation differences
were 3.76e-6 and 1.23e-5, while N=11 top occupation stayed below 1.04e-6.
The exact finite moment correction and bath-only infinite-mean bound are
separate from coupled cutoff stability; neither certifies the coupled infinite
density. The 40-step finite-model objective decreased from 0.0160018399 to
0.0000712776 with exact detached JSON replay in the admitted local environment.
No calibrated ion-trap heating, global Gibbs state, physical-parameter recovery,
capability promotion or full M3/M4 completion follows.

## Accepted driven three-level relaxation scenario

On 2026-10-26 the user approved all five decisions in the
[next-scenario contract](M3_NEXT_SCENARIO_CONTRACT_REVIEW.md). Implementation
[#886](https://github.com/flagos-ai/FlagQuantum/pull/886) implements the driven
three-level fixed `2->1->0` relaxation example and independent integration tests
through existing APIs. The accepted scenario table now includes this bounded dissipative reference
alongside the unchanged closed model.

At implementation head `5f27e41b7b8b912d82f7880446b2d41f14db21bb`, local default
verification passed 16,017 cases/224 skipped and integration passed
2,515/177 skipped, with zero XML failures/errors. Python 3.10 and 3.12 each passed
the 20 new scenario cases plus four unchanged closed-model cases. Full-density
maximum error was 1.89e-15 and the maximum eight-control gradient error was
4.19e-13, within the unchanged 1e-10 and 1e-9 budgets. The 40-step training
objective decreased from 0.87460332 to 0.09593434 and detached numeric JSON replay
was exact on the local admitted environment. Four independent reference-function
ASTs and both old closed-model file blobs remain unchanged.

Latest complete CI `40269085812` and GPU `40269085833` succeeded at head
`2bb50f2fd54a5315bf80772569b0527bcef91488`, independently of earlier-head
results. All 54 standard CI jobs, including the remote Python 3.10/3.11/3.12
matrix, succeeded. All 60 applicable checks completed: 57 success and only the
original three manual-pool skips under their existing rules. Under the PR's
independent human authorization it merged as
`05bca74d08ac905f4ccdbb03235f2a478b2f89b7` on 2026-10-26. Main smoke
`40271009760` then succeeded; this is not full main CI, a release or device
validation. Capability maturity and the remaining M3/M4 boundaries are unchanged.
No device-calibration, gate-fidelity or higher-level truncation claim follows
from this finite phenomenological model.

## Accepted native unread observation

[Proposal 089](API_CHANGE_PROPOSAL_089_ISING_UNREAD_OBSERVATION.md) approved the
bounded unread slice before implementation. The delivery reused the existing
boundaries: [#798](https://github.com/flagos-ai/FlagQuantum/pull/798) established
replaceable consumers and captures;
[#802](https://github.com/flagos-ai/FlagQuantum/pull/802) bound source versions at
the Runtime reader; [#804](https://github.com/flagos-ai/FlagQuantum/pull/804) added
the Core operation/strict codec;
[#807](https://github.com/flagos-ai/FlagQuantum/pull/807) composed the fixed
channel in Simulation; [#809](https://github.com/flagos-ai/FlagQuantum/pull/809)
admitted native plan 1.2 and replaced all four test factories with the real path.
The obsolete test-only source/recipe/execution implementations were removed.
Independent consumer assertions and F/R oracles remain unchanged.

`dephase_pair()` observes the charge of the **current** positions 0/1 at its
source-order boundary, not the initial anyon names. It retains both Kraus
contributions, including a zero-weight branch as zero, without drawing an
outcome. An inverse braid after this channel therefore need not restore a pure
state. The existing ExecutionResult may contain a mixed `(1,2,2)` CPU complex128
density; it gains no event, branch, classical bit or feedback field.

Only sources containing exactly one such operation use native source/plan 1.2.
Braid-only 1.1, CircuitIR/circuit plans and existing Rydberg/Lindblad artifacts
retain their meanings and exact captured bytes. Rehashed mismatched versions or
a braid-only source relabelled 1.2 are rejected. Existing memory, environment,
output and options checks remain in the same planning/execution path.

Each implementation merged after its applicable complete CI/GPU gates. For
#809, standard CI `40084969445` and GPU `40084969438` completed successfully on
head `10c45e6b4622ca06d7cdd4826553ebe65f4f8eea`; merge
`01f7e7d1dbe9478eb48883f90fb8108696e5f4a0` then passed Main smoke
`40086625921`. That last workflow is a smoke check, not a full main CI or M4
performance/device acceptance. The [delivery record](../api-changes/FQ-ISING-UNREAD-OBSERVATION-20261024.md)
contains the runnable native workflow and preserved support limits.

## Accepted single sampled native trajectory

[Proposal 091](API_CHANGE_PROPOSAL_091_ISING_SAMPLED_OBSERVATION.md) was approved
before its staged deliveries: document
[#818](https://github.com/flagos-ai/FlagQuantum/pull/818), replaceable baseline
[#821](https://github.com/flagos-ai/FlagQuantum/pull/821), Core source
[#824](https://github.com/flagos-ai/FlagQuantum/pull/824), Simulation continuation
[#827](https://github.com/flagos-ai/FlagQuantum/pull/827), and Runtime
[#838](https://github.com/flagos-ai/FlagQuantum/pull/838). The four baseline
factories now use real Core construction, root planning/execution and JSON replay;
the disposable reference was deleted without changing independent consumer or
F/R-oracle assertions.

`measure_pair()` observes current positions 0/1 once, mutually exclusive with
`dephase_pair()`. The CPU profile requires explicitly supplied `shots=1` and a
Python integer seed in `[0, 2**64-1]`. Simulation evolves the prefix, applies the
selected fixed Kraus projector, normalizes by its original positive trace, and
continues the suffix. Runtime reuses terminal-instrument probability validation
and the private CPU draw, preserving original branch weights separately from
corrected draw probabilities. A zero-probability selected branch is refused.
No new IR, controller, solver, registry or result class is introduced.

The existing result holds a conditional `(1,2,2)` complex128 density and requested
conditional expectations. One `mid_circuit_measurement` record follows the
requested expectations: CPU int64 index `(1,1)`, ordered labels `1`/`psi`, current
anyon identities and source-operation boundary. Empty qubits do not represent a
classical register; the result has no automatic terminal samples, physical event
time or feedback. This record does not claim atom-presence detection.

Only this sampled source/plan uses 1.3 and execution-environment protocol 1.1
with the exact `rng_torch_version`. Readers and preflight refuse incompatible
builds before numerical evolution or drawing. Braid-only 1.1, unread 1.2 and old
Circuit/Rydberg/Lindblad captures remain byte-identical. The numeric sampling
subtotal is 216 bytes, not process RSS; existing state storage remains 64 bytes.

Latest #838 head `fd8ca1e10acc32cf7a5b5da9346a1e3a37d30c96` passed complete
PR CI `40149401926` (54 successful jobs), GPU `40149401931`, and all applicable
checks (57 successful, three original manual-pool skips). It merged as
`2cde712535badbe1ef8e2a0e56dcde74911d4e7d`. Its post-merge Main smoke
`40151647085` was pending at this review; container publication `40151647103`
was queued. Neither is reported as successful or as complete main CI. The
[delivery record](../api-changes/FQ-ISING-SAMPLED-OBSERVATION-20261024.md)
retains the staged acceptance, executable workflow and support limits.

## Keep error budgets separate

| Error or uncertainty | Existing witness | Interpretation |
| --- | --- | --- |
| Integration | Three-level Krylov/full-density comparison at 1e-10; Runge-Kutta error shrinks by more than tenfold per step doubling, final error below 1e-6. | Subdivision repeats identical constant pulses, so this is not waveform approximation. |
| State-space truncation | Spin thermal cutoff 3/5/7; cutoff-7 probability error below 5e-5 and explicit geometric-tail/top-block bound. Photonic three-/four-level embedding checked at 1e-10. | The spin's conditioned finite initial state and blocked top transition are reported. A sector-closed optical embedding does not establish general thermal convergence. |
| Exact unread channel | Both initial fusion channels, noncommuting prefixes/suffixes, weighted conditional continuations, pure/mixed echo and full complex density at 1e-12, rtol=0. | This finite algebra check has no time grid, sampled choice or integration error; exact JSON replay is checked separately. |
| Conditional trajectory | Independent full complex density and conditional expectations for both initial channels, noncommuting suffixes and zero-weight refusal at 1e-12, rtol=0; weighted conditional branches recover the unread channel. | Probability correction does not replace the original positive normalization trace. JSON replay is exact only within the admitted RNG environment. |
| Statistical | Existing terminal-instrument/binomial checks; 4,096 shared single-index draws retain the predeclared count interval 1,856..2,240 and private-seed replay. | Statistical intervals are not substitutes for deterministic numerical tolerances. Do not increase the shot bound to conceal a solver error. |
| Gradients | Three-level independent four-point differences at 1e-9; spin closed-block derivative at 1e-9; native pair independent frozen-unitary derivative at 1e-10. | These certify declared continuous controls, not discrete braid gradients or hardware calibration. |
| Loss and selection | Native pair exponential survival, observed efficiency model, unconditional RR, no-flag-selected RR and physical survival-conditioned RR. | Undetected physical loss must not become a detected erasure record. Selection denominators are explicit. |

## Current-API user workflow

Run without credentials or hardware:

```python
import torch
import flagquantum.lindblad as fql
from examples.driven_three_level import initial_controls, pulse_plan

plan = pulse_plan(initial_controls())
result = fql.run(plan)
restored = fql.LindbladPlan.from_json(plan.to_json())
replay = fql.run(restored)
print(round(float(result.expectation("leakage")[-1]), 8))
print(torch.equal(result.density_matrices, replay.density_matrices))
```

Representative executed output is `0.00189653` then `True`. This calls existing
example/model helpers and shared APIs; it does not introduce a new facade.
The native unread workflow also uses the existing root API:

```python
import flagquantum as fq
from flagquantum.experimental.simulation import IsingBraidProgram

program = IsingBraidProgram().exchange(1).dephase_pair().exchange(1, clockwise=True)
plan = fq.plan(program, outputs=fq.expectation(program.projector("psi"), name="psi"))
result = fq.run(fq.ExecutionPlan.from_json(plan.to_json()))
print(plan.schema_version)
print(round(float(result.expectation("psi")[0]), 12))
```

Executed output is `1.2` then `0.5`. The unread channel destroys the coherence
needed by the inverse-braid echo; no intermediate outcome was selected.

The single sampled trajectory uses the same root entries and existing results:

```python
import torch
import flagquantum as fq
from flagquantum.experimental.simulation import IsingBraidProgram

program = IsingBraidProgram().exchange(1).measure_pair().exchange(0)
plan = fq.plan(
    program,
    options=fq.ExecutionOptions(shots=1, seed=17),
    outputs=fq.expectation(program.projector("psi"), name="psi"),
)
result = fq.run(fq.ExecutionPlan.from_json(plan.to_json()))
observed = result.measurement("mid_circuit_measurement")
index = int(observed.value[0, 0])
print(plan.schema_version)
print(observed.metadata["outcome_labels"][index] in ("1", "psi"))
print(abs(float(result.expectation("psi")[0]) - index) < 1e-12)
print(torch.equal(result.state, fq.run(plan).state))
```

Executed output is `1.3`, `True`, `True`, `True`. The selected label is deliberately
not fixed across PyTorch builds. The final expectation is conditional on the
actual outcome, whereas the unread example returns an ensemble expectation.

For all six scenario witnesses, run the following existing tests rather than
creating a second manifest or a test that merely asserts this document's text:

```bash
python -m pytest -q \
  tests/integration/test_train_then_deploy_golden_path.py \
  tests/unit/test_train_deploy_error_metrics.py \
  tests/integration/test_driven_three_level.py \
  tests/integration/test_spin_mode_convergence.py \
  tests/integration/test_boson_interference.py \
  tests/integration/test_rydberg_control_training.py \
  tests/integration/test_rydberg_transport_visualization.py \
  tests/unit/test_native_rydberg_dynamics.py \
  tests/unit/test_ising_braid_contract.py \
  tests/integration/test_ising_braid_execution.py \
  tests/integration/test_ising_fusion_measurement.py \
  tests/unit/test_ising_unread_contract.py \
  tests/unit/test_ising_unread_runtime.py \
  tests/unit/test_ising_unread_compatibility.py \
  tests/unit/test_ising_sampled_contract.py \
  tests/unit/test_ising_sampled_runtime.py \
  tests/unit/test_ising_sampled_compatibility.py \
  tests/integration/test_ising_sampled_execution.py
```

## Approved intermediate atom-presence delivery

The user approved all five decisions in
[proposal 092](API_CHANGE_PROPOSAL_092_RYDBERG_PRESENCE_TRAJECTORY.md).
The replaceable baseline [#854](https://github.com/flagos-ai/FlagQuantum/pull/854)
preserved independent complex-density consumers and original source/plan captures.
Core [#869](https://github.com/flagos-ai/FlagQuantum/pull/869) added one named
`measure_presence(atom, at=..., efficiency=...)` operation and strict source 1.4.
Simulation [#875](https://github.com/flagos-ai/FlagQuantum/pull/875) added private
prefix and selected-branch continuation through existing numerical authorities.
It merged as `f8e69c6c9153e4c5570a486c5673b27c72ae5dcd` after its latest
complete CI `40209946126` and GPU `40209946116` succeeded; Main smoke
`40251813632` then succeeded. That last workflow is only a smoke check.

The accepted [Runtime implementation #879](https://github.com/flagos-ai/FlagQuantum/pull/879)
joins source 1.4 to sealed LindbladPlan 1.8, explicit shots=1/seed, environment
1.1 and existing canonical results through `fq.run(plan)`. It merged as
`21b8e7f6e34a638e5120a0f4d7fff64fe196adb2` under its independent human grant,
after complete CI `40255122443` (attempt 2) and GPU `40255122490` succeeded on
head `757736cd20f8aa958e626ea0465d9988d56b2c04`. All applicable checks passed;
only the original three manual-pool gates skipped under their existing rules.
The remote Python 3.10/3.11/3.12 matrix passed. Local default verification was
15,995 passed/224 skipped and integration was 2,481 passed/177 skipped, both
with zero XML failures/errors. Main smoke `40258387265` then succeeded; this is
only a smoke result, not a full main CI, release or hardware certification.
The first CI attempt failed while downloading setup-action archives; its logs
were retained and a single bounded failed-job retry completed successfully.
See [the staged delivery record and executed public example](../api-changes/FQ-RYDBERG-PRESENCE-TRAJECTORY-20261025.md).

The numerical seams retain the complete native schedule, original absolute
time grid and all interval bindings. The selected branch is normalized by its
original positive trace, with zero-probability refusal. Earlier rows are forward
states and the observation boundary is post-observation. Detector no-flag includes
undetected loss; it does not establish physical survival or trap activity.
Independent Kraus algebra, full complex densities, weighted unread equivalence,
roster identity, changing bindings, convergence, old bytes and private RNG
invariance remain separate acceptance witnesses. Runtime replacement changes
only the four factories and deletes the disposable reference; all 29 independent
physical consumer/oracle function ASTs are preserved. The final state has shape
`(1,D,D)`, expectations retain their time axis, and a retained density record has
`(T,D,D)` axes. Old unconditional Lindblad profiles remain outside root execution;
`fql.run` and adjoint refuse the sampled profile.

No parallel IR, controller, event registry, solver or result class is introduced.
Multi-event trajectories, conditional feedback, detector latency, erasure
routing/decoding, motional heating, Remote/hardware execution and gradients
through stochastic choices remain separate decisions. This inventory records
approved staged delivery, does not retroactively change proposals 081/083/085,
and does not declare M3 complete. Device validation and performance claims still
require the roadmap's separate M4 evidence.
