# M3 scenario acceptance and remaining boundaries

This review inventories the bounded CPU paths required by the
[shared roadmap](../roadmap/FLAGOS_ALIGNED_RELEASE_TRAIN.md#reference-scenarios).
It changes no API, schema, capability declaration, maturity level or approval.
An accepted reference scenario is not certification of a hardware technology or
of all M3/M4 exit gates. The inspected source baseline is main
`4be2375eff47c112ce8d8e24f8206ed5592130d8`; later work must refresh this evidence.
Historical design reviews and contract approval records remain historical.

## Executable evidence by user scenario

| Scenario | Runnable example (`python -m examples.` prefix) | Independent acceptance and merge evidence | Remaining boundary |
| --- | --- | --- | --- |
| Circuit training and deployment packaging | `train_parameterized_circuit_then_deploy` | [Golden path](../../tests/integration/test_train_then_deploy_golden_path.py), [metrics/RNG regressions](../../tests/unit/test_train_deploy_error_metrics.py); [#771](https://github.com/flagos-ai/FlagQuantum/pull/771) separates variational error from finite-shot error and retains small exported angles. | The in-memory provider is local contract execution, not evidence of a submitted cloud job or a QPU measurement. |
| Driven three-level superconducting model | `driven_three_level` | [Full density, all eight control derivatives, pulse order and refinement](../../tests/integration/test_driven_three_level.py); [#787](https://github.com/flagos-ai/FlagQuantum/pull/787). | Closed rotating-frame three-level model; no calibrated device, pulse emission, dissipation, gate-fidelity or higher-level truncation certification. |
| Moving neutral-atom pair | `train_rydberg_controls --steps 40 --intervals 128 --shots 4096`; `visualize_rydberg_transport` | [Native source/dynamics](../../tests/unit/test_native_rydberg_dynamics.py), [training, loss, explicit observed flags and conditioning](../../tests/integration/test_rydberg_control_training.py), [commanded-path view](../../tests/integration/test_rydberg_transport_visualization.py); [#707](https://github.com/flagos-ai/FlagQuantum/pull/707)/[#714](https://github.com/flagos-ai/FlagQuantum/pull/714)/[#715](https://github.com/flagos-ai/FlagQuantum/pull/715)/[#738](https://github.com/flagos-ai/FlagQuantum/pull/738). | One/two-actor imposed paths and declared Markov loss; no native intermediate instrument, inferred trap activity, motional heating, species calibration or hardware execution. |
| Spin coupled to a motional mode | `spin_mode_convergence` | [Complex Rabi blocks, total excitation, finite-mode dark top, analytic gradient and replay](../../tests/integration/test_spin_mode_convergence.py); [#782](https://github.com/flagos-ai/FlagQuantum/pull/782). | Resonant lossless rotating-wave model and explicit finite thermal conditioning; no ion-trap pulse frontend, heating or hardware calibration. |
| Two-mode photonic interference | `boson_interference` | [Closed attenuation/HOM density, cutoff embedding, unconditional QND counts and back-action](../../tests/integration/test_boson_interference.py); [#741](https://github.com/flagos-ai/FlagQuantum/pull/741). | Finite Fock reference; no Gaussian backend, optical-program frontend, destructive detector or photonic hardware submission. |
| Small anyon braiding and fusion | `ising_braid_execution`; `ising_fusion_measurement` | [Unchanged physical contract consumers](../../tests/unit/test_ising_braid_contract.py), [native execution/replay](../../tests/integration/test_ising_braid_execution.py), [fusion coherence and QND workflow](../../tests/integration/test_ising_fusion_measurement.py); [#769](https://github.com/flagos-ai/FlagQuantum/pull/769)/[#778](https://github.com/flagos-ai/FlagQuantum/pull/778). | Exactly four sigma anyons, total vacuum and fixed gauge, CPU ideal braids; no other sectors, continuous gradients, universal braid set or Majorana-device claim. |

The neutral-atom row has a working terminal-observation slice, not all the
roadmap's event requirements. The other rows likewise establish the named
bounded reference, not unrestricted route support. None is a distributed
capacity or performance acceptance result.

## What is actually shared

- Circuit programs and the native Ising program use `fq.plan/run`; analog and
  mixed-dimension models use `flagquantum.lindblad.plan/run`. Do not claim that
  every source is accepted by every root entry point. Existing refusals matter.
- Controls reuse Core parameters and interval binding. Numeric matrices and
  physical sources reach their existing domain owners rather than a new
  technology registry. Ising exchange words are discrete and do not acquire
  control gradients merely because another route trains through PyTorch.
- Matrix requests reuse `fq.expectation` and named outputs. Analog evolution
  retains its existing evolution result; native Ising retains ExecutionResult.
  Shared vocabulary does not mean identical result shapes or result classes.
- Analog models reuse Liouvillian assembly and existing integrators. Ideal
  Ising braids reuse finite SuperOperator actions, not an invented Hamiltonian
  or an artificial time coordinate. Circuit execution keeps its native path.
- Terminal observations reuse `sample_terminal` and MeasurementResult. Explicit
  Kraus maps explain conditional/unread back-action; the terminal sampler itself
  returns indices, not conditional states, detector trajectories or feedback.
- Saved numeric plans capture values and their supported semantics. Rebuild
  live plans after optimizer updates; JSON is not a saved training graph.
  Circuit, Lindblad and native plan versions remain their respective contracts.

No new framework, solver, binder, result envelope, registry, evidence schema or
capability catalog is needed to express this inventory.

## Keep error budgets separate

| Error or uncertainty | Existing witness | Interpretation |
| --- | --- | --- |
| Integration | Three-level Krylov/full-density comparison at 1e-10; Runge-Kutta error shrinks by more than tenfold per step doubling, final error below 1e-6. | Subdivision repeats identical constant pulses, so this is not waveform approximation. |
| State-space truncation | Spin thermal cutoff 3/5/7; cutoff-7 probability error below 5e-5 and explicit geometric-tail/top-block bound. Photonic three-/four-level embedding checked at 1e-10. | The spin's conditioned finite initial state and blocked top transition are reported. A sector-closed optical embedding does not establish general thermal convergence. |
| Statistical | Existing terminal-instrument/binomial checks and unchanged private-seed replay. | Statistical intervals are not substitutes for deterministic numerical tolerances. Do not increase the shot bound to conceal a solver error. |
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
  tests/integration/test_ising_fusion_measurement.py
```

## Next boundary: review one intermediate native observation

The next proposed vertical slice is **one instrument at a declared boundary of
an existing native finite-basis program**, with explicit probability and
post-observation continuation. This is a review target, not an approved API.
Qubit dynamic execution already has intermediate measurement, classical state,
feedback traces and result projection under
[Runtime dynamic execution](../../flagquantum/runtime/dynamic/execution.py) and
[its result](../../flagquantum/runtime/dynamic/result.py). The gap is their
relationship to native finite-basis physics, not a global lack of measurement
or control flow. Coordinate with that workstream before selecting a surface.

A concrete contract review must settle these points before implementation:

1. Which existing ProgramRecord operation and dynamic execution/result boundary
   can own the instrument? Preserve atom/trap or fusion labels across the join;
   do not add a parallel program IR, controller, event registry or result type.
2. Is the first path an unread exact channel or a sampled conditional branch?
   Specify `J_j(rho)`, its trace and the normalization denominator. Refuse a
   zero-probability selected branch; do not silently discard unobserved sectors.
3. Capture the explicit instrument model, boundary ordering, outcome-label order
   and any private RNG seed in existing versioned plan/result authorities.
   Distinguish exact density replay from sample replay and preserve old bytes.
4. Restrict the first reviewed case to local CPU and a bounded declared space.
   Real-time feedback, Remote execution, detector latency, erasure decoding,
   motional heating and gradients through stochastic choices remain separate
   decisions; refusing them is preferable to inferred support.
5. Establish an independent two-stage channel reference, a replaceable consumer
   baseline, probability/trace and unread-versus-conditional checks, replay and
   refusal tests. Then split domain-owned implementations under the existing
   architecture. New protected APIs require explicit approval; each PR retains
   independent merge authorization and complete applicable CI/GPU gates.

This review does not authorize that contract, retroactively change proposals
081/083/085, or declare M3 complete. Device validation and performance claims
still require the roadmap's separate M4 evidence.
