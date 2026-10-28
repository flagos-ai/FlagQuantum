# API Change Proposal 099: Bounded Twin parameter estimation

## Status

**Implementation authorized; additive to the Twin framework API.** This slice
fits only the parameters of one frozen, full-rank local probe design and returns
an immutable estimate with simultaneous finite-shot intervals.

## Problem

A full-rank probe bank states which restricted parameters can be separated, but
it does not turn hardware counts into values. Applications should not implement
their own matrix inversion, silently pool different snapshots, reuse one task,
or present unconstrained coefficients as physical error probabilities.

The framework needs one fail-closed estimator that binds reports to the exact
target, physical mapping, probe circuits, executed programs, and Twin snapshot.
Its uncertainty statement must expose the model and sampling assumptions. It
must not mutate a Twin or promote a candidate.

## Decision

- Add `fq.twin.estimate_probe_parameters(bank, hardware_reports, *,
  confidence_level=0.95)` and immutable `TwinParameterEstimate`.
- Upgrade the runtime-only probe-bank schema to
  `flagquantum.twin_probe_bank.v2`. Version 2 freezes the noiseless Z-marginal
  and directed-coupler ZZ distributions aligned with every design row. Version
  1 intentionally had no persistence loader; a bank must be prepared again
  from its original circuits before estimation.
- Pool distinct reports only within the same probe identity. Require one target,
  ordered physical mapping, backend, snapshot, and authoritative executed
  program across the complete fit.
- Use the exact full-column-rank least-squares left inverse of the declared
  design. Constrain returned local burden parameters to `[0, 1]` and name every
  parameter whose point estimate reaches or is clipped to a boundary.
- Compute a distribution-free multinomial TV radius for every Z or ZZ row.
  Bonferroni correction makes the requested confidence simultaneous across all
  frozen rows. Propagate those row radii through the fixed left inverse.
- Report observed and fitted row TVs, residual RMSE, maximum absolute residual,
  and `goodness_of_fit = max(0, 1 - RMSE)`. The score is descriptive within the
  declared model, not a model-selection probability.

## Public API

The complete offline executable is
[`examples/twin_parameter_estimation.py`](../../examples/twin_parameter_estimation.py):

```bash
python examples/twin_parameter_estimation.py
```

The core application call is:

```python
import flagquantum as fq

bank = fq.twin.prepare_probe_bank(
    target="quafu:Shenglian",
    physical_qubits=(20, 27),
    probes=probe_circuits,
    parameter_family="local_pauli_readout",
)

# Each item is a distinct result previously returned by
# TwinExperiment.validate_result(...). No submission occurs here.
estimate = fq.twin.estimate_probe_parameters(
    bank,
    hardware_reports,
    confidence_level=0.95,
)

print(estimate.parameter_values)
print(estimate.confidence_intervals)
print(estimate.identifiability_status)
print(estimate.goodness_of_fit)
print(estimate.boundary_parameters)
```

## Statistical interpretation

The fit assumes a fixed first-order linear local-error-burden model,
independent multinomial hardware reports, a simultaneous Bonferroni finite-shot
bound, and parameters in the unit interval. These assumptions are stored in the
result. The confidence intervals target simultaneous coverage under those
assumptions; they do not cover model misspecification, temporal drift,
correlated tasks, leakage, crosstalk, or arbitrary Kraus channels.

The fitted observations are total-variation distances between hardware and
noiseless simulated Z/ZZ marginal distributions. They concern classical
measurement distributions, not quantum-state amplitudes or state fidelity.

## Fail-closed behavior

Estimation rejects an underdetermined bank and names its unidentified
parameters. It also rejects missing probes, extra circuits, reused report
identities, reused provider task IDs, invalid counts, count/probability
disagreement, target or ordered-mapping mismatch, mixed snapshots, missing
executed-program identity, and invalid confidence levels.

`TwinExperiment.validate_result(...)` now copies the receipt-bound ordered
physical mapping into the normalized report metadata and rejects a conflicting
provider mapping. This supplies the mapping identity required by the estimator
without changing the hardware-report schema.

## Result contract

`TwinParameterEstimate` binds the probe-bank identity, target, ordered mapping,
family, parameter values and intervals, boundary parameters, report and task
identities, snapshot, shots per probe, row observations and radii, residual
metrics, confidence level, assumptions, schema, and deterministic identity.

The object contains no Twin, noise model, provider client, receipt, queue state,
candidate, promotion, routing, retry, or control action.

## Acceptance

- Synthetic known-parameter recovery succeeds with zero model residual.
- Every interval contains its corresponding bounded estimate.
- Reordered reports and repeated distinct observations produce deterministic
  pooled results.
- Invalid counts, reused reports or tasks, missing probes, mapping mismatch,
  and mixed snapshots fail closed.
- Underdetermined designs name missing parameters and produce no estimate.
- Boundary parameters are explicit.
- Focused Twin tests, API contracts, strict typing, formatting, documentation,
  default CI, and runtime CI pass.

## Non-goals

- No arbitrary Kraus, leakage, correlated-noise, pulse, many-body, or temporal
  model reconstruction.
- No claim that confidence intervals include model error or drift.
- No QPU submission, polling, retry, cancellation, scheduling, routing, model
  mutation, candidate construction, or promotion.
- No Agent, MCP, natural-language, REST, UI, or application policy.
