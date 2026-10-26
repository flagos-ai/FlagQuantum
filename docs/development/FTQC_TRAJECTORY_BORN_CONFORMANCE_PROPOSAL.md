# FTS-11C: Finite Local Trajectory Born-Conformance Proposal

Status: proposed, not approved or implemented. E-H qualified private coherent
references and exact aggregate density, not inputs to the trajectory sampler.
This new finite Runtime/reference boundary and its comparison values require
specific approval. Document merge or green CI does not activate it.

## Existing owners and scope

Reuse the three literal E0/E1/E2 circuits and independent analytic/direct
references from [E-H](FTQC_DYNAMIC_COHERENT_REFERENCE_PROPOSAL.md). Reuse the
current internal `run_dynamic` trajectory/batched implementations, not another
sampler or an independently sampled coherent term mixture. The
[existing schedule conformance](../../tests/qec/test_schedule_feedback_conformance.py)
already observes `torch.multinomial` inputs while delegating to the original
sampler. Extend that test technique only for this finite domain.

The [stable execution contract](API_CHANGE_PROPOSAL_084_DYNAMIC_EXECUTION_UNIFIED_RUN.md)
continues to expose `ExecutionResult` samples and measurement records, with no
stable per-shot state or exact Born-probability fields. Native final states and
strategy selection are used only in internal conformance. No stable strategy,
phase, RNG stream, result schema, provider or postselection API is introduced.

This would qualify current local statevector trajectories on the named cases.
It would not qualify a production coherent-decomposition sampler, efficient
large-scale sampling, random-number-generator quality, hardware or all dynamic
programs. FTS-10 selection, general FTS-11 product sampling and FTS-02/09 remain
open. No benchmark observer, process campaign or performance target is requested.

## Decision requested: groups I-L

Approve the finite matrix below and CPU complex128 state/density comparisons
and float64 Born-vector comparisons with atol=1e-12, rtol=0. This value needs
explicit admission for the Runtime observation boundary, even though it was
approved for earlier private references. Proposed per-module invocation cap:

| Group | Existing execution entry | Calls | Shots per call | Total trajectories |
| --- | --- | --- | --- | --- |
| I natural draws | Internal `run_dynamic`, E0/E1/E2 x trajectory/batched | 6 | 8 | 48 |
| J controlled branch coverage | Same internal entries; E0 record0, E1/E2 record0/1 x both strategies | 10 | 1 | 10 |
| K stable projection | Current `fq.run`, E0/E1/E2 | 3 | 8 | 24 |
| L discriminating negatives | Reuse captured/reference values; no extra execution | 0 | 0 | 0 |

Total: at most 19 dynamic calls and 82 trajectories, fixed seed 53 for natural
and terminal draws, one local process and one numerical CPU thread restored
after verification. No retry, seed search, frequency tolerance, optional SDK,
noise, gradients, extra qubits, second classical record or resource experiment.
Reuse E-H reference vectors once; do not allocate a new family of coherent
branches. The fixed calls are correctness tests, not retained benchmark data.

### I: observe real Born inputs and continuation

Wrap `torch.multinomial` only inside the test context. Copy each input and
returned selection, then delegate unchanged arguments and generator to the
original function. Do not replace gate evolution, measurement collapse, reset,
conditional dispatch or final sampling. Finish imports and construction before
the observation context; reject unexpected call shapes/order rather than
silently ignoring draws. Restore the original function on exit, including error.

The existing scalar path makes one measurement draw, optional reset draw, and
one terminal draw per trajectory. The batched path makes the corresponding
row-wise draws. Compare the actual finite raw inputs, without renormalizing them:

- First measurement: (1,0) for E0, (h,l) for E1/E2, using E-H's analytic h/l.
- E2 reset: the one-hot distribution for the already measured qubit m, not a
  second independent (h,l) draw. Reset changes qubit 0 without erasing c0.
- Terminal sampling: the computational-basis probabilities of the independent
  normalized conditional state for that row's recorded m.

Check input shape, CPU float64, finite nonnegative entries and sum=1 separately.
For the native result, require samples shape (shots,2), classical bits shape
(shots,1), and final states shape (shots,4), finite CPU complex128. Compare every
row's state norm and phase-invariant full density to the E-H conditional state.
E0 cannot report record1. Final sample qubit0 equals c0 for E0/E1 and is zero
for E2; terminal selections must agree with both returned sample columns.

Do not compare empirical frequencies to h/l, require both outcomes in eight
natural draws, or promise identical samples across versions/strategies. A test
can pass its natural-draw check with only one observed nonzero branch; J supplies
deterministic coverage instead of additional random attempts.

### J: controlled coverage is not a random-sampling certificate

For each positive analytic record in the table, intercept only the first
measurement selection and choose that record after independently checking the
actual input probabilities. Use shots=1 for each strategy. Leave subsequent
reset and terminal draws delegated to the original function. Exercise production
collapse, reset and continuation, then compare norm, density, record and terminal
probability inputs exactly as in I.

The test-only selector must refuse a zero-support outcome before returning a
selection. Do not force E0 record1, divide its zero state, resample another
outcome, or change production refusal behavior. This guard is a test-fixture
boundary, not evidence for a public postselection feature. Label J as controlled
coverage wherever results are described; never combine its outcomes into an
empirical frequency or RNG claim.

### K: preserve the existing stable result projection

Run E0/E1/E2 once each through `fq.run` with eight shots and seed 53, explicit CPU,
approximation/fallback disabled, and dtype on the `DynamicCircuit` constructor.
Do not set dynamic `ExecutionOptions.precision` or request density mode/outputs;
those remain outside the stable local slice. Reuse I's read-only draw observation
and independently check recorded outcomes, terminal inputs and sample columns.

The canonical result keeps state/value absent, its effective plan attached,
final samples in `measurements[0]`, and c0 in ascending-bit-order
`measurements[1]`. Do not expose the native `final_states` field or freeze which
internal strategy auto currently selects. K does not add direct-versus-restored
plan repetitions or alter the already tested plan serialization contract.

### L: meaningful negatives, no extra runtime executions

Reuse captured probability vectors, states and analytic records. The checker
must distinguish a (1/2,1/2) substitution for E1's first draw, a reset that redraws
from (h,l), wrong-record continuation, a swapped terminal qubit order and a
relative phase error invisible in computational-basis probabilities. Refuse
nonfinite, negative, zero-total or incorrectly shaped draw data before reduction.
Do not normalize bad probabilities into agreement, widen tolerances, add skips
or mutate production instructions/fixtures merely to pass the checker.

These negatives validate the finite checker, not a general statistical testing
framework. Any discovered production defect is reported before changing public
behavior or a protected contract.

## Review and remaining work

First review I-L, observation independence, controlled-versus-natural labeling,
numerical comparisons and the 19-call/82-trajectory cap. After exact-head full CI
and review make this proposal reviewable, request specific human approval.
Only after that answer and main admission may a separate test PR implement it.
No new observation hook, selector, sampler conformance or matrix execution has
been implemented for this proposal.

E-H remains a delivered private reference, not a production phase-capable
representation. Method selection, phase/overlap cost evidence at W4 scale,
noise/QEC integration, large-scale sampling and production decomposition remain
separate gaps. Both prior research campaigns and collectors are immutable;
their consumed budgets do not admit more recovery, probes or collection here.

## Runnable current user example

Requirements: supported Python, core FlagQuantum and CPU; no optional SDK,
device, credentials or dataset. Save as `/tmp/ftqc_trajectory_example.py` and run
`python /tmp/ftqc_trajectory_example.py` after installation, or
`PYTHONPATH=. python /tmp/ftqc_trajectory_example.py` from the checkout.

```python
import torch
import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit

for reset in (False, True):
    circuit = DynamicCircuit(2, dtype=torch.complex128, device="cpu")
    circuit.h(0).t(0).h(0).cx(0, 1).measure(0, classical_bit=0)
    if reset:
        circuit.reset(0)
    circuit.conditional("x", 1, classical_bit=0).h(1).t(1).h(1)
    result = fq.run(circuit, options=fq.ExecutionOptions(
        shots=8, seed=53, device="cpu",
        allow_approximate=False, allow_backend_fallback=False,
    ))
    assert result.samples is not None and result.samples.shape == (8, 2)
    record = result.measurements[1]
    assert record.kind == "mid_circuit_measurement" and record.value.shape == (8, 1)
    expected = torch.zeros_like(record.value[:, 0]) if reset else record.value[:, 0]
    preserved = bool(torch.equal(result.samples[:, 0], expected))
    assert preserved
    print("reset" if reset else "retained", result.samples.shape, preserved)
```

Expected stdout:

```text
retained torch.Size([8, 2]) True
reset torch.Size([8, 2]) True
```

This current stable user workflow is an engineering example, not execution or
certification of the proposed I-L observation matrix or a retained study point.
