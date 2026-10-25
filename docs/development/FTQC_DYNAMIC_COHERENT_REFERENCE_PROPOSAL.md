# FTS-11B: Bounded Dynamic Coherent-Reference Proposal

Status: proposed, not approved or implemented. This new finite private reference
domain and its comparison values are not covered by the static A-D approval.
No API, production branch engine, sampling certificate or collection budget is
activated by this document.

## Existing owners and missing boundary

[Static coherent conformance](FTQC_COHERENT_BRANCH_CONFORMANCE_PROPOSAL.md)
qualifies interference before measurement, not Born weights or conditional
continuation. The [master plan](../roadmap/FTQC_SIMULATION_MASTER_PLAN.md#52-mixed-clifford--non-clifford-simulation)
requires those semantics before product mixed-circuit sampling acceptance.

Reuse existing dynamic IR and `DynamicCircuit`. Its exact `density_matrix()`
method already projects records, applies conditions and reset, and sums outcome
densities. The [current tests](../../tests/unit/test_dynamic_density_matrix.py)
cover measurement, reset, feedback and classical def-use. Do not build another
density engine or generic dynamic interpreter.

The [stable `fq.run` contract](API_CHANGE_PROPOSAL_084_DYNAMIC_EXECUTION_UNIFIED_RUN.md)
returns trajectories through canonical samples and measurements, without stable
exact branch-probability or conditional-state fields. This proposal does not
add those fields or a `mode="density_matrix"` option. Exact reference algebra
and public aggregate-density equality do not certify trajectory sampling, RNG
quality, per-shot conditional states or a production decomposition.

## Decision requested: finite groups E-H

Approve only these three noise-free two-qubit cases, their private coherent
measurement/continuation algebra and CPU complex128 comparisons with
atol=1e-12, rtol=0. The values require specific approval for this domain and are
not a new runtime accuracy guarantee. Each future module verification uses
three existing public `DynamicCircuit.density_matrix()` calls, one per case,
and exactly 20 private projected Clifford vectors of four amplitudes each.
A single-thread fixture restores the previous setting. No shots, statistical
tolerance, timing repetition, SDK, device or formal study process is proposed.

### E: literal programs and record identity

Start in |00>, qubit order (0,1), c0 undefined. All listed gates execute in
order. `measure(0,c0)` records its outcome. The later conditional reads that
same record even when reset changes qubit 0. Tail Q is H(1), T(1), H(1).

| Case | Prefix | Measurement and continuation | T count |
| --- | --- | --- | --- |
| E0 impossible-record anchor | H(0), H(0), CX(0,1) | measure(0,c0); X(1) if c0=1; Q | 1 |
| E1 biased-record continuation | H(0), T(0), H(0), CX(0,1) | measure(0,c0); X(1) if c0=1; Q | 2 |
| E2 reset preserves record | Same as E1 | measure(0,c0); reset(0); X(1) if c0=1; Q | 2 |

Only these named descriptions are admitted. No batch, second measurement,
record overwrite, noise, decoder, frame, injection, loop, larger circuit family
or new dynamic branch API is implied. Reuse current def-use refusals rather
than restating a public exception contract.

### F: coherent sum before Born weight

Expand every literal T with the static identity T=aI+bS, retaining complex
coefficients. For record m apply Pm on qubit 0 to each private Clifford term,
then its record-specific continuation. For E2, reset on the already measured
qubit is identity for m=0 and X(0) for m=1; it changes the qubit, not c0.
This private identity does not replace the public reset implementation.

```text
v_m = sum_s c_s U_m P_m V_s |00>
p_m = <v_m | v_m>
rho_final = sum_m |v_m><v_m|
```

Both prefix and tail choices belong to s: E0 needs two terms per record, E1/E2
four each, for 4+8+8=20 private vectors. These are scalar references, not timed
workloads or a production representation. Reuse static private test helpers
where their canonical phase semantics apply; any test-only factoring must
preserve existing static conformance. The oracle must not call production gate,
evolution, compiler or projector helpers.

Independently evolve direct scalar T phases with explicit basis projection and
conditional permutations; do not obtain that reference solely from the coherent
sum or production density engine. Compare canonical amplitudes and Born weights.
Retain cross terms within each record; sum different record densities
incoherently because they are exclusive measurement outcomes.

### G: analytic conditional states and zero outcome

Let w=exp(i*pi/4), alpha=(1+w)/2, beta=(1-w)/2, u=(alpha,beta),
h=|alpha|^2=(2+sqrt(2))/4 and l=|beta|^2=(2-sqrt(2))/4. Canonical answers:

- E0: v0=|0> tensor u, v1=0; weights (1,0).
- E1: v0=alpha |0> tensor u, v1=beta |1> tensor u; weights (h,l).
- E2: v0=alpha |0> tensor u, v1=beta |0> tensor u; weights (h,l).

Final densities are respectively |0><0| tensor |u><u|,
diag(h,l) tensor |u><u|, and |0><0| tensor |u><u|. E1 populations are
(h*h,h*l,l*h,l*l). E2 is pure although its record has two nonzero alternatives.

Check ordered shape, CPU complex128, finite entries, nonnegative finite Born
weights, sum(p)=1 and final trace=1 before reducing errors. Compare the complete
public density with the independent aggregate, without claiming access to the
engine's record states. For an analytically nonzero outcome normalize v_m by
sqrt(p_m) only after checking its unnormalized weight against the analytic and
direct reference. Compare conditional norm and full density separately. Never
normalize individual coherent terms or final trace to hide loss.

E0 record 1 stays zero probability with no conditional state. A private request
to normalize it must fail before division. Do not floor the probability or
resample another outcome. This is not a public postselection API.

### H: discriminating negatives without extra public calls

Reuse the already computed vectors. Each wrong construction must disagree with
the analytic/direct answer at the proposed comparison value:

- Squared term weights before interference, or independently normalized terms
  before summation: E1 Born weights or aggregate density must differ.
- Cross terms between different measurement records: E1 must expose the
  nonphysical inter-record coherence instead of a measured density.
- Lost c0 during reset, wrong-record conditioning or omitted correction:
  E1/E2 full density must expose the incorrect continuation.
- Zero-branch normalization, nonfinite or incorrectly shaped reference data:
  refuse before division or reduction.

Aggregate equality alone cannot prove branch weights: E2's correct conditional
states coincide, so a wrong record distribution can leave its density unchanged.
The independent analytic Born checks and E1 expose that limitation. No sampling
certification follows.

## Review sequence and remaining decisions

Review E-H, independence, record/reset semantics and numerical values first.
After exact-head CI/review makes this proposal reviewable, request explicit
human approval. Document merge and green CI are not approval. Only after human
approval and main admission may a separate test-only PR implement this slice.

Production phase representation, conditional sampling, trajectory/Born
validation and target W4 selection/cost remain open. FTS-10/11 system acceptance,
FTS-02/09 and R01-R15 remain partial. No new private projector, branch enumeration
or proposed conformance has run for this document. Both completed study
campaigns and all raw evidence remain immutable; no collector, probe or recovery
attempt is authorized here.

## Runnable existing user workflow

Requirements: supported Python, core FlagQuantum and CPU; no optional SDK,
credentials or dataset. Save as `/tmp/ftqc_dynamic_density_example.py` and run
`python /tmp/ftqc_dynamic_density_example.py` after installation, or
`PYTHONPATH=. python /tmp/ftqc_dynamic_density_example.py` from the checkout.
This is the current exact density method, not a proposed branch API.

```python
import torch
from flagquantum.dynamic import DynamicCircuit

for reset in (False, True):
    circuit = DynamicCircuit(2, dtype=torch.complex128)
    circuit.h(0).t(0).h(0).cx(0, 1).measure(0, classical_bit=0)
    if reset:
        circuit.reset(0)
    circuit.conditional("x", 1, classical_bit=0).h(1).t(1).h(1)
    density = circuit.density_matrix()
    assert density.shape == (1, 4, 4)
    assert bool(torch.isfinite(density).all())
    trace = float(density[0].diagonal().sum().real)
    assert abs(trace - 1.0) <= 1e-12
    print("reset" if reset else "retained", *[
        f"{float(p):.8f}" for p in density[0].diagonal().real
    ])
```

Expected stdout:

```text
retained 0.72855339 0.12500000 0.12500000 0.02144661
reset 0.85355339 0.14644661 0.00000000 0.00000000
```

This current user example is an engineering smoke check, not an implementation
or certification of the proposed independent E-H reference.
