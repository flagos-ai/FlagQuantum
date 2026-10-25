# FTS-10B / FTS-11A: Bounded Coherent-Branch Reference Proposal

Status: proposed finite private-test semantics and numerical admission; not
approved or implemented. No new engine, public API, benchmark campaign or
performance target is activated. Document merge or green CI is not approval.

The [completed static pilot](FTQC_MIXED_CPU_COMPLETED_PILOT.md) establishes small
existing-route observations, not a coherent decomposition candidate. The
[master plan](../roadmap/FTQC_SIMULATION_MASTER_PLAN.md#52-mixed-clifford--non-clifford-simulation)
requires interference, phase and normalization before selecting that candidate.
This proposal specifies the next missing reference boundary without waiving
FTS-02/09 or final FTS-10 dependencies.

## Decision requested and selection consequence

Approve only groups A-D below: eight noise-free static circuits with n=2/3 and
T count 0/1/2/4, two named analytic anchors, private phase/mixture/cancellation
negative cases, and CPU complex128 comparisons with atol=1e-12 and rtol=0.
The numerical value needs explicit approval for this coherent-branch domain;
its use in the prior QEC conformance or static pilot is not inherited admission.

Implementation would be one private conformance slice in the existing Simulation
test tree, reusing current public unexpanded execution and independent small
reference infrastructure. It would not create a production branch executor or
modify the n=4/8 benchmark observer envelope. At most ten unexpanded public CPU
executions are needed per verification invocation: eight matrix cases and two
anchors. Clifford branch vectors are private scalar test references, not timed
production workloads. Tests use no shots, warm-up/measurement campaigns,
independent-process timing repeats, hardware or optional SDK profile.

A pass would qualify only the reference algebra for the declared finite cases.
FTS-10 selection still requires a phase-capable production representation,
branch/overlap cost evidence on the target W4 tasks and resource/refusal review.
The default FTS-12 decomposition candidate remains conditional. A dense sum of
small branch vectors cannot establish a stabilizer algorithm or capacity gain.
No further research collection or recovery budget is requested here.

## Current evidence and missing boundary

| Candidate | What is now established | What is missing for selection |
| --- | --- | --- |
| Statevector / MPS / tensor network | The admitted n=4/8 static matrix has 48 complete points across two campaigns, preserving one original monitor failure and all exact profiles. | Target W4 n=32/64, code-derived workloads, comparable per-workload outer wall for every campaign, capacity and method-selection criteria. |
| Coherent Clifford decomposition | The scalar gate identity below and an existing analytic interference example specify a correctness question. | No phase-capable production branch/overlap representation, bounded executor or measured comparison exists. Clifford sampling/fault propagation does not provide that capability. |
| Independent references | Existing scalar basis reference and analytic Bell/GHZ cases can be reused where their semantics apply. | The coherent cross terms, coefficient phase and cancellation failure boundaries proposed here are not yet qualified. |

No row selects a winner. Existing declared-DEM samplers, QEC decoder work and
native PyMatching are different domains and are not a T-state evolution path.
The independent Quafu SDK work in current main remains outside this proposal.

## Algebra and phase boundary

Use the canonical matrices H = [[1,1],[1,-1]]/sqrt(2), S = diag(1,i),
T = diag(1,exp(i*pi/4)) and the conventional CX permutation, with initial
computational zero state. The exact identity is

```text
b = (exp(i*pi/4) - 1) / (i - 1)
a = 1 - b
T = a I + b S
```

For t T instructions, enumerate all 2**t binary choices of I/S at the literal
authored positions. Each branch coefficient is the ordered product of its a/b
choices. Retain those complex coefficients, including cancellation, without
renormalizing individual terms. t=0 is one branch with coefficient one.

The coherent state is the sum of coefficient times canonical Clifford branch
vector. Its density includes every off-diagonal branch cross term. A classical
mixture of branch probabilities omits these terms. For the Bell-T-HH anchor,
normalizing the squared-coefficient mixture gives (3/8,1/8,1/8,3/8), whereas the
correct probabilities are ((2+sqrt(2))/8,(2-sqrt(2))/8,(2-sqrt(2))/8,
(2+sqrt(2))/8). This is a discriminating analytic negative, not a proposed
approximation.

Private scalar Clifford vectors carry the explicit canonical phase convention.
This proposal does **not** freeze a new canonical-phase guarantee for the public
`ExecutionResult.to_statevector()` accessor or a stabilizer representation.
The public unexpanded result is compared through probability, normalization and
density, which are invariant under one common global phase. Arbitrary independent
branch rephasing is not permitted unless its coefficient is inversely rephased.
A future production branch boundary must separately account for that phase; this
small reference test is not that boundary's implementation or approval.

## Group A: eight static descriptions

The fixed cases are the Cartesian product n in {2,3}, t in {0,1,2,4}. Reuse the
static pilot's four-layer open-chain gate ordering, specialized to these sizes:

1. Start at zero; apply H to each qubit in ascending order.
2. For each layer l=0,1,2,3, apply S to `(l+1) mod n`, then CX `(q,q+1)`
   for q=0..n-2 in ascending order. Apply T to `l mod n` only when l<t.
3. Apply H to every qubit in ascending order; retain computational-basis
   probability order q=0..n-1.

This defines exactly the authored t count. Expanding T changes only its literal
position to I/S, leaving Clifford order fixed. There are at most 16 branches and
8 amplitudes per branch; that bounds the private reference size, not total
runtime RSS. No ring, t=8, noise, gradient, sample, reset or adaptive control is
included. The existing benchmark CLI remains restricted to its approved matrix.

For each case compare three quantities without implementation self-consistency
as the only evidence:

- A private coherent sum of independently evolved canonical scalar Clifford
  branch vectors with the explicit coefficients above.
- An independent direct scalar T-evolution reference, reusing the existing
  basis oracle where it matches this exact ordering. It must not call the
  production gate/evolution/compiler/decoder being checked.
- Existing forced exact CPU complex128 statevector execution of the unexpanded
  circuit, through `fq.Circuit`, `fq.run` and its existing result accessor.
  Approximation and backend fallback stay disabled. Use one CPU numerical
  thread in the private fixture and restore prior thread settings afterwards.

Enumerating the fixed matrix requires 46 scalar branch vectors and eight
unexpanded public executions. It is a tiny reference experiment, not a fourth
numerical core, alternate builder, new runtime mode or coherent engine benchmark.

## Group B: two analytic anchors

- H-T-H on one qubit: probabilities ((2+sqrt(2))/4,(2-sqrt(2))/4).
- Bell preparation H(0), CX(0,1), then T(0), H(0), H(1): the four probabilities
  given above. The current runnable public example below is this anchor.

Compare the analytic answer independently with the coherent scalar sum and the
unexpanded public result. Canonical scalar branch vectors for these anchors
require at most four additional tiny vectors, two per anchor. No public branch
execution or phase correction API is introduced.

## Group C: meaningful phase and mixture negatives

Reuse the already computed private vectors; these negatives add no public
execution calls. Each negative must actually distinguish the wrong result on its
named anchor, not merely assert an implementation detail.

- Rephase one scalar branch and inversely rephase its coefficient: the coherent
  total must be unchanged. One common global phase must leave density unchanged.
- Rephase one branch without compensating its coefficient: the Bell anchor must
  expose the incorrect interference; do not align branch phases independently
  to the answer.
- Replace the Bell coherent sum with the normalized squared-coefficient mixture:
  it must produce the wrong (3/8,1/8,1/8,3/8) distribution above.
- Remove off-diagonal branch cross terms without normalizing: the norm/trace
  failure must remain visible. Do not normalize the wrong mixture into a pass.
- A synthetic identical-vector pair with coefficients (1,-1) has zero total
  vector. A private normalization request must refuse it before division; an
  impossible zero-norm reference is not a physical successful state.

## Group D: numerical acceptance and refusal boundary

The specifically requested test values are complex128 atol=1e-12/rtol=0 for
absolute probability, squared-norm residual and full density-matrix error. For
the coherent and direct scalar references, their explicit canonical amplitude
vectors must also agree within that absolute value. Public unexpanded state
comparison remains global-phase invariant; no per-branch public phase promise
is inferred. Check finite entries and ordered shape before reducing errors.

Do not normalize the coherent sum or direct reference to hide norm loss.
Normalization is permitted only after independently verifying nonzero unit norm
within the approved value, and density error must not replace the norm check.
Keep coefficient/gate-order mistakes observable through both analytic anchors.
Refuse private matrix requests outside group A before allocating branch
vectors; group B admits only its two explicitly named anchors, not generic
one-qubit support; never add a truncation, approximate mixture, backend fallback or
benchmark-envelope extension to make a case pass.

These are proposed private verification gates, not runtime guarantees or a new
public exception/result schema. Any discovered production defect is reported
before changing protected behavior, fixtures or thresholds. If a new production
phase contract is needed, form and approve it separately.

## Review sequence and remaining decisions

1. Review this finite contract and the independence/phase boundary.
2. After exact-head CI and review gates make the proposal reviewable, obtain
   explicit human approval for groups A-D and the comparison values.
3. Only then add the missing private conformance tests in an independent direct
   main PR with a current runnable public example and exact-head evidence.
4. Review the tests/main admission. Keep production branch representation,
   target-scale candidate comparison, dynamic Born/conditional semantics and
   final FTS-10 selection as separate unmet decisions.

No new tests, branch enumeration or candidate research run has been implemented
or executed for this proposal. Both prior raw campaigns and their collectors
remain immutable; their consumed budgets do not authorize another attempt.

## Runnable current user example

Requirements: supported Python, core FlagQuantum and CPU; no optional SDK,
physical device, credentials or dataset. Save as `/tmp/ftqc_interference_example.py`.
Installed command: `python /tmp/ftqc_interference_example.py`; source command:
`PYTHONPATH=. python /tmp/ftqc_interference_example.py`.

```python
import flagquantum as fq

circuit = fq.Circuit(2).h(0).cx(0, 1).t(0).h(0).h(1)
for mode in ("statevector", "mps", "tensor_network"):
    options = fq.ExecutionOptions(
        mode=mode, device="cpu", precision="complex128",
        allow_approximate=False, allow_backend_fallback=False,
    )
    probabilities = fq.run(
        circuit, outputs=fq.probabilities([0, 1]), options=options,
    ).probabilities
    assert probabilities is not None and probabilities.shape == (1, 4)
    print(mode, *[f"{float(p):.8f}" for p in probabilities[0]])
```

Expected stdout:

```text
statevector 0.42677670 0.07322330 0.07322330 0.42677670
mps 0.42677670 0.07322330 0.07322330 0.42677670
tensor_network 0.42677670 0.07322330 0.07322330 0.42677670
```

This existing two-qubit sanity example is not a failed-study-point retry or a
continuation launch. Its success does not establish supervision availability.


