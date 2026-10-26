# M3 next scenario: driven three-level relaxation

Status: **all five decisions approved by the user on 2026-10-26**.
Approval authorizes freezing independent acceptance and composing the example
and integration tests through existing APIs. It authorizes no public API
expansion or production modification; each PR needs its own merge grant.
Reviewed on 2026-10-26 against accepted main
`f22e05929d3924e759b0e7032532ddca7085e8c7` (#882).
This document freezes the approved scenario semantics and acceptance criteria.
The accompanying example and integration test compose existing APIs; no library
production code, API, schema, capability or maturity level changes.

## Remaining gaps and selection

The [accepted M3 inventory](M3_SCENARIO_ACCEPTANCE.md) already contains six
bounded physical references plus circuit training/deployment packaging. Proposal
092's single intermediate presence trajectory is accepted; it does not authorize
feedback or additional observations.

| Existing scenario | Remaining gap | Candidate next slice and dependency |
| --- | --- | --- |
| Driven three-level model | Closed evolution only; no dissipation or device calibration | Fixed two-channel relaxation with existing numeric Lindblad planning; recommended. |
| Moving neutral-atom pair | No feedback, heating, calibrated motion or hardware | Feedback needs branch-dependent control semantics and separate protected-contract review; heating needs a motional state space. |
| Spin and motional mode | Lossless rotating-wave model; no heating | A heating bath needs finite-cutoff boundary and tail-error acceptance in addition to dynamics. |
| Two-mode photonic interference | Finite Fock reference; no destructive detector/frontend/hardware | Destructive detection needs an explicit output-space and conditional-state contract. |
| Ising braiding/fusion | Fixed four-anyon vacuum sector; one observation only | Other sectors or feedback require new source/result semantics and independent sector oracles. |
| Circuit deployment packaging | Local provider contract, no real submitted job | External service access and provider evidence are a separate deployment workstream. |

Selection uses three criteria: a new physical effect visible to the user, a
complete input-to-replay workflow, and the fewest unproven contracts. Relaxation
adds a genuinely open-system effect to the existing driven model without a new
state space, detector, branching language, provider or serialized profile.
It is a proposed next slice, not a claim that other gaps are less important or
that M3 is complete.

## User outcome and bounded physical contract

A user declares four complex pulse envelopes and fixed positive relaxation
rates, inspects excitation/leakage and relaxation, differentiates a terminal
objective with respect to all eight pulse controls, performs a short optimizer
loop, and saves/replays a detached numeric plan on CPU without credentials.

Reuse the ordered basis `(0, 1, 2)` and four duration-0.25 constant segments from
[driven_three_level.py](../../examples/driven_three_level.py). Set total time to
1 and use consistent arbitrary time units. Angular frequencies have inverse-time
units; relaxation rates have inverse-time units and are not angular frequencies.
For segment k, define the rotating-frame Hamiltonian explicitly as

```text
H_k = [[0,            (x_k - i y_k)/2,       0],
       [(x_k+i y_k)/2, 0.15,                 sqrt(2)(x_k-i y_k)/2],
       [0,             sqrt(2)(x_k+i y_k)/2, -2.7]]
L_10 = |0><1|, gamma_10 = 0.4
L_21 = |1><2|, gamma_21 = 0.7
d rho/dt = -i[H_k,rho] + sum_j gamma_j *
           (L_j rho L_j^dagger - {L_j^dagger L_j,rho}/2)
```

Supply unscaled L matrices with their separate rates through existing
`collapse_operators`; do not multiply by square-root rates as well. The rates
are fixed and state-independent. Zero rates are a separate limiting-case
witness; negative/nonfinite rates retain existing admission failures. No new
rate validation rules or trainable-rate promise is proposed.

This phenomenological zero-temperature cascade has no upward jumps, pure
dephasing, detector events, pulse emission or device-calibration meaning.
The levels are retained coordinates, not a certified truncation of an infinite
oscillator. Excitation/leakage must not be described as gate fidelity.

The primary training input is `|0><0|`; deterministic acceptance additionally
uses all three basis-state densities and a full-rank coherent state
`0.7 I/3 + 0.3 |v><v|`, where `v=(1,i,1)/sqrt(3)`. This explicitly exercises
complex off-diagonal elements and mixed inputs.

## Existing contract and ownership

Simulation owns this scenario composition and its numerical witnesses.
Integration reviews this document and any changes to shared acceptance prose.
Reuse the existing `fql.plan/run`, numeric mixed-dimension Hamiltonian and
collapse inputs, `(3,)` dimensions, interval `parameter_values`, `fq.Parameter`,
`fq.expectation`, existing integrators, autograd and `LindbladPlan` JSON reader.

The current planner explicitly supports fixed collapse operators/rates with
interval-bound Hamiltonian coefficients. Its existing `EvolutionResult`,
population/expectation/time/density axes and applicable numeric plan version
remain authoritative. Root `fq.run` is not expanded to ordinary Lindblad plans.
Numeric JSON replay is detached from training; rebuild live plans after each
optimizer update.

Proposed delivery is a separate readable example and scenario integration test,
linked from the existing M3 inventory. Leave the closed-model example, its
independent unitary reference and all current tests unchanged. Do not add a
general frontend, model class, registry, solver, facade, result class or helper
layer merely for this example. Do not modify protected snapshots or versions.
If existing APIs cannot express the declared workflow, report that mismatch
and obtain separate review before any production fix or contract expansion.

## Independent acceptance criteria

These are approved pass/fail budgets, not measured results. Freeze
the independent witnesses and thresholds before implementation. A failure is
diagnostic evidence; do not weaken a threshold or change the model to hide it.

| Witness | Required assertion |
| --- | --- |
| Exact undriven cascade | With H=0 and initial level 2, compare every population against `p2=exp(-gamma21*t)`, `p1=gamma21*(exp(-gamma21*t)-exp(-gamma10*t))/(gamma10-gamma21)`, `p0=1-p1-p2`, at times 0, 0.25, 0.5, 0.75, 1. Maximum absolute error <=1e-10, rtol=0. Include level-1 decay and zero-rate limits without dividing by zero. |
| Coherence damping | With H=0, independently check `rho01(t)=rho01(0)*exp(-gamma10*t/2)`, `rho02(t)=rho02(0)*exp(-gamma21*t/2)` and `rho12(t)=rho12(0)*exp(-(gamma10+gamma21)*t/2)` for the coherent input; same 1e-10 budget. |
| Driven full density | Construct the 9x9 reference generator by applying the displayed scalar equation to each elementary matrix E_ab, then compose independent matrix exponentials in chronological pulse order. Do not import production Liouvillian/model/Kraus/integrator builders or production-captured matrices into the oracle. Compare all complex density entries at all original pulse boundaries for all four inputs at atol=1e-10, rtol=0. |
| Trace, Hermiticity, positivity | Trace error <=1e-10, maximum anti-Hermitian entry <=1e-10, smallest Hermitian eigenvalue >=-1e-10 at retained rows. Assert visible population decay and nonzero leakage separately; trace preservation alone is insufficient. No positivity projection or renormalization may conceal a failure. |
| Pulse ordering | The existing noncommuting pulse sequence and its reversal must produce distinct independent terminal densities (maximum-entry difference >1e-3); each production result must match its own ordered reference. |
| Integration error | Subdivide each identical constant segment by 1, 2, 4, 8; compare Runge-Kutta density errors over original boundaries with the independent exponential. Each step doubling reduces error by >10 and final error <1e-6. Preserve pulse boundaries and fixed rates; report integration error separately from physical-model error. |
| Continuous gradients | For terminal excited population and leakage, check all eight real x/y control derivatives against independent four-point finite differences, step 1e-4, absolute error <=1e-9. Finite differences use only the independent reference; no gradient promise for stochastic choices, rates or times. |
| Training workflow | From the existing `initial_controls`, optimize `(1-P1(T))**2 + 2*P2(T)` using Adam lr=0.08 for 40 steps under the declared dissipative dynamics; require finite losses/gradients and final objective <=0.9 of initial objective. Validate final populations with the independent reference. This is a proposed deterministic acceptance target, not a measured performance claim. |
| Detached replay | After training, build a fresh detached numeric plan, JSON round-trip through the existing reader and reproduce density/expectation tensors exactly on the admitted same-build CPU environment. Cross-build physics uses numerical budgets, not byte-identical floating-point claims. |
| Compatibility and refusals | Preserve old source/plan captures and closed-model outputs; existing malformed shape/dtype/rate/binding/nonfinite failures retain their exception and failure-stage contracts. Do not add tests that only assert document wording. |

The reference may share a low-level `torch.matrix_exp` primitive with the
implementation; independence resides in the scalar model, generator construction,
analytic cascade/coherence checks and chronological composition. State that
limitation explicitly rather than calling it an independent solver backend.

No sampling interval, Fock cutoff, hardware error or scalability budget follows
from these tolerances. The finite qutrit has no new cutoff-convergence claim.
Keep existing three-level 1e-10 density and 1e-9 gradient budgets intact.

## Proposed delivery and review gates

1. Human review approves or revises the scenario, reuse-only boundary and budgets
   above. This draft grants no implementation or merge permission.
2. Establish the independent analytic/density/gradient acceptance witnesses
   before composing the new example. Keep the old closed reference unchanged.
3. After approval only, deliver the example and integration scenario through
   existing APIs; add no production changes under this authorization.
4. Run focused scenario and existing closed-model tests, Python 3.10/3.11/3.12
   coverage, appropriate default/integration tiers, ownership/architecture,
   language/links/style/type gates and complete applicable CI/GPU. A missing
   local Python environment is not a passing result; use the actual remote matrix.
5. Record actual output, oracle errors, gradient errors and convergence results.
   Mark the implementation PR ready only after its current head passes; request
   that PR's own conditional merge authorization once. Verify main smoke after
   merge and update the existing M3 inventory without declaring M3/M4 complete.

The user approved these five review decisions: (1) select this scenario next; (2) accept the fixed
two-jump physics and explicit exclusions; (3) accept unchanged public/schema
surfaces and composition-only delivery; (4) accept the independent witnesses and
separate budgets; (5) permit implementation only after explicit approval, with
independent merge grants retained for every PR.

## Implementation review evidence

The independent reference was written and analytically checked before the
example: four rate combinations passed cascade/coherence checks. Its source and
function AST hashes were captured externally before example implementation.
The initial consumer run failed because the new example did not yet exist;
that negative log is retained. The four reference functions remain unchanged.
Consumer-only mechanical changes make `zip(..., strict=False)` explicit, sort
an import and use the typed `torch.autograd.backward(loss)` entry; physical
assertions, inputs and budgets are unchanged.

Run the public workflow with:

```bash
python -m examples.driven_three_level_relaxation
```

Executed local CPU output:

```text
Fixed relaxation rates: gamma_10=0.4, gamma_21=0.7
Training objective: 0.87460332 -> 0.09593434
Final ground/excited/leakage: [0.2220132, 0.75904846, 0.01893835]
Exact detached numeric replay: True
```

The new 20 scenario cases and four unchanged closed-model cases passed on local
Python 3.10 and 3.12. Full local and remote acceptance remains pending at this
draft stage. Do not infer merge approval or accepted M3 completion from this
record. The matrix-exponential reference shares the low-level Torch primitive,
not production generator assembly. Physical budgets above remain authoritative.
