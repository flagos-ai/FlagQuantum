# M3 next scenario: a spin and mode in a fixed thermal bath

Status: **all five decisions approved on 2026-10-26; implementation #897 accepted**.
Approval permits independent acceptance followed by existing-interface example
and integration-test composition. It grants no public API/production change or
PR merge; each PR still needs its own conditional merge authorization.
Prepared on 2026-10-26 after accepted relaxation implementation #886 and its
accepted progress record #893. Latest remote main was
`b3961dc7c491b5fb5153288e00759265feb7b57f`; comparison with reviewed documentation
head `f3b83b85a2aec480a01f14b7370575f8a42947bc` showed no changes to the relevant
examples, Lindblad/boson interfaces or M3/roadmap records. The approved delivery
adds an example and independent integration tests through existing APIs; library
production code, public schemas and capability maturity remain unchanged.

## Remaining gaps and recommended next slice

The [M3 inventory](M3_SCENARIO_ACCEPTANCE.md) now accepts both closed and fixed
two-channel dissipative qutrit references, and one intermediate Rydberg presence
trajectory. These references do not certify full M3/M4 or a physical device.

| Scenario | Remaining gap | Small next slice and contract cost |
| --- | --- | --- |
| Driven qutrit | No calibration, higher-level truncation, pulse emission or hardware | Another calibrated model needs external device evidence; adding another fixed channel alone adds little new acceptance. |
| Moving neutral atoms | No feedback, motional heating, calibrated paths or device execution | Feedback changes branch/control semantics; motional heating introduces a new state space beyond the current g/r/lost basis. |
| Spin and oscillator | Current exchange model is lossless; no bath-induced heating | Fixed downward/upward oscillator jumps reuse the same state space, operators, planner and results; recommended. |
| Photonic interference | No destructive detector or optical frontend | Destructive observation requires explicit output-space and conditional-result decisions; hardware needs separate access/evidence. |
| Ising fusion | Fixed sector and one observation; no feedback | Further sectors or conditional controls need independent source/sector contracts. |
| Circuit deployment | Local packaging does not establish a submitted service job | Remote credentials, provider access and job evidence belong to a separate deployment slice. |

The recommendation extends a previously accepted complete workflow with a new
physical effect and a new independent cutoff witness, without adding a frontend,
source operation, result field or external service. It is a finite local-bath
reference, not an ion-trap heating model or a generic thermalization feature.

## User workflow and reuse-only boundary

After approval, a user would select a finite mode cutoff, prepare the same
conditioned thermal mode and excited spin, declare fixed bath rates, inspect
spin population/mode occupation/boundary occupation, check cutoff sensitivity,
fit one continuous exchange coupling, and save/replay a detached numeric plan.

Reuse [spin_mode_convergence.py](../../examples/spin_mode_convergence.py), existing
boson matrix construction, `fq.Parameter`, numeric mixed-dimension `fql.plan/run`,
fixed `collapse_operators`, `fq.expectation`, autograd, `EvolutionResult` and
`LindbladPlan` JSON reading. Simulation owns this example and its scenario tests;
Integration reviews shared progress prose. Keep the old lossless example,
analytic Rabi blocks and existing tests unchanged.

The proposed delivery is a separate example and integration test, not a library
production change. Reuse current schemas and actual planner-selected versions.
Add no root exports, detector/event records, IR, model class, registry, solver,
result class or dependency. Root `fq.run` admission does not broaden. If the
existing public path cannot express this model, stop and report the mismatch
for separate review; do not silently expand its contract.

## Fixed model and units

Use CPU complex128, subsystem order `(spin, mode)` and dimensions `(2, N)`.
Spin basis is ground/excited; mode basis is `0,...,N-1`, with spin most significant.
Time is in consistent arbitrary units and exchange g has inverse-time angular
frequency units. Bath rates have inverse-time units, not angular frequencies.

```text
a_N |n> = sqrt(n) |n-1>; a_N^dagger |N-1> = 0
s_- = |ground><excited|; s_+ = s_-^dagger
H = g (s_+ tensor a_N + s_- tensor a_N^dagger)
L_down = I_spin tensor a_N,       gamma_down = 0.3
L_up   = I_spin tensor a_N^dagger, gamma_up   = 0.1
d rho/dt = -i[H,rho] + gamma_down D_down(rho)
                        + gamma_up D_up(rho)
D_L(rho) = L rho L^dagger - {L^dagger L,rho}/2
D_down = D_L for L=L_down; D_up = D_L for L=L_up
```

Set g=0.6, final time T=0.7 and primary output times `(0,T/2,T)`.
The primary initial state is `|excited><excited| tensor rho_q,N`, where q=0.2 and
`rho_q,N = sum_n (1-q)q^n/(1-q^N) |n><n|`. Include the coherent-spin input
`(|ground>+i|excited>)/sqrt(2)` with the same mode state as a full-density witness.
For bath-only analytic checks set g=0. Zero-bath checks set both rates to zero.

Supply unscaled full-space L matrices plus their separate rates. Never scale L
by a square-root rate and also supply the rate. The conventional bath notation
is `gamma_down=kappa*(n_bath+1)` and `gamma_up=kappa*n_bath`; here kappa=0.2 and
n_bath=0.5. This two-jump convention is documented in the primary
[QuTiP thermal-oscillator example](https://qutip.org/docs/4.2/guide/guide-steady.html#example-harmonic-oscillator-in-thermal-bath).
QuTiP is a physics convention reference, not a new runtime/test dependency.

Initial mean occupation tends to 0.25 before finite conditioning, so the bath
raises mode occupation when g=0. At nonzero g, spin exchange and bath flow both
affect occupation; do not assert every observed trace is monotone heating.
No bath rate, cutoff, time or temperature is trainable in this slice. There is
no Kelvin calibration, spin damping, dephasing, non-Markov bath, detector event,
trajectory sampling or gradients through random choices. This local dissipator
does not certify a global Gibbs state of the coupled interacting Hamiltonian.

## Finite boundary is part of the declared model

Do not replace `[a_N,a_N^dagger]` by identity: it equals
`I-N|N-1><N-1|`. The top has no upward transition, and normalization must not
silently redistribute blocked upward flux. The finite stationary oscillator
distribution at g=0 is proportional to `(gamma_up/gamma_down)^n`, truncated and
normalized on the same declared coordinates; it is not the infinite bath state.

With `M = spin_excitation + mode_number` and
`p_top = <I_spin tensor |N-1><N-1|>`, the exact finite moment equation is

```text
d <M>/dt = -(gamma_down-gamma_up) <mode_number>
           + gamma_up (1 - N*p_top).
```

Exchange commutes with M even at finite N. Requiring M to remain constant in
this open model would be a wrong acceptance test. Report occupation flow, not
physical heat/energy or entropy production without additional frequency and
thermodynamic definitions.

## Independent reference and separate acceptance budgets

All numbers below are approved acceptance thresholds, not measured results.
Validate independent witnesses, then freeze their source,
AST hashes, inputs and thresholds before composing the example. Preserve failed
evidence and do not relax a budget to make the implementation pass.

| Witness | Approved required assertion |
| --- | --- |
| Scalar finite generator | Independently fill exchange entries `H[excited,n;ground,n+1]=g*sqrt(n+1)` and their conjugates, and jump entries from scalar n. Apply the displayed equation to every elementary matrix E_ab; propagate with independent full-matrix exponentials. Import no production model/Liouvillian/kernel/integrator builder or captured matrices into the oracle. Sharing low-level `torch.matrix_exp` is disclosed; this is independent assembly, not an independent solver backend. |
| Bath-only populations | At g=0, independently assemble an N-state birth/death generator: downward rate `gamma_down*n`, upward rate `gamma_up*(n+1)` only for n<N-1; diagonal is minus the actual outgoing sum. Compare every mode population at all three times with its exponential, atol=1e-10, rtol=0, for N=3,5,7,9,11. The bath leaves the spin density unchanged at g=0. |
| Analytic bath checks | At zero rates the mode state is unchanged. The finite normalized geometric state with q_bath=1/3 is stationary at g=0. Check the infinite-mode mean formula and its explicit finite-boundary bound below; do not demand exact equality to the infinite formula at finite N. |
| Coupled complete density | For g=0.6, all N=3,5,7,9,11 and both declared initial spin states, compare every complex density entry at all primary times against the scalar oracle, atol=1e-10, rtol=0. Output order, basis order, complex coherence, dimensions, times and retained-density axes must match the existing result contract. |
| Physical state and boundary | At retained rows: trace/Hermiticity maximum error <=1e-10 and smallest Hermitian eigenvalue >=-1e-10. Independently check the finite adjoint-generator moment identity, maximum entry error <=1e-12, including its nonzero top correction. No clipping, PSD projection or silent renormalization may conceal a failure. |
| Zero-bath compatibility | At both zero rates, the excited-thermal input matches the unchanged lossless analytic Rabi-block reference at atol=1e-10, rtol=0; reproduce its finite dark top rather than changing it. Existing lossless example, tests and stored public artifacts remain unchanged. |
| Cutoff sensitivity | Report spin excitation, mode occupation and p_top at N=3,5,7,9,11. For N=9 vs11 at the primary times require absolute spin-population difference <5e-5 and occupation difference <1e-4. At N=11 require p_top<1e-5. These are finite-cutoff stability checks, not a rigorous error certificate for the coupled infinite-dimensional density. Do not demand monotone errors or compare differently sized matrices directly. |
| Integration error | At fixed N=3, g=0.6 and the excited-thermal input, compare Runge-Kutta results to the independent finite exponential on uniform grids with 8,16,32,64 steps over T. Maximum full-density error at original times 0,T/2,T decreases by >10 per doubling; final error <1e-6. Refinement changes no physical parameter or cutoff. |
| Continuous gradient | At N=5, g=0.6, compare the derivative of terminal spin excitation against independent four-point finite differences with step 1e-4; absolute error <=1e-9. The oracle recomputes its scalar finite generator for each shifted g. No rate/time/cutoff gradient promise follows. |
| Fit and detached replay | At N=5, generate the terminal spin target using the independent reference at g=0.6. Fit from g=0.35 with Adam lr=0.04 for 40 steps; all objectives/gradients finite, final squared target error <=0.1 of initial. Validate the fitted terminal density against the independent reference at 1e-10. Rebuild a detached numeric plan and require exact same-environment JSON replay of retained densities/expectations. Report cutoff sensitivity at the declared g separately from parameter fitting; do not claim infinite-mode parameter recovery. |
| Existing refusals | Reuse current invalid dimension/state/operator/rate/nonfinite/binding and sealed-plan behavior. No new refusal promise, snapshot regeneration or public validation change is authorized. |

### Bath-only infinite-mean check and boundary bound

For g=0 and the unconditioned geometric initial mode, define
`alpha=gamma_down-gamma_up=0.2` and
`n_inf(t)=0.5+(0.25-0.5)*exp(-alpha*t)`.
For the declared conditioned finite state the initial mean deficit is
`delta0=N*q^N/(1-q^N)`. Subtracting the finite moment equation gives

```text
n_inf(t)-n_N(t) = delta0*exp(-alpha*t)
  + gamma_up*N*integral_0^t exp(-alpha*(t-s))*p_top(s) ds.
```

A pure upward birth process with rates `gamma_up*(n+1)`, the unconditioned
geometric initial state and no downward jumps dominates this g=0 finite chain.
Its geometric ratio is `b(t)=1-(1-q)*exp(-gamma_up*t)`. Therefore

```text
0 <= n_inf(t)-n_N(t) <= delta0 + gamma_up*N*t*b(t)^(N-1).
```

Independently check this declared formula/bound at primary times for every N,
allowing 1e-10 numerical slack. It applies only to bath-only occupation for the
specified initial family. It is not a bound on coupled density, spin population,
gradient, physical-model error or hardware heating. For coupled cutoff checks,
report actual successive-cutoff differences and boundary occupation separately.

## Review and staged delivery

The user approved these five decisions:

1. Select this fixed local thermal-bath spin-mode scenario as the next bounded
   M3 slice, with the declared inputs, rates, cutoff sequence and exclusions.
2. Keep public interfaces, schemas, production code and capability maturity
   unchanged; deliver only an example, independent integration tests and honest
   progress prose after approval. Any unmet API requirement returns for review.
3. Approve the explicit blocked-top model, finite moment correction, bath-only
   analytic bound and distinction between cutoff stability and infinite-space
   certification.
4. Approve the proposed independent witnesses, fixed numerical/gradient/cutoff
   budgets and deterministic fit/replay workflow; freeze them before example
   implementation. Actual validation is recorded separately below.
5. Require complete applicable local tiers, actual Python 3.10/3.11/3.12
   coverage, ownership/architecture/API/type/style/language/link checks and
   latest complete CI/GPU before readiness. Ask once for each resulting PR's
   own conditional merge grant; approval of a scenario is not a merge grant.

The explicit human approval above authorizes this bounded example/test delivery.
The prior #886 and #893 merge grants do not authorize its PR merge. No dependency
or library production implementation is authorized. Update the existing
M3 inventory to accepted only after independently authorized delivery and actual
post-merge smoke, retaining all remaining M3/M4 and device boundaries.

## Accepted implementation evidence

The independent scalar reference first passed all five analytic cutoff cases.
Its source and all acceptance-function ASTs were captured before the example
existed. The six scalar reference functions remain unchanged; after the example
became an importable first-party module, six consumers received import-order-only
Ruff fixes. No physical input, assertion or threshold changed. The pre-example
missing-module run and initial import diagnostics are retained externally.

The new 24 cases and five unchanged lossless cases passed on Python 3.10/3.12.
Measured complete-density maximum error was 2.48e-14; gradient error was
4.12e-12. At primary times, N=9 versus N=11 spin and occupation differences
were 3.76e-6 and 1.23e-5; N=11 top occupation stayed below 1.04e-6.
RK errors for 8/16/32/64 steps were 2.14e-8, 1.33e-9, 8.23e-11 and 5.13e-12.
These are separate witnesses, not an infinite coupled-density error certificate.

Run the implemented public example:

```bash
python -m examples.spin_mode_thermal_bath
```

Representative executed CPU output ends with:

```text
Population objective: 0.0160018399 -> 0.0000712776
Fitted finite-model coupling: 0.6139171498
Cutoff stability checked separately; exact detached numeric replay: True
```

The fitting target 0.8022880147390464 was calculated using the frozen scalar
reference at N=5, g=0.6 before example implementation; its provenance is explicit
in the example. The fitted value is a finite-model fit, not infinite-mode or
physical parameter recovery. Complete default verification at code-identical head
`ed557965c5d1dae93a862b77495f999503240257` passed 16,049 cases/224 skipped
(412.16s), and integration passed 2,539/177 skipped (193.38s); both XML reports
have zero failures/errors. The first default attempt retained 16,048 passed,
224 skipped and one document-link failure: the hygiene scanner read the original
D-bracket notation as a link to rho. Equivalent D_L notation corrected only the
Markdown equation, and the unchanged hygiene test then passed. Original negative
logs/XML remain preserved. Public APIs, scientific code and frozen budgets did
not change. Latest head `bf9afadc8cca81b5f0d7ae48462f32e2df7e71a1` then passed
complete CI `40277142842` and GPU `40277142859` (attempt 1), including all 54
standard CI jobs and the actual remote Python 3.10/3.11/3.12 matrix. All 60
checks completed with 57 success and only the original three manual-pool skips.
The final progress-doc commit retained identical physical source/test blobs.
Local Python 3.11 is not claimed.

[#897](https://github.com/flagos-ai/FlagQuantum/pull/897) was externally merged by
FlagQuantum as `5ee74364cdbab17d1f0bebbee253bf400b467800` on 2026-10-26.
Main smoke `40279655585`
succeeded on the merge commit; this is only smoke, not full main CI or hardware
validation. This accepted slice grants no future PR merges or additional
protected capability, and does not declare all M3/M4 gates complete.
