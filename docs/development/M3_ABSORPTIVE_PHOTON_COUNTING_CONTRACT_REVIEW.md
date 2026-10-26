# M3 next scenario: terminal absorptive photon counting

Status: **approved for existing-interface implementation; delivery accepted through #921 on 2026-10-26**.
The user instructed "do" in direct response to this five-decision review. This
authorizes the bounded example/test delivery below, not production/API changes
or any PR merge; each PR still needs its own conditional human grant.
Prepared on 2026-10-26 after thermal-bath scenario #897 and accepted progress
record #903. Latest remote main was
`4bb6657d3004cb111439e1e711b9d7a90c24d3a3`. Comparison with reviewed
documentation head `93c595d6ee3c8cf7d15321ede8e99ce82dda51db` showed no changes
to the relevant photonic example, terminal sampler, SuperOperator or M3 records.
The implementation may add only the reviewed example, independent tests and
honest progress prose. Public APIs, schemas, dependencies and capability
claims remain unchanged. Earlier approvals are not its merge authorization.

## Remaining gaps and selection at the original review

The [accepted M3 inventory](M3_SCENARIO_ACCEPTANCE.md) now covers fixed
dissipation in the qutrit and spin-mode references, one intermediate atom-presence
trajectory and one sampled Ising observation. It retains these separate gaps:

| Scenario | Remaining boundary | Candidate cost |
| --- | --- | --- |
| Driven qutrit | No calibrated pulses, higher-level truncation or device execution | Needs a separately justified physical model or external device evidence. |
| Moving atoms | No feedback, motional states or calibrated transport | Feedback requires branch/control contracts; motional states exceed g/r/lost. |
| Spin and mode | No global interacting Gibbs or calibrated heating model | Another fixed bath is not a new complete acceptance outcome; microscopic calibration needs separate evidence. |
| Two-mode photons | Current number instrument is QND, with no absorptive detector reference | A terminal vacuum-reset instrument fits existing square Kraus maps and sampler; recommended. |
| Ising fusion | Fixed sector and one observation; no feedback | New sectors or branches need new independent source/sector contracts. |
| Deployment | Local package/provider does not prove an external submitted job | Needs explicit service access and provider/job evidence. |

Recommend one finite two-mode absorptive counting workflow. It adds physically
distinct back-action and heralded conditional-state checks using existing
owners, without a new frontend, controller, rectangular channel or result type.
It does not establish general destructive-detector or photonic-device support.

## Existing boundaries and the user outcome

[Proposal 083](API_CHANGE_PROPOSAL_083_TERMINAL_INSTRUMENT_SAMPLING.md) and the
current implementation already distinguish probabilities from physical
back-action. `SuperOperator.from_kraus` accepts square matrices of one shared
dimension. `sample_terminal` accepts complete ordered full-space Kraus branches
and returns the existing `MeasurementResult` with indices and captured metadata;
it does not return conditional states or infer physical loss.

The approved user workflow is:

1. Reuse [boson_interference.py](../../examples/boson_interference.py) to evolve
   the declared two-mode input, including declared propagation loss.
2. Declare a complete terminal absorptive instrument for mode 0 and a detector
   efficiency, while preserving mode 1 and the existing joint matrix coordinates.
3. Use existing `sample_terminal` for ordered observed counts and inspect all
   outcomes, including zero counts and unobserved physical loss.
4. Separately use `SuperOperator.from_kraus(...).apply(rho)` to analyze every
   deterministic branch and its conditional remaining-mode state.
5. Replay the detached numeric evolution plan and explicitly reconstruct the
   same declared instrument and sampling inputs in the admitted environment.

This is example/integration-test composition owned by Simulation, with
Integration reviewing shared acceptance prose. Keep the existing QND counting
example, tests and terminal-sampler contract unchanged. Add no library production
code, root export, source/plan version, detector registry, result field or helper
framework. Existing `EvolutionResult` and `MeasurementResult` retain their
different meanings. If an unmet existing-interface requirement is found, return
for separate review instead of expanding the contract.

The measured mode remains in the joint coordinates, reset to vacuum. Reading
its vacuum block for analysis does not create a reduced-state result contract
or mean that the runtime removed a mode. The sampler still returns indices only.
No rectangular Kraus matrix or new public conditional-state runner is proposed.

## Fixed finite physical model

Use CPU complex128, ordered mode basis `|a,b>` with index `N*a+b` and N=3 or 4.
The primary input is `|1,1>` and the existing balanced coupling g=0.7 evolves
for `T=pi/(4*g)`. Each mode has the same fixed propagation-loss rate
lambda in `{0,0.2}`, so transmission is `tau=exp(-lambda*T)`.
Propagation loss and detection efficiency are separate physical inputs.

Measure mode 0 once, terminally, with efficiency eta in `{0,0.65,1}`. It absorbs
all incoming photons in that mode; eta is the per-photon registration probability,
not the absorbed fraction or propagation transmission. Ordered labels are the
strings `0,...,N-1`, including zero-probability labels. There is no dark count,
saturation, timing, feed-forward, detector calibration or hardware submission.

For observed count m and actual incoming mode occupation n>=m, define

```text
w(m|n) = binomial(n,m) * eta^m * (1-eta)^(n-m)
K_mn = sqrt(w(m|n)) * (|0><n| tensor I_mode1)
J_m(rho) = sum_(n=m..N-1) K_mn rho K_mn^dagger.
```

Each outcome contains a sequence of distinct Kraus matrices, summed
incoherently. Replacing that sequence by one coherent sum is a different map
and is refused by the physical acceptance. At eta=0, w is one only for m=0;
at eta=1, it is one only for m=n. State these endpoint rules explicitly.
Keep every branch sequence nonempty, including valid zero matrices.

The total effect is identity on the whole declared finite space:
`sum_mn K_mn^dagger K_mn = I`. The unread action is vacuum replacement on mode 0
and tracing out its incoming occupation, with mode-1 coherences retained within
each fixed incoming n block. Zero registered photons do not imply no absorption
or an originally empty mode. In particular eta=0 still absorbs the measured mode.

Use the original branch trace p_m to normalize a positive branch; corrected
draw probabilities are not normalization denominators. A zero branch contributes
the zero matrix to the ensemble and has no normalized conditional state. Do not
divide by zero, discard its label or invent a conditional state. This is an
example analysis rule, not a new runtime result or public refusal contract.

For evolved floating-point inputs, an analytically zero branch can have a tiny
residual trace. Check its unnormalized matrix/weight against the declared
1e-10 error budget and report the residual; assign no conditional-state accuracy
claim and do not clamp it into a physical outcome or divide numerical noise by
itself. Conditional comparisons apply to analytically positive branches, all of
which must independently be verified to exceed 1e-8 for this finite test family.
This separation is an accuracy statement for these witnesses, not a sampler
threshold or a change to original trace/probability handling. Exact algebraic
zero-matrix witnesses remain exact and are never normalized.

Vacuum-reset number measurement is also described by the primary
[Strawberry Fields MeasureFock documentation](https://strawberryfields.readthedocs.io/en/latest/code/api/strawberryfields.ops.MeasureFock.html).
That convention is a reference, not a new dependency or a claim that this
example implements Strawberry Fields or its feedback semantics. The inefficient
instrument above is defined explicitly here; count effects alone would not
determine this back-action.

## Independent finite oracle

Construct the reference from scalar blocks, not production-generated instrument
or solver matrices. For arbitrary joint input rho, its only nonzero output block
for observed m is the measured-mode vacuum block:

```text
J_m(rho)[0,b;0,d] = sum_(n=m..N-1) w(m|n) * rho[n,b;n,d].
```

All other output entries are zero. This provides an independent full complex
branch oracle without a Kraus builder or production `SuperOperator`. Its unread
sum equals `|0><0| tensor sum_n rho[n,:,n,:]`.

For the primary balanced input with propagation loss, construct the pre-detector
reference independently as

```text
rho = tau^2 |HOM><HOM|
    + tau*(1-tau) (|1,0><1,0| + |0,1><0,1|)
    + (1-tau)^2 |0,0><0,0|
|HOM> = (|2,0> + |0,2>)/sqrt(2).
```

Its observed probabilities have the closed form
`P2=(eta*tau)^2/2`, `P1=eta*tau*(1-eta*tau)`, `P0=1-P1-P2`, with all higher
labels zero. These probability checks do not replace full branch-density checks.

Additional density witnesses are vacuum, each Fock `|n,b>` in the declared
coordinates, `( |0,0> + |1,1> )/sqrt(2)`, and
`( |1,0> + i|1,1> )/sqrt(2)`. The first coherent witness detects an incorrect
coherent sum across unresolved incoming photon numbers; the second checks that
mode-1 complex coherence is retained. Also exchange the modes and explicitly
measure the other named coordinate to expose ordering mistakes.

## Approved independent acceptance standards

These are approved acceptance thresholds, not results. Validate and
freeze independent references, physical assertions and budgets before composing
the new example. Preserve all failures and do not tune thresholds to pass.

| Witness | Approved pass/fail criterion |
| --- | --- |
| Completeness and branch algebra | For N=3,4 and all three eta values, full-space completeness error <=1e-12. Compare every complex entry of all J_m on the declared density witnesses to the scalar-block oracle at atol=1e-12, rtol=0, before normalization. |
| Absorption versus QND | At eta=1 and input `|2,0>`, both instruments give count 2 with probability one, but the proposed branch is vacuum in mode 0 while the unchanged QND branch retains occupation 2. At eta=0, the sole observed count is 0 while mode 0 still becomes vacuum. Require exact declared support, not just equal probabilities. |
| Unresolved alternatives and coherence | For the Bell-like witness, compare full m=0 output with the incoherent scalar-block sum; no spurious mode-1 off-diagonal term may be introduced. For the fixed-left-occupation coherent witness, retain its complex mode-1 coherence in every positive branch. Budget 1e-12, rtol=0. |
| Evolved HOM input | For both lambda values and both cutoffs, compare the existing evolved density with the independent attenuated HOM reference at 1e-10, rtol=0. Then compare every instrument branch and its original trace to the independent scalar oracle at 1e-10, rtol=0. Do not modify the existing HOM/loss oracle. |
| Conditional and unread action | For every analytically positive branch in the declared family compare normalized full joint densities and the remaining-mode vacuum block against the independent branch divided by its original trace at 1e-10, rtol=0. Require trace/Hermiticity error <=1e-10 and minimum eigenvalue >=-1e-10 without projection. Analytically zero branches are compared only as unnormalized matrices/weights within the declared budget, without assigning a conditional state; exact zero Kraus witnesses stay exact. The weighted conditional ensemble reproduces the unread scalar partial-block sum at 1e-10. |
| Closed count law | Compare all ordered probabilities against P0/P1/P2 above, at 1e-10, rtol=0, including eta=0/1 and all exact zero support. Retain vacuum and lost-photon sectors; no hidden survivor or click postselection. |
| Finite embedding | For all declared inputs with total photon number <=2, embed N=3 coordinates into N=4 and compare evolved/branch/positive conditional densities at 1e-10, rtol=0. The count-3 branch at N=4 remains zero. This is sector-closed embedding, not thermal or general infinite-Fock convergence. |
| Sampling and statistics | Use existing `sample_terminal` with explicit shots=4096, seed=7. Indices/labels/probabilities follow its existing rank-two input contract. Histograms retain all N labels and sum to 4096. For 0<p<1 require frequency error <6*sqrt(p*(1-p)/4096); p=0/1 obey exact support. No fixed individual sampled label is promised across Torch builds. Statistical intervals never replace deterministic density/probability budgets. |
| Replay and RNG | Existing numeric LindbladPlan JSON replay is exact on the admitted same-build CPU environment; explicitly reconstruct the same numeric instrument and sampler inputs, then require identical samples and deterministic branch states. Preserve global Torch and Python RNG states. Do not claim a new sealed combined evolution/detector plan, instrument reader or cross-build bitwise guarantee. |
| Existing API compatibility | Old QND example/tests/captures and all library source remain unchanged. Keep existing CPU/dtype/completeness/gradient/resource/shape refusals, including refusal of rectangular matrices. No new result metadata or accepted public types are authorized. Sampling receives detached state/operators; no gradient-through-detector promise is made. |

There is no optimizer requirement for this measurement slice. Its complete
workflow is physical input, validation, existing evolution, declared absorption,
deterministic branch analysis, terminal sampling/statistics and reproducible
replay. Continuous training and stochastic gradients retain their existing
separate contracts. Report numerical, finite-space and statistical budgets
separately; make no performance or scalability claim.

## Five review decisions

1. Select this terminal absorptive counting reference next, with finite N=3/4,
   the declared inputs/loss/efficiency values and explicit exclusions.
2. Approve the square vacuum-reset Kraus model and incoherent unresolved-n sum;
   keep measured-mode vacuum coordinates instead of adding rectangular maps or
   a removed-mode result contract.
3. Preserve current sampler/result/plan semantics. Conditional densities remain
   explicit deterministic example analysis, not inferred runtime fields, native
   event trajectories or feedback. No production/public/schema change is allowed.
4. Approve and freeze the scalar full-density, coherence, QND distinction,
   conditional/unread, count-law, embedding, statistical and replay witnesses
   with the separate proposed budgets before implementation.
5. After explicit approval only, implement the example and independent
   integration tests, synchronize honest planning progress, and require complete
   applicable local tiers, actual Python 3.10/3.11/3.12 coverage, repository
   gates and latest complete CI/GPU before readiness. Every PR still requires
   its own independent conditional merge grant and actual post-merge smoke.

The approval above authorizes only this existing-interface example/test slice.
It grants no library production change, API snapshot change or PR merge. M3 acceptance changes
only after separately authorized delivery; device validation and full M3/M4
remain open decisions.

## Accepted implementation and delivery

Independent analytic reference closure passed six cases before the example
existed; the pre-example consumers failed on the missing module as expected.
The initial 34 scenario cases and seven unchanged QND/HOM cases passed together.
Five scalar reference ASTs remain byte-identical. The sixth, `reference_inputs`,
received only an explicit complex128 tensor wrapper for the same scalar witness
assignment after strict typing rejected Python complex assignment. Consumer
imports were sorted; the deliberately incorrect coherent-sum witness uses
`stack.sum` instead of Python's integer-initialized `sum`. Captured before/after
sources retain these mechanical changes; no input, physics assertion or budget
changed. Library production and public API surfaces are unchanged.

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
