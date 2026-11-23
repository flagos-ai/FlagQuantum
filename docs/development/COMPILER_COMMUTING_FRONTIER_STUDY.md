# Bounded commuting-frontier routing study

Status: rejected for default admission. Production routing is unchanged.

The candidate can reduce routing cost on generated application circuits but did
not reduce CZ count or dependency depth on any of six inspected external inputs.
Its DNN line compilation median regressed by 18.6 percent on the shared local
machine. Generated inputs also expose final-artifact CZ and depth regressions.
These outcomes do not justify adding the candidate to ordinary compilation.

## Source review and integration boundary

Qiskit 2.6.2's `Commuting2qGateRouter` was inspected at revision
`0c26a5a158c4d62b2b303a8c2f3c6eb2b14653b4`, file
`qiskit/transpiler/passes/routing/commuting_2q_gate_routing/commuting_2q_gate_router.py`,
SHA-256 `27ba2ce60f16a770a8a75b004498a30f684e6d59a959caa660a33feedc8afe27`.
It routes packaged commuting two-qubit blocks using swap strategies and
sublayers. Copyright IBM 2022, Apache-2.0. No upstream source was copied. The
review motivated relaxing submitted order for proven commuting operations; this
prototype is not a port or a claim of equivalent algorithmic behavior.

The experiment reuses FlagQuantum's existing structural commutation predicate
and SABRE trials. Per-wire commuting groups are bounded to 32 instructions.
A new group follows every member of the preceding group; the existing classical
RAW/WAR/WAW constraints are preserved. Matrix overrides, nonunitary operations,
channels, dynamic instructions, routing-phase instructions and conditions are
excluded from relaxed groups. One additional candidate competes with all five
existing trials using the incumbent restoration/orientation-aware score. Zero
incumbent cost and an unchanged dependency graph skip redundant trials. The
prototype is injected only inside isolated development drivers; it changes no
production API, IR, dependencies, layouts or serialization.

## Evidence and acceptance

The [retained study](evidence/compiler-commuting-frontier-study.json) records
all fixtures, drivers, per-arm acceptance results, audit, timing samples,
traced allocation observations, fresh-process RSS and output hashes. Compiler
baseline: `b8e57f85acc8e0fe05fa67be4482788dc12a9486`. SDKBench revision:
`5ba03de33f8a9193f86b26d2cb3fa841ede7c98a`. The workflows field's SDKBench
revision is the harness source pin, not the compiler source pin. Full emitted
payloads are omitted from this checked-in report; their canonical hashes and
the unabridged local study's SHA-256 are retained. This is not a published
full-output archive.

There are 122 graph assertions covering ordering, noncommuting barriers,
classical dependencies and excluded instruction classes. An initial graph
control used an invalid barrier without its dynamic metadata; correcting the
control made it exercise valid IR. An initial AST-cloned trial required future
annotations. These development corrections do not change acceptance policy.

The paired application run covers five generated families, widths 6/8/12,
four topologies and both ordinary restored-layout and target-artifact workflows:
120 pairs, all 240 outputs accepted by unchanged SDKBench. Width 6 uses full
operators; widths 8/12 use three probes only. The maximum observed residual is
`6.9184e-14`. Generated fixtures use inspected development nonce `01` repeated
32 times and are not held out.

| Workflow, 60 pairs each | CZ better / equal / worse | Dependency depth better / equal / worse |
| --- | --- | --- |
| Ordinary restored layout | 21 / 39 / 0 | 18 / 38 / 4 |
| Target artifact | 21 / 38 / 1 | 16 / 40 / 4 |

The six-wire QFT line artifact illustrates a limitation of selecting by routed
native cost: routing CZ decreases from 111 to 99, but final CZ after synthesis
increases from 75 to 77; depth decreases from 52 to 50. The twelve-wire QFT grid
artifact improves CZ from 352 to 324 but worsens depth from 150 to 196. A routing
objective alone does not guarantee the best downstream synthesized artifact.

## Resource observations and decision

After other local tests finished, 25 cases received one warmup and nine
alternating timing blocks per arm: six external inputs, sixteen seeded controls
and three representative artifact workflows. All timing samples are retained.
On the six external inputs, CZ and depth are unchanged. Candidate/baseline median
time ratios are 0.9282 and 1.1855 for DNN grid/line, 1.008 and 1.0317 for QAOA,
and 1.0478 and 1.0305 for Simon. Control ratios span 0.9302 to 1.1749.

Three separate allocation observations per arm cover the external and artifact
representatives. These are Python traced allocations, not process RSS. An
additional 54 fresh workers, each bounded to 20 seconds, observe peak RSS:
baseline 199.53–209.91 MiB and candidate 199.48–209.03 MiB. Repeated outputs within
each arm match their hashes. The machine is shared macOS with one numerical
thread requested, no exclusive reservation and no memory cgroup. Timing and RSS
are development observations, not qualified comparative performance claims.

The result rejects this candidate for default admission. There is no
fixture-specific dispatch, acceptance relaxation, production patch, or claim of
superiority. Further scheduling work needs an independently justified objective
that accounts for delivered target cost and depth, plus prospective external
workloads and enforced resource bounds. The existing user workflow remains the compiler's
[golden path](../../flagquantum/compiler/README.md),
`python -m examples.target_aware_compilation`: ordinary target compilation
continues to use the current SABRE candidates. This report supports that workflow
by documenting why the proposed extra default trial is not adopted.

Documentation-link, public-evidence-privacy and CI-isolation checks pass: 16 tests
in 4.32 seconds. Full exact-head CI remains required before merge.
