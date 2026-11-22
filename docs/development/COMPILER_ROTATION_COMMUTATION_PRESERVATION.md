# Rejected rotation-analysis preservation

Owner: Compiler. Classification: `single_device_fast_path`.

**Rejected resource admission.** The new preservation helper and its key
extraction are withdrawn. Both production modules are restored byte for byte
to the earlier draft head `9f307bbd6cbb4aaf39d04e2defb155c875118cb2`.
The controlled-gate extension itself remains unpromoted. The candidate below is
a historical experiment, not enabled production behavior.

## Requirement and mechanism

The earlier composed controlled-gate extension remains rejected by its first
resource observation: ordinary repetition6/ring has candidate/main median ratio
1.162137 above the predeclared 1.05 limit. See
[the retained first outcome](COMPILER_CONTROLLED_COMMUTATION_INTEGRATED_REUSE.md).
A subsequent independent fixed-two-qubit numerical-validation experiment did not
establish the declared complete-workflow benefit and was withdrawn in PR #1310.
Neither result is erased or replaced by a favorable repeat.

An untimed pass trace of the first failing input identifies exactly one change
before routing: `merge_commuting_rotations` adds two RZ(pi/4) rotations into one
RZ(pi/2), taking 58 operations to 57. Every remaining pass and the subsequent
fixed-point round make no additional change. Do not bypass that fixed-point
round: another workflow may expose further optimization.

Instead, update the existing compilation-scoped commutation memo after a proven
rotation merge. It remains one entry, expires on scope exit, stores only structural
keys and immutable analysis, and never retains IR, tensors or numerical values.
Standalone analysis and arbitrary structural changes still recompute normally.
No generic PassManager, public signature, analysis field, declared pass sequence,
policy, replay check, dependency or external SDK adapter changes.

The proof is deliberately narrow. Every removed structural class must have an
earlier retained equivalent member in the same original block on every wire.
Thus each block keeps its first member and every removed boundary witness has an
equivalent retained witness. Deleting later duplicates cannot make an internal
commuting pair false, erase an existing boundary or move its first candidate.
Retained blocks therefore equal the original partition with positions remapped.
Recheck every retained boundary to recover its exact first-failing offset;
internal logical pair counts follow n*(n-1)/2. Both the complete partition and
its public logical comparison counters must equal a fresh full sweep. A missing
witness, zero-sum deletion, mismatched structural key, checker change, expired
scope or unproven boundary leaves normal recomputation in force. Numerical
parameter values remain outside the proof, as in the existing commutation rules.

This is a FlagQuantum-owned incremental update to the existing Qiskit-style
partition; it is not a new upstream implementation port. Existing source/license
attribution remains in place. It must earn admission through the original
complete-workflow resource gate rather than through a reduced sweep count.

## Prospective validation

Publish the candidate and this declaration before observing new complete-workflow
resources. Freeze source hashes, runner hashes, versions, exact adapter options,
compiler seed 11, unchanged SDKBench
`9aa9260920984e2ca05bcb293abb885fccd9c144`, and exact integrated-main baseline
`d8941cea0950447ef4e3bb73ad42a5d5efe4b2ae`. This candidate combines the
controlled rule and the new analysis preservation. Baseline substitutes main's
original checker and complete rotation-merge function. Both variants retain the
same scoped lifecycle, native routing, target optimization and artifact replay.
AST checks restrict other changes to extracting the identical structural-key
expression into a private helper. The new preservation helper is reached only
through the candidate rotation-merge function. No cross-call cache is introduced.

Run all 60 application inputs (widths six/eight/twelve, five families, four
topologies), both ordinary-restored/target-artifact workflows, both variants:
240 observations with inspected nonce `01` repeated 32 times. After publication,
generate exactly one fresh 32-byte nonce and freeze it before the second complete
240-observation draw. Do not inspect, redraw, tune, omit cases or discard failures.
Save every output, audit, fixture hash, metric and original compile-wall value.
Alternate variant order by input. Candidate CZ count and dependency depth may
not increase in any pair. Report every changed output separately. Acceptance
still uses unchanged 1e-8 semantics, topology/layout checks, capacity caps and
120-second compiler limits. Width six uses full operators; widths eight/twelve
use three limited probes, which are not full-operator evidence. Replay both
complete reports in a fresh process importing no producer SDK.

Use the inspected QFT6/grid, repetition6/ring and QFT12/grid inputs for both
complete workflows. Each variant gets two warmups, nine alternating retained
rounds and one separate traced round, fresh adapters and identical patch lifetimes.
Independently accept and match the frozen quality output hash, audit and metrics
of all 144 executions outside timing. Store every sample, not only the final one.
Keep the existing prospective gates unchanged: each of six candidate/main median
ratios at most 1.05, and each Python peak within both 1.10 times baseline and
128 KiB above baseline. All quality and producer-free replay checks must pass.
Tracing excludes adapter construction; Python peaks do not measure native
allocation, RSS or cgroup memory. Report local ratios without claiming statistical
significance, formal host qualification or cross-SDK rank.

If a gate fails, keep the extension unpromoted, retain the first complete outcome
and remove this added preservation machinery unless independent evidence justifies
it. Do not rerun for a favorable result, widen limits or select cases afterwards.
The prior resource failures remain separate historical outcomes. Formal Linux
qualification remains deferred. No machine/service restart or kernel upgrade.
No superiority, scalability, new hardware support or full-suite claim follows.

Use one numerical thread, immutable source, no simultaneous local tests, bounded
process groups (900 seconds per quality draw; 300 for resources/replay) and a
16 MiB raw-report ceiling. Create outputs exclusively. Retain raw reports or
byte-exact reconstructable hashed chunks below existing repository artifact
limits. Full required exact-head CI and Integration PR #1308 remain separate
merge prerequisites. Setup-only failures remain reported as failures, not tests
or acceptance evidence.

Frozen control: `fq-rotation-preservation-control-20261122.py`, SHA-256
`abb20f0830c4941e1fb1efb1699696ca3e0958525c9e28900ab103c21c77186b`.
Producer-free replay: `fq-rotation-preservation-replay-20261122.py`, SHA-256
`757496f4b1fd792b9de418724881e2785e3a7e40fdaaef9aecd299b18407043a`. Reports embed both complete sources and the frozen protocol.

## Existing user workflow

```python
import flagquantum as fq
from flagquantum.compiler import compile

source = fq.Circuit(2).rz(0, theta=0.2).cx(0, 1).rz(0, theta=0.3).h(0)
result = compile(source)
assert len(result) == 3
```

Expected: one RZ(0.5), the CX and H preserve the complete operator. The retained
workflow test compares this result against a replacement that disables memo
preservation and an independent complete operator. No fixed-point verification
is skipped.

Before freeze, all 391 selected local compiler and integration tests pass,
including the original complete-density thermal-reset workflow. The 32 new tests
cover all twelve rotation families in both operand orders, 300 deterministic
random programs, exact analysis counters, zero-sum invalidation, mutable matrix
presence, symbolic parameters, trainable gradients, standalone scope and the
public compile workflow. Strict Linux-targeted mypy passes for both complete
production modules and the complete new test module. These sparse local tests do
not certify complete repository CI.

Initial test setup used undeclared `barrier` and `reset` opcodes; corrected to
an explicit custom matrix fence and declared amplitude-damping channel before
observations. Initial strict typing found an untyped Tensor.backward call;
replaced with typed torch.autograd.grad. Two expanded-suite attempts named absent
sparse files; neither collected tests. The corrected verified file selection is
the 391-test result above. No scientific threshold or failure assertion changed.

The default sparse-checkout `smoke or unit` command passes 426 tests, with
246 deselected. The documented public workflow executes and returns RZ, CX, H.
Ruff, Black, English-content checks, Markdown links and whitespace checks pass.
None of these local selections replaces complete exact-head CI.


## First outcome and withdrawal

The candidate and declaration were published at
`3737f8d0262b14b5d34a2afdf5abc5812095f2b2` before observations. Both complete
draws accept all 480 observations. Every paired CZ count and dependency depth
is non-increasing; twelve ordinary repetition payloads change per draw and the
other 108 payload pairs are identical. Producer-free replay accepts all 480,
with maximum residuals 6.072657351995283e-14 and 3.8949521648662045e-14
against unchanged 1e-8. There are 160 full-operator and 320 limited three-probe
observations. No application CZ/depth improvement is established.

All 144 warmup/timed/traced resource executions are independently accepted and
matched to frozen quality hashes, metrics and audits. The first and only resource
control fails admission: ordinary repetition6/ring ratio 1.170783 exceeds 1.05.
The other five time gates and every Python-memory gate pass. No favorable rerun,
case omission, new allowance or changed threshold follows. Do not infer a
significant change from the earlier 1.162137 ratio with a different candidate.

| Input | Workflow | Main median ms | Candidate median ms | Candidate/main | Main Python peak bytes | Candidate Python peak bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 06/qft/grid | ordinary-restored | 6.953 | 6.935 | 0.997375 | 479760 | 485526 |
| 06/qft/grid | target-artifact | 90.418 | 89.082 | 0.985220 | 2199144 | 2199078 |
| 06/repetition/ring | ordinary-restored | 6.085 | 7.124 | 1.170783 | 565333 | 581428 |
| 06/repetition/ring | target-artifact | 113.739 | 114.033 | 1.002587 | 2567421 | 2567421 |
| 12/qft/grid | ordinary-restored | 28.204 | 27.511 | 0.975407 | 2136627 | 2160984 |
| 12/qft/grid | target-artifact | 553.978 | 570.980 | 1.030690 | 10118214 | 10118164 |

Remove the added preservation machinery and the new experiment-specific test
module. Its exact 32-test source remains in the published freeze with the measured
candidate. Existing tests and every independent scientific acceptance assertion
remain unchanged. The incremental proof can be correct while its implementation
fails to improve the required complete workflow; adding this complexity is not
justified by these observations.

The [inspected index](evidence/rotation-preservation-inspected-screen.json) and
[fresh index](evidence/rotation-preservation-fresh-screen.json) preserve every row,
input pin, audit, original compile-wall value and hashed payload. Their payload
chunks reconstruct the complete original reports byte for byte, including field
order and original SHA-256. The [resource report](evidence/rotation-preservation-resources.json)
contains every sample and per-sample check. The [producer-free replay](evidence/rotation-preservation-replay.json)
retains its complete source, draw hashes and residuals. Reports embed the frozen
protocol and complete control source. This retrospective decision does not alter
any observed numerical data.

The extra fixed-point round remains necessary for the observed nonzero merge.
Repeated attempts to reduce ancillary work have not met the unchanged resource
gate. Further compiler work should address delivered routing/target-gate cost or
the full optimization algorithm rather than accumulating unproven memo layers.
Full exact-head CI, the separate Integration prerequisite and formal Linux
qualification remain unresolved. This does not admit draft PR #1304.

The [untimed trigger trace](evidence/rotation-preservation-trigger-trace.json)
records the earlier clean draft source, both accepted output hashes/audits, and
every pass change. Its only pre-routing change is the two RZ(pi/4) instructions
merged into RZ(pi/2). It is causal work evidence, not a latency measurement.

After withdrawal, the default sparse `smoke or unit` selection passes 394 tests
(246 deselected); the selected physical-evidence and original thermal-reset
integration files pass all 49 tests. Both production modules exactly match the
earlier clean draft head. Language, local Markdown links, byte-exact archive
reconstruction, existing revision-field vocabulary and whitespace checks pass.
The removed candidate's prior strict checks are historical evidence, not a claim
that the withdrawn helper remains enabled. Full exact-head CI is outstanding.
