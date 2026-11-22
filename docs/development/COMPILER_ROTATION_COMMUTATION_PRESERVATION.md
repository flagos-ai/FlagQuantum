# Controlled commutation with rotation-analysis preservation

Owner: Compiler. Classification: `single_device_fast_path`.

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
