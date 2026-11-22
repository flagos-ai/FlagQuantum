# Native Notebook integration handoff

This delivery makes the quantum program, its classical inputs, compilation,
execution and result interpretation visible in the five synthesis catalogs.
It adds six FlagQuantum teaching lessons and reusable Compiler improvements;
the lessons are separate from upstream migration acceptance.

## Scope and counting

| Measure | Accepted scope | Interpretation |
| --- | --- | --- |
| Approved upstream workflows | 212/212 | Historical original-scale acceptance and editorial replay; applicable current-source regressions remain required |
| Original Qmod paths | 216 | 210 execution smokes and six expected original refusals |
| Approved alternative paths | 6 | Execution smokes counted separately from original paths |
| New teaching lessons | 6 | Independent state, cleanup, phase, topology, sampling and gradient checks |

The original fault-tolerance calibration workflow remains deferred outside the
approved 212. A local calibration model does not establish equivalence to the
unavailable upstream model. Source-path smokes are not full-state acceptance.

## Runnable user example

Run this as a saved Python file from an installed FlagQuantum checkout:

```python
import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Const, Output, QNum, allocate, hadamard_transform, qfunc


@qfunc
def energy(x: Const[QNum], y: Const[QNum], result: Output[QNum]) -> None:
    result |= x * x + y


@qfunc
def main(x: Output[QNum], y: Output[QNum], result: Output[QNum]) -> None:
    allocate(2, x)
    allocate(2, y)
    hadamard_transform(x)
    hadamard_transform(y)
    energy(x, y, result)


program = synth.synthesize(main, max_qubits=24, memory_limit_bytes=64 * 1024**2)
result = fq.run(program)
outcomes = dict(program.probabilities(result, registers=("x", "y", "result")))
assert set(outcomes) == {(x, y, x * x + y) for x in range(4) for y in range(4)}
assert all(abs(p - 1 / 16) < 1e-12 for p in outcomes.values())
print(outcomes)
print(program.report["resources"])
```

Users declare widths, fresh outputs, immutable borrowed inputs and coherent
preparation. The compiler supplies arithmetic, temporary allocation and cleanup.
Classical qfunc arguments are synthesis-time specialization; this delivery does
not introduce symbolic runtime qfunc parameters. Lesson six demonstrates the
existing Circuit execution-plan and gradient reuse instead.

## Reusable implementation changes

- Reuse input bits and fuse supported terminal arithmetic, including a bounded
  Fourier route when a workspace budget excludes the default ripple route.
- Use Walsh butterflies in real-state preparation and signed binary places for
  constant modular updates.
- Avoid repeated validation within the same readout or inspection boundary.
- Admit the approved bounded fresh native preparation and ordinary effect
  capacities, retaining separate helper, QPE, Qmod and resource guards.
- Compose the original image detector and complete 102/256-phase nozzle inverse
  streams as native programs; Circuit constructions remain independent references.

The capacity changes are governed by the approved
[preparation proposal](../../docs/api-changes/FQ-SYNTH-NATIVE-PREPARATION-CAPACITY-20261121.md)
and [effect proposal](../../docs/api-changes/FQ-SYNTH-NATIVE-EFFECT-CAPACITY-20261121.md).

## Retained verification and limitations

The [delivery plan](NOTEBOOK_UX_PLAN.md) records exact source checkpoints,
resource counts and numerical errors. The
[editorial ledger](NOTEBOOK_UX_REVIEW.md) links all 212 reviewed Notebooks.
The [retained closeout receipt](NOTEBOOK_UX_ACCEPTANCE.json) at source
`fea2d0a30` binds 41 source hashes and 15 raw evidence hashes. It is copied
without alteration from local acceptance; raw logs remain local artifacts.
Its scope and status describe that historical checkpoint, before this PR.

The original image's 17 code cells and the nozzle's 14 code cells pass on the
formatted sources. Compiler/QPE checks pass 2152 tests; nested helper/public
flow checks pass ten. The final catalog/audit selection passes 23 tests.
The 222-path raw batch contains one Vlasov observer timeout; a separate retained
240-second recheck passes with unchanged source and resources. All final expected
stages match, with zero resource-profile differences. The raw batch is not green.

The retained default run reports 21261 passed, 801 skipped, seven failed and six
errors. Its failed/error node set has no additions against the prior run. Six
launch failures and six Gloo errors concern local socket permissions; one failure
concerns historical evidence origins in shared private Git refs. These are
unresolved checks, not passes, and require verification on a suitable runner.

No gate-count change is claimed as a hardware or runtime speedup. Runtime memory
budgets bound planned state buffers, not total process RSS. Full acceptance
replays remain offline evidence; ordinary PR CI follows the existing repository
workflow without adding a 212-Notebook replay job or weakening required checks.

## Integration checkpoint

On November 22, 2026, `main` at
`d14bd25c647e7d2ec6f82b087cb32f8fe4fa4277` was merged without conflicts as
`1d0613a4c995d7a67e227f9cc2aeca0ae148a639`.
All 719 focused tests across the changed Compiler, Services and integration
test files pass in 189.74 seconds, including the six lessons and capacity guards.
The displayed user example and the matching Qmod program both pass complete-state
and clean-work checks against an independent Circuit reference. The native
example uses nine qubits, 14 logical instructions and depth nine in this run.

The public API snapshot, production/test/tool Ruff and Black checks, repository
hygiene, documentation source checks and handoff links pass. An exploratory Ruff
scan of all changed Notebooks is not clean: it reports 183 import-order,
72 unused-import, 419 late-import, 32 redefinition and two loop-callback
diagnostics. Notebook-wide lint is not a configured PR gate; no lint rules are
disabled or changed by this delivery. This scan is separate from executed-cell
and numerical acceptance and remains a presentation-maintenance item.

The first post-sync default attempt exhausted local temporary storage; its
interrupted log is preserved and is not acceptance evidence. After removing
only stale pytest scratch directories with no living lock owner, the default
selection was restarted. Its final result is recorded separately below.

The restarted default selection reports 21447 passed, 801 skipped, one failed
and no errors in 374.57 seconds with four workers. The remaining failure is
`test_checked_in_evidence_accounts_for_every_recorded_revision`; extra local
branches/tags in the shared repository make historically unpublished revisions
reachable. The first independent clone also copied local tags and retained
16 diagnostics. In a disposable clone with only the respective branch ref and
no extra local tags, unchanged origin checks report zero diagnostics for both
the exact main and integration trees, and the original test passes in 8.50 s.
The checked local calibration-baseline tag is absent from the GitHub remote.
No refs or evidence declarations in the original shared repository were changed.
The raw default run still remains failed; the independent test is a separately
recorded environment check, not a rewrite of that result.

Catalog, Notebook audit, CI replay-selection and frontend-boundary checks pass
42 tests in 9.25 seconds. The
[post-sync check record](NOTEBOOK_UX_INTEGRATION_CHECKS.json) binds the source
commits, selections, results and raw-log hashes. No full 212-workflow replay on
the integration commit is claimed.

Historical acceptance above remains bound to its original commits; it is not
relabeled as a replay of a later integration tree. Raw post-sync logs are in the
local `notebook-integration-20261122` evidence directory.
