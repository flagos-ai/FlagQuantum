# Original Qmod corpus progress

## Current integration candidate

The [classified catalog](../../examples/synthesis/CATALOG.md) now retains every
one of the 216 original paths and their exact source hashes. The strict original
snapshot has 210 execution smoke passes and six source-semantic refusals. The
separately approved six-source numerical overlay has 216 execution smoke passes;
it does not replace the original files. Recorded physical resources remain tied
to their implementation revision.

Runnable corresponding native qfunc/Circuit examples cover 23/216 original
paths. The first completion batch adds 17 cases, with nine standard gates,
four preparations, unsigned multiplication and three quadratic tutorials.
`tests/integration/test_synthesis_native_catalog.py` compares all 23 native
counterparts against Qmod and independent Circuit references, and checks
preparation, phase, initialization, Const and resource boundaries. These native
counts are independent of the complete Qmod execution smoke inventory.
General native arrays/views, mutable helper borrows and runtime classical
parameters remain separate gaps. The integration candidate is not a merge,
release authorization or independent-user-trial result.

## Retained 2026-11-02 measurement

As of 2026-11-02, following #1002 merged at
6df1f2ae6296334d087fdf39918f8b1e46559a19 after 57 successful checks and
three configured skips. The denominator is the retained Classiq library snapshot
2bfd835e73b6882fcc515a725098b04766c238e7, not the moving upstream main.

## What has actually run

Eight of 216 original file paths (3.70%) have complete parse, lowering, real CPU
complex128 execution and full-state verification at their original dimensions.
Both file and text inputs are covered. This is seven distinct source blobs out
of 208 (3.37%), not eight independent algorithms.

| Original filename | Full execution evidence | Allocated qubits |
| --- | --- | --- |
| RZ.qmod | New literal-angle original, both input forms and native high-level counterpart | 1 |
| PHASE.qmod | New constant-phase original, both input forms and coherent probes | 1 |
| X.qmod | Existing baseline | 1 |
| synthesis_tutorial.qmod | Existing baseline | 23 |
| multiplication_2vars_example.qmod | Existing baseline | 24 |
| comparator_integer_example.qmod | Existing additive comparator | 10 |
| classiq_overview_tutorial.qmod | Added original tutorial, both input forms | 23 |
| execution_tutorial.qmod | Added original tutorial, both input forms | 23 |

The two added tutorials compute the already supported x*x+1 on a uniform
three-bit input. They exercise implicit and explicit allocation widths.
execution_tutorial.qmod is byte-identical to synthesis_tutorial.qmod; it increases
path coverage only. Those two tutorials did not add an algorithm family or language capability.
The later PHASE cohort adds a new constant-phase gate path, with native angles
limited to integer literals 0..2**53 radians; it is not general CReal support.
Recent arithmetic optimizations improved capacity but did not increase this
original-source count before these missing evidence cases were added.

## Complete user workflow

From a repository checkout:

```python
from pathlib import Path
import flagquantum as fq
from flagquantum import synth

program = synth.synthesize(
    Path("tests/fixtures/classiq_qmod/classiq_overview_tutorial.qmod"),
    max_qubits=24,
    memory_limit_bytes=512 * 1024**2,
)
result = fq.run(program)
print(dict(program.probabilities(
    result, registers=("x", "y"), max_amplitudes=1 << 24,
)))
```

Expected pairs (x,x*x+1) for x=0..7, each probability 1/8. Original x/y widths
are 3/6. Each added tutorial plans 23 qubits and a 128 MiB state buffer; the
512 MiB budget is a planned-buffer guard, not a process peak-memory guarantee.
Both added originals measure 23 used qubits, 14 hidden workspace bits,
379 operations (3 H, 218 CX, 156 CCX, 2 X) and depth 288 on CPU complex128.
The four integration cases verify every complex amplitude and zero hidden work
bits, rather than probabilities alone. No source edit, reduced input size or
substitute hand-built circuit is counted as an original execution.

## Whole-corpus frontend audit

All 216 retained input blobs were checked against their pinned Git blob hashes
and passed individually through the current native frontend. Eight are admitted;
208 are refused. This audit does no synthesis, planning or execution. Refusals
are grouped by their first diagnostic, so later missing dependencies are hidden:

| First diagnostic | Files |
| --- | ---: |
| qmod.syntax | 64 |
| qmod.role | 61 |
| qmod.type | 23 |
| qmod.token | 19 |
| qmod.operator | 10 |
| qmod.expression | 8 |
| qmod.budget.tokens | 7 |
| qmod.call | 6 |
| qmod.width | 5 |
| qmod.statement | 2 |
| qmod.budget.source | 2 |
| qmod.signature | 1 |

The original scan used the merged #1000 implementation; the PHASE follow-up
reran all 216 hash-checked sources and changed admission from six to seven. Per-file diagnostics and pinned-source
inventory remain in the project audit ledger; these aggregate results are a
local development audit, not a CI-certified corpus runner or compatibility
claim. The eight admitted files now have separate real execution evidence.

## Next delivery priorities

Keep original-file coverage separate from unit-test counts and frontend
admission. Each compatibility PR should name newly passing original paths,
retain original bytes/license, provide a reference oracle, execute unchanged
sizes or record an explicit capacity blocker, and report real physical costs.

Use the first-blocker list to select the next small original cohort, then
review all its transitive semantics and standard-library dependencies before
adding syntax. Arrays/views, consuming roles, signed/fixed-point types,
parameterized gates, classical parameters, structs and higher-order algorithm
combinators remain separate gaps. Do not weaken budgets or admit syntax without
an execution path just to raise the frontend count. Further kernel optimization
should be tied to a named original case or explicit user capacity requirement.

The new sources have a separate tutorial_sources.json pin and reuse the retained
MIT license. Old manifests, license bytes and the original seven digest checks
are unchanged. No production implementation changes in this evidence slice.
Independent developer trials remain 0/2 and release_authorized remains false.
The retained #1001 tutorial evidence remains four file/text cases passing in
92.50 seconds and two independent source/manifest/license unit cases. The new
PHASE focused suite passes 53 cases in 0.69 seconds, including its original
file/text executions and coherent/global-phase probes. See the
[PHASE delivery](SYNTH_CONSTANT_PHASE_GATE_DELIVERY.md) for the complete boundary.

The PHASE follow-up also passes strict typing across 879 production/example
sources and existing static/documentation/source digest checks. Required default
smoke/unit records 5051 passed, 121 skipped and the same three local Simulation
QFT fusion assertion failures in 212.38 seconds, stopped at the configured
three-failure limit. These expect 28 state applies and observe 44; no Simulation
implementation or assertion changes. The default tier is not green.
Log: /private/tmp/fq-phase-gate-default.log.

## UP-00 native compatibility decision

The [unified contract](../api-changes/FQ-SYNTH-UNIFIED-PROGRAM-UP00-20261102.md)
preserves unchanged Qmod source as the migration path and familiar qfunc/role/
phase/control concepts as the future native program design. RZ/decimal import
remains; the unmerged float-return native phase prototype is withdrawn. General
explicit native phase is a named gap, not a Circuit alias or a mandatory rewrite.

Run `python -m examples.synthesis.tutorials.qmod.native_and_qmod_rz.main` for current Python
high-level arithmetic, its equivalent Qmod, and an independent RZ Circuit
reference. These distinct tasks are labeled; the arithmetic example is not
claimed to be a native implementation of arbitrary-angle phase. Corpus counts
and source pins do not change because a Python convenience profile is withdrawn.

The earlier 118 targeted/2689 related runs and 5064 passed/121 skipped/3 local
QFT assertion failures belong to prototype head
4c923bf111ace9c29bf0d784bbfabfc0fee940d9. They are preserved in the
[historical delivery record](SYNTH_LITERAL_RZ_DELIVERY.md) and are not current
revised-head results. Current UP-00 checks are recorded separately.
