# Original Qmod corpus progress

As of 2026-11-02, following #1000 merged at
67feba502d4a5071ffef19ce09e5ca4f58677b64 after 57 successful checks and
three configured skips. The denominator is the retained Classiq library snapshot
2bfd835e73b6882fcc515a725098b04766c238e7, not the moving upstream main.

## What has actually run

Six of 216 original file paths (2.78%) have complete parse, lowering, real CPU
complex128 execution and full-state verification at their original dimensions.
Both file and text inputs are covered. This is five distinct source blobs out
of 208 (2.40%), not six independent algorithms.

| Original filename | Full execution evidence | Allocated qubits |
| --- | --- | --- |
| X.qmod | Existing baseline | 1 |
| synthesis_tutorial.qmod | Existing baseline | 23 |
| multiplication_2vars_example.qmod | Existing baseline | 24 |
| comparator_integer_example.qmod | Existing additive comparator | 10 |
| classiq_overview_tutorial.qmod | Added original tutorial, both input forms | 23 |
| execution_tutorial.qmod | Added original tutorial, both input forms | 23 |

The two added tutorials compute the already supported x*x+1 on a uniform
three-bit input. They exercise implicit and explicit allocation widths.
execution_tutorial.qmod is byte-identical to synthesis_tutorial.qmod; it increases
path coverage only. No new algorithm family or language capability is claimed.
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
and passed individually through the current native frontend. Six are admitted;
210 are refused. This audit does no synthesis, planning or execution. Refusals
are grouped by their first diagnostic, so later missing dependencies are hidden:

| First diagnostic | Files |
| --- | ---: |
| qmod.syntax | 64 |
| qmod.role | 61 |
| qmod.type | 23 |
| qmod.token | 19 |
| qmod.operator | 11 |
| qmod.expression | 8 |
| qmod.budget.tokens | 7 |
| qmod.call | 7 |
| qmod.width | 5 |
| qmod.statement | 2 |
| qmod.budget.source | 2 |
| qmod.signature | 1 |

The scan used implementation 7a7a133bddc013d2e286fe5eeede585d35acaba3,
whose tree equals the merged #1000 tree. Per-file diagnostics and pinned-source
inventory remain in the project audit ledger; these aggregate results are a
local development audit, not a CI-certified corpus runner or compatibility
claim. The six admitted files now have separate real execution evidence.

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
Four real file/text integration cases pass in 92.50 seconds; two independent
pinned-source/manifest/license unit cases pass. Existing seven digest checks,
Black 26.10.1, Ruff, architecture, unchanged API snapshot, generated docs,
maturity, language, local links and whitespace pass. Production is unchanged;
no repeat of the parent arithmetic regression or strict typing is claimed.

Required default smoke/unit: 5030 passed, 121 skipped and the same three local
Simulation QFT fusion assertion failures in 207.97 seconds; stopped at its
configured three-failure limit. These expect 28 state applies and observe 44.
No Simulation code or assertion is changed. The default tier is not green.
Log: /private/tmp/fq-tutorial-default.log.
