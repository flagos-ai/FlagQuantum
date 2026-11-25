# Bounded native translation reuse control

Compiler-owned `single_device_fast_path` engineering proposal. No public API,
rule, tolerance, budget, routing replay or evidence policy change is proposed.
The production source remains unchanged until acceptance and resource review.

## Fixed proposal and inputs

The baseline is integrated main
`006c3268e803bd8dc13258b913cccfa9b69bb718`. The proposed native legalization
module SHA-256 is `1501d2d1fff450e6ba750245670bd670cbfae8397ad23b917ac6da182e22cf81`.
Its successful parameterless named-gate translations are reused by opcode and
exact wires in a dictionary local to one legalization call, capped at 128 entries.
Each cache hit constructs fresh Instructions with the current source metadata.
No cache survives a call. Parameterized instructions and custom matrices retain
the existing path. Output support checks, combined budgets, decomposition
records, source hashes and subsequent independent replay remain unchanged.
This reuses the existing equivalence search rather than adding a registry or
external compiler dependency. It does not reproduce Qiskit's full BasisTranslator.

The unchanged SDKBench checkout is
`9aa9260920984e2ca05bcb293abb885fccd9c144`. Use the already inspected nonce
`01` repeated 32 times. Widths 6/8/12, QAOA/Ising/hopping/QFT/repetition and
line/ring/grid/sparse yield 60 cases per ordinary-restored and target-artifact
workflow: 120 pairs and 240 independent acceptances. Require identical evaluator
metrics, normalized operations/layout hashes and adapter audits for every pair.
Six-wire cases use full operators up to phase; larger cases use three probes.
This is inspected development evidence, not a held-out or formal study.

## Observer and resource decision

Both variants use AST-cloned functions and the same cached module-attribute
patch list, equal setup/teardown and restoration assertions. Do not scan module
globals for only one variant. Keep Python 3.12.14, one numerical thread and the
existing garbage collection policy. Do not overlap local tests or transfers.

After acceptance use target-artifact QFT6/grid, repetition6/ring and QFT12/grid,
two warmups and nine retained alternating samples per variant. Retain the first
completed run, all samples and all failures. Separately measure whole-observation
Python traced peaks including fixture, adapter and independent acceptance; these
are not process RSS or compiler-only peaks. Require at least 5% lower median
adapter time in each inspected timing case and no larger traced peak in any
case before proposing production adoption. Passing these development checks is
necessary, not sufficient for release performance qualification. Missing,
inconsistent or failing evidence leaves adoption unproven; do not change cases
or rerun to obtain a favorable result.

Use a separate 600-second process-group supervisor with bounded TERM/KILL,
leaving SDKBench's stage timer intact. Formal dedicated-host qualification is
user-deferred. No machine, service, kernel or driver may be restarted or changed.

## Existing workflow

```python
import flagquantum as fq
from flagquantum.compiler import optimize
source = fq.Circuit(3).h(0).h(1).h(2).to_ir()
assert optimize(source).instructions == source.instructions
```

Expected: independent gates remain unchanged. This proposal supports existing
CPU compilation and offline target-artifact workflows; it adds no public API.

## First completed outcome

This declaration was published at
`3761d6de0f888e983179b9e0fe8dc0bccefa505e` before execution.
[The retained exact record](evidence/compiler-translation-reuse.json) contains
the driver, complete proposed module, baseline hash, all first-run observations
and the balanced observer configuration. The bounded supervisor completed once
with status zero. No local test run or artifact transfer overlapped execution.

All 120 pairs and 240 independent acceptances retain identical metrics,
normalized operation/layout hashes and adapter audits. This finite equality
does not by itself establish all metadata, autograd, concurrency or tamper
behavior. Production adoption would additionally require focused behavior tests.

| Inspected case | Baseline / proposed median | Observed reduction | Whole-observation Python traced peak, baseline / proposed |
| --- | ---: | ---: | ---: |
| 6-wire qft, grid | 91.751 / 90.058 ms | 1.85% | 35,363,902 / 35,364,777 bytes |
| 6-wire repetition, ring | 115.695 / 112.168 ms | 3.05% | 35,364,741 / 35,365,598 bytes |
| 12-wire qft, grid | 578.236 / 556.588 ms | 3.74% | 35,365,980 / 35,364,804 bytes |

No timing case meets the declared 5% reduction. The candidate also has
slightly larger observed traced peaks. Neither fact proves a statistically
significant resource change on a shared host, but neither permits relaxing
the declared admission rule. Do not adopt this proposal, rerun for a favorable
result, or tune the fixture selection. The production legalizer stays unchanged.
The finite result suggests repeated rule search alone is insufficient to close
the remaining native-CZ selection overhead. A later design must address a
verified larger cost while retaining independent baseline and routing replay;
it needs its own fixed proposal and first-run control. No performance ranking,
release claim or formal-host qualification follows.
