# Local native quantum-program acceptance

All **216 frozen quantum-program counterparts** passed their documented native
and independent Circuit comparison modules in one fresh local run on
`65a2ccdc56587fd6c49b419b48efe5c72c954ce4`. The run completed on
2026-11-11 UTC (2026-11-12 Asia/Shanghai); all 1,580 pinned production and native
example Python files remained unchanged. No prior module exits were reused.

| Category | Compared programs | Passed |
| --- | ---: | ---: |
| Algorithms | 43 | 43 |
| Applications | 59 | 59 |
| Community | 12 | 12 |
| Functions | 55 | 55 |
| Tutorials | 47 | 47 |
| Total | 216 | 216 |

## What was verified

The modules check complete complex amplitudes, absolute phase, declared register
order and retained environments or clean workspace as appropriate. Mixed-state
examples compare complete density operators. The 150-qubit basis-state oracle
uses an exact, untruncated bond-one MPS comparison over every site, including the
absolute minus sign and all 83 work sites; it does not substitute a small dense
problem. Two byte-identical source counterparts share reviewed native builders
and independent reference runners. Every documented module executes assertions;
reference-only modules and empty entry points are rejected by the catalog.

The unchanged original defaults and existing resource/time limits were retained.
In particular, whitebox fuzzing also executes its original ten Grover rounds;
Rainbow executes all three documented iteration counts within its existing
1,800-second module limit. No approximate backend fallback or phase alignment
was introduced for acceptance.

Source replay separately checks all 216 originals and six approved repairs.
**210 unchanged originals pass; six originals retain semantic refusals, and
their six explicitly approved repairs pass.** The 216 native comparisons use
those same original/approved identities. This is not a claim that all unchanged
originals are accepted. Source hashes, repair approvals and licenses remain in
[test fixtures](../../tests/fixtures/classiq_qmod).

The frozen quantum corpus uses upstream revision
`2bfd835e73b6882fcc515a725098b04766c238e7`. The expanded five-category inventory
uses the separate revision documented in [migration status](MIGRATION_STATUS.md).
Their source identities are tracked independently.

## Other gates and limits

- Focused compiler acceptance: 141 passed; related regressions: 529 passed.
- Rainbow/Grover and autocallable scenarios: 42 and 27 passed, respectively.
- Typing, formatting, lint, ownership, documentation, architecture, maturity and
  catalog checks passed. Source replay retained its previous resource results.
- The complete default smoke/unit tier ran: 18,814 passed, 52 failed, 71 errors
  and 615 skipped. Its 123 failure/error node identities exactly match the
  preserved baseline; no failure was added or removed. This tier remains red,
  so this report is not a full repository CI or release approval.
- Local Notebook workflows remain **24/213**. The expanded inventory contains
  **577 artifacts in 417 provisional groups**, including **148 Python source
  artifacts** that still require case/helper classification. Quantum-program
  acceptance does not certify data preparation, training or other Notebook
  workflow steps. See [migration status](MIGRATION_STATUS.md).

The delivery ledger archives all 216 module logs, source pins, source replay,
failure-node comparison, named checks and the requirement-by-requirement local
completion audit. Earlier timeouts, failed assertions and incomplete audits are
retained separately rather than overwritten. Remote integration CI remains a
subsequent gate.

## Subsequent subset-majority workflow acceptance

After the quantum-only acceptance above, the subset-majority QNN completed its
local training workflow. The original data split, architecture, pinned checkpoint,
MSE, SGD and rounded-label evaluation are retained. Exact CPU complex128 readout
and float64 classical tensors explicitly replace the original 4096-shot/float32
path. The current inventory therefore records **25/213** local workflows; the
historical 24/213 figure above describes the earlier run.

See the [workflow guide](algorithms/machine_learning/hybrid_qnn/hybrid_qnn_for_subset_majority/WORKFLOW.md)
for reproduction, independent gradient/update checks and actual metrics. The
30-epoch scratch run does not exceed the held-out constant-zero accuracy
baseline; its completed training is not convergence evidence. No other Notebook
workflow or maturity status is promoted by this addition.

## Reproduce a comparison

Use each case's comparison command from the [catalog](CATALOG.md), for example:

```sh
python -m examples.synthesis.applications.cybersecurity.whitebox_fuzzing.comparisons.circuit
python -m examples.synthesis.tutorials.basics.grover_graph_coloring.comparisons.circuit
python -m tools.synthesis_catalog
```

Run the complete documented module list for a fresh corpus claim; the catalog
validator alone checks structure and provenance rather than numerical agreement.
