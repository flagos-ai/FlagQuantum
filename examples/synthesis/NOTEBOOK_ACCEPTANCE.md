# Native Notebook acceptance

On November 19, 2026, the maintainer approved formal local acceptance of 212
native Notebook workflows. Each accepted workflow has an editable native
Notebook, a runnable Python entry, an independent numerical acceptance test and
source provenance in the migration inventory.

| Category | Accepted | Original Notebook inventory |
| --- | ---: | ---: |
| algorithms | 45 | 45 |
| applications | 69 | 69 |
| community | 18 | 18 |
| functions | 31 | 31 |
| tutorials | 49 | 50 |
| Total | 212 | 213 |

The approved scope is 212/212 (100%). The original upstream inventory remains
213, so the generated catalog correctly reports 212/213. Neither denominator
certifies unclassified standalone Python sources or a product release.

The deferred workflow is
`tutorials/technology_demonstrations/fault_tolerance_engine_end_to_end/fault_tolerance_engine_end_to_end`.
Its original `public/depolarize_1e-3` calibration is unavailable. Original-model
acceptance remains required; no synthetic calibration or substituted model
receives acceptance credit.

## User workflow

From the repository root, run a native high-level quantum example:

```sh
python -m examples.synthesis.community.basic_examples.superposition.main
```

Its reusable `@qfunc` allocates three bits, prepares a uniform superposition and
reports eight named-register outcomes with probability approximately 0.125 each.
Open the adjacent `walkthrough.ipynb` for an editable explanation. The five
category READMEs and [catalog](CATALOG.md) link to all accepted workflows.
Examples use FlagQuantum Python; original Qmod sources remain test fixtures.

## Evidence and integration boundary

The formal local acceptance snapshot is commit
`917547a1c940bfdcf426f256e02db3494c7ce2d5`. Its source audit checked all 212
Notebook files, 212 upstream blobs and 2,721 original code-cell hashes. Per-case
`provenance.json` records the source identity and numerical acceptance test.
This snapshot is preserved; synchronization with main requires fresh integration
checks and does not rewrite historical evidence.

The retained Qmod replay distinguishes 210 original execution passes, six
explicit original semantic refusals and six passing approved replacement paths.
Replacement paths do not turn original refusals into original execution passes.

The last pre-integration default run had 17 failures, 19,691 passes and 769
skips. An eighteenth historical failing node was independently reproduced.
There were no new failure nodes in that batch. These are historical measurements,
not a passing result or a statement about the current integration revision.

Local acceptance, pull-request CI, merge approval and release certification are
separate gates. See the integration PR for checks of the synchronized revision.
