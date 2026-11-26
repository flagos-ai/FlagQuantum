# Independent structure migration

This ledger records small, complete migration work packages that can proceed
while [PR #1084](https://github.com/flagos-ai/FlagQuantum/pull/1084) remains open.
It does not certify completion of the repository-wide architecture migration.
Rust implementation remains deferred until #1084 closes and the integrated
architecture migration passes acceptance.

## Frozen review baseline

- Main: `7278efd00f6768fbbacff6a41eb4dec57a59783e`.
- Open, draft #1084 head: `e27f1096dff90586f125f5629f8f2dd401730f41`.
- #1084 base: `03017ddf96dd2d9604f2dbc4d2cdfe2f3837f878`.
- Audit date: 2026-11-26 (Asia/Shanghai).

Independence requires auditing every implementation, importer, test, package
reference, build reference, and new consumer in the PR snapshot. An empty
changed-file intersection alone is insufficient. Repeat the audit if either
head advances before integration.

## Allocator peak observation: implemented, awaiting integration

| Previous authoritative path | New authoritative path |
| --- | --- |
| `flagquantum/compute/_memory.py` | `flagquantum/compute/local/_memory.py` |

The helper observes memory on directly controlled devices, so Local Compute owns
it. The old private module is deleted. No compatibility shim, new public type,
new dependency, API snapshot update, or second implementation is introduced.
Only its relative imports and its consumers' imports change; the function's
signature and body are AST-identical. Existing platform contracts and registry
remain at their current locations.

The complete executable reference closure is:

- `flagquantum/compute/_memory.py` (implementation before the move);
- `flagquantum/runtime/executors/statevector/training.py`;
- `flagquantum/runtime/executors/mps/training_engine.py`;
- `tests/unit/test_training_peak_memory.py`.

All four files are byte-identical between the frozen main and #1084 snapshots.
A repository-wide search of each snapshot finds the same three consumers and
no additional references to the old module. The package is discovered by the
existing setuptools package discovery. Local contributor instructions are in
[`compute/local/README.md`](../../flagquantum/compute/local/README.md).

Validation on Python 3.12.14 and Torch 2.14.1, using the isolated migration
checkout as `PYTHONPATH`:

```sh
python -m pytest tests/unit/test_training_peak_memory.py tests/unit/test_statevector_training.py tests/unit/test_mps_training.py tests/integration/test_training_parameter_lifecycle.py -q -p no:cacheprovider
python tools/public_api_snapshot.py
python tools/check_architecture.py
```

The same 70 tests pass before and after the move. API, architecture, and
Integration ownership checks pass. The allocator function AST is unchanged, and
setuptools discovers `flagquantum.compute.local`. These checks establish CPU
regression coverage and fake-provider behavior, including FlagOS observation
ordering; they do not certify physical accelerators, distributed execution,
wheel installation, or the full #1084 integration.

## Deferred work

- Root Compose relocation: #1084 adds `.github/workflows/publish-dev-container.yml`
  references to `compose.dev.yaml`. Migrating only the file and current main's
  README would leave future CI consumers stale. Reconcile the complete workflow
  closure after #1084 closes. Preserve Compose project directory, build context,
  bind mount, environment loading, and project identity during that migration.
- Remaining direct-device modules, Remote/Realtime consolidation, compiler
  construction, Core IR, Simulation engines, build metadata, and whole-tree
  tooling: execute their complete work packages against the integrated baseline;
  this allocator move does not establish their independence.
- Rust optimization: retain the separately documented post-integration gate.

Rollback of this work package is one commit revert. No persisted data, public
schema, checkpoint, or deployment package requires conversion.
