# Public API Policy

FlagQuantum's stable root API is the exact snapshot in `public_api_v1.json`.
Removing or renaming one of those exports requires a versioned migration and
release-note review.

`flagquantum.experimental` contains explicitly unstable planner, rank, shard,
transport, and evidence structures. They may evolve without the stable API
compatibility guarantee. Historical root access to listed internal structures
emits `DeprecationWarning` and remains available only through version 0.3.0.

Modules below `flagquantum.runtime`, implementation helpers prefixed with
an underscore, and symbols absent from the stable/experimental snapshots are
internal APIs.

Maintainer and coding-agent enforcement is defined by
[`PUBLIC_API_PROTECTION.md`](../development/PUBLIC_API_PROTECTION.md). Updating
the API snapshot is not, by itself, authorization to change the stable
contract.

## Removed pre-release device API

`DistributedQuantumDevice`, `GeneralEncoder`, `InvertibleUnitary`,
`measure_allZ`, and the DTensor interchange helpers are historical v0.1
interfaces. They were removed before the first public alpha, are absent from
`public_api_v1.json`, and have no compatibility layer. Applications must use
`Circuit`, `Module`, and the plan/run/result interfaces in the stable snapshot.
