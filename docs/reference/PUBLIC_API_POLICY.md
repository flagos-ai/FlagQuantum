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
