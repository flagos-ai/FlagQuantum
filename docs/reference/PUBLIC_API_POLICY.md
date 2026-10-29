# Public API Policy

FlagQuantum's stable root API is the exact snapshot in `public_api_v1.json`.
Removing or renaming one of those exports requires a versioned migration and
release-note review.

`flagquantum.experimental` contains explicitly unstable planner, rank, shard,
transport, and evidence structures. They may evolve without the stable API
compatibility guarantee. Historical internal root exports were removed before
the first public alpha; users import maintained extensions from their explicit
domain namespaces.

The unreleased maintained `flagquantum.synth` extension and its narrow
`fq.run(Program)` admission are recorded in
[`synthesis-program-run-v1-candidate.json`](../../contracts/synthesis-program-run-v1-candidate.json).
Its seven domain symbols are not root exports. Services owns their single
implementation; Runtime continues to execute an existing ExecutionPlan. This
implementation candidate does not claim completed independent user trials,
release approval, Program serialization or broader backend/Qmod support.

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

## Removed pre-release root aggregator

The historical `flagquantum.api` module and implicit root fallback were removed
before the first public alpha. The package root exposes exactly the stable
snapshot plus `flagquantum.experimental`; algorithms, compiler, deployment,
noise, providers, and backend tools use their explicit package namespaces.
