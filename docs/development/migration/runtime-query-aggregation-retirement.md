# Runtime query and trajectory aggregation ownership

This package removes two internal relays while preserving existing configuration
and rank-local noisy-MPS workflows. It changes neither the tensor backend registry
nor trajectory statistics, sampling, RNG state, numerical kernels or result schema.

## Actual owners

| Previous internal definition | Final operation |
| --- | --- |
| `runtime/backend_registry.py::set_active_backend` | Delete the no-op selection relay. `configuration.set_backend` queries the existing `get_backend_capabilities(name)` directly before its unchanged context update and error handling. The registry remains the capability owner. |
| `runtime/executors/mps/noisy.py::merge_noisy_mps_results` | Delete the executor relay. The existing aggregation implementation in `runtime/trajectories/mps.py` retains the operation name `merge_noisy_mps_results`; its redundant `_runtime` suffix disappears. All callers use that owner. |

Paths are relative to `flagquantum`. The complete aggregation definition is
unchanged apart from its name. The public selection method is AST-identical
apart from its internal capability-query binding. The original query relay never
selected a backend: context mutation already belongs to `configuration.set_backend`.
No setter semantics are moved into the registry, and no validation moves earlier.

The existing guide and refusal message refer to the merger by its operation name,
which remains available at the authoritative Runtime owner. The one maintained
trajectory test changes only its import; its full original 12-trajectory,
seed-113, three-rank-local workflow and every assertion remain intact.
CPU rank-local statistics are not evidence of distributed capacity expansion.
No new alias, wrapper, module facade or compatibility hook replaces either relay.

## Public and contract boundaries

The protected root/configuration operations, backend capability types, extension
execution admission, exception chain, default selection, ContextVar isolation and
published result/checkpoint schemas remain unchanged. The obsolete registry
helper is absent from the protected baseline and selected stable/experimental
namespace contracts. Proposal 063 remains explicitly proposed; its compatibility
prose must describe current implementation accurately, without treating an
unapproved proposal as authorization for a Stable Core change or as a new
permanent compatibility promise.

`runtime/executors/mps/training.py::train_distributed_mps` remains at its recorded
locator in `contracts/public-api-v0.2-baseline.json` and experimental contracts.
It owns the existing public lazy boundary to the training engine. Removing that
locator requires the public API process; this package introduces no additional
training compatibility layer. Existing Circuit/root operations, public plan
methods, evolution namespaces and registered compiler callback identities keep
their previously documented bases.

A same-module deletion or implementation rename is not a source-file relocation.
Do not manufacture a moved definition or change a source-identity contract merely
to count either change as a directory migration. Record the executor's partial
relay retirement against its actual trajectory owner; keep the rest of the
executor's substantive compilation/scheduling lifecycle intact.

## Original behavior and required acceptance

The frozen original corpus contains 32 complete outcomes from the qualified
#1352 native installation. All four relevant defining/caller files are
byte-identical to the current #1354 prerequisite source. The digest is
`dd6fe92d6ab62654b512a5f49991074d636df550f280fd06a973e7419dce3783`.
It covers backend aliases, missing/invalid names, default lookup and public
context selection, plus complete, reordered, partial, failed and invalid
rank-local merges. Full tensors, MPS states, statistics, input graphs and
refusals are encoded; original 12 trajectories and actual seeds are retained.

The external complete-product preview reproduces all 32 outcomes byte-for-byte.
It adapts only verifier owner/default-call/provider/output bindings and preserves
the original capture filename/line positions. The original relay's no-argument
`pytorch` default is replayed through an explicit existing-owner argument, not
confused with the capability query's distinct current-context default.
The preview reuses unchanged native binaries and dependencies; it is not a newly
built candidate wheel or formal implementation qualification.

Before completion, require complete current/incoming consumer review, unchanged
public contracts, original32 replay from actual source and a newly packaged
external installation, focused/default tests, optional-import checks and this
package's own exact-head CI. Preserve original evidence and report failures and
conditional skips truthfully. No predecessor green status substitutes for the
new head. The incoming #1084 native/developer/Construction closure remains open.

## Existing user workflow

```python
import flagquantum as fq

result = fq.run(
    fq.Circuit(1).x(0),
    options=fq.ExecutionOptions(backend="pytorch"),
)
print(result.state.abs().square())
# tensor([[0., 1.]])
```

The same registered capability selection and execution/refusal rules apply.
