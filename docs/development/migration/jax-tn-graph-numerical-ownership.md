# JAX tensor-network graph numerical ownership

Status: actual source and independent native installed artifact validated after
PR #1348 passed exact-head CI at
`ae49084621dd7681236fda4a2c2f4cd210b68963`. This change still requires its
own committed-source binding and exact published-head CI acceptance.

## Final ownership

| Donor | Existing numerical owner | Remaining Runtime responsibility |
| --- | --- | --- |
| `runtime/executors/jax/tensor_network/gradients.py` state graph | `simulation/tensor_network/jax/kernels.py::jax_parameterized_tn_state_nodes` | Backend admission, precision policy and parameter bridge |
| Same donor, expectation graph | `simulation/tensor_network/jax/kernels.py::jax_parameterized_tn_expectation_nodes` | Observable selection, static slicing/tasks, execution and result evidence |
| `runtime/executors/jax/tensor_network/contraction.py` local rank sum | `simulation/tensor_network/jax/contraction.py::sum_tensor_network_partials` | Empty-rank refusal, device admission, real pmap collectives and execution labels |

Simulation constructs the physical zero-state/gate/Pauli tensors and their
contraction labels. It accepts an explicit admitted dtype. Its lazy JAX imports
neither select Runtime policy nor enable x64. Gate numerics continue to use the
existing active compute dtype, independently of the initial-state precision.
No dependency, module, registry, public schema or native ABI is added.

Runtime retains classical objective weights/aggregation, static Torch graph
planning, slice/rank assignment, real collective order, autograd bridge,
parameter lifetime, timers and evidence. Generic objective assembly is not a
second physical tensor-network implementation.

The two old private graph definitions and all current bindings are deleted.
Consumers access the defining owner modules directly, so replacement does not
require changing the consumer. No forwarding names or compatibility layer are
retained. The two retired-member ledger rows describe these actual removals;
the inline reduction extraction is not recorded as a definition relocation.

## Acceptance and limits

Preserve complete graph tensors, labels, names, metadata, order, amplitudes,
expectations, derivatives, dtypes and refusal behavior. The retained original
66-scenario record covers complex64/128, independently selected graph/gate
precision, batch sizes zero/one/two, conjugation, observables, channel refusal,
negative batch admission and empty/nonempty local reductions. Original records
must not be regenerated from the migrated implementation.

Run the replacement tests in `tests/unit/test_jax_tn_graph_ownership.py`, the
existing JAX distributed-plan and local numerical suites, default unit/smoke,
architecture/API/ownership gates, and an independently installed native wheel.
The installed artifact must reproduce complete source outcomes and load product
providers from its own installation. Exact-head CI must verify all preserved
selection nodes and whole executed job times, including setup and upload.

CPU local simulations and replacement fakes are development correctness
evidence. They do not certify real multi-device capacity, new public batching,
new gradient coverage or scalability. Scientific tests, markers, tolerances,
iteration budgets, safety limits and coverage thresholds remain unchanged.

## Incoming branch

Main remains `945932e33552245fd82d2760221e29dad9628dc6`. PR #1084 at
`e8ffeb01629254bb9463c223b8b459f81b6735fd` is draft and conflicting. Its last
eight commits changed fourteen SHA-verified blobs with no overlap with this
migration. The unfiltered changed-incoming Python audit still reports seventeen
retired-import integration obligations for both the parent 586-row ledger and
active 589-row ledger. These are retained obligations, not G0 acceptance.

The overall architecture migration and the ten-minute CI target remain open.
The qualified parent still had Maxwell at 815 seconds and logical-cat source-RX
at 619 seconds; no scientific input reduction is authorized. Rust work remains
deferred until the incoming branch and architecture migration close.

## Actual source and installed acceptance

The complete default smoke/unit selection passed: 21,896 passed and 865
conditional skips, 440 seconds whole-process time, four workstealing workers
and one calculation thread each. All previous 22,748 selection nodes remain;
the only additions are thirteen owner-replacement cases. These thirteen need
optional JAX and are also executed in the real JAX source and installed suites.
An initial serial invocation reached its local 720-second observer bound; its
process group was terminated and waited, and its partial log is retained. The
complete retry uses the already qualified parent execution profile. No test,
per-test limit, marker, oracle, optimizer budget or assertion was changed.

Eight complete real JAX modules passed in source and in an independently
installed native wheel: 207 passed, six existing device-condition skips. All
213 outcomes match, both native extensions and all 509 loaded product providers
come from the wheel's own installation, and all 1,265 product Python files match
source, wheel and installed bytes. The wheel SHA-256 is
`9ff45ecfb066f5eda4176402fe581a6741f01683eed5769eebfed1ccf57a0888`.
Dependencies are borrowed from existing task-owned third-party installations;
this is not a new dependency-resolution claim.

Both actual source owners and the installed owners reproduce all 66 complete
original outcomes, including every tensor byte, graph field and refusal. The
source replay inherited an inaccurate console label for its provider count;
its actual assertion requires the source checkout, and its exact file bindings
and separate parity receipt record this explicitly. The installed replay
requires its own site-packages. No original outcome record was regenerated.
The first replacement-test attempt failed on metadata=None, and an existing
boundary test initially referenced the deleted Runtime import. Both raw failures
are retained; tests now preserve None and inspect the true Simulation owner.

Ruff, Black, strict Mypy, architecture, public API, vocabulary, ownership, docs,
language, hygiene, fixed-source hooks, Black24 and supplementary contract gates
passed. AST review confirms unchanged pre-existing Simulation definitions and
only the prescribed Runtime caller/reduction substitutions. No public
signature, result schema, native source or ABI changes. This local acceptance
does not stand in for this PR's own actual merge-source CI reports.
