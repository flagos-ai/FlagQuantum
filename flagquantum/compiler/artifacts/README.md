# Compiler artifacts

This directory owns target artifact construction, emitted-program conformance,
physical plans and compilation-evidence generation. Core owns the artifact and
evidence schemas. Runtime and Deployment own read-only handoff validation; this
directory must not execute programs or contact providers.

Dependencies may point to Core and other Compiler stages. Numerical block
verification stays in Compiler lowering; the evidence schema is not a numerical
oracle and must not import Compiler or Simulation.

The existing offline user workflow is:

```bash
python -m examples.target_aware_compilation
python -m pytest tests/integration/test_artifact_compilation_pipeline.py -q
```

For a typical evidence change, inspect `compilation_evidence.py`, the applicable
Core evidence schema, and `physical_plan.py`. Preserve version 1–3 encodings and
identities. Add a scenario that constructs actual compilation objects, checks
serialization and rejects a rehashed false lineage.

`native_block_evidence.py` privately generates Core's version 4 physical plan
from a retained native baseline plan, verified block rewrites and the final
schedule. It replays baseline plan generation, routing, native legalization,
block operators and output scheduling. Output instructions name either one
unchanged native input or a collective block; no fused instruction receives a
fictitious single source. Mapping transitions remain in topology-program space.

This builder is not yet connected to automatic target compilation, executable
artifact creation or Runtime/Deployment handoffs. Its admitted baseline is the
existing undirected, unallocated version 1 physical plan, which requires identity
initial and final placement. Private optimizer support for other initial
placements does not expand the version 4 reader's admission.

The shortest development path for this builder is:

```bash
python -m pytest tests/integration/test_native_block_physical_evidence.py tests/hybrid_compiler/test_physical_plan.py -q
```

Start with the SWAP followed by a reversed CX scenario: four native CZ gates
become two, with many-to-many lineage and independent stage verification. The
test also exercises disjoint spectators, routing transitions, complete
cancellation, resource refusal and altered baseline ancestry. This is a local
correctness example, not an SDKBench performance result. Replay and synthesis
costs must be included in later compile-time and memory measurements.
