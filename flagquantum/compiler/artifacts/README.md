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

The target workflow automatically optimizes admitted full-CZ targets after
per-instruction native legalization and before scheduling. A successful rewrite
produces a version 4 physical plan and compilation-evidence bundle. An absent or
unsuccessful rewrite preserves version 1–3 identities and encodings.

`physical_plan.py` constructs the authoritative Core evidence type;
`native_block_evidence.py` derives its native-input and collective-output records.
Output instructions name either one unchanged native input or a collective
block; no fused instruction receives a fictitious single source. Mapping
transitions remain in topology-program space. Bundle generation replays routing,
native legalization, block operators, final scheduling and physical lineage.
This serialization check uses the retained snapshot's capture time; it does not
authorize executing against an expired target snapshot.

The admitted artifact profile remains undirected and unallocated, with identity
initial and final placement. Private optimizer support for other initial
placements does not expand the version 4 reader's admission. Runtime accepts v4
through its existing read-only source/target/output identity checks. Deployment
can prepare the handoff offline while retaining snapshot-expiry and shot-limit
checks. Neither adapter recompiles or independently proves a block operator,
and these local checks do not certify remote submission or hardware execution.

The shortest development path for this workflow is:

```bash
python -m pytest tests/integration/test_native_block_physical_evidence.py tests/hybrid_compiler/test_physical_plan.py -q
```

Start with the SWAP followed by a reversed CX scenario: four native CZ gates
become two, with many-to-many lineage and independent stage verification. The
test also exercises disjoint spectators, routing transitions, complete
cancellation, resource refusal, parameter boundaries and altered baseline
ancestry. It replaces the block implementation with an identity implementation
without modifying target, artifact or emission consumers, and compares unchanged
artifact bytes when no rewrite is accepted. This is a local
correctness example, not an SDKBench performance result. Replay and synthesis
costs must be included in later compile-time and memory measurements.
