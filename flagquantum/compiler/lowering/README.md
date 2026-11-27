# Circuit lowering and target adaptation

This directory owns program lowering, native gate legalization, direction
legalization integration, and instruction scheduling. `pipeline.py` composes
initial circuit lowering, optimization, routing, and post-routing evidence for
the existing `flagquantum.compiler.compile` entry.

The root `compiler/pipeline.py` forwards the optimizer and router at call time.
This preserves existing replacement consumers while the stage implementation
owns option validation, transformation order, and metadata.

Dependencies may include Core IR, operator and capability semantics, compiler
optimization, topology, formats, and artifact validation at their existing
boundaries. Runtime selects execution backends. Numerical execution and provider
submission belong to their owning domains.

For a typical change, edit the relevant lowering module and its semantic tests.
For orchestration changes, start with the runnable topology example and compiler
replacement tests.

```bash
python -m examples.circuits.compiler.target_aware_compilation
python -m pytest -q tests/hybrid_compiler/test_passes.py tests/hybrid_compiler/test_target_legalization.py tests/team/compiler/test_compiler_pipeline_replacement.py tests/team/compiler/test_routing_conformance.py
```

The example checks legal target edges, restored mappings, and CPU numerical
equivalence. Artifact compilation enters through
`artifacts/artifact_compilation.py` and uses `target_legalization.py` for its
verified snapshot-driven path.

## Target admission during decomposition

The existing U3 candidate checks its first concrete target leaf before building
its scalar correction. Accepted U3 words still validate every correction leaf;
target-independent optimization retains the complete word. See the
[workflow, phase contract and tests](../../../docs/development/COMPILER_U3_CANDIDATE_ADMISSION.md).

## Direct CZ parity words

`basis_translation.py` retains the CX definitions and offers direct CZ words
for RZZ, CRZ and controlled phase. The existing shortest-word search expands and
checks both; target predicates, angle objects, metadata and growth limits remain
authoritative. See [identities, application observations and limits](../../../docs/development/COMPILER_DIRECT_CZ_PARITY.md).

## Native CZ block stage

`native_block_optimization.py` retains the exact per-instruction native baseline
and records collective RX/RZ/CZ rewrites with the Core block evidence type. It
reuses the existing bounded collector and KAK synthesis. Its verifier replays
native legalization, checks exclusive memberships and unchanged instructions,
and recomputes the 4x4 operators, phase witness, CZ costs and combined budget.
Dependency validation scans the source once; it does not compare every block
against every spectator.

Native target declarations of ProgramIR measurement and reset capabilities do
not count as additional entanglers. They do not enable dynamic instructions in
this CircuitIR stage. Terminal measurement requests are retained, while noise
channels and unknown annotations remain global collection boundaries.

Routing annotations are admitted only with the retained undirected topology
result and its coupling map. The stage replays the existing router with the
recorded selected strategy, compares the complete retained result and program
hash, and replays native legalization before deriving eligible native leaves.
It never strips annotations from the retained IR. Original source annotations
remain boundaries, including source metadata that imitates reserved router keys.
Directed routing and allocated workspace remain outside this stage's admission.
Replay repeats the selected routing strategy; its compile-time and memory cost
must be included in subsequent performance measurements.

The stage is automatic in `legalize_circuit_for_target` for full-CZ targets with
absent or undirected, unallocated topology and identity initial placement.
`TargetLegalizationResult.native_block_optimization` retains the native baseline,
output and rejection counters. Final scheduling uses the accepted output.
An accepted rewrite enters the [v4 artifact evidence workflow](../artifacts/README.md).
When no rewrite is accepted, existing identities and encodings are preserved.
Runtime and Deployment accept the v4 handoff through their existing read-only
identity checks and offline preparation; no remote execution is certified here.

Run its focused offline checks with:

```bash
python -m pytest -q tests/unit/test_native_block_optimization.py tests/unit/test_compilation_target_cz_resynthesis.py
```

They exercise the existing SWAP/reversed-CX workflow, reducing its retained
28-instruction baseline from four CZ gates to two, plus cancellation, spectators,
parameter boundaries, tampering and resource refusals. Independent simulation
matrices verify the complete operator, including both operand orders and all
five routing selections. Nonidentity initial placements are applied explicitly
to every input basis state; final layouts remain restored. Rehashed mutations
of routing parameters, ordering, ancestry, coverage and restoration are rejected
before annotations can enter a block. The existing user-facing target workflow
above retains its prior behavior for targets outside the admitted CZ profile.
