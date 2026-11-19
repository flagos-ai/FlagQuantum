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
python -m examples.target_aware_compilation
python -m pytest -q tests/hybrid_compiler/test_passes.py tests/hybrid_compiler/test_target_legalization.py tests/team/compiler/test_compiler_pipeline_replacement.py tests/team/compiler/test_routing_conformance.py
```

The example checks legal target edges, restored mappings, and CPU numerical
equivalence. Artifact compilation enters through
`artifacts/artifact_compilation.py` and uses `target_legalization.py` for its
verified snapshot-driven path.

## Native CZ block stage

`native_block_optimization.py` retains the exact per-instruction native baseline
and records collective RX/RZ/CZ rewrites with the Core block evidence type. It
reuses the existing bounded collector and KAK synthesis. Its verifier replays
native legalization, checks exclusive memberships and unchanged instructions,
and recomputes the 4x4 operators, phase witness, CZ costs and combined budget.
Dependency validation scans the source once; it does not compare every block
against every spectator.

The implementation currently remains private and is not connected to automatic
target compilation. Routing annotations are still boundaries. Enabling it in
that workflow requires validated routing provenance, v4 physical-plan generation
and the subsequent Runtime/Deployment handoff work from the approved proposal.
No new execution or deployment support is claimed by this stage alone.

Run its focused offline checks with:

```bash
python -m pytest -q tests/unit/test_native_block_optimization.py tests/unit/test_compilation_target_cz_resynthesis.py
```

They exercise the existing SWAP/reversed-CX workflow, reducing its retained
28-instruction baseline from four CZ gates to two, plus cancellation, spectators,
parameter boundaries, tampering and resource refusals. Independent simulation
matrices verify the complete operator, including both operand orders. The
existing user-facing target workflow above continues to use its prior behavior.
