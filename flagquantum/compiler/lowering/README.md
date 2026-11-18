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
