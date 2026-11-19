# Circuit optimization

This directory owns target-independent circuit rewrites, optimization levels,
pass and analysis declarations, and fixed-point optimization. Target routing and
execution belong to the topology and runtime domains.

- `pass_registry.py` binds built-in passes and their analysis requirements.
- `pipeline.py` resolves optimization levels and admitted passes, then delegates
  convergence to `PassManager` in `pass_manager.py`.
- Individual rewrite modules own their numerical and measurement semantics.
- `_cz_blocks.py` owns verified native CZ block proposals. Its private
  `_cz_local_cleanup.py` adapts Cirq local-run merging for arbitrary-angle
  RX/RZ targets; raw budgets and independent whole-block acceptance remain in
  `_cz_blocks.py`. See the [source and evidence record](../../../docs/development/COMPILER_CZ_CIRQ_REUSE.md).

Dependencies may include Core IR and operator semantics, compiler register and
lowering primitives required by a rewrite, and the circuit math those transforms
use. This directory must not choose runtime backends, submit jobs, or execute
simulation kernels.

The supported expert entries are `flagquantum.compiler.optimize`,
`default_pass_registry`, and the pass-manager exports. The root
`compiler/pipeline.py` preserves their entry signatures and forwards replacement
registries used by existing compiler consumers.

For a typical change, edit a rewrite and its focused semantic tests. Changes to
pass composition also require the level and admission tests.

```bash
python -m examples.compiler_optimize
python -m pytest -q tests/hybrid_compiler/test_passes.py tests/unit/test_compiler_pass_manager.py tests/unit/test_compiler_pass_plugins.py tests/team/compiler/test_optimization_levels.py
```

The example checks numerical equivalence and the recorded optimization level.
The tests exercise analyses, admitted passes, convergence, and failure behavior.
