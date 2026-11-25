# Torch capture and callable compilation

Compiler owns `module_builder.py` and `training.py`. Module capture converts
supported Circuit construction into Core CircuitIR placeholders and an ephemeral
Torch FX parameter-binding graph. Training compilation wraps an existing
callable using Torch's graph compiler and records compilation diagnostics.

Allowed dependencies include the standard library, Torch and Torch FX, native
Circuit construction, Core IR and parameters, and shared errors. Runtime owns
execution, capture reuse and invalidation. This directory must not select
devices, implement numerical simulation kernels or submit jobs.

The supported user workflow is `fq.Module` and ordinary Torch training. Existing
expert module paths `flagquantum.compiler.module_builder` and
`flagquantum.compiler.training` remain available. Internal consumers should
import directly from this directory.

```bash
python -m pytest -q tests/unit/test_module_ir_capture.py tests/unit/test_module_ir_replacement.py tests/unit/test_training_callable_compiler.py tests/unit/test_training_compilation.py
```

These executable scenarios build public Modules, compare parameter binding and
gradients with independent Torch arithmetic, and run optimizer steps through a
replacement compiler. For a small change, extend one supported binding operation
or a callable failure case in the matching test. Preserve detach semantics,
optimizer mutation counts and user exceptions. The Module replacement test also
checks the existing JAX consumer when that optional dependency is installed.
