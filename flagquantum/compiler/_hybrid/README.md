# Private hybrid program IR

This directory owns the missing program-level semantics for bounded hybrid
quantum-classical compilation: typed SSA values, structured `if`/`for` regions,
and ordered quantum effects. It is private Compiler machinery and is not
exported from `flagquantum` or `flagquantum.compiler`.

It does not own static circuit semantics, target execution, numerical state,
devices, providers, gradients, or framework capture. Static quantum regions
will lower to the existing Core-owned `CircuitIR`; existing Compiler, Runtime,
and Simulation boundaries remain authoritative.

The historical `_compiler` tree is reference evidence only. Do not restore or
copy its package structure, `QuantumIR`, `TargetIR`, artifact types, provider
models, or runtime contracts here.

## Ten-minute change path

1. Add or adjust the smallest required type/operation in `model.py` and
   `schemas.py`.
2. Add its semantic invariant to `verifier.py`.
3. Add one readable accepted scenario and one rejected scenario under
   `tests/hybrid_compiler`.
4. Run:

   ```bash
   python -m pytest tests/hybrid_compiler -q
   python tools/check_architecture.py
   ```

The first golden scenario is nested tensor-indexed loops with data-dependent
RX/RY selection, a CX loop, and a scalar expectation. Phase 1 represents and
verifies that structure; it does not execute it.
