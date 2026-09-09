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
RX/RY selection, a CX loop, and a scalar expectation. `capture.py` parses the
restricted Python source into that representation without invoking the function
or reading tensor values. It remains compilation only and does not execute the
program.

Straight-line local temporaries are supported. A branch or loop may not rewrite
an outer classical binding yet: that requires explicit region-carried values
and therefore fails closed instead of silently using a stale value.

`specialize.py` selects one bounded runtime path and records an ephemeral gate
trace. `lowering.py` converts that trace to the existing `CircuitIR` with Core
`Parameter` slots. Runtime tensor slices stay in a separate binding map by
reference; reusable structure identity never serializes or hashes their values.
The optional structure cache is bounded and reports hit, miss, and eviction.

`dynamic_lowering.py` owns the separate bounded measurement-feedback profile.
Capture represents a top-level measurement as a boolean SSA value plus a new
linear quantum effect. A direct `if` on that value lowers both branches to
classically conditioned instructions on one Core `CircuitIR`; it does not
sample or collapse state in Compiler. The private Runtime dynamic-session
handoff performs that execution through the existing trajectory machinery.

This first dynamic profile is deliberately small: no runtime inputs,
top-level measurements only, direct measurement-bool conditions, H/X/CX gates,
and no stochastic gradients. Unsupported nesting, trainable tensors, durable
sessions, accelerators, and distributed execution fail closed or remain
outside the contract.
