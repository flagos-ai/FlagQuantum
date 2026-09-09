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

Straight-line local temporaries are supported. Bounded `for` loops may rebind
outer scalar, index, and bool names through explicit `scf.for` region arguments,
results, and yields. Compile-time-resolved `if` branches may merge the same
classical types through explicit `scf.if` operands, region arguments, yields,
and results, including pass-through on a side that does not assign the name.
Assignment discovery follows nested `if` and `for` regions, so each enclosing
operation explicitly carries any outer state updated below it. Tensor state,
loop-target shadowing, and classical state escaping a measurement-dependent
branch fail closed.

Measurement booleans may be assigned to names and composed with `not`, `and`,
`or`, bool equality/inequality, and equality/inequality between measurements.
Compiler canonicalizes the result as bounded disjunctive normal form. A single
conjunction retains existing `conditions` metadata; multiple clauses use
private `condition_clauses` metadata consumed by both local dynamic trajectory
strategies. The default expansion ceiling is 64 clauses. Inline measurement
composition, conditional measurement, excessive expansion, and provider
dialects without explicit complex-predicate support fail closed.

Scalar, index, and bool values may leave a measurement-dependent branch as
bounded condition-partitioned SSA cases. Later arithmetic and comparisons are
evaluated per case; quantum parameters and wires lower to mutually exclusive
conditioned instructions. Static bounded loops may carry these values.
Conditional measurement, measurement-derived loop bounds or program returns,
case expansion beyond the configured ceiling, and stochastic gradients fail
closed.

Statically bounded loops may contain measurement, same-round conditioned
feedback, and unconditional ancilla reset. Lowering unrolls each round and
allocates dense classical bits in source order; a syndrome bool may be carried
through the loop and returned after its final iteration. Unrolling,
measurement count, and predicate expansion have independent configured
ceilings. Conditional measurement/reset and measurement-dependent termination
remain outside the profile.

`specialize.py` selects one bounded runtime path and records an ephemeral gate
trace. `lowering.py` converts that trace to the existing `CircuitIR` with Core
`Parameter` slots. Runtime tensor slices stay in a separate binding map by
reference; reusable structure identity never serializes or hashes their values.
The optional structure cache is bounded and reports hit, miss, and eviction.

`passes.py` owns the verified Program IR normalization stage used by both static
specialization and dynamic lowering. The first concrete analyses are constant
facts and whole-program SSA use counts. The first transformations fold supported
constant-only arithmetic, inline a compile-time-selected `scf.if` region,
eliminate statically empty `scf.for` regions, and remove only unused constants.
Control simplification rewires explicit carried SSA values and the linear
quantum effect to the selected yield or initial loop values. Verification runs
before the pipeline and after every pass; at most 32 passes are accepted. Each
lowered result retains both the source and optimized semantic identities plus
per-pass operation counts. The private `optimize=False` path is a differential
oracle, not a second compiler mode or public API commitment.

This stage deliberately keeps `HybridProgram` as its input and output. It does
not introduce a target dialect, `TargetIR`, general pass registry, stable pass
extension API, device optimization, or performance claim. Operations that may
fail at runtime are not removed merely because their result is unused.

`dynamic_lowering.py` owns the separate bounded measurement-feedback profile.
Capture represents a top-level measurement as a boolean SSA value plus a new
linear quantum effect. A direct `if` on that value lowers both branches to
classically conditioned instructions on one Core `CircuitIR`; it does not
sample or collapse state in Compiler. The private Runtime dynamic-session
handoff performs that execution through the existing trajectory machinery.

The base dynamic profile is deliberately small: top-level measurements only,
direct measurement-bool conditions, H/X/CX gates, and no stochastic gradients.
Its bounded parameterized extension accepts non-trainable scalar, index, and
bool inputs, specializes positive `range` loops, and lowers RX/RY values through
ordered Core `Parameter` slots. Unsupported nesting, tensor inputs, trainable
values, loop measurements, durable sessions, accelerators, and distributed
execution fail closed or remain outside the contract.
