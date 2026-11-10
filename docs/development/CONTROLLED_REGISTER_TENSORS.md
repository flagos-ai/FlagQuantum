# Controlled register tensors (GRAPH56)

Simulation owns explicit dense tensor inspection of a finite register signature.
The two owners are `simulation/tensor_network/register_tensor.py` (word axes and
zero-port projection) and `controlled_tensor.py` (identity plus active block).
They consume existing Core `Signature` and `Register` metadata. They do not
construct a foreign component graph, choose a backend, lower gates, allocate
workspace, or modify the public Circuit/Program contract.

The implementation follows the unchanged Qualtran v0.7.0
`simulation/tensor/_tensor_data_manipulation.py` and
`_infra/controlled.py::_ControlledBase._tensor_data`, locked at
bcb17bb25360ed68ce3bc084645db3e929799389. Copyright and source links remain in
the owning files. Source Git blobs are checked against the original inventory.

## Layout and interface

A tensor has all output axes followed by all input axes. Each row-major
register element contributes one axis of size `2**bitsize`; its word basis uses
MSB-first bit order. A scalar zero-width word contributes a dimension of one;
a zero-length register array contributes no axes. The matrix projection helper
selects output zero at a LEFT port and input zero at a RIGHT port. This is a
zero-amplitude projection, not a reset or a partial trace. The all-scalar result
retains the source NumPy scalar type.

`controlled_register_tensor(controls, child_signature, patterns, child_tensor)`
requires THRU controls and child ports, with controls preceding child ports.
The caller supplies one activation array per control declaration, with matching
shape. The callback supplies the already contracted child tensor; it is called
once after identity creation and before active-space symbolic rejection. The
child output/input axes are reshaped to the existing signature. Both sides of
the control axes are indexed with the raw integer pattern, including NumPy
negative indexing. No signed or finite-field codec is inferred.

Finite nonnegative widths/shapes, control/pattern arity and pattern shapes are
checked at this native boundary. Non-THRU controls and child ports are rejected.
Unknown symbolic dimensions cannot be densely allocated. Raw index overflow and
child reshape errors remain failures. This finite inspection interface does not
claim complete upstream protocol, drawing or Cirq-unitary fallback coverage.

Dense identity creation uses `16 * 4**n` bytes for n total declared bits before
projection. This is an explicit small-system inspection tool, with no claim of
large-system capacity, primary execution replacement or differentiability
through a NumPy child callback. The result can be supplied to the existing
PyTorch `ContractionGraph`; no second contraction engine is added.

## Existing workflow

Users continue to append register operations through `circuit.append(op)`.
`tests/unit/test_controlled_tensor.py` binds the existing two-control X emitter,
compares its gates/feedback with the original `ControlledViaAnd`, and checks
both measurement branches against the dense controlled tensor. An entangled
spectator is retained, clean work returns to zero, and actual DynamicCircuit
append execution agrees. There are four activation patterns, eight forced
branches and sixteen sampled trajectories. Per circuit the emitted counts
include four T/Tdag and one measurement/reset, with one clean work wire. The
independent source QEC model reports four T-equivalents and one measurement.
The tensor inspection itself emits no gates and has no logical-resource model.

Run the focused checks:

```bash
python -m pytest tests/unit/test_register_tensor.py tests/unit/test_controlled_tensor.py -q
```

The tests also compare all 27 LEFT/RIGHT/THRU combinations, zero/empty axes,
scalar/word/matrix/multiple controls, Hadamard, global phase and an independently
supplied shaped child. Complete complex joint states and their adjoint recovery
pass through the existing contraction graph. Replacing a child's tensor supplier
does not modify the controlled consumer.

## Original failure retained

A zero-length control array around a global-phase child has no tensor axes.
The source identity helper returns a NumPy complex scalar; the source controlled
method then fails because that scalar does not support item assignment. Native
retains TypeError at the corresponding point after the child callback. This
scenario is diagnostic evidence and is not counted as successful controlled
execution. The original reference is not repaired. Other scalar-only controlled
signatures remain subject to that same boundary.

## Verification evidence

Implementation commits are 566c3a7c (register projection) and fea0b624
(controlled child supplier). Sixty-eight new tests pass in 4.30 seconds;
they include valid execution, failure-order checks and the scalar source
defect diagnostic. Both new owners have 100% statement/branch coverage
(72 statements, 32 branches). Related regression passes 658 tests in 9.81
seconds, retaining two existing dependency/deprecation warnings. All 64 local
integration tests pass in 4.58 seconds. Strict mypy covers both new owners and
both test files; Black, Ruff, Simulation scope, architecture, Program contract
and protected API checks pass.

Eleven unchanged upstream tensor tests pass in 4.86 seconds, with 35 unrelated
tests deselected and 13 upstream environment warnings retained. Four original
Git blobs match the source inventory. Default smoke/unit stops at maxfail with
21507 passed, 300 skipped, 1037 warnings and seven failures in 222.62 seconds.
The seven failure IDs exactly match the independent baseline. This is not a
whole-suite success claim.

Evidence is retained under ftqc-reference-env/g2-controlled-tensor-* (source
hashes, focused coverage, related regression, integration, original tests,
resource JSON, quality checks and default baseline comparison). G2 remains open;
the next source audit covers tensor index binding into the existing
ContractionGraph. No CI wait or automatic integration-PR merge is implied.
