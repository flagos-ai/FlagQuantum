# Static classical bindings for native quantum functions

The user authorized the next native completion batch after #1104: finite real
scalars, one-dimensional real arrays, parameterized writable helpers and the
original QML guide with two nonzero bindings. This additive unreleased successor
is recorded in [the candidate contract](../../contracts/synthesis-native-classical-candidate.json).
Historical contracts, source bytes, licenses and resource baselines remain fixed.
Implementation is authorized; release authorization remains false.

## Semantics

The existing public `synthesize` accepts one additional optional keyword,
`classical_inputs=None`, forwarding to the existing bound-source workflow. It
applies to typed Python qfunc and Qmod sources; pure integer-expression and phase
convenience functions refuse it. No root fq exports or run/read signatures change.
`float` annotates a synthesis-time scalar and `list[float]` a synthesis-time real
array. These are separate from quantum QNum/QArray storage and use no quantum bits.
The root still requires one or more Output quantum ports; bindings must exactly
match the classical parameter names. The standard annotation objects are checked
by identity, including quoted annotations and imported aliases.

Accept only built-in int/float numbers, excluding bool, in the finite ±2**53
range. Array bindings are built-in lists or tuples with 1..4096 elements. Total
bound scalar/array elements cannot exceed 4096. Values are copied to immutable
compiler data for each synthesis; rebinding the same qfunc creates a separate
program. No mutable global binding table, tensor differentiation or runtime
rebinding is introduced. Qmod retains its existing admitted type/shape checks.

Gate angles and helper actuals accept finite literals, bound scalars, in-bounds
array elements, unary signs and +, -, *, /, with expression depth at most 32.
Division by zero refuses. Helpers may receive whole bound arrays or bounded
literal arrays. `len(array)` supplies a static loop bound under the existing
loop/expansion budgets; real values cannot become allocation widths or indices.
Integer-index syntax and quantum array views retain their existing restrictions.
No host function evaluation, dynamic attributes, negative indices, empty/nested
arrays, array slices, defaults, keyword actuals or arbitrary callbacks are added.
`RX(angle, q)` is a captured marker for the existing checked qreg rx gate, using
exp(-i angle X/2). It retains absolute phase and has no host execution behavior.

Classical arguments may be interleaved with plain writable quantum helper ports.
Helpers remain gate/loop-only with disjoint borrowed quantum locations. Existing
Const-only arithmetic helpers retain their separate profile; parameterized
Const/Output helpers, allocation, Output escape, adjoint and coherent control of
parameterized mutable helpers refuse. Helper scopes restore bindings, source,
views and loop variables on return, including failures. Classical arguments may
not shadow gates, helper callables, len or range. The register verifier and
lowering remain authoritative for initialization, aliasing, lifetimes and budgets.
No new IR, gate compiler, executor or vendor adapter is added.

## User workflow and acceptance

Run the [native QML example](../../examples/synthesis/tutorials/basics/qml/main.py)
and its [native/Circuit comparison](../../examples/synthesis/tutorials/basics/qml/comparisons/circuit.py).
The untouched Qmod and native qfunc must agree with an independent single
RX(input_0 + weight_0) Circuit for (0.37, -0.81) and (1.2, 6.1), comparing every
complex amplitude at absolute tolerance 1e-12 without aligning global phase.
The second sum exceeds 2*pi to expose accidental phase-dropping angle reduction.
Nested real-array helpers additionally verify noncommuting gates, asymmetric
bit order, caller/callee scope restoration and zero workspace.

Reject missing/extra bindings, bool/nonfinite/nested/oversized arrays, bounds,
shadowed callables, overlap, Output escape and budget violations before execution.
Run the existing native/compatibility suites and replay all 216 approved corpus
cases against their recorded resource counts, separately retaining the six
unchanged-source semantic refusals. This batch adds one native counterpart;
the 70 bound cases identified in the previous dependency audit are candidates,
not a claim that their other missing native operations are implemented.
