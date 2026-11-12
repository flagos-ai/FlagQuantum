# Classical vectors around a typed component

Compiler owns the graph transformation corresponding to pinned Qualtran v0.7.0
`xcheck_classical_quimb.py`: prepare input integer kets, call the original
component once, determine output values if necessary, and attach integer effects.
The source is unchanged at `bcb17bb25360ed68ce3bc084645db3e929799389`.

## Existing contracts and ownership

`component_basis_vectors.integer_basis_function` represents a concrete ket or
effect with existing Program operations. A ket allocates a clean owned word,
splits it, uses the existing `int_state_ir` loader's ordered X gates, and joins
it. An effect performs the same X gates before a clean release. It uses the
existing executable register admission; there is no new dtype, opcode, serialized
schema or public export. Widths are positive and constants are built-in integers.

`ComponentProgramBuilder.define` applies its constructor's existing duplicate
name rule to definitions added before a call. Port checking remains at call time
and full verification at finalization. This allows the wrapper to follow source
construction order in a single pass, without rebuilding a graph after output
inference or mutating private builder state. It is an internal builder method,
not an extension of the protected public API.

`component_classical_vectors.flank_component_with_classical_vectors` returns an
ordinary `ProgramRecord` with a scalar wrapper entry. Original function objects
and physical metadata are retained; `canonical_payload` is invalidated. Generated
names avoid original definitions and the wrapper name. Input/output arrays are
read in signature and row-major order using source `asarray` indexing. Bool and
NumPy integer indices become built-in integers; no signed, field or fixed-point
value codec is substituted for an unsigned basis index.

When outputs are omitted, an explicit caller-owned evaluator receives the exact
original input mapping once, after input preparations and the original child
call have been constructed. Explicit outputs, including `{}`, bypass evaluation.
The evaluator may compose the existing native classical state. Its exceptions
propagate unchanged. Compiler imports no Simulation implementation; numerical
execution and RNG policies remain outside graph construction.

Simulation consumes the returned graph through existing tensor suppliers,
selection, label binding and contraction. The previous batch's integer ket/effect
supplier provides the vector data. Tests replace the Compiler vector factory to
observe its exact leaf definitions and supply them to that existing numerical
boundary; no new graph interpreter, provider registry or foreign method lookup
is introduced. The child supplier can be replaced by a differentiable tensor
without changing graph construction.

## Boundaries

Only verified typed logical quantum entries are admitted. Classical ports,
physical records, missing entries and conflicting wrapper names fail explicitly.
The existing loader refuses zero-width words and floating/symbolic constants;
these native refusals are not presented as the original diagnostic strings.
Empty shaped ports create no vector calls. Their native declared shape remains
valid, while the original builder can fail inferring empty-array final types.
Automatic foreign decomposition discovery and source-wide dtype admission remain
outside this slice.

Tensor effects are exact, unnormalized projections, including mismatched output
values. Physical binding represents an effect as uncompute followed by clean
release: it requires an explicit caller-owned pool and a valid cleanliness
promise, and does not implement postselection, trace or reset. A mismatched
physical effect must not be advertised as a deterministic projection circuit.
Normal users still append the bound existing operation with `circuit.append(op)`.

## Verification and resource accounts

Every 1–4-bit input is checked against the original IntState decomposition,
ordered physical preparation wires and complete vector. Preparation followed by
its matching effect restores the zero pool. Shapes scalar/vector/matrix, exact
input mapping identity, once-only inference, explicit output bypass, failure
order, name collisions and unchanged original function identities are covered.

Six shaped H/S graph cases compare original flank call order and full complex
scalar contractions. Differentiable child phase replacement preserves autograd.
The native classical evaluator is exercised through the inference callback.
For the existing two-add component, all 64 input combinations and all 16
measurement histories close to the original complex zero-pool amplitude divided
by four. The existing original-source Add gate/feedback checks remain in the
regression suite. Source and native empty/zero-width boundaries stay explicit.

The wrapper itself emits no primitives. For integer preparation, actual physical
X count is `popcount(value)`; the native matching physical effect uses the same
number of X gates before release. The independent source model charges **one
arbitrary Clifford per integer vector**, rather than popcount. The closed Add
example adds six such source model Cliffords (three states, three effects), while
actual extra X gates are the popcounts of the six configured words. Source
IntEffect tensor decomposition is compared numerically; no nonexistent upstream
physical effect circuit is used as a primitive-stream reference. No T or
measurement is added by these vector wrappers. Existing Add's measurements,
feedback and work demand remain separate from those costs.

G2 remains open. The source module's unrestricted symbolic/foreign dtype and
error interfaces are not certified, and this batch does not start G3.

Local verification: 31 new cases; 4137 concentrated cases plus two appended original-X scenarios and 84 CPU/TN tests passed. Strict typing of five files, Black, Ruff, Compiler scope, architecture, internal IR and API checks passed. Integer basis construction (26 statements/8 branches) and flank construction (72/30) each have 100% coverage. Twelve source blobs matched the inventory.

Bounded default smoke/unit: `7 failed, 22385 passed, 300 skipped, 1048 warnings in 210.84s (0:03:30)`. All observed failures are in the seven-case baseline; this is not a green full suite. Evidence prefix: `g2-classical-vector-flank-`.


## Original source examples through native graphs

The unchanged upstream `test_xgate`, `test_cnot`, `test_toffoli` and
`test_add` now run with only their flank factory replaced by the native graph
constructor and existing numerical suppliers. The replacement retains real
executable child bodies. Original Add remains `Add(QUInt(5))`; its complete
1024-by-1024 tensor agrees with the native matrix. X also retains its mismatched
output effect with overlap zero.

Bound physical gates, operands and measurement feedback match the existing
source decomposition. Arbitrary complex logical inputs entangled with a spectator
agree on every forward measurement branch, with all work bits zero. Add has
16 forward branches; all 16 inverse branches restore the input from one verified
forward branch. This is not a claim of explicitly executing all 256 history pairs.
Closed original inputs also restore the pool on every branch and in four
`circuit.append(op)` trajectories per example.

| Example | Actual body primitives | Extra preparation/effect X | Source body model | Source closed model | Peak qubits |
|---|---|---|---|---|---|
| X | X: 1 | 1 | Clifford: 1 | Clifford: 3 | 1 |
| CNOT | CX: 1 | 3 | Clifford: 1 | Clifford: 5 | 2 |
| Toffoli | CCX: 1 | 5 | Toffoli: 1 | Toffoli: 1, Clifford: 6 | 3 |
| Add(5) | H: 12, T: 8, T-adjoint: 8, CX: 45, S: 4, measurement: 4, conditional CZ: 4, reset: 4 | 6 | AND: 4, Clifford: 25, measurement: 4 | AND: 4, Clifford: 29, measurement: 4 | 14 |

Counts are deliberately separate abstraction levels. Atomic CCX is not certified
as a fully lowered zero-T operation. The source model charges one Clifford per
integer vector, irrespective of popcount. The physical effect still requires a
valid clean-release promise.

This batch adds 16 tests: 4155 concentrated, 1398 coverage-run and 84 CPU/TN
checks pass. Strict typing of six files, formatting, scope, architecture, IR and
API checks pass. Six relevant source blobs match the pinned inventory. Upstream
Add tensor contraction emits three autoray matrix-multiplication warnings;
they are retained and the complete finite matrix comparison passes. The default
suite was not rerun for this tests-only batch; its prior baseline above remains.
Evidence prefix: `g2-classical-vector-source-examples-`.

The source notebook test remains unverified, so Q-0899 is not closed. Wider
symbolic/foreign dtype and diagnostic interfaces keep Q-0898 open.

The [factorized addition graph](ADDITION_COMPONENTS.md) now supports the original
notebook's five-/32-bit closed numerical examples with bounded tensor inputs and
intermediates. All 13 original notebook code cells execute in the reference
environment. [Native component drawing](COMPONENT_REGISTER_SCORE.md) now covers the original
closed examples and expanded Add through 32 bits. Automatic provider composition
remains open; the source files are not marked closed on those cases alone.

## Shared scenario checks

`tests/unit/component_vector_checks.py` owns the test-only `tracked_flank`
observation helper and explicit `contract` helper. Classical-vector, Add,
component-score and original-example scenarios import it directly instead of
importing these helpers from a collected test module. The helper imports no
other test modules and contains no collected scenarios. Other scenario-specific
fixtures remain in their existing owners; this is not a claim that every
cross-test dependency has been removed.

The observation helper forwards explicit evaluator, entry and wrapper-name
options to the real Compiler constructor. The numerical helper keeps the
existing basis-vector and register-tensor suppliers and Simulation contraction.
Callback connections and results are typed; no `Any` is needed in either helper.
The Add append scenario now checks that its evaluator inputs are integers before
calling the existing integer action, rather than relying on `Any` to pass typing.
No production program, decomposition, physical instruction, resource model,
source implementation or public API is changed.

Local verification: all 85 affected cases pass without skips. Before/after ordered
collection IDs, parameterization/decorators and all original assertions are
preserved. Exactly one scenario body gains the typed evaluator input check;
this is not a claim that every scenario AST is identical. Strict typing of five
files, Black, Ruff, Compiler scope, architecture, IR and protected API checks pass.
Five relevant pinned source blobs remain unchanged; nine existing reference
numerical warnings remain visible. Evidence prefix: `g2-vector-checks-`.

This refactor adds no acceptance scenario or completed source file. The previous
600-second default-suite timeout remains unresolved and was not repeated for a
test-helper relocation; no full-suite pass is claimed.
