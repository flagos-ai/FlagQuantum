# Component expansion for tensor contraction (GRAPH58)

`compiler/registers/component_tensor_expansion.py` translates the selection
policy of Qualtran v0.7.0 `flatten_for_tensor_contraction` onto the existing
`expand_component_calls` owner. Source is locked at
bcb17bb25360ed68ce3bc084645db3e929799389. Program transformation stays in
Compiler; tensor binding and numerical contraction stay in Simulation. There
is no new graph record, tensor execution engine, provider registry or public API.

## Interface and precedence

`expand_components_for_tensor(record, has_custom_tensor, entry="main",
full_flatten=True)` takes an already-composite typed entry. The existing tensor
provider identifies declared custom tensor callees by name using the supplied
classifier. This explicit declaration replaces the source's Python
`my_tensors.__qualname__` heuristic at the native boundary; no foreign objects or
Python method names enter Program IR. This classifier does not execute a tensor
supplier or inspect numerical values.

The returned labels preserve occurrence identity and may differ from operation
positions after expansion. Pass traversal positions from `iter_component_calls`
to Simulation, not these IDs. See the expanded provider consumer in
[component tensor providers](COMPONENT_TENSOR_PROVIDERS.md#expanded-graph-visitation-graph82).

Full expansion ignores custom declarations, does not query the classifier and
expands every available typed-call decomposition. Partial expansion queries the
classifier in existing traversal order and retains calls with custom tensors.
Atomic or mixed bodies remain leaves under the existing expansion owner's rules.
A retained composite's internal definitions remain available; retained call
identity is not equivalent to deleting its decomposition. Entry outputs,
register directions, instance labels, private-work ownership and depth bounds
are handled by the existing transformer. Immutable definitions stay shared.

Classifier exceptions propagate. The depth limit includes the final no-change
probe and propagates the existing ValueError on exhaustion. Primitive bodies
without a typed-call decomposition remain atomic; this does not emulate lazy
foreign decomposition callbacks or their exception categories. This interface
does not wrap an arbitrary atomic root, infer a tensor provider from reflection,
flatten primitive instructions, contract tensors or silently fall back to a
simulator. At GRAPH58, source root wrapping and general provider dispatch
remained open; GRAPH78 below adds explicit typed-root wrapping.

## Real consumers

Tests build a source component with both custom identity tensors and a real
H-H decomposition, then compose one to three layers with one or two calls per
layer. Both choices are tested against the unchanged original
`flatten_for_tensor_contraction`. Actual retained callee order, instance IDs,
shared native definitions and complete dense operators match. Selected native
leaf tensors pass through the existing register tensor binder and integer-label
greedy contraction owner. Original custom tensors and decompositions agree;
this fixture does not claim that arbitrary inconsistent providers are equivalent.

The same graphs retain the existing `circuit.append(op)` workflow. Primitive H
counts and the independent source Clifford model match; an entangled spectator
is preserved and caller clean storage remains zero. Declaring a custom tensor
changes the contraction boundary, not the semantics or gate lowering of a call.

A second consumer declares exact Add/adjoint tensors as stopping boundaries in
partial mode and expands their typed wrappers in full mode. This declaration is
explicit native provider metadata; the source Add is not claimed to have a
custom `my_tensors` override. The unchanged source arithmetic decompositions are
used as the gate oracle for both modes. Primitive operands, parameters and
feedback match before and after selection. Each mode covers sixteen forward
measurement histories and all 256 forward/inverse history pairs, including
complete complex amplitudes, the entangled spectator and two clean work wires.
Across two modes this totals 32 forward branches and 512 history pairs.
Eight forward and eight forward/inverse DynamicCircuit trajectories also pass.

Per direction this measured consumer emits sixteen T/Tdag and four
measurements/resets with two reused work wires. Independent source QEC models
report sixteen T-equivalents and four measurements, separately from primitive
Clifford counts. The source peak is eight, including the explicit spectator.
Selection itself emits no gate and defines no resource model.

Run the focused source-backed checks from the repository root:

```bash
python -m pytest tests/unit/test_component_tensor_expansion.py -q
```

The finite suite also checks source depth failures, atomic retention, input
immutability and classifier errors. Complete foreign lazy-decomposition error
translation, open-system tensor selection and whole parent-file acceptance
remain outside this slice.

## Verification evidence

Implementation: 7956ca55. All 25 new tests pass. Related regression passes
819 tests in 8.62 seconds, retaining two existing warnings. Focused coverage
passes 132 tests in 7.45 seconds; the four selected tensor/selection owners
cover all 119 statements and 48 branches. The selection function itself has
five statements and no separately instrumented branch; behavioral tests verify
both modes rather than using that zero branch count as semantic evidence.
All 64 local integration tests pass in 4.09 seconds. Strict mypy checks eight
files; Black, Ruff, Compiler scope, architecture, Program and protected API
checks pass. Six unchanged upstream flatten/dense tests pass in 1.04 seconds;
eight original source Git blobs match the source inventory.

Default smoke/unit stops at maxfail with 21400 passed, 300 skipped, 1037 warnings
and five failures in 127.27 seconds. All five IDs belong to the independent
seven-failure baseline; the remaining two were not reached. This is not a
whole-suite success claim. Evidence and four forward/inverse resource scenarios
are retained at ftqc-reference-env/g2-tensor-expansion-*.

G2 remains open. Next is source direct-through unused-wire identity handling,
reusing existing nodes and register-axis binding. Automatic provider execution,
At that batch, root wrapping, lazy foreign decomposition failures and open-system
traversal remained separate obligations; subsequent evidence records each slice. No CI wait or automatic merge is implied.


## Explicit root entry (GRAPH78)

`expand_component_root_for_tensor(record, has_custom_tensor, entry="main",
name="tensor_root", full_flatten=True, max_depth=1000)` translates the original
`Bloq.as_composite_bloq()` single-call wrapper followed by the existing flatten
policy. It belongs to Compiler and reuses `ComponentProgramBuilder` and
`expand_components_for_tensor`. It creates one typed call to the selected root,
retains the root signature, then expands that call according to the existing
policy. No new Core record, numerical supplier, registry or public API is added.

Callers explicitly distinguish a component root from an already-composite entry.
The latter continues to use `expand_components_for_tensor`; it corresponds to
`CompositeBloq.as_composite_bloq()` returning itself. No body-length or operation
name heuristic decides which path to take. A partial-mode classifier can retain
the root's own custom tensor at occurrence zero. Full mode never queries that
classifier. Atomic primitive bodies remain typed leaf calls, even in full mode.
The inserted call counts toward the existing expansion depth budget. Original
definitions remain shared; a transformed record invalidates canonical_payload.

Inputs, outputs and ownership use the existing builder: LEFT inputs are consumed,
RIGHT outputs are defined, THRU successors are returned, arrays keep row-major
element order and words retain MSB-first bit order. Quantum and classical ports,
zero-width words and empty layouts keep their declared metadata. The wrapper
adds zero physical instructions and zero work bits. The selected child still
owns measurement, cleanup, classical initialization and actual resource costs.
Invalid modules, unknown roots, untyped/physical entries and wrapper-name
collisions are refused; classifier exceptions preserve their original identity.
No lazy foreign decomposition exception translation is claimed.

### Reference and execution acceptance

Forty-one new cases compare the locked, unchanged root-wrapping/flatten chain.
Custom H-H identity roots exercise both retaining the root and full expansion;
atomic H and phaseful S roots retain the source occurrence IDs. Complete tensors,
relative-phase derivatives, entangled spectator amplitudes and unused clean-bit
zero are checked through existing tensor binders and `circuit.append(op)`.
Sixteen shaped/zero-width phaseful graphs compare every source connection and
full dense operator, including scalar, (2,), (1,2) and (0,) idle layouts.

Mixed quantum/classical roots with retained or discarded measurement results
compare the complete source superoperator in complex64 and complex128. The
numerical leaf implementation is replaced by the unchanged source tensors while
keeping graph, consumer and bindings fixed. An atomic MeasureZ root also feeds
the same superoperator consumer. All two-outcome atomic and four-outcome mixed
measurement histories are checked as full complex amplitudes, including the
source X-measurement basis frame, both classical idle inputs and unused clean
storage. Across the two modes and configurations there are 36 measured branch
executions. Consumed measured inputs are not claimed zero or reset.

Actual bound instructions, operand order, measurement slots and feedback are
identical before and after wrapping/selection. For the phaseful shaped example,
actual H3/S2 matches source Clifford5/T0. Atomic MeasureZ emits one measurement,
matching its source model. Mixed MeasureZ/MeasureX emits H1/measurement2 and one
classical entry XOR; the source QEC model reports measurement2/Clifford0/T0.
These categories remain separate: the source measurement abstraction does not
charge that basis H, and external classical initialization is a native entry
cost. Source QubitCount is 3 for the phase example, 1 for MeasureZ and 4 for the
mixed graph, whose count includes a classical port. It is not interchangeable
with caller storage including spectator and unused clean bits. Wrapping has no
independent tensor-storage allocation or resource model.

The existing user workflow remains:

```bash
python -m examples.algorithms.binary_field_arithmetic
python -m pytest tests/unit/test_tensor_root_entry.py -q
```

This is a finite `single_device_fast_path` slice. Automatic provider discovery,
lazy foreign decomposition error translation, empty numerical graph scalar
semantics and full parent-file acceptance remain open. G2 continues; G3 is not
entered. Evidence is retained under `ftqc-reference-env/g2-tensor-root-*`.

### GRAPH78 verification evidence

Implementation: `bf2ee8d6`. All 41 new source-backed cases pass. Final related
regression: 1413 passed in 15.75 s (41 existing/source warnings). Focused
verification: 688 passed in 20.51 s (39 warnings). All 64 local integration
tests pass in 3.20 s. The changed expansion owner covers all 22 statements and
six branches; the 16 selected owners cover 511 statements and 206 branches at
99%, retaining two historical GRAPH60 profile-refusal misses. Strict mypy passes
39 related files plus three source-following files. Black/Ruff, Compiler scope,
architecture, Program IR and public API checks pass.

Thirty locked source blobs, including Bloq.as_composite_bloq and CompositeBloq,
match the unchanged inventory. Twenty-five unchanged source tests pass in
1.05 s, with five existing/source warnings. Qualtran 0.7.0 and native Kickmix
0.1 Simulator were checked; the existing binary-field append example passes.
Eight full/partial resource scenarios preserve exact physical lowering and
separately record the independent source costs.

Default smoke/unit: seven failed, 21507 passed, 300 skipped and 1037 warnings in
122.95 s. All seven failure IDs exactly match the independent baseline; no new
failure ID was observed. The maxfail run is not a green default suite. It ran
before the final shaped dense-operator assertions were added, without any later
production behavior change. Raw evidence is at `ftqc-reference-env/g2-tensor-root-*`.
