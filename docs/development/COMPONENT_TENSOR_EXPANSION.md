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
simulator. Source root wrapping and general provider dispatch remain open.

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
root wrapping, lazy foreign decomposition failures and open-system traversal
remain separate obligations. No CI wait or automatic merge is implied.
