# Component control

This package owns control predicates, multi-control, indexed and controlled component composition.

Core, shared Compiler facilities and the register components used by these
implementations are allowed dependencies. Backend selection and numerical
execution belong to Runtime and Simulation. Public APIs and IR schemas follow
their existing change-proposal process.

Start with [controlled_component.py](controlled_component.py). Preserve register order, feedback slots and caller-owned
workspace. From the repository root, run the focused scenario tests.

```bash
python -m pytest tests/unit/test_register_controlled_component.py -q
```

The [register package guide](../README.md) provides executable consumer examples,
source-parity requirements, resource-counting boundaries and the relocation map.

`multi_controlled_component_program` finds the scalar input/output control by
name, as one THRU declaration or a same-named LEFT/RIGHT pair. This includes
one-qubit opaque ports exposed by `AutoPartition`. The returned
signature puts its QBit array first, with new control before old control; other
ports retain their declared order. Child work and predicate work keep separate
lifetimes, and the predicate owner measurement-erases its workspace. Packing
adds no primitive gates. Exact adjoints of packed callbacks use the existing
adjoint graph owner with explicit inverse Partition pairs and exact child
leaves; the selected child and the callback's packed entry remain distinct.

Run `tests/unit/test_packed_specialized_control.py` for wrapped controls, actual
specialized-adjoint callbacks, all small measurement histories, and the existing
`circuit.append(op)` execution path. Primitive counts and source resource models
are checked separately.

`test_packed_adjoint_control.py` follows the whole existing-control returned
adder through `AdjointWithSpecializedCtrl`, for all four old/new polarities.
The packed current entry and selected helper use the same explicit entry name.
Supply exact inverse pairs for its And compute/erase functions, Partition
functions and child leaves to the existing adjoint owner. The original callback
returns the existing control before the data. With the three-qubit packed data
port tested here, a nonleading control misassigns that returned bit to the data
port while finalizing the source adjoint graph. Native rejects the same
connection. `test_equal_width_control_callbacks.py` shows that equal-width
nonleading ports silently pass type checks but give a callback equivalent to
the intended inverse after an input SWAP. Its returned root, gate counts and
resource model do not expose the difference. Native reproduces that graph;
these cases are defect evidence, not acceptance of the intended operation.

`test_directional_control_callbacks.py` verifies valid `left_only` composition
with a same-named control input/output pair. Existing-control predicates expose
one THRU control array, while packed data is consumed into separate child right
ports. The exact adjoint reverses this flow using supplied inverse Partition
leaves. Caller data wires are not freed; predicate and child work are erased and
released by their existing owners. A renamed control lacking a same-named output
and nesting a directional wrapper as an AutoPartition child remain invalid in
the source. Those failures are diagnosed separately.
