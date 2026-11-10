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

`multi_controlled_component_program` finds the existing scalar THRU control by
name, including a one-qubit opaque port exposed by `AutoPartition`. The returned
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
connection. Equal-width nonleading ports remain unaudited. The source failure
is diagnosed separately, without reordering ports or claiming execution parity.
