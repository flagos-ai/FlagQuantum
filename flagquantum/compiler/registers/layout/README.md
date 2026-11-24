# Register layout and workspace

This package owns partition, cast and bit views, clean and borrowed workspace ownership.

`wire_allocation.WireAllocation` grows one physical ID space for incremental
construction. It reuses `RegisterWorkspace` leases for temporary storage and
reserves permanent named-register IDs independently. Freeing individual wires
retains source LIFO reuse. Formats owns register metadata and operation columns.
Allocation emits no reset, and callers must restore wires before freeing them.
Run `tests/unit/test_physical_builder_binding.py` for complete source lifecycle
transactions, partial failures, copied builders and normal append replacement.

`component_cast.cast_word_function(name, source, target, shape=())` defines a
typed LEFT-to-RIGHT quantum view using existing Program split/join operations.
Array elements are row-major; each word retains MSB-first physical wire order.
It consumes input handles, creates typed output handles, and emits no primitive
gates or workspace. Core executable dtype and shape admission remains authoritative.
Use the existing bookkeeping control callback for unconditional control passthrough.
Specify a reverse cast with swapped types and the same shape: pinned Qualtran
v0.7.0 `Cast.adjoint()` drops shape, so its shaped failure is recorded separately
from explicit reverse-call parity. Classical conversion and signed execution
are outside this finite quantum slice.

Mixed directional compositions preserve explicit input/output ownership. The
existing graph scheduler may free an independent input before allocation and
reuse that wire; declaration order does not impose a dependency. Free performs
no reset: callers must establish clean zero or borrowed joint-state restoration.
Run `python -m pytest tests/unit/test_component_cast.py -q` with the locked
reference environment to check callbacks, wire order, full complex states and
the independent source resource model. Empty logical arrays are admitted;
physical execution still requires at least one physical wire.

Core, shared Compiler facilities and the register components used by these
implementations are allowed dependencies. Backend selection and numerical
execution belong to Runtime and Simulation. Public APIs and IR schemas follow
their existing change-proposal process.

Start with [partition.py](partition.py). Preserve register order, feedback slots and caller-owned
workspace. From the repository root, run the focused scenario tests.

```bash
python -m pytest tests/unit/test_register_signature.py -q
```

The [register package guide](../README.md) provides executable consumer examples,
source-parity requirements, resource-counting boundaries and the relocation map.

`component_word_views.gather_word_binding` assembles requested views of existing
owned words. It reuses an exact word, splits intersecting words when necessary,
casts single-bit views and rejoins only the requested positions. Allocation stays
with the caller. Scaled phase addition uses this source-ordered path, including
repeated whole-word access and overlapping shorter slices. Run the phase-gradient
and variable-rotation decomposition tests to verify view graphs and execution.

Stored AutoPartition accepts validated typed child declarations as well as
primitive and call bodies. Symbolic partitions retain declarations, and exact packed adjoints
reverse the original partition sequence. Run `test_component_auto_partition.py`
and `test_packed_prepare_component.py` when changing that boundary.

Raw construction results share the physical ID cursor with explicit classical
allocations and retain a separate reuse bank. The existing dirty workspace bank
owns these helper slots. `KmxCircuitBuilder.result_bit` provides deterministic
release, optional clean reuse and explicit ownership transfer. `measure_result`
adds HMR and optional condition ownership. Formats emits stores and condition
records, while layout owns allocation alone. Run
`tests/unit/test_physical_builder_results.py` and
`tests/unit/test_kmx_result_lifetimes.py` for the locked lifecycle and all small
complex measurement branches through the construction replacement.

`packed_oracle_component.py` owns the common PREPARE/SELECT AutoPartition
wrapper and exact partition adjoints. Callers provide a Core Signature, explicit
ordered register groups and complete directional child Programs. Logical junk
and packed words remain caller-owned; packing adds no primitive gates or work.
The existing Algorithms packed PREPARE entry delegates to this owner. Run
`test_packed_prepare_component.py` and `test_legacy_packed_oracle_factories.py`
for source calls, constructor replacement and execution evidence.
