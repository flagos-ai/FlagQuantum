# Register layout and workspace

This package owns partition, cast and bit views, clean and borrowed workspace ownership.

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
