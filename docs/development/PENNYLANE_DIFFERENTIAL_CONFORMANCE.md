# PennyLane Differential Conformance

FlagQuantum certifies its static PennyLane boundary with deterministic numerical
comparisons rather than conversion success alone. Run the focused suite with a
supported PennyLane installation:

```bash
python -m pytest tests/test_pennylane_interop.py \
  tests/test_pennylane_interop_conformance.py -q
```

`run_pennylane_conformance()` retains two readable golden cases and adds six
fixed-seed programs over three to five wires. Every seeded program initializes
all wires, includes one-, two-, and three-wire gates, and varies parameter values
and wire order. A failing case reports its seed in the case name so the exact
program can be reproduced.

Each seeded case compares three paths against an independently constructed
PennyLane `QuantumScript`:

1. native FlagQuantum execution;
2. FlagQuantum IR exported to PennyLane; and
3. the independent PennyLane program imported into FlagQuantum.

The independent script is built directly from the generated operation record;
it does not call the adapter to construct the reference program. The maximum
absolute statevector error across all three comparisons determines whether the
case passes. The same programs also run through the framework-neutral adapter
round-trip contract, alongside the fail-closed unsupported-operation case.

CI executes this suite against PennyLane 0.44.1 and 0.45.1. This evidence covers
static, bound-parameter `QuantumScript` semantics at complex128 precision. It
does not certify QNodes, differentiation, finite shots, measurements, devices,
provider hardware, or arbitrary wire-label preservation.
