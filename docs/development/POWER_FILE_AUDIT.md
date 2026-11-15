# Register Power source family

The unchanged reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Both source files retain the
Google LLC 2023 Apache-2.0 attribution.

| Inventory | Frozen Git blob | Disposition |
|---|---|---|
| Q-0248 `power.py` | `5331c3afd2ce0dffba17c4adb92659d15b76311b` | Concrete owned repetition, numeric and modeled repetition, signed exponent action, and label suffixing verified. General source wrapper protocol remains open. |
| Q-0249 `power_test.py` | `7c26011bd2f6d0a0ca557a343dd3cb8f116c2a31` | Closed. The unchanged four tests and the corresponding native cases passed, including fresh-process imports. |

## Source methods and native boundaries

Power retains all THRU registers of the child. The constructor checks port
direction before validating the positive repetition count. The source numeric
adapter retains the locked constructor and decomposition distinction for
symbolic and NumPy integer counts. The native typed Program entry requires a
concrete Python integer and caps the number of generated calls at the shared
Power limit. It threads the complete output binding map from each call into
the next. Register order, row-major shaped elements and intra-word bit order
hold across every iteration. Child definitions own any work or feedback.

`register_power_ir` handles already-bound primitive streams. Coherent
adjoints use the core inversion rules, while measured bodies supply their
explicit inverse. Signed nested powers multiply absolute repeat counts and
select the underlying adjoint according to the sign of the product. Native
signed emission matches that source behavior without merging the rotations in
the tested stream. `PowerCall` preserves symbolic weights and counts repeated
child calls independently of physical emission. Locked symbolic peak-width
queries still have the previously documented source refusal.

`power_wire_symbols` applies the source `^power` suffix in wire order after
the display owner has resolved the child's labels. The source wrapper's
foreign diagram callback and automatic fallback label discovery remain open.
The native paths also lack a general foreign `GateWithRegisters` wrapper and
its overloaded `__pow__` object semantics. These boundaries keep Q-0248 open.

## Original test coverage

| Original test | Native counterpart |
|---|---|
| `test_power` | Direction rejection precedes count validation. The ten-call entry retains the scalar, 2-by-2 array of two-bit words, and three-bit pass-through port. Each call consumes the previous SSA results. The physical stream retains the source fixture's X and four CX actions in each round. |
| `test_power_of_power` | Positive powers, double negatives and a single negative flatten to the exact locked Rz leaf stream. The full complex operator and modeled rotations agree. |
| `test_power_circuit_diagram` | The one-wire atom and all twelve shaped wire labels agree with the original Cirq diagram metadata after suffixing. The unchanged suite also checks the complete Cirq diagrams. |
| `test_no_circular_import` | Both the locked child process and a fresh interpreter importing the native compiler, simulation and drawer owners succeed. |

All four unchanged source tests, 193 concentrated native cases and 109
integration cases passed. Strict typing, formatting, Compiler scope,
architecture, multilevel Program IR and public API checks passed.

## Execution and cost evidence

The shaped fixture binds ten X and forty CX instructions, with no extra
work wires or measurements. Its synthetic source test atoms declare their own
test costs, independently of that physical equivalent. The signed Rz cases
each emit six rotations and the source model counts six rotations. The
previously verified measured example still emits 16 T/Tdg and four
measurements for two rounds, reusing one work wire. Its modeled peak is four
logical wires and the five-wire circuit includes a spectator. Existing
measured regression covers every tested history, complex amplitudes and work
cleanup through a full inverse.

```bash
python -m examples.algorithms.classical_power_values
```
