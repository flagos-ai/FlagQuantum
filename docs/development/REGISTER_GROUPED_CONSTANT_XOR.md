# Grouped constant XOR

The compiler owns `grouped_controlled_xor_constant_ir` in
`compiler/registers/grouped_constant_xor.py`. It binds existing `ControlRegister`
word descriptors to the shared `controlled_component_ir` predicate schedule and
existing positive `controlled_xor_constant_ir`. It introduces no new control
representation, execution engine or resource registry.

Control groups and the target use MSB-first wire order. Words within a group
retain descriptor order. The constant is an unsigned stored target bit pattern;
encoding a signed constant belongs to the caller. Original controls are logical
inputs/outputs. For m nonempty control bits, the caller owns max(0,m-1) initially
zero predicate wires and equally many distinct classical slots. Target, control
and work wires must be globally distinct. Predicate work returns to zero;
classical slots retain the erasure outcomes. A single control uses no work or
measurement and preserves its declared polarity. XOR is self-adjoint; applying
the same operation twice restores any target.

The exact source chain is Qualtran v0.7.0, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`:
`XorK.controlled` -> `ControlledViaAnd` -> `CtrlSpecAnd` ->
positive controlled XorK -> `CtrlSpecAnd.adjoint`. The implementation reuses the
existing FlagQuantum owners for these steps. Source copyright and license are
retained. Reference files are unmodified.

## Validation

G2-BIT03 has 124 tests. Ninety configurations compare the original primitive
sequence, operands, dependencies and feedback: ten shaped/integer/signed control
specifications, target widths 1/2/5 and zero/full/alternating constants. Every
configuration compares all seven QEC categories and peak qubits in both source
profiles. Coherent tests cover every control/target/spectator input simultaneously
with complex amplitudes, every measurement history for one to three controls,
all constants for one/two-bit targets, clean work and double application. Invalid
wire ownership, work, slots and constants fail closed.

The concentrated bitwise/AddK/control regression passes 1,046 tests without skips.
Six existing LCU reference tensor warnings remain visible; no new numerical LCU
claim is made. Strict production/example typing, Black, Ruff, scope and
architecture checks pass.

Run:

```sh
python -m examples.algorithms.grouped_constant_xor
```

The user writes `circuit.append(GroupedConstantXor(3))`. The example preserves
control word 2, maps target 1 to 2, and clears its predicate wire in all 16 sampled
trajectories; exhaustive branch evidence comes from the tests, not sampling.

| Quantity | Emitted primitive instructions | Original logical model |
|---|---:|---:|
| T + TDG | 4 | 1 And, converted to 4 T |
| Measurement | 1 | 1 |
| Total live qubits | 5 | 5 |
| Predicate work | 1 | Included in live-qubit peak |
| Clifford categories | 4 X, 3 H, 8 CX, 1 S, 1 conditional CZ | 3 Clifford plus the separate And category |
| Reset | 1 | Included in measured erasure semantics |

Primitive Clifford instruction counts are not the abstract model's Clifford
category: the latter leaves And as its own category. No physical count is
replaced by an estimate. Symbolic target widths/constants and arbitrary controlled
register-to-register XOR are outside this concrete slice; the parent file remains
open in the migration ledger.
