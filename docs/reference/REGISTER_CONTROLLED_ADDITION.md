# Controlled register addition and shared logical And

`compiler/registers/controlled_addition.py` ports `CAdd` from Qualtran v0.7.0,
commit `bcb17bb25360ed68ce3bc084645db3e929799389`. The exact call chain is
`And(cv,1)` per input bit, the existing `Add`, and `And(cv,1).adjoint()` in
reverse input order. Google copyright and Apache-2.0 notices are preserved.

`and_compute.py` owns the four-T And network and its measurement-assisted
uncomputation, for all four control-polarity pairs. These gates were extracted
from the existing adder. Both `arithmetic.add_ir` and the new controlled adder
consume this one implementation; the old adder's gates and metadata remain
unchanged. Erasure retains the source H, measurement, conditional CZ and reset
sequence rather than substituting another clear operation.

## Concrete register interface

`controlled_add_ir(control, left, right, loaded, carries, load_bits, carry_bits,
control_value=1)` adds zero-extended a to b when control equals the selected
zero/one value. Registers use big-endian bit patterns, with input width m no
larger than target width n. Supply m clean loaded wires and n-1 clean carry
wires, plus one distinct nonnegative classical measurement slot per work wire.
Quantum operands are disjoint. Control and a are preserved, and all work ends
in zero, including for a coherent superposition of the control values.

Equal-width bit patterns also implement signed modular addition. Unequal-width
signed extension is not provided by this interface. Its source limitation is
recorded below. Symbolic and general multi-control construction remain pending.

Run without reference dependencies:

```sh
python -m examples.algorithms.register_controlled_addition
```

The example uses `circuit.append(ControlledAddition(cv))` for both control
values. A relative imaginary phase separates the coherent control branches;
the active branch produces `2+3=5`, the inactive branch retains 2, and every
sampled trajectory preserves the full complex state and clears four work wires.

## Measured counts versus model

| m -> n, control value | Measured CNOT | Measured X | Measured T+TDG | Model And count | Model T at 4 T/And | Measured / model measurements | Clean quantum work |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2 -> 3, 1 | 32 | 0 | 16 | 4 | 16 | 4 / 4 | 4 |
| 3 -> 8, 1 | 86 | 0 | 40 | 10 | 40 | 10 / 10 | 10 |
| 8 -> 8, 1 | 129 | 0 | 60 | 15 | 60 | 15 / 15 | 15 |
| 8 -> 8, 0 | 129 | 32 | 60 | 15 | 60 | 15 / 15 | 15 |

The executable gate set is X/H/S, CNOT, T/TDG, conditional CZ, measurement and
reset. Counts match the unmodified source's complete primitive gate stream,
operands, per-wire dependencies and classical feedback. Disjoint gates may be
reordered by the source DAG exporter; no cancellations or gate combinations
are introduced in this comparison. The zero-control path retains each source
And's own pre/post X sequence rather than globally moving or cancelling flips.

Independent `QECGatesCost` T values use explicit `total_t_count()` conversion.
Its raw `t` field excludes And lowering, and its raw Clifford field is not a
complete primitive count. For example the model reports 54 Clifford units for
both eight-bit polarities, while the executable zero-control circuit contains
32 additional X gates. These raw estimates are retained separately; they are
not described as primitive Clifford agreement. The direct executable comparison
includes these extra X gates. Classical slots number m+n-1 in this binding and
are distinct even though some measurements occur before other work is erased.

## Signed source boundary

Pinned `CAdd(QInt(1), QInt(2))` has inconsistent classical and executable actions:
for control=1, a=-1, b=0, its classical method returns b=-1 (pattern 3), while its
executable decomposition zero-extends a's one-bit pattern and returns pattern 1.
The native bit-pattern entry reproduces the executable decomposition, with a
regression for the discrepancy; it does not claim to implement that signed
classical action. Equal-width signed cases are separately compared successfully.
No patch to upstream code is used to obtain gate equality.

## Local acceptance

58 new tests include CAdd source comparisons at widths (1,1), (2,2), (3,3),
(5,5), (8,8), (1,3) and (3,8), both control values and all three upstream integer
dtypes. Executable parity for the narrow signed cases is kept distinct from
mathematical signed-action acceptance. Four standalone And polarity cases check
literal compute/erase source gates, complete complex compute amplitudes and
both measured erase outcomes. Native CAdd tests cover every input/control
amplitude with an entangled spectator, work cleanup and inverse-by-controlled
subtraction. All measurement outcomes in the small cases appear in the 512
sampled trajectories; finite samples do not establish exact probabilities.
Invalid work layouts, aliases, polarities and classical slots are rejected.

All 444 combined regression tests pass. The And and controlled-addition modules
have 100% statement coverage. Strict mypy passes on 929 source files. Both Black
versions, Ruff, architecture boundaries and the multi-level IR contract pass.
Whole-PR remote CI/GPU and review remain independent acceptance conditions.

## Controlled unsigned constants

`controlled_add_constant_ir(control, target, value, constant, carries,
classical_bits, control_value=1)` ports the separate controlled `AddK` path:
controlled `XorK` load, **uncontrolled** `Always(Add)`, controlled `XorK` unload.
It does not replace the middle Add with CAdd. `arithmetic.py` owns a single
internal load/add/unload implementation consumed by both plain and controlled
constant addition. The existing `add_constant_ir` signature and emitted sequence
remain unchanged.

For an n-bit target, supply n clean constant wires, n-1 clean carries and n-1
classical slots, all explicit and disjoint as required by the plain entry.
Only loading and unloading use CNOTs from the control. The control-on-zero source
wraps the whole operation in two X gates on the control; those are retained.
Negative constants are normalized modulo `2**n`; nonnegative constants must fit.
The supported interface is concrete and unsigned. Other dtypes, symbolic values
and multiple controls are not counted as completed.

The runnable example above also uses
`circuit.append(ControlledConstantAddition(cv))`. With three-bit target 2 and
constant -3, its active branch becomes 7 while the inactive branch remains 2.
Both control polarities preserve the branches' imaginary relative phase and
clear all five quantum work wires in every sampled trajectory.

| n, k, control value | Measured CNOT | Measured X | Measured T/TDG | Model T at 4 T/And | Measured / model measurements | Clean quantum work |
|---|---:|---:|---:|---:|---:|---:|
| 3, -3, 1 | 25 | 0 | 8 | 8 | 2 / 2 | 5 |
| 8, 83, 1 | 89 | 0 | 28 | 28 | 7 / 7 | 15 |
| 8, 83, 0 | 89 | 2 | 28 | 28 | 7 / 7 | 15 |
| 8, 0, 1 | 81 | 0 | 28 | 28 | 7 / 7 | 15 |

These T and measurement estimates come independently from the pinned controlled
AddK's `QECGatesCost`, with its explicit And-to-T conversion. Raw model fields
are not primitive gate counts. The complete native instruction multiset and
per-wire dependencies, including operands and all feedback metadata, match
unmodified source. Zero constants retain the source's complete carry network;
its functional identity is not used to silently remove T gates or measurements.

65 new cases compare source networks at widths 1, 2, 3, 5 and 8, for zero, one,
maximum and negative constants with both controls. Native widths 1 through 4
check every target/control complex amplitude, an entangled spectator, clean
work, and inverse-by-opposite-constant. Every carry outcome appears in the 64
sampled trajectories used for these small cases; no exact probability claim is
inferred from sampling. Validation rejects invalid controls, aliases, constants
and workspace. Plain AddK and all earlier arithmetic consumers are rechecked.

The combined suite passes 509 tests. Controlled-addition statement coverage is
100%; arithmetic coverage is 97%, including full coverage of the shared constant
body and the new entry. Strict mypy passes on 929 source files. Both Black
versions, Ruff, architecture and multi-level IR checks pass. Whole-PR remote
acceptance remains separate.
