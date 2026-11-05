# Controlled modular arithmetic

The compiler owns two concrete register transformations:
`controlled_modular_addition.controlled_modular_add_ir` and
`controlled_modular_subtraction.controlled_modular_subtract_ir`. Both consume
explicit wires and classical slots and return native primitive IR. They reuse
controlled Add, constant Add, controlled constant Add, controlled greater-than,
BitNot, and the exact source AddK adjoint. There is no compiler-owned executor,
backend selection, or new public root API. Users append an operation normally:

```bash
python -m examples.algorithms.register_controlled_modular_arithmetic
```

The example uses `circuit.append(ControlledModularArithmetic(operation, cv))`,
coherent controls and complex residue amplitudes, both control polarities, and
addition/subtraction restoration through the existing dynamic runtime.

## Interface and ownership

Arguments are `control, left, right, modulus, workspace, classical_bits`, with
keyword `control_value=0|1`. Words are big-endian, equal-width concrete unsigned
or Montgomery bit representations; these operations implement ordinary residue
addition/subtraction without Montgomery scaling. Control and left are preserved;
right becomes `(right +/- left) % modulus` on the selected control branch.
Both branches must satisfy the input promise: addition `0 <= left <= modulus`,
subtraction `0 <= left < modulus`, and `0 <= right < modulus`.
Require `0 < modulus < 2**n`; subtraction also requires `modulus+1 < 2**n`.

`controlled_modular_requirements(n)` returns quantum work
`max(2*n+2, n+6)` and reusable classical slots `max(2*n, n+2)`.
Work is caller-owned, initially zero and returned zero. A retained sign wire
and a shared pool serve successive loaded constants, carries, and comparator
roles. The three unmeasured comparator roles reuse other zero pool wires.
A physical measured wire keeps its classical slot across roles; slots contain
the latest measurement, not the complete outcome history. Input/work aliases,
invalid widths, polarities, modulus, and insufficient work/slots are rejected.

## Pinned source and gate-level comparison

Apache-2.0 Google copyright and source URLs are retained in both modules.
Reference is unmodified Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`,
`mod_arithmetic/mod_addition.py::CModAdd` and
`mod_arithmetic/mod_subtraction.py::CModSub`.
Subtraction preserves the source's unconditional BitNot / AddK(modulus+1),
CModAdd, exact AddK adjoint / BitNot stages, including inactive controls.
It does not substitute another equivalent subtraction network.

The reference adapter preserves parent component lifetimes, then expands each
component through its original Bloq decomposition. A Cirq-exported controlled
AddK wrapper is returned to that same upstream controlled Bloq decomposition,
preventing generic controlled-unitary synthesis from changing the primitive set.
Only temporary names and classical keys are bound to native physical roles.
Tests compare gate multisets and each wire's ordered stream, including operands
and feedback metadata; disjoint source DAG gates may commute. QUInt and
QMontgomeryUInt, both polarities, and widths through eight are covered.

## Complex channels and every measurement branch

Tests independently verify every valid column of the literal three-qubit 4T And
and both unnormalized erase outcomes. A test-only certificate then recognizes
those exact primitive blocks in the full emitted IR, checking zero compute
outputs and the And invariant before each erase on every promised basis input.
Only unconditional X/CX outside the blocks are accepted; unknown gates fail.
By linearity and composition, every one of the `2**m` outcome branches has the
same full complex map multiplied by `2**(-m/2)`, with work zero, on any coherent
input in the tested domain, including a spectator. This is an algebraic
all-branch proof, not exhaustive enumeration of `2**m` complete trajectories.
Raw dense primitive projection independently checks all 64 one-bit-add branches
and selected zero/one/alternating branches at widths two and three, including
nonuniform complex input phases and spectator entanglement. The runtime example
also exercises `append` and sampled trajectories. Eight-bit promised basis
inputs are exhaustive: addition 126,504, subtraction 126,002 for modulus 251.

## Measured primitives and independent source estimates

Counts below are emitted primitive counts, not model-derived substitutions.
T includes T and T-dagger. Conditional CZ counts are emitted instructions;
actual executed feedback depends on measurement outcomes. One reset per measure.
For these emitted streams H = 3*m, S = m, conditional CZ = m, where m is the
measurement count; this closes the primitive inventory in the table.

| n=8, modulus=251 | Control value | CX | X | T | Measurements | Quantum work | Total qubits | Classical slots |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CModAdd | 1 | 479 | 45 | 164 | 41 | 18 | 35 | 16 |
| CModAdd | 0 | 479 | 81 | 164 | 41 | 18 | 35 | 16 |
| CModSub | 1 | 641 | 85 | 220 | 55 | 18 | 35 | 16 |
| CModSub | 0 | 641 | 121 | 220 | 55 | 18 | 35 | 16 |

Separately evaluated Qualtran QECGatesCost estimates give CModAdd `20*n+4` T,
`5*n+1` measurements, and CModSub `28*n-4` T, `7*n-1` measurements.
QubitCount independently agrees with `1+2*n+max(2*n+2,n+6)` total qubits.
These agree for tested instances. Clifford model accounting remains separate
from measured CX/X/H/S and conditional CZ; it is not labeled an executed total.

## Explicit source boundaries

The inherited `right=modulus` promise mismatch and all-ones-modulus subtraction
constant overflow remain open in Q-0570.02 / Q-0579.02. Source all-ones failures
are reproduced at widths one, two, three and both control polarities. Native
subtraction rejects the overflow explicitly. No reference source is patched or
silently extended, and the parent upstream files remain incomplete in G2.
