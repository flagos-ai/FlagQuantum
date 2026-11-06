# Modular negation

`modular_negation.modular_negate_ir` and
`controlled_modular_negation.controlled_modular_negate_ir` emit the original
Qualtran v0.7.0 ModNeg/CModNeg register algorithms as native IR. They directly
reuse multi-controlled X reduction, the existing MultiTargetCNOT fanout tree,
controlled AddK, and And/measurement-assisted erasure. The compiler does not
execute circuits, choose devices, or add another runtime or public root API.

```bash
python -m examples.algorithms.register_modular_negation
```

The runnable example uses `circuit.append(ModularNegation(control_value))` for
ordinary negation (`None`) and both coherent control polarities (0/1). It checks
all valid residues with nonuniform complex phases, work zero and involution
through the existing dynamic runtime, without Qualtran or NumPy installed.

## Interface and ownership

Ordinary arguments: `target, modulus, workspace, classical_bits`. Controlled
arguments add `control` and keyword `control_value=0|1`. Words are big-endian,
with unsigned or Montgomery bit representations; no Montgomery scaling is
performed. The input promise on both control branches is `0 <= x < p`, with
`0 < p` and `p+1 < 2**n`. The compiler cannot inspect quantum input amplitudes.
Output is `(-x) % p`, or x on the inactive control branch. Control is preserved.

`modular_negation_requirements(n)` returns `(2*n, n-1)`. Its work consists of a
nonzero flag, n loaded-constant wires, and n-1 wires shared by the zero-detection
And ladder and the AddK carries. `controlled_modular_negation_requirements(n)`
adds one retained apply-And target and returns `(2*n+1, max(1,n-1))`.
All input/control/work wires are disjoint and explicit. Work starts and ends
zero on the promised domain. Classical slots hold latest outcomes, rather than
the complete measurement history. The outer apply target reuses a classical
slot only after inner AddK cleanup, before the last zero check.

Source order is retained: initialize nonzero flag; zero-pattern multi-control X;
original depth-layer MultiTargetCNOT fanout; controlled AddK(p+1); zero check;
flag cleanup. CModNeg computes/erases And(control,flag) around the fanout and AddK.
It keeps both zero checks on inactive branches and the literal negative-control
X gates. MultiControlX.adjoint() is the same source Bloq, so no dynamic primitive
IR is mechanically reversed. Fanout uses the shared source tree (`2*n-1` CX),
not an alternative n-CX network.

## Source and primitive comparison

Google Apache-2.0 copyright and source links are retained. Reference is unmodified
Qualtran v0.7.0 commit `bcb17bb25360ed68ce3bc084645db3e929799389`,
`mod_arithmetic/mod_subtraction.py::ModNeg/CModNeg`. Tests cover QUInt and
QMontgomeryUInt, ordinary and both controlled polarities, widths 2/3/5/8.
Gate multisets, per-wire ordered gates, operands and feedback are compared;
independent gates in the source DAG may commute.

The test adapter keeps source component lifetimes and unfolds original Bloq
control reductions. When the source Cirq exporter wraps an encoded X control,
it reconstructs that same upstream controlled Bloq from the encoded bit pattern
before decomposition, preserving the original And ladder rather than Cirq's
generic unitary synthesis. This does not patch upstream source or replace the
original algorithm. MultiControlX uses its own original decomposition; the
existing source fanout tree is retained unchanged.

## Complex channels, work and all measurement branches

The strict test-only And-channel certificate verifies every compute/erase
invariant for all promised basis inputs through width eight, both polarities,
and applies the same map twice to establish involution. Independent literal
And primitive columns and both erasure Kraus outcomes establish positive phase
and identical branch maps. By linearity/composition every branch has the ideal
full complex map multiplied by `2**(-m/2)` and clean work.

Raw full-circuit postselection additionally enumerates **every** measurement
branch at widths two and three on complete coherent residue inputs entangled
with a spectator: ordinary 8/64 branches and controlled 16/128 branches for
each polarity. These compare unnormalized complex amplitudes with absolute
tolerance 1e-12, including branch probabilities and work zero. Width-eight
all-branch coverage is a composition proof, not a claim to execute 2**22 dense
statevector trajectories. The separate runtime example exercises append and
sampled dynamic trajectories.

## Measured primitives and independent resource models

The table scans emitted IR for n=8,p=251. T includes T-dagger; conditional CZ
counts are emitted instructions and runtime execution depends on outcomes.
H=3m and S=CZ=reset=m, with m the measurement count.

| Native circuit | CX | X | T | Measurements | Clean quantum work | Total qubits | Reused classical slots |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ModNeg | 194 | 66 | 84 | 21 | 16 | 24 | 7 |
| CModNeg, cv=1 | 200 | 66 | 88 | 22 | 17 | 26 | 7 |
| CModNeg, cv=0 | 200 | 70 | 88 | 22 | 17 | 26 | 7 |

Separately evaluated QECGatesCost gives ordinary `12*(n-1)` T and `3*(n-1)`
measurements; controlled adds four T and one measurement. QubitCount gives 3n
ordinary total and 3n+2 controlled total. All agree on tested instances.
Clifford model fields are retained separately from expanded CX/X/H/S/CZ counts;
source And blocks are abstract in the model, and its Clifford field is not an
executed primitive total. At n=8,p=251 the source And/Clifford fields are
21/141 for ordinary negation and 22/142 for controlled negation (both polarities).
No model estimate substitutes for measured gates.

## Explicit open boundaries

The original AddK(p+1) load fails for p=2**n-1, including width one. Both source
failures and native explicit rejection are tested at widths 1/2/3 and both
control polarities. The classical ModNeg/CModNeg model is identity outside the
residue domain, but the decomposition does not fulfill that total-domain claim:
n=2,p=2,x=2 produces x=0 and a dirty flag (on an active controlled branch as
well). Unmodified source classical decomposition rejects its Free, and native
raw gate evidence retains that flag instead of silently resetting it. These
remain open upstream parity boundaries; native promises only valid residues.
Symbolic and other unfinished modular families remain outside this acceptance.
