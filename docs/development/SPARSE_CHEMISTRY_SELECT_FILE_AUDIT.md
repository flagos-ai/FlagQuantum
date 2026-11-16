# Sparse chemistry SELECT and package audit

The frozen reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`, under
`qualtran/bloqs/chemistry/sparse`.

| Inventory | Source file | Git blob |
| --- | --- | --- |
| Q-0407 | `__init__.py` | `6b0c0373aa2cdbfb24bc314c6eb4b47d4aadb674` |
| Q-0410 | `select_bloq.py` | `1c7fd5595aa87a6c647070cffb43d941a2d7cc9f` |
| Q-0411 | `select_bloq_test.py` | `4afa771527bfa51d37cc7a85f64b15fd7471db4c` |

## Package boundary

The source package documents a second-quantized Hamiltonian with
one-electron coefficients T and two-electron integrals V in chemists'
notation. The Jordan–Wigner formulation uses modified coefficients
T′_{pq} = T_{pq} − Σ_r V_{pqrr} and products of spin-resolved orbital
pair operators. The diagonal pair has a minus-Z sign. In the two
off-diagonal index orders, the corresponding parity string joins X
or Y endpoints. Sparse PREPARE keeps the symmetry-inequivalent
integrals above the specified threshold, and the symmetry swap flags
restore the permutations in superposition.

The native `sparse_chemistry_data` owner accepts T′ and the chemists'
integrals. Its representative rows preserve source order and signed
alias data. `PrepareSparseChemistry` owns preparation and cleanup;
`SelectSparse` owns the Majorana pair binding. These are the native
owners of the frozen package's two exports. They are composed by the
sparse chemistry walk. The thresholds, row selection, and sign and swap
paths are covered by the PREPARE and SELECT regressions.

## SELECT declaration and execution

The source signature uses four bounded orbital words, two spin
bits, the one-body flag, and an orbital-major system word. Both
optional external control polarities add a through bit. The source
composite returns that bit directly from its input. Its S phase and
the four Majorana calls are unchanged. The tests compare names,
dtypes, widths, sides, per-wire operands and feedback for four and
six spin orbitals in all three control modes. The four-spin fixture
emits 40 T/T† and ten measurements with three work wires. The
direct source estimate is obtained separately. The coherent state
test includes control and spectator bits and compares forward and
adjoint branches.

At one spatial orbital the source declares the empty orbital words,
but its QEC and peak queries fail inside the Majorana children.
The native append path reduces each same-orbital Y/X pair to iZ
on the selected spin, since the Jordan–Wigner prefix cancels.
The one-body S phase makes its conditional pair a controlled minus-Z.
The native path emits two controlled swaps and uses no measurement or
work wires. Complete basis-superposition amplitudes include the
unconditional scalar i in the forward direction and its conjugate
in the adjoint, with both optional control polarities tested.
There is no locked primitive stream or resource estimate for this
specialization.

## Verification and remaining boundary

The unchanged original suite returned five passes and one expected
failure with Protobuf 5.29.5. The serialization test now passes in
that reference environment. The focused regression passed 126
cases with strict typing, format, ownership, architecture, IR, and
public API checks.

Q-0407 and Q-0411 close the package mapping and runnable original
tests. Q-0410 remains open for the original one-orbital resource
boundary and independently controlled executable semantics. Native
source-mode binding follows the original pass-through behavior.
