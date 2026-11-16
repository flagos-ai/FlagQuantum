# Fault-tolerant chemistry authoring tutorial

Q-0452 covers the complete chemistry authoring notebook from Qualtran
v0.7.0 at `bcb17bb25360ed68ce3bc084645db3e929799389`. The original
27-cell notebook matched blob `262f4e1c4ebc5dbc1114ae062b87698588e1935d`
and executed in an independent kernel.

The migrated tutorial preserves the selection-register sketch, both lookup
scaling sweeps, the contiguous-index declaration, the detailed alias-table
wiring, and the comparison with executable sparse PREPARE. All 60 sampled
lookup-model budgets match the locked code at the original spin-orbital
counts. The detailed signature includes the four alternate orbital values
and their return paths through the final comparator. The native render
uses the same register dependencies and separates the comparator from
its final uncomputation step.

The example uses the original random distributions and tensor symmetry
with an explicit seed of 7. The fixture tests verify the coefficient
arrays and all 650 keep and alternate rows at both precision settings
against the locked helper. They also compare the clean SelectSwapQROM
model at the original block choice for that data. Both detailed
signatures, peak widths (152 and 116) and budgets match the original declaration and its decomposition.

| Scope | Modeled CCZ equivalents |
| --- | ---: |
| Original 16-bit selection sketch | 328 |
| Detailed dense at 10 bits | 562 |
| Detailed dense at 4 bits | 502 |
| Sparse at 4 bits | 153 |

The tutorial uses the source approximate-swap convention and excludes
arbitrary-rotation synthesis from these budgets. The dense
`ContiguousIndexFourD` leaf declares its signature and Toffoli count;
the source leaves index arithmetic and classical execution as exercises.
The detailed dense diagram and cost graph preserve that modeled boundary.
The executable example uses the existing native sparse PREPARE binder
with 135 retained coefficient rows. For this fixture it reserves 11
clean workspace wires and emits 96 measurements. These values describe
the emitted circuit separately from the 153-equivalent model. Full
complex amplitudes and cleanup of the shared sparse operator are covered
in the sparse PREPARE regression fixtures.

The native independent kernel rejects reference-library imports. It
executes the graphs, two scaling figures, register-score figure and
`circuit.append(op)` example with numerical assertions.
