# Tensor hypercontraction SELECT audit

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`, under
`qualtran/bloqs/chemistry/thc`.

| Inventory | Source file | Git blob |
| --- | --- | --- |
| Q-0419 | `select_bloq.py` | `a3b825950fc09a429f829637ed14d7c73e922bb6` |
| Q-0420 | `select_bloq_test.py` | `f1b2b7a84dc7d831216470214ca02f6b07e02349` |

SELECT declares bounded orbital addresses and six selection flags
alongside the alias precision word and the two opaque spin banks.
An optional external control bit precedes the selection ports.
Each of the four directional rotation leaves has a one-body flag,
an angle-data word, an address word, and one spin bank. The tests
match the ordered ports, dtypes, domains, and register directions.

The source allocates one angle word, passes it through all four
rotation leaves and frees it. Tests identify the allocation and
release and compare the intermediate angle connections. The model
now derives both SELECT and leaf widths from their source
signatures, keeping the shared angle word live at the top level.
For either external control polarity, the original composite
returns the bit directly from the input. The model includes the
extra live bit and retains the original gate categories. Its tests
cover both polarities and the uncontrolled case at four geometries.

## Emitted operations and source limits

The executable SELECT continues to use `circuit.append(op)`.
The exact and eight-bit paths take the existing Givens networks,
compare their complex amplitudes, and clear the reused work. The
unblocked and blocked lookups exercise both coherent and measured
cleanup. In the measured fixtures the default arithmetic emits
32 or 64 measurements respectively. The upstream-style carry
path adds 28 measurements and changes the emitted Toffoli count.
The walk tests check feedback-slot ownership and complex phases
in both directions.

The reference rotation leaf gives scalar Toffoli counts and
does not supply a decomposition. The representative source model
with rank two, four spin orbitals and eight angle bits estimates
107 Toffoli and 11 controlled swaps, with 24 peak logical qubits.
This is separate from the physical instruction and measurement
counts. The optional external control on the source declaration
does not implement an independently controlled executable SELECT.

The locked original tests returned three passes and two skips.
The frozen 40-cell tutorial was already executed independently
during PREPARE acceptance. Q-0420 closes the original tests.
Q-0419 remains open for the source rotation decomposition and the
independently controlled native binding.

The concentrated native regression passed 185 cases. Strict typing,
format, ownership, architecture, IR, and public API checks passed.
