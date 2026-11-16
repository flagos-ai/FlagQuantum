# Quantum variable rotation source-file audit

The reference is unchanged Qualtran v0.7.0 at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| File | Frozen Git blob | Ledger item |
| --- | --- | --- |
| `rotations/__init__.py` | `e6c63bdf43262fd6e3771a9eebd5c8e7ad75d4b8` | Q-0666 |
| `quantum_variable_rotation.ipynb` | `e04a9a4f90b717eb8dbd654b445699edc4268773` | Q-0681 |
| `quantum_variable_rotation.py` | `8177ea80e8150c79dac75e79e0868054fa2db8a4` | Q-0682 |
| `quantum_variable_rotation_test.py` | `365027236f79d3723e619f6175654f2b285c3d14` | Q-0683 |

Run `jupyter notebook examples/notebooks/quantum_variable_rotation.ipynb`.
The native notebook binds both original 12-bit factories with
`circuit.append(op)`, reproduces their signatures and direct call graphs,
and evaluates the four original data sweeps. The sizing charts report
both the formula and fixed-point search, including the 54-bit cutoff used
by the executable width selector. The first chart labels each coefficient
as `1 - 2**(-bits)`, matching the source's data. The other two charts use
the source precision-dependent terminal T estimates. Both notebooks ran
in independent kernels. The native run rendered four SVGs and four plots.

## Execution and resources

The cost word is big endian. The gradient route binds a caller-owned
through Fourier gradient and clean measured add carries. The test oracle
walks the source scaled-add children to the bound Add leaves. Direct
Z powers preserve the original retained bits, exponents and tolerance
allocation. The source declares a repeated representative Z power for
costing, while the emitter uses the individual angle on each retained wire.

| Original 12-bit body | Literal T/T† | Literal CX | Literal phase | Literal measurements | Source QEC categories | Reserved / source wires |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Direct Z powers | 0 | 0 | 6 | 0 | 6 rotations | 12 / 12 |
| Gradient, 19-bit through word | 484 | 1244 | 0 | 121 | 119 Toffoli | 49 / 31 |

The source T plots model synthesis and omit the caller's gradient
preparation. The execution tests compare complete complex amplitudes
on the supplied source basis and superposition, with per-history
projection on small fixtures. Larger cases check the measured carry
invariants and sparse coherent executions. The original one-bit direct
and gradient grids, the 12-bit direct precision grid, 48 worst-case
width/tolerance pairs, and the four T-complexity cases are mapped.
Across the four original charts, 1,248 gradient data points and 88
direct T-estimate data points match the locked reference. The source
suite returned 86 passes and three expected failures.

## Source precision and remaining boundaries

The default gradient gives maximum complex phase error
`0.00010785802409596983` at unsigned cost word `4095`, over all 4,096
cost words with requested tolerance `0.0001`. The native gate stream
matches that source fixture. The two nine-bit narrow-gradient fixtures
marked as expected failures in the source's Cirq suite have maximum
errors approximately 0.242 (integer) and 0.233 (fractional) with
requested tolerance 0.1. The tests check translation and work cleanup
against the actual source and record these discrepancies separately
from ideal phase error.

Q-0681 and Q-0683 are covered. Q-0682 remains open for the source
precision overruns, signed direct rotation indexing, zero or quantized
zero sizing, fixed-point search overflow, recursive width on affected
mixed dtypes, and symbolic execution. The symbolic declarations used
by the original tutorial are modeled. Q-0666 remains open for other
rotation exports.
