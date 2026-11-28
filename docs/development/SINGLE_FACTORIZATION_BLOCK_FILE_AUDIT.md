# Single-factorization block encoding audit

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`, under
`qualtran/bloqs/chemistry/sf`.

| Inventory | Source file | Git blob |
| --- | --- | --- |
| Q-0404 | `single_factorization.ipynb` | `ebc36a07b6bb637148c644d25cad68991823aaf1` |
| Q-0405 | `single_factorization.py` | `5e1104df8ac20e98c1e263a83214c7d1ec72a51f` |
| Q-0406 | `single_factorization_test.py` | `1c35e9d97c7866aca0830bbdfed19619abd23942` |

The one-body declaration preserves the ordering of the three individual
control flags, the bounded factor and spatial addresses, the three
bounded single-qubit selectors, and the two spin banks. The full
block groups the three control bits, the two opaque spatial words,
and the two rotation flags. Both retain explicit zero-width address
ports when applicable. The signal-state identity uses exactly the
selection registers of each declaration. The original models now
derive their local widths from these native signatures.

The resource tests compare the original top-level decomposition and
each immediate child in order. They cover both one-body direction
settings, distinct forward and inverse lookup blocks, the two tutorial
examples, and the 108-orbital OpenFermion case. The full source reflection
uses the outer success flag. The frozen one-body adjoint setting retains
the same composite schedule and distinct forward and inverse block
parameters.

## Resources and physical circuit

| Tutorial model | Toffoli | Controlled swap | And and measurements | Peak logical qubits |
| --- | ---: | ---: | ---: | ---: |
| One body | 1,517 | 18 | 46 each | 75 |
| Full block | 3,095 | 50 | 179 each | 80 |

The executable full block uses the existing coefficient-based binding
with `circuit.append(op)`. Its register layout includes coherent alias
and sign outputs and interleaves the system spin orbitals. Its middle
reflection uses both outer success and the nonzero-factor flag. The
paper projection tests compare the complete complex system matrix and
the nonprojected output states across four histories, checking lookup
and temporary cleanup. The two-spatial-orbital case emits 408 T/T† and
67 measurements using six work wires. The source PREPARE and SELECT
leaves are declaration-only, so there is no complete source primitive
stream for that physical comparison.

The standalone native `SingleFactorizationOneBody` binds the same body
used in the full block. It applies inner PREPARE, the two Hadamards, the
controlled orbital exchange, signed SELECT, and the matching inverse
preparation. It retains the input factor address and success flag. With
success zero its system action is the identity. The projected active
block equals the signed alias-quantized Pauli average. Measurements
clear borrowed work and lookup outputs, while the coherent signal can
have nonzero components outside the projected subspace. The complete
native block used the same instruction sequences before and after the
extraction in all three existing fixtures.

| Standalone circuit | Emitted T/T† | Measurements | Work wires | Feedback slots | Branch comparison |
| --- | ---: | ---: | ---: | ---: | --- |
| One spatial orbital with one or two rows | 26 | 3 | 2 | 1 | All eight histories |
| Two spatial orbitals with one row | 148 | 23 | 4 | 3 | Four representative histories |

All tested histories agree on the complete complex output for both
success values and every system basis input. The test also compares the
projected complex matrix and checks the unchanged address and cleanup.
The locked standalone graph passes `sys`, `l_ne_zero`, and `rot_aa`
through, and its SELECT declaration has no `sys` port. The native
system action uses the independently validated physical SELECT,
with emitted counts recorded separately from the source model.

## Tutorial, tests, and remaining boundary

The unmodified 18-cell source tutorial passed in an independent kernel.
The native 18-cell tutorial produced both signatures and both call
graphs, plus a small example with 94 emitted T/T†, 13 measurements,
and four work wires. The original non-notebook suite reported nine
passes, three skips and one notebook deselection with OpenFermion
1.6.1. The concentrated native suite passed 48 cases. Strict typing,
format, ownership, architecture, IR, and public API checks passed.

Q-0404 and Q-0406 close the notebook and runnable original tests.
Q-0405 remains open for the end-to-end source-to-physical boundary.
The locked source alpha
and epsilon getters are both unimplemented and remain explicitly
outside the verified numerical contract.


## Complete source disposition (2026-11-28)

The [complete constructor and dependency closure](RESOLVER_SINGLE_FACTORIZATION_AUDIT.md#complete-factorization-source-closure-2026-11-28) supersedes the earlier missing generic inverse and symbolic declaration paths.
Original absent bodies, unavailable queries and reproduced defects remain
explicit under the approved faithful-source rule. Whole-root physical inventory
and numerical execution remain unavailable. Acceptance requires the batch checks
and verified publication.
