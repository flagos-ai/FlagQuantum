# Phase-gradient source-file audit

The reference is unchanged Qualtran v0.7.0 at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| File | Frozen Git blob | Ledger item |
| --- | --- | --- |
| `rotations/__init__.py` | `e6c63bdf43262fd6e3771a9eebd5c8e7ad75d4b8` | Q-0666 |
| `phase_gradient.ipynb` | `ffe4116032691edf96d3ede42c32b03e2fbfa3d6` | Q-0670 |
| `phase_gradient.py` | `97885bc9d4c054df375854ea6d9d4d3afecb63e4` | Q-0671 |
| `phase_gradient_test.py` | `af6677a79bdc1bff2a02dd44dd22b618ea417263` | Q-0672 |

## Tutorial and register bindings

Run `jupyter notebook examples/notebooks/phase_gradient.ipynb`. The native
notebook retains the 35 original cells and adds an append example and its
explanation. It compares four concrete fixtures and one symbolic declaration.
All concrete fixtures run through `circuit.append(op)`. The reference notebook
ran separately unchanged, and the native notebook rendered eight SVGs in an
independent kernel with reference imports forbidden.

All words use big-endian fixed-point order. Preparation writes the new gradient
word from zero with the source default exponent `-1`. The existing native
positive default remains available for callers that use it. Addition keeps the
input and gradient as through ports, preserving the input plus any spectator
wires. Measured carry work starts and returns at zero. The scaled fixture
keeps the source's two-bit fractional coefficient format, which saturates the
supplied coefficient 2 to 0.75.

## Original tests and direct dependencies

The locked source suite passed 57 tests and skipped three symbolic
decomposition and dtype checks. The native tests map the original grids:

| Source cases | Native verification |
| --- | --- |
| State widths 6, 7, 8 and exponents ±0.124, ±1 | Source H and Z-power sequence, full complex state against the source tensor, inverse to zero |
| Unitary widths 6, 7, 8, exponents −0.5, 1, 0.1, controlled and uncontrolled | Source CZ/Z expansion and phase operands, precise angles, full forward and inverse amplitudes and clean work. All raw histories through four measurements, three specified histories above that |
| Addition widths 4 and 7, no control and both control values | Source fixed-point permutation and declared cost, with Add/CAdd and their adjoints compared at the primitive layer. Exhaustive small histories and structural certificates plus sparse checks for larger cases |
| All five original scaled-addition fixtures | Source coefficient schedule, slice binding and adjoint addition leaves. Declaration multiplicity compared independently of total Toffoli count. Structural clean-work certificates, complex forward and inverse checks, and full small history projection |
| Symbolic unitary | Source symbolic rotation multiplicities and width formulas, including the controlled variant |

The native resource model preserves a source graph quirk at zero exponent:
the identical Z or CZ keys coalesce to one declared child. The emitted stream
still retains one zero-angle operation per wire. A symbolic controlled unitary
keeps the reference width formula `Max(3, n + 1)`. Its concrete declared cost
includes a clean reusable predicate qubit and reports `n + 2`.

| Concrete fixture | Literal phase | Literal H | Literal T/T† | Literal CX | Literal measurements | Source QEC categories | Reserved / declared wires |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Four-bit unitary | 4 | 0 | 0 | 0 | 0 | 1 T, 2 Clifford, 1 rotation | 4 / 4 |
| Four-bit state | 4 | 4 | 0 | 0 | 0 | 1 T, 6 Clifford, 1 rotation | 4 / 4 |
| Four-bit plain addition | 0 | 9 | 12 | 33 | 3 | 2 Toffoli | 11 / 8 |
| Two-bit scaled addition | 0 | 3 | 4 | 8 | 1 | 0 Toffoli | 5 / 4 |

## Exact precision, port and symbolic declarations

The unitary declaration retains each typed Z or CZ child exponent and the
source tolerance `epsilon / width`. The symbolic controlled leaf has exponent
`exponent / 2**width` and multiplicity `width`. At width four and exponent
zero the controlled declared graph has one child, while the execution visits
all four target wires. The unitary signature drops a zero-width through word.
The state signature retains the explicit zero-width output word.

The arithmetic model includes `p - 2` Toffoli for a plain addition into a
gradient of width `p` and `2*p - 4` for either control polarity. Peak width is
`Max(3, n + p)` or `Max(3, n + p + 1)` respectively for input width `n`.
The scaled symbolic example with a `QFxp(4,2)` coefficient declares three
plain additions and preserves the `QFxp(5,3)` input through port. Its source
model gives `3*p - 6` Toffoli and peak width `Max(3, 5 + p)`. These estimates
describe the abstract source declarations. The measured primitive counts
above come from the previous execution comparison. No emitter changed.

## Remaining source boundaries

Q-0670 and Q-0672 are covered. Q-0671 remains open for symbolic execution
and source decomposition boundaries. `AddIntoPhaseGrad` exposes a classical
permutation and a Toffoli model, with no gate decomposition. Its one-bit model
may declare a negative Toffoli count. The scaled source Soquet graph rejects
the existing mixed QFxp cases with inputs `QFxp(5,3)` and `QFxp(3,2)`,
while its Cirq generator and numeric action can be verified. The native
arithmetic comparison follows the corresponding Add/CAdd leaves with these
boundaries recorded explicitly. Q-0666 remains open for the remaining exports.
