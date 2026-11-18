# ZPow via phase gradient source-file audit

The reference is unchanged Qualtran v0.7.0 at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| File | Frozen Git blob | Ledger item |
| --- | --- | --- |
| `rotations/__init__.py` | `e6c63bdf43262fd6e3771a9eebd5c8e7ad75d4b8` | Q-0666 |
| `zpow_via_phase_gradient.ipynb` | `52d26d103692d41efd47ca719ad24102cf984b79` | Q-0687 |
| `zpow_via_phase_gradient.py` | `b9fe0ca61754f68aef838ef23abf2e6ab14e7690` | Q-0688 |
| `zpow_via_phase_gradient_test.py` | `17fa1e59be1b4d5b0b845531897485f688f41724` | Q-0689 |

Run `jupyter notebook examples/notebooks/zpow_via_phase_gradient.ipynb`.
The native notebook retains the twelve original cells, displays the
three original signatures and declared calls, and appends the 40-bit
body on caller-owned wires. Both notebooks ran in independent kernels.
The native kernel rendered two SVGs with Qualtran, Cirq and quimb imports
forbidden.

## Source binding

The original precision `1e-11` selects 40 gradient bits. The data and
gradient registers are through. The quantized half-exponent for the
original `3/8` angle is `206158430208`. The source allocates a clean
angle word, applies the first controlled XorK, adds the loaded value to
the gradient, applies the same XorK again, and frees the cleared word.
The native compiler borrows one angle word and the measured adder
carries. The adjoint reuses the same constant and reverses the direction
of the phase addition. The tests compare the bound source child order,
operands and classical feedback slots in both directions.

The 40-bit body was certified on both data values and four representative
gradient words including wrap. Sparse complex trajectories check the
full supplied state with all clean storage returned. The original three-,
four- and seven-bit tensor cases were reproduced with exact relative
phase from the stored constant. The three- and four-bit cases project
every measurement history. The seven-bit certificate covers every
gradient basis word and both data values, followed by two sparse
coherent executions on each input vector. The ideal rotation comparison
uses the original suite's global-phase allowance. The original suite,
including its notebook, passed 19 cases and skipped three checks for
symbolic decomposition, inferred counts or dtype checking.

## Literal resources and model

| 40-bit body or source model | T/T† | CX | Measurements | Declared QEC categories | Wires |
| --- | ---: | ---: | ---: | --- | --- |
| Native measured body | 156 | 469 | 39 | — | 120 reserved |
| Source concrete | — | — | 0 | 38 Toffoli, 4 Clifford | 81 logical |
| Source symbolic angle | — | — | 0 | 38 Toffoli, 80 Clifford | 80 logical |
| Source symbolic precision | — | — | 0 | `b-2` Toffoli, `2b` Clifford | Source live-width maximum |

The symbolic angle uses the source call-graph width fallback. The
symbolic precision reaches the composite width query, which keeps the
maximum over its source children. Both use the unencoded half-exponent
as the controlled-load declaration. Native execution binds concrete
parameters before emitting gates.

## Remaining source boundaries

Q-0687 and Q-0689 are covered. Q-0688 retains unresolved symbolic execution and
the unsigned angle format boundary. Explicit factory resolution is covered below.
A negative forward exponent is
rejected in the source unsigned conversion. Native reversal uses the
adjoint of the nonnegative encoded operation. Q-0666 remains open
while its other rotation exports retain their registered boundaries.

## Complete factory resolution and replacement

The complete class, three example factories, documentation specification and
original test file map to the existing native catalog, compiler and file tests.
The two source symbolic factories are resolved explicitly with `attrs.evolve`
before import. The same operation binds the concrete factory to the small
validation cases. Native execution uses `circuit.append(op)` and the existing
controlled-XOR and measured phase-addition owners. The source reference remains
unchanged.

`test_zpow_source_binding.py` covers widths one, two and three with exponent
`3/8`, and three-bit exponents `1.25` and `3.0`. The latter exercises the original
unsigned wrapping conversion. Each resolved object is imported in both directions,
round-tripped through owned JSON IR and replaced by the native emitter at the same
named append consumer. Complete logical operators, exact primitive histograms,
all-history channel certificates and independently checked final-zero storage
agree. Explicit projections enumerate every raw measurement history on a complex
superposition of all logical words. Every Fourier-gradient input also has the
analytic complex phase from the stored half-exponent. The existing source-graph
walker checks child order, physical operands and feedback at the shared arithmetic
lowering depth. Resource categories and live width are checked independently
against the original model.

| Concrete width and exponent | Literal T/T† | Literal CX | Literal measurements | Reserved wires |
| --- | ---: | ---: | ---: | ---: |
| 1, 3/8 | 0 | 1 | 0 | 3 |
| 2, 3/8 | 4 | 9 | 1 | 6 |
| 3, 3/8 | 8 | 23 | 2 | 9 |

The same histograms hold in both directions. These measured primitive counts
remain separate from the source abstract Toffoli/Clifford model above. The
one-bit source Add model anomaly is inherited. An unresolved symbolic precision
fails physical shape validation, while an unresolved symbolic angle reaches the
source decomposition rejection. Negative forward exponents fail the original
unsigned conversion and the native admission check. These cases remain explicit
boundaries under Q-0688. The previously accepted 40-bit body, original tensor
tests, independent notebook and symbolic model evidence are reused.
