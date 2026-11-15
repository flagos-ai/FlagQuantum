# Chemistry resource estimation tutorial audit

Q-0398 refers to Qualtran v0.7.0
`qualtran/bloqs/chemistry/resource_estimation.ipynb` at
`bcb17bb25360ed68ce3bc084645db3e929799389`.
The original Git blob is `86889c2116aed7385ef814e42ddec5c1895eb149`.

The immutable 34-cell source was executed in an independent kernel.
Strict execution stopped at zero-based cell 12. The source passes
the legend handles positionally and the labels as a keyword, which
Matplotlib rejects. A diagnostic execution kept the original cell
contents and recorded that same exception. Every other cell ran.
The native chart uses the actual pie wedges and passes both legend
arguments positionally.

## Model comparisons

The native tutorial covers all three studies. Its THC fixture uses the
original synthetic dimensions, precisions, contractions, and a fixed
random seed. Both implementations receive the same arrays for the
reference comparison, including coefficient normalization and alias
preprocessing.

| Modeled CCZ-equivalent case | FlagQuantum and locked source |
| --- | ---: |
| THC SELECT | 7,105 |
| THC PREPARE | 3,195 |
| THC PREPARE adjoint | 839 |
| THC LCU total | 11,139 |
| THC forward rotation leaf | 1,914 |
| Double factorization | 21,728 |

The THC rotation leaf splits into 1,512 rotation and 402 lookup
CCZ-equivalent units. The PREPARE classification independently agrees
in all seven source T-equivalent categories. Data loading contributes
11,396 of 12,802 T equivalents. This chart uses the original
classification and default T-equivalent weights.

The separate CCZ comparison generalizes the modeled QROAM bank
routing from approximate controlled swaps to controlled swaps.
The raw model assigns 7,560 T to those approximate swaps; the tutorial
generalization converts those to 1,890 CCZ units. It is a budget
transformation, so the two columns must stay separate. The example
does not emit a gate stream or simulate a FeMoCo-sized state.

The first-quantization study retains the cubic-cell formula and five
electron counts. Each SELECT, PREPARE, and PREPARE adjoint cost is
compared to the pinned source. The three direct SELECT contributions
are checked separately for every size.

| Electrons | SELECT | PREPARE | PREPARE adjoint |
| --- | ---: | ---: | ---: |
| 10 | 1,364 | 506 | 506 |
| 20 | 2,124 | 527 | 527 |
| 40 | 3,644 | 558 | 558 |
| 60 | 5,164 | 582 | 582 |
| 80 | 6,684 | 609 | 609 |

The native register scores cover THC SELECT, double factorization,
and the two-particle first-quantized SELECT example. The double
factorization score reuses the existing stage and operand map at
the larger tutorial dimensions.

## Verification and boundary

The native tutorial generated seven PNG figures and one SVG in an
independent kernel without importing upstream packages. The existing
double factorization notebook also passed. The concentrated regression
reported 254 passes. Strict typing, format, ownership, architecture,
IR contract, and public API checks passed.

Q-0398 closes the tutorial and its resource analysis. The executable
gaps previously recorded for first-quantization leaves and whole
chemistry blocks remain on their own inventory rows.
