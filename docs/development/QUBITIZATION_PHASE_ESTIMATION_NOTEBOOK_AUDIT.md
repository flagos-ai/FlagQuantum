# Complete qubitization phase-estimation notebook audit

The frozen `phase_estimation/qubitization_qpe.ipynb` has blob
`b99e4b7c6cf28cb1224c1afc8cf455042d602b6f` at Qualtran v0.7.0
`bcb17bb25360ed68ce3bc084645db3e929799389`. All thirteen original cells map
one-to-one to `examples/notebooks/qubitization_phase_estimation.ipynb`, with
one additional evidence-boundary Markdown cell. The original source remains
untouched. Algorithms owns the examples and composition. Rendering reuses the
existing signature and call-graph owners.

## Complete cell closure

| Original cells | Native responsibility |
| --- | --- |
| 0–4 | Title, owned imports, complete source description, diagram and unresolved paper Fig. 2 discrepancy |
| 5 | Small Hubbard factory with the same lattice, hopping, interaction, normalization and LP accuracy |
| 6 | Sparse factory with the identical seed-7 integral arrays, probability width 16 and rotation width 8 |
| 7 | THC factory with 450 factors, 152 spin orbitals, probability width 10 and rotation width 20, identical seeded arrays and normalization |
| 8 | Four-site Ising factory and four-bit rectangular window |
| 9–10 | All four graphical signatures, preserving complete named register descriptors |
| 11–12 | Complete Hubbard one-level graph and all five direct child counts |

The native factories reuse HubbardWalk, SparseChemistryWalk, THC alias-data
and PREPARE metadata, THC SELECT metadata, Ising walk construction and the
accepted shared QPE/window declarations. The THC inverse-QROM blocking choice
retains the original adjacent-integer optimization, producing 32. No large
THC SELECT networks are fabricated for a declaration-only notebook. Original
rotation precision and erasure blocking are retained as declaration parameters.
Source walk selection unions iterate a set. Native declarations use their
existing deterministic order, while comparing complete descriptors by name.
No source order stability is asserted.

## Acceptance and reused evidence

The untouched original notebook runs completely in a bounded isolated process,
including its original imported chemistry test fixtures. The native notebook
runs with Qualtran, Cirq and OpenFermion imports forbidden. Every original code
cell executes. All five integral arrays match byte-for-byte. Each of the four
walk normalizations, complete QPE and reflection descriptors, accuracy widths
14/21/40/4 and the THC blocking choice match the source. All four signature
SVGs and the one-level graph render through the existing drawer.

The Hubbard graph retains PREPARE once, positive-controlled walk once,
zero-controlled reflection 26 times, unconditional walk 16382 times and inverse
QFT once. These are declared component calls. No primitive count is inferred
from this graph.

Physical composition acceptance is reused from
`QUBITIZATION_PHASE_ESTIMATION_FILE_AUDIT.md`, including exact gate parameters,
full complex amplitudes, applicable exhaustive measurement histories, scratch
cleanup, native walk replacement and independently checked models. The batch
regression includes those executable checks. The notebook separately reports
the previously measured two-bit reference-child inventory of 105 primitives
and the LP inventory of 227, with their independent model categories. Those
inventories belong to the audited small fixtures.

## Remaining boundaries

The original notebook constructs large declarations and graphs only. Large
factory physical QPE admission, complete SDK-default leaf synthesis and the
original paper discrepancy remain open under Q-0631. These declaration objects
have explicit failing physical child emitters. Binding without window PREPARE
fails, and invoking an unresolved child fails with its admission boundary.
The notebook makes no large-state amplitude, clean-work or primitive-count
claim. Existing Q-0634 and Q-0622 boundaries remain recorded separately.

Q-0629 closes the complete notebook responsibility. It does not close the
remaining physical-source test acceptance tracked by Q-0631.

Batch validation passes 232 concentrated tests, strict mypy for the complete
extracted notebook and its test, Black, Ruff, Algorithms scope, architecture,
multi-level IR contract and the protected API snapshot. All five rendered
figures were inspected. Logs use the `g2-qubitization-notebook-` prefix in the
external reference evidence directory.
