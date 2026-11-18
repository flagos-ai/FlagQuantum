# Complete textbook phase-estimation notebook audit

The frozen source is `phase_estimation/text_book_qpe.ipynb`, blob
`139af0f36d2671d02ce95abbedcbeda60b1d19ff`, at Qualtran v0.7.0
`bcb17bb25360ed68ce3bc084645db3e929799389`. Its 24 cells map one-to-one to
`examples/notebooks/textbook_phase_estimation.ipynb`. Every mapped cell retains
its original index. Two added cells demonstrate physical append and separate
resource accounts. Copyright and Apache-2.0 provenance remain present.

## Complete responsibilities

All four QPE instances retain theta and exact width expressions, including both
precision/error factories. Both rectangular instances retain their signatures.
The six signature drawings preserve register name, width, shape and side.
Concrete diagrams use the existing signature renderer. Symbolic diagrams use
the shared graph annotation renderer to label declaration metadata explicitly.
They do not resolve symbolic widths into physical circuits.

Both original one-level call graphs retain original direct weights. The QPE
children have weights 1,7,1. The rectangular Allocate and OnEach children each
have weight one, with the existing shared graph separately retaining its five
H leaves. Tests normalize those labels and check both graphs against untouched
source execution. All eight diagrams are rendered and visually inspected.

The original appendix and variance derivation remain present, with their source
conventions and mathematical steps. This migration does not certify a new
variance theorem. Register and state ownership are inherited from the complete
source-file audit. Production imports contain no Qualtran, Cirq or OpenFermion.
The reference test blocks all three SDK imports while executing every native
cell, after executing every original cell independently.

## Ordinary append and separate resources

The added numeric example prepares the target eigenstate, then uses
`circuit.append(textbook_qpe_small.make_on(...))`. Its full complex phase output
matches the direct source Fourier expression at theta=0.234, with three phase
bits and most-likely output two. No phase alignment is applied. The actual QPE
body has 17 primitives, including six H, ten cphase and one swap, with no work.
The initial target X is counted separately.

The independently declared source model reports one T, ten And, 19 Clifford,
seven rotations, ten measurements and peak five. All categories and peak match
the original instance. The direct callback repeats controlled-phase operations
according to the locked `Power` schedule. The SDK-default controlled-Bloq leaf
stream and native original walk/GQSP integration remain Q-0634 gaps recorded in
the source-file audit. This notebook adds no claim that closes those gaps.

## Verification

Every original and native cell executes. Concentrated notebook, QPE and direct
window/LP/QFT/GQSP-file dependencies pass, with strict mypy on the test and all
extracted notebook code. Black, Ruff, Algorithms scope, architecture, multi-level
IR and protected API checks pass. Logs use `g2-textbook-notebook-` in the
reference workspace. Q-0632 covers this complete notebook.

The concentrated batch reports 104 passed. Its only warning concerns the local
IPython directory, which uses its existing temporary-directory fallback. All
strict and architecture checks listed above pass.
