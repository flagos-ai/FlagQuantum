# Recursive phase-gradient QFT

The emitter follows the [locked QFTPhaseGradient](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/qft/qft_phase_gradient.py)
recursion. Transform the left half without swaps, accumulate the product
of the bit-reversed left half and the right half into the full gradient,
then transform the right half. Each child uses the appropriate gradient
prefix. Only the outer call may reverse the result order. The inverse
reverses the stages and subtracts the product.

The caller binds equal-width big-endian data and gradient registers.
The negative gradient eigenstate is preserved and induces the textbook
QFT. The joint operator is also defined for other gradient states.
The shared product accumulator borrows a clean pool sized at the root,
reuses it at every recursion level, and restores it after every product.

At width four the emitted circuit uses 52 T-family gates, 13 measurements,
nine clean work qubits and three classical slots. The recursive source
leaves estimate 48 T in total. The locked aggregate graph estimates 40 T
because the equal child entries collide in its dictionary. Its child
entries also default to bit reversal, while the executable children omit
it. The source product leaf supplies an arithmetic gate and model without
a primitive decomposition. Tests compare the executable parent bindings,
the native composed stream, the full source operator for small registers,
and cleanup of a prepared gradient through the append entry point.

## Whole-file evidence

| Ledger | Source file | Git blob | Coverage |
| --- | --- | --- | --- |
| Q-0639 | `qft_phase_gradient.ipynb` | `ee2fd8acc417d2c7fe552357f91b07bc71b998f6` | Original three-bit example, graphical signature and call graph, plus native append and cleanup |
| Q-0640 | `qft_phase_gradient.py` | `6597eba42bf90995e7e857ea452405a657fc87df` | Recursive emission, typed ports and the concrete and symbolic source declarations |
| Q-0641 | `qft_phase_gradient_test.py` | `d6fee8c159f98ceb8d5bbac7fcd04faf39d7c4aa` | Original arithmetic checks, model bound and source count-autotest boundary |

The new model retains the source unsigned data and fractional gradient
through ports. It uses the existing call-graph engine with an independent
source declaration. Concrete oracle checks cover widths one to five,
ten and 123, with both reversal settings. The unrestricted symbolic width
matches the source logarithmic estimate
`8 * max(floor(n/2), n - floor(n/2))**2 * log(n, 2)`
and the symbolic peak `max(1, 2*n)`. No symbolic circuit allocation is
attempted. The pinned source supplies the arithmetic product leaf as an
operator and a cost bound without primitive gates. The existing recursive
gate-order test compares each operand binding and expands the product
using the accepted native implementation.

The unchanged source suite passed 14 tests with one expected failure
for the disabled general count-autotest enforcement. The old executable
tests and 17 new model checks passed, for 23 focused cases. The 5,497-test concentrated regression, strict
typing, format, scope, architecture, internal IR and public API checks also
passed. The executable tests retain
the earlier full complex branch, gradient restoration and clean workspace
evidence. All five original notebook code cells executed. The migrated
notebook ran six code cells and rendered two SVG figures in an isolated
kernel with reference imports blocked.

Q-0640 remains open for the source aggregate defect. Source dictionary
keys collapse the two equal child calls at even widths. It also declares
bit reversal on recursive children that execute without reversal. The
native cost faithfully reports the aggregate declaration and the native
emitter preserves the executable recursion. The width-four case above
keeps the aggregate 40 T, the 48 T sum of executed product-leaf models,
and the 52 emitted T-family gates separate. Q-0641 remains open for the
source general count-autotest gap.


## Stored catalog dependency interface

This batch retains the complete locked source-file graph through the existing
Program and append paths. Words use MSB-first physical labels and caller-owned
through storage. Textbook q is unsigned. Gradient QFT and approximate QFT keep
a separate unsigned fractional gradient through word owned by the caller.
Shared multiplication and controlled addition own their measured carry work,
which must return to zero. TwoBitFFFT has x/y QBit through ports and no work.
Its original ignored-adjoint decomposition remains separate from the native
exact inverse. Existing primitive execution and source-model evidence is reused.
Compiler owns stored word/call composition, Algorithms owns the already accepted
fermionic Fourier sequence, and the example provider owns catalog glyphs and
literal-definition qualification. No protected contract change is required.


## Complete catalog graph closure

The concrete three-bit root retains all eighteen original calls. Recursive
width-one and width-two children preserve with_reverse=False, independent of
the source model's equal-key collision and default-reversal declarations.
Four complete source-success graphs and fresh drawings agree. PlusEqualProduct
remains an original atomic refusal. Shared native product accumulation below
that boundary owns and erases its clean work. Literal execution emits 24
T/T-dagger, 46 CX and six measurements on ten physical wires. Separate source
model T-equivalent is 40 and QubitCount six. Complete small arbitrary-gradient
joint operators, all tested raw histories and exact existing-emitter inverse
recovery pass. No equality with a nonexistent original product recipe is claimed.
