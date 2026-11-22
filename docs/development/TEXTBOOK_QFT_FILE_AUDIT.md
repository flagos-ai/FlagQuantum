# Textbook QFT whole-file acceptance

This audit covers the textbook QFT files at the locked Qualtran v0.7.0
revision `bcb17bb25360ed68ce3bc084645db3e929799389`.

| Ledger | Original file | Git blob | Acceptance |
| --- | --- | --- | --- |
| Q-0642 | `qft_text_book.ipynb` | `6c741f6fbef15cc60c3d42bc2fb6ecf240695179` | All six original code cells run. The FlagQuantum notebook retains the concrete and symbolic instances, graphical signature, and one-level call graph, and adds native append execution. |
| Q-0643 | `qft_text_book.py` | `c560deab207982dff006944015f99e59eb262943` | Source schedule, both examples, through signature and concrete/symbolic graph and costs. |
| Q-0644 | `qft_text_book_test.py` | `031b499182d27681ed3add569a4d0125c548b704` | Original assertions mapped below. The width-123 small-angle assertion remains open. |

The algorithm owns the coherent Fourier circuit. It emits Hadamards,
controlled phases on earlier wires in reversed order and an optional final
bit reversal. Existing `qft`, `append_qft`, and `Circuit.append` paths use
the same emitter. The executable transform requires positive width and uses
no extra work or result bits. The resource model and single unsigned
through signature also accept the source's unrestricted symbolic factory.
That symbolic instance cannot be decomposed into a concrete circuit.

The resource module reuses the source controlled phase-gradient model at
exponent one half. The direct declaration has one child per smaller gradient
size for concrete widths. The symbolic declaration groups them under width
`n - 1` with multiplicity `floor(n/2)`. Reversal adds its independent swap
count. The existing MeasureQFT cost consumer now shares this textbook model
and adds its Z measurements. The shared classifier also gives MeasureQFT the
locked source's small-angle classification at larger concrete widths.

## Original assertions and boundaries

- The quick and slow unitary tests map to complete complex basis columns
  through seven wires, both directions, with and without reversal, using the
  native append path. Existing gate-level tests compare the entire ordered
  source decomposition at widths one, three and five.
- The ten-wire source complexity assertion and the symbolic resource
  assertion match the actual source calls, QEC cost categories and modeled
  peak. The width-123 source model reclassifies tiny phases, yielding 3,774
  general rotations where the test's closed form expects 7,260. Both the
  source result and its expected failure are recorded. The circuit still
  emits all 7,503 controlled phases.
- The factory and symbolic autotests map to the concrete/symbolic signatures,
  direct call graphs and notebook instances. Upstream skips symbolic
  decomposition and dtype checks that require decomposition. Its general
  autotest count check is an expected failure because count enforcement is
  disabled. The explicit count tests here exercise both directions of the
  reversal option with the frozen declarations.
- With the unchanged source, 22 cases passed, three were skipped for the
  source admission conditions above and two were expected failures.
- The native notebook runs seven code cells and renders three SVGs in a
  clean kernel with Qualtran, Cirq and quimb imports blocked.

| Width with reversal | Emitted H | Emitted controlled phase | Emitted SWAP | Modeled T | Modeled And | Modeled rotation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 3 | 3 | 1 | 1 | 3 | 0 |
| 10 | 10 | 45 | 5 | 8 | 45 | 28 |
| 123 | 123 | 7,503 | 61 | 121 | 7,503 | 3,774 |

The emitted operations contain no measurements or work allocation. The
source-controlled-gradient cost model can count both measurement cleanup
and extra logical width. These modeled costs and emitted primitives are
reported separately. The concentrated regression passed 5,479 cases.
The final whole-file suite passed 21 cases, including the full-unitary
assertions added after that regression began. Strict typing, format, scope,
architecture, internal IR and public API checks passed.


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

The concrete three-bit root retains all ten original calls. Both controlled
PhaseGradientUnitary children, their CZPowGate bodies and fresh drawings agree.
Shared one-bit fractional views are reused without an extra Split/Join pair.
The unchanged symbolic root still raises DecomposeTypeError. Literal three-bit
execution emits 12 T/T-dagger, 18 CX, three phases and three measurements on
four physical wires. The separate source model reports one T, three And, nine
Clifford and three measurements, T-equivalent 13 and QubitCount four. Specialized
source phase classification and literal retained phase primitives are distinct.

The four-file catalog group retains five concrete source-success roots and one
symbolic original refusal. Thirteen exact recursive graphs/drawings and 248
original atomic contexts are recorded. The 1960 additional reachable contextual
children include 1704 unqualified/native-only descendants below original atomic
boundaries. Unreachable retained helper definitions are excluded.

The shared layout owner can retain the immutable nominal dtype of the original
_QReg dictionary key separately from the latest SSA handle dtype. Python equal-key
assignment retains that key, and the frozen adapter chooses later casts from it.
This metadata is explicit only for the new source-ordered Fourier compositions.
Existing gather callers retain their previous behavior. Complete definition
mutation tests alter both CX operands and H atoms so even a one-H recursive child
is changed. The first mutation-only failure left that H-only child intact;
the corrected test rejects every altered qualified child.
