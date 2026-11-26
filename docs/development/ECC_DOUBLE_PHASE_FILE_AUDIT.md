# Elliptic-curve double phase estimation

## Original constructor and direct dependency closure

The complete `find_ecc_private_key.py` constructor now binds through the shared
typed phase roots. Its exact six calls are two raw base-coordinate IntStates,
base-point phase estimation, public-key phase estimation, and two dirty Frees.
The original symbolic example retains this graph, with symbolic initialization
and phase children declared rather than assigned numerical bodies. The source
point fields survive the owned source library; no automatic Montgomery
conversion or classical output was added.

Source shallow libraries and the complete call graph round-trip with optional
SDKs blocked. Original modulus and curve consistency checks remain properties
used by the grouped model. They are not silently promoted into constructor or
graph guards: a mismatched-modulus direct graph still has the source six-call
shape, while the source model property rejects it. The existing numerical
Algorithms operation keeps its separate admission rules.

The root inherits scalar ECAddR opacity, model-only QROAM erasure and the original
MeasureQFT unused-CBit failure. Whole physical primitive counts, all-branch
execution and private-state cleanup are unavailable, not zero. Concrete grouped
windowed source QEC cost additionally fails when the generic point requires an
integer conversion. These failures are retained alongside the existing explicit
readout workflow and its independently verified joint terminal projections.

The source-shaped roots share MeasureQFT through an independent Ecosystem owner,
not the RSA adapter. Whole old RSA and three shim records remain exactly equal
after that move. Selected actual window-child conformance checks all 32 small
complex columns, every applicable erasure history, returned layouts and clean
private work. This does not certify the unavailable whole double-phase root.
See [the phase audit](ECC_PHASE_ESTIMATION_FILE_AUDIT.md) for the shared boundary.
Q-0466 remains open; the original dirty coordinate exits are never reported as
zero-work promises.

The locked reference is Qualtran v0.7.0,
`bcb17bb25360ed68ce3bc084645db3e929799389`. This batch covers
`qualtran/bloqs/cryptography/ecc/ecc.ipynb`, `find_ecc_private_key.py`
and `find_ecc_private_key_test.py`.

## Source calls and physical binding

The symbolic source signature is empty. The source requires common
modulus and curve coefficient and declares two calls to phase estimation
at a generic point. Its concrete graph initializes the two data words
from the raw base coordinates, estimates at the actual base point and
then at the actual public point, and dirty-releases both data words.
The test independently checks the initializer values and the shared
Soquet operands at the two calls and releases. The native append binding
makes the caller's data, phase storage and both ordered readout words
explicit. It reuses clean arithmetic scratch and preserves dirty data for
inspection at the end.

The source initializer preserves the stored unsigned words. At width 3
and modulus 3, its initial `(0,1)` is interpreted as `(0,2)` by the
windowed Montgomery child. This encoding behavior is visible in the
joint state check. The one-bit fixed-point branch remains a source
declaration because ECAddR has no gate decomposition. Both points'
phase operations must satisfy the inherited arithmetic cleanup domain.

## Joint projections and resources

For (P=(0,1)), (Q=(1,0)=2P) on (y^2=x^3+x+1) over the field of
size 3, the native circuit uses two three-bit address windows.
The complete joint state is computed independently from frozen
windowed coordinate maps and two positive-exponent DFT sums.
The test projects each first readout before executing the second
stage, using its actual measurement slot and physical basis order.
It checks all 64 joint projected states including zero-probability
branches, exact complex amplitudes and all arithmetic-work qubits.
Previous window and QFT tests independently check component gate order
and measurement feedback.

| Scope | Resource evidence |
|---|---|
| Native emitted circuit | 80 bound wires, 17 reusable arithmetic slots, 6 readout slots, 15,376 T/T†, 38 CCX, 31,362 CX, 9,378 H, 6 CP, 2 SWAP, and 2,692 measurements |
| Two concrete source phase child models | Combined 2 T, 42 Toffoli, 672 CSwap, 2,662 AND, 17,626 Clifford and 2,668 measurements |
| Full source model | Top-level gate count fails when its generic symbolic point is converted to a concrete row; QubitCount fails at the unused MeasureQFT results |

The child numbers give a comparison of two concrete calls. The source
top-level declaration itself counts two generic calls.

## Tutorial and test-file disposition

The locked 39-cell notebook runs in its source kernel. The
[native tutorial](../../examples/notebooks/register_elliptic_curve_cryptography.ipynb)
retains all four sections, symbolic and concrete source declarations,
signatures, call graphs, the initial-state and dirty-release diagram,
and the 20 fixed-point examples. A concrete append example reports both
measurements and the physical wire requirement. All 12 code cells
and nine SVGs run in a separate native kernel with reference imports
forbidden.

The full source test file reports three passes, one source skip and
two expected failures. Its symbolic construction and notebook checks
have native counterparts, plus the physical joint projection test.
Q-0465 and Q-0467 close. Q-0466 stays open for the source's scalar gate
decomposition, resource-query and basis-encoding boundaries, and the
inherited wider-domain cleanup issues.

## Parameter-preserving model traversal

`source_callees(allocator)` follows the original fresh `Rx`/`Ry` allocation and
returns two calls to one generic point phase operation with the complete width,
curve, addition-window and multiplication-window parameters. The allocator is
caller owned, so names already allocated in an enclosing query are respected.
`source_call_graph()` expands that phase declaration through the existing
symbolic graph engine. Its terminal multiplicities are twice the representative
phase child's counts. Repeated queries are deterministic. `cost_attrs()` retains
the source width column. The module-level grouped graph is still available.

The regression compares both declaration levels against the unchanged source,
including preallocated symbols and two window settings. The generic point is
kept distinct from the two physical execution points. The full physical joint
projection regression above is rerun with this batch. No additional primitive
counts are inferred from these model edges, and the source resource-query,
scalar decomposition and clean-domain boundaries remain open.

Batch validation on 2026-11-21 passed 44 concentrated cases and 29 integration
cases, strict typing for all three changed Python files, formatting, Algorithms
scope, architecture, IR and public API checks. The Cirq 1.6.1 lane passed all
20 selected model and physical phase cases with the native Kickmix extension
present.
