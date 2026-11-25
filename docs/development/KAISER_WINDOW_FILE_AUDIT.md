# Complete Kaiser window source and explicit rotation dependency

## Frozen identity and ownership

Q-0621 maps `phase_estimation/kaiser_window_state.py`, blob
`38e851f815ca3d6833215563738b481f39801a47`, at Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. Q-0622 maps its tests, blob
`6a4917cf85cd2f8f50c016a8b930d598680ee775`.

Simulation owns the periodic SciPy window calculation and normalization in
`simulation/numerics/kaiser_window.py`. Algorithms owns the shared phase-window
subclass, complete parameter factory, output descriptor and declared preparation
child in `algorithms/primitives/kaiser_window.py`. It reuses the existing dense
rotation preparation, its concrete and symbolic resource models, and the shared
bound operation. No external quantum SDK enters production.

The RIGHT output is unsigned fixed point, with all n bits fractional and
big-endian physical order. Symbolic coefficients use the existing shape-only
sequence length. Numeric coefficients retain the source's cached NumPy array.
The complete source expression is SciPy kaiser(2**n, alpha*pi, sym=False), divided
by its NumPy norm. Its overflow, NaN and negative-width behavior is retained.
Both original example constructors and the full precision/delta ln-ln factory
map to the same native data and expressions.

The source exact tensor is represented by these coefficients. Its single
declared child uses those coefficients and n phase bits. Native source graphs
accept caller-owned child identity keys and retain the single-call edge. The
shape resource formula includes the parent signature maximum, including the
original unreduced symbolic Max expressions. Concrete resources wrap the same
rotation child in the Kaiser parent.

## Explicit append contract

The source declares no physical Kaiser decomposition. Native `make_on` explicitly
binds the source's declared quantized rotation child. The caller supplies the
clean phase output, a negative-exponent phase-gradient eigenstate, clean work
and reusable classical slots. `circuit.append(state.make_on(...))` lowers through
the existing measured rotation path. The gradient is retained and all scratch
returns to zero in every branch. The uncompute flag selects the original
rotation child's uncompute path. It is separate from the source's generic
adjoint wrapper.

At n=2, work is five qubits and two reusable slots. Each direction has seven
measurement events, which reuse those slots. Full complex amplitudes and clean
work are checked for all 128 histories independently in each direction. The
uncompute test starts from the source forward output and returns to the original
coherent target and gradient, including absolute phase. Every forward history
produces the same normalized source output, so the independently checked inverse
histories establish their Cartesian composition by linearity. Gate streams,
operands, angles and feedback match the untouched rotation child. IR round-trip
preserves that stream. Existing complete rotation-file and channel evidence is
reused for the shared dependency.

## Separate exact tensor, actual primitives and declared model

Normalized coefficients match the exact source array at widths 0,1,2,5,8 and
alpha 0,0.25,2,-2. Source tensor comparisons cover widths 1,2,3,5. Concrete and
symbolic factories, single child data/precision, metadata and every QEC category
are compared with the frozen source. Symbolic peak width retains the parent's
own width alongside the child's maximum.

The two-bit emitted child has 158 primitives. Its inventory is 67 CX, 21 H,
14 T, 14 T-dagger, ten X, seven S, seven CZ, seven measurement, seven reset and
four Rx. Independently, the source QEC model reports four And, 26 Clifford,
four measurement, zero continuous rotations and peak nine. Measurement events
from concrete phase addition differ from its abstract source cost annotation,
as already recorded by the rotation dependency. These models retain their
original convention.

For alpha=2, the declared n-bit-precision rotation child differs from the exact
Kaiser coefficient tensor. Phase-preserving maximum absolute joint-amplitude
errors are 0.2561492360899099 at n=2 and 0.09675450972341836 at n=3. These are
actual errors measured from the untouched source child with its negative phase
gradient, without global-phase alignment. They are not exact-tensor acceptance.
The exact log is `g2-kaiser-quantization.log`.

## Original skips, failures and unclosed boundaries

The complete original suite reports one pass, three original skips and one
original serialization xfail. It skips physical decomposition because none is
declared, exact graph/decomposition comparison because only build_call_graph is
available, and type checking because there is no decomposition. The log is
`g2-kaiser-source.log`.

At one phase bit, the exact source coefficients and tensor exist, but constructing
its declared rotation child raises AssertionError. Native physical binding fails
with a domain ValueError requiring at least two phase bits. Zero-width and
symbolic windows have no admitted physical child. Overflow produces the same
NaNs as the original expression. The exact ideal tensor and the quantized child
remain separate. Q-0622 was initially retained for these physical and original-test boundaries.
The notebook is tracked separately by Q-0620. Q-0621 covers complete source
responsibility mapping with these explicit source limitations.

## Batch checks

The concentrated Kaiser, LP, shared window, rotation-file, table and resource
regression reports 132 passed. Strict mypy covers all four new Python files.
Black, Ruff, separate Simulation and Algorithms scope checks, architecture,
multi-level IR and protected API checks pass. Replacing the numerical owner
with the original coefficient expression preserves both appended instructions
and the independent resource model. Logs use `g2-kaiser-` in the reference
workspace. Simulation and Algorithms changes are committed independently.

## Complete original test-file closure, November 25

The full original test file has one autotester function covering numeric and
symbolic examples in five modes. All ten raw example/mode pairs now run
independently, so an early skip cannot hide the second example. Both construct;
neither declares a decomposition. Count checks remain UNVERIFIED with only a
manual model, and type checks remain NA. Numeric serialization passes; symbolic
serialization fails on original HasLength(2**n). Unchanged source pytest reports
one pass, three skips and one xfail.

Existing native owners cover both exact constructors/signatures, concrete and
symbolic model edges, coefficients, qualified wire transport and bodyless
physical refusal. The quantized child retains its separate gate, full complex,
every-branch, clean-work and resource certificates. Its ideal-tensor error,
one-bit child failure, symbolic limits and source HasLength failure remain
recorded. No exact physical Kaiser root or symbolic full-wire reconstruction is
claimed. Q-0622 closes its complete actual test-file responsibility with these
source failures explicitly disposed. See [the complete test matrix](WINDOW_SOURCE_TEST_ACCEPTANCE.md).
