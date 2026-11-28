# Reversible elliptic-curve addition

The reference is the immutable Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. The batch covers
`ec_add.py`, its tests and tutorial, and their direct arithmetic dependencies.

## Stored constructor and direct dependency closure (2026-11-27)

All seven original constructor identifiers now resolve through native typed
owners. The six stages retain 20, 24, 24, 28, 16 and 61 literal calls at width
three; the ECAdd root retains its six calls and bystanders. Actual original
libraries at levels 0, 1 and 2 resolve with SDK imports blocked. The unchanged
symbolic example remains a declaration; the original five-bit/modulus-seven
example has the complete six-call body. Extra constructor fields and incorrect
signatures reject. A registration is not full source-file acceptance.

Compiler owns one module per stage and the root, with one shared source word-copy
loop. Algorithms continues to own the seven signatures. Ecosystem translates
foreign fields and configures existing predicates, arithmetic and exact inverses.
The required adjoints were added to the existing linear-comparison, modular-add
and modular-double owners. Nine complete forward Programs remain identical to
the pre-batch snapshot. Both signed and unsigned exact comparator inverses have
source evidence, including the complete signed three-bit complex operator.

Coordinates are MSB-first Montgomery words without an embedded modulus in the
outer declarations. Stage one transfers five RIGHT flags; stage two transfers
RIGHT lam. Stage five consumes LEFT lam, and stage six consumes all five LEFT
flags. Kaliski and doubling may physically release and rebind data. Explicit
input-pool ownership and workspace-output transfer remain required; consumers
use the returned output layouts. No ownership guard or public operation signature
was weakened. The existing `circuit.append(op)` path stays available; the typed
replacement is also appended and executed with Qualtran, Cirq and Qiskit blocked.

### Complex channels, cleanup and resource scope

All 81 flag inputs, 243 slope inputs, 486 coordinate inputs, 72 product inputs
(including every dirty three-bit y), and 486 inputs satisfying the slope-release
promise pass. Flag cleanup and the complete root cover all 16 point pairs on the
recorded modulus-three curve. Actual primitive And obligations and lookup
outcome phase coefficients certify every erasure history on these domains.
Two sparse complex executions independently compare the complete output support.
The original control leaves additionally compare primitive operands and feedback
on all 1,024 dirty logical inputs each. Private work and consumed logical storage
return to zero on the recorded promises; logical RIGHT outputs remain live.

The source QROAM adjoint is model-only. Its native measured implementation adds
24 measurement events per width-three product versus the historical reverse
lookup route. The five products in the complete chain add 120 H, 120 measurements,
120 conditional X and 60 conditional Z and remove 60 CX, with unchanged T-family
and CCX counts. There is no asserted upstream physical stream through the absent
lookup-adjoint body. Comparable source calls and primitive dependencies are
checked at their actual boundaries. Lookup phases are retained until actual
feedback cancels them; missing or incorrect feedback fails the certificate.

The following counts are measured from the typed width-three/modulus-three
Programs, with multiplication window one. CCX remains an emitted primitive.
Source default T estimates use four T per Toffoli and CSwap; the literal exact
CSwap implementation uses seven T. Estimates are never substituted for emitted
counts. Source and native live peaks agree in these fixtures.

| Component | Emitted instructions | Emitted T/T† | Emitted CCX | Emitted measurements | Source default T estimate | Source measurements | Native/source peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Flags | 665 | 112 | 0 | 28 | 112 | 28 | 21/21 |
| Slope | 13,742 | 3,056 | 6 | 494 | 2,576 | 470 | 55/55 |
| Coordinates | 3,174 | 544 | 0 | 160 | 544 | 136 | 35/35 |
| Products | 3,257 | 572 | 0 | 191 | 572 | 143 | 27/27 |
| Slope cleanup | 13,529 | 3,016 | 3 | 484 | 2,524 | 460 | 54/54 |
| Flag cleanup | 1,949 | 340 | 10 | 85 | 380 | 85 | 26/26 |
| Complete root | 36,316 | 7,640 | 19 | 1,442 | 6,708 | 1,322 | 58/58 |

### Retained failures and verification

The original modulus-five infinity witness still releases f1=1. A direct sparse
execution preserves the correct coordinates and that dirty physical flag; the
source final-stage classical graph rejects its Free. The lease is located using
the actual flag binding, not the scheduler's release order. The five-bit/window-
five reduction witness still produces y=27 where the one-level callback gives
1. These are retained failures, not successful cleanup. Other original coordinate,
release, parameter, larger-input and symbolic aggregate-model boundaries below
remain unwaived; Q-0457 and Q-0460 stay open.

The final concentrated snapshot passes 175 cases in each actual Cirq 1.7.0 /
Qiskit 2.5.0 and Cirq 1.6.1 / Qiskit 2.0.0 profile, with the real Qualtran 0.7.0
and Kickmix 0.1 Simulator asserted. Local integration passes 145 cases. Twenty
Python files pass strict typing, formatting, owner scopes, language, architecture,
IR and public API checks. Twenty-seven frozen source/test and direct-dependency
blobs match the locked inventory. Prior full
original tests, slow classical matrices and notebook receipts are reused rather
than rerun. `g2-ecc-complete-resources-20261127.json` records all seven inventories;
`g2-ecc-forward-compatibility-20261127.json` records unchanged complete Programs.

## Stage one

The operation reads four big-endian Montgomery residues `a, b, x, y` and
returns them unchanged. It owns five zero-initialized outputs in source signature
order: `f1, f2, f3, f4, ctrl`. The predicates are `a == x`, `b == -y mod p`,
the two zero-coordinate pairs, and all-zero `(f2, f3, f4)`. The temporary negation
and restoration of `y` follows the source order. The caller separately provides
`2n` zeroed scratch qubits and `2n - 1` reusable measurement slots. The
source's all-ones modulus still fails the `p+1` constant load.

The original full test file and unchanged 11-cell notebook have executed. All six native stages and the composed primitive trace have evidence below. The implementation and test rows remain open for the listed source defects and wider-window physical boundaries.

### Stage one evidence

- Bound all seven operation bodies from their locked Soquet connections, including
  the negation/restoration of `y` and both all-zero coordinate checks. Compared
  wire order, gate operands and feedback slots at widths 2, 3, and 5. The
  reference graph permits independent operations to move; the native circuit
  follows source construction order while reusing one workspace.
- At width 3 and modulus 3, all 81 residue quadruples agree with the source
  callback. The per-AND certificate proves equal normalized complex output on
  every erasure branch and zero scratch. Three complete sparse executions on
  reversed wire bindings reproduce the full complex state.
- At width 3 and modulus 5, the emitted stream has 225 CX, 136 X, 84 H,
  56 T, 56 T†, 28 S, 28 conditional CZ, 28 measurements and 28 resets.
  The source model reports 28 AND, 213 Clifford and 28 measurements, with
  a 21-qubit modeled peak. The fixed native layout reserves 23 wires,
  including the five result flags and six scratch wires. The model can
  schedule zero-state allocations at their first use.
- The focused dependency regression passed 1,248 tests. Strict typing,
  formatting, team scope and architecture, IR and public API checks passed.

## Stage two

The native slope stage consumes the two through flags and five big-endian
Montgomery words, and writes the fresh `lam` output. It follows the source
subtractions, Kaliski inverse, dirty windowed multiply, and the two slope
selection loops. The equality predicate stays live through slope selection and
the negative-control update of `f1`. The adjoints consume the original inverse
history and multiplication index/reduction outputs. The original `x` word is
restored to its physical port; integration tests check zero on every work wire.

At width 3 and modulus 3, composing the flag and slope stages on reversed
physical wire bindings matches the source's first two stages for all 243
coordinate/tangent combinations. The per-AND certificate verifies all 498
erasures on this domain; two complete sparse executions compare complex states
with phase-sensitive inputs. The negative-controlled equality was compared
independently for primitive order on each wire, operands and feedback. The
existing component regressions cover the two subtractions, both inverse and
both windowed multiplication directions.

| Width, modulus, window | Native emitted operations | Source model |
|---|---|---|
| 3, 3, 1 | 3,056 T/T†, 6 CCX, 470 measurements, 71 bound wires | 470 AND, 6 Toffoli, 168 CSwap, 3,175 Clifford, 470 measurements, 55 peak qubits |
| 5, 13, 5 | 8,296 T/T†, 10 CCX, 1,194 measurements, 131 bound wires | 1,166 AND, 26 Toffoli, 480 CSwap, 8,407 Clifford, 80 T, 1,166 measurements, 97 peak qubits |

The measured adjoint streams and static workspace reservation explain why
primitive counts and modeled counts need their own columns. The first row has
5,920 CX, 1,746 H, 1,062 X, 470 S, 470 conditional CZ and 470 resets in addition
to the listed T/T†, CCX and measurements.

Two observable source boundaries remain:

- For `n=3, p=3, window=1, f1=ctrl=a=b=lam_r=0, x=y=1`, the high-level callback
  reports `f1=0`; the original one-level decomposition and the native circuit
  both produce `f1=1`, with zero slope.
- For `n=5, p=13, window=5, f1=a=b=lam_r=0, ctrl=1, x=2, y=9`, the source
  Kaliski inverse supplies 5 to the multiplier. Its decomposed multiply on
  `(5,9)` reaches 27 via the locked +p final reduction, and the circuit selects
  27. The stage callback and the one-level graph invoking the high-level child
  method both report 1. The native stage preserves the decomposed behavior.
  This leaves full mathematical curve-addition parity open.

## Stage three

The controlled coordinate update computes the dirty Montgomery product
`x * lam`, subtracts it from `y` under `ctrl`, and consumes the product and
its retained indices and reduction bit in the adjoint. The same layout
helper now serves stages two and three. A second chain copies `a`, doubles
the copy, adds `a`, and conditionally adds the resulting `3a` to `x`.
Add-adjoint and double-adjoint clear the temporary before the final XOR.
The double result uses the source's rotated word. Its former most
significant bit is in the returned work tuple until the adjoint restores
the original ordering.

At width 3, modulus 3 and window 1, the complete gate dependency order,
operands, including the rotated word, and measurement feedback agree with
the locked component bodies. All 243 initial residue/tangent combinations
through the first three stages match the source composition. The channel
certificate checks all 634 measurement erasures and zero work; two sparse
executions check full complex amplitudes on permuted physical wires. At width
4 and modulus 5, 250 inputs exercise both control values and nonzero tripling
with the source slope relation. The zero-control branch preserves both
coordinates and the remaining inputs.

| Width, modulus, window | Native emitted operations | Source model |
|---|---|---|
| 3, 3, 1 | 544 T/T†, 1,454 CX, 408 H, 152 X, 136 S, 136 conditional CZ, 136 measurements and 136 resets; 40 bound wires | 136 AND, 910 Clifford, 136 measurements, 35 peak qubits |
| 4, 5, 1 | 824 T/T†, 206 measurements, 50 bound wires | 206 AND, 1,334 Clifford, 206 measurements, 44 peak qubits |
| 5, 13, 5 | 1,408 T/T†, 312 measurements, 83 bound wires | 284 AND, 16 Toffoli, 80 T, 2,348 Clifford, 284 measurements, 76 peak qubits |

For width 5, modulus 13 and window 5, `ctrl=1, a=b=0, x=5, y=1, lam=9`,
the locked multiplication's decomposed result is 27 with reduction flag 1.
This supplies an out-of-residue operand to the controlled subtraction.
The primitive execution returns `y=6`, while the source stage callback
returns `y=0`. The measured native branch preserves the amplitude and returns
all temporary wires to zero. This is a counterexample for the executable
arithmetic chain outside the multiplication-to-subtraction residue promise.

## Stage four

The native stage copies the slope into a separate word before squaring.
It subtracts the rotated square from `x`, runs the multiplication adjoint
with its original index and reduction outputs, and clears the copy.
A second paired multiplication uses the updated `x`; the product is XORed
into `y` before clearing the multiplication history again. The two products
share the same workspace, and the input slope is preserved.

At width 3, modulus 3, window 1, the locked source component bodies agree
with the complete emitted sequence on wire order, operands and feedback.
The certificate verifies all 72 width-3 and 400 width-4 inputs with residue
`x` and slope and arbitrary `y` against the source decomposition. Sparse
executions compare full complex amplitudes and zero workspace. The first
four stages also compose on permuted wires for all 243 small-domain input
tuples, with all 777 erasures certified and complete sample amplitudes.

| Width, modulus, window | Native emitted operations | Source model |
|---|---|---|
| 3, 3, 1 | 572 T/T†, 1,437 CX, 429 H, 103 X, 143 S, 143 conditional CZ, 143 measurements and 143 resets; 35 bound wires | 143 AND, 789 Clifford, 143 measurements, 27 peak qubits |
| 4, 5, 1 | 964 T/T†, 241 measurements, 45 bound wires | 241 AND, 1,300 Clifford, 241 measurements, 35 peak qubits |
| 5, 13, 5 | 1,964 T/T†, 411 measurements, 77 bound wires | 355 AND, 32 Toffoli, 160 T, 2,991 Clifford, 355 measurements, 65 peak qubits |

Two source behaviors are recorded separately. For `(x,y,lam)=(0,1,1)`
at width 3 and modulus 3, the source callback returns `(1,2,1)` and the
decomposition returns `(1,3,1)` because the final operation is XOR. The
native stream follows the decomposition. For `(12,0,9)` at width 5,
modulus 13, window 5, the physical square is 7 and the updated coordinate
is 5. The lowered product `(5,9)` is 27; the native state therefore has
`y=27` and zero work. The one-level source composition uses the child's
high-level result 1 and yields `y=1`.

## Stage five

The slope cleanup owns a consumed `lam` port and keeps `lam_r` through.
The Kaliski inverse and window multiplication retain their physical output
and history roles until the literal adjoints clear them. The first slope XOR
uses per-bit two-control X gates. The tangent XOR uses a shared predicate
with the scalar control first and the big-endian inverse word second,
active on `(1, 0...0)`. The source then frees the slope. After both
adjoints, the circuit applies controlled negation, adds `a` and conditionally
subtracts `b` in source order.

The source implementations of both slope selectors agree with the native
primitive dependencies, operands and feedback at width 3. The first five
stages were appended on reversed wires for every point pair of
`y^2=x^3+x+1` over fields of size 3 and 5, including infinity and encoded
tangents. All 16 and 81 pairs respectively match the source decomposition.
The channel certificate establishes zero slope and work on every erasure
branch. Four sparse executions also compare the full complex state.

| Width, modulus, window | Native emitted operations | Source model |
|---|---|---|
| 3, 3, 1 | 3,016 T/T†, 3 CCX, 5,842 CX, 1,716 H, 1,040 X, 460 S, 460 conditional CZ, 460 measurements and 460 resets; 69 bound wires | 460 AND, 3 Toffoli, 168 CSwap, 3,121 Clifford, 460 measurements, 54 peak qubits |
| 4, 5, 1 | 5,200 T/T†, 4 CCX, 768 measurements, 88 bound wires | 768 AND, 4 Toffoli, 304 CSwap, 5,190 Clifford, 768 measurements, 69 peak qubits |
| 5, 13, 5 | 8,224 T/T†, 5 CCX, 1,176 measurements, 129 bound wires | 1,148 AND, 21 Toffoli, 480 CSwap, 80 T, 8,307 Clifford, 1,148 measurements, 96 peak qubits |

For `ctrl=1, a=0, b=1, x=y=lam_r=0, lam=2` at width 3 and modulus 3,
the slope promise fails. The source decomposition rejects `Free(2)`. The
bound native wires retain the slope 2 and the primitive execution preserves
its complex amplitude while clearing scratch. The classical stage callback
omits the consumed slope; it cannot certify that it was erased.
The arbitrary-residue intake independently found 32 such rejected tuples
out of 243 after the first four stages.

## Stage six and the composed operation

The last stage unsets the common control, applies the four ordered
three-word zero predicates and restores the right-infinity coordinates via
controlled XOR. Two independent `f1 & f2` conjunction lifetimes control
the coordinate subtraction and addition. The source equality and final
fanout-and-erase sequence act on the five logical flags. The primitive
comparison covers all four three-word predicates and their first-use
control order and feedback. The controlled arithmetic, concatenated
equality and multi-And use the previously verified component bodies.

The user-facing operation appends all six steps with one
`circuit.append(op)`. It binds five Montgomery through registers in source
order and reserves the five flags and slope beside the largest local
scratch requirement. It preserves the actual source release operands when
a source promise fails. The one-level symbolic signature and all six
weighted call-graph edges were independently checked against Qualtran.

The curve (y^2=x^3+x+1) gives two independent input sets. At modulus 3
all 16 point pairs match the affine point formula and clear the flags,
slope and scratch. At modulus 5 the 81 native outputs match the staged
source decomposition, including its actual flag values at `Free`.
The source rejects 18 pairs because a flag reaches `Free` nonzero, and
its staged coordinates disagree with the high-level callback for 12
pairs. For example, infinity plus `(2,1)` returns the right coordinates
and retains `f1=1`. Adding `(0,1)` to `(2,4)` produces `(2,4)`
in the staged source and `(4,3)` in the affine callback. Width 4 has
Montgomery factor 1 modulo 5 in this example. The native primitive
certificate covers all 97 pairs across all erasure branches. Sparse
executions additionally check full complex amplitudes on permuted wires.

| Width, modulus, window | Native emitted operations | Source model |
|---|---|---|
| 3, 3, 1 | 7,640 T/T†, 19 CCX, 15,590 CX, 4,638 H, 2,781 X, 1,322 S, 1,322 conditional CZ, 1,322 measurements and 1,322 resets; 74 bound wires | 1,322 AND, 19 Toffoli, 336 CSwap, 8,783 Clifford, 1,322 measurements, 58 peak qubits |
| 4, 5, 1 | 12,864 T/T†, 24 CCX, 26,048 CX, 7,672 H, 4,109 X, 2,152 S, 2,152 conditional CZ, 2,152 measurements and 2,152 resets; 93 bound wires | 2,152 AND, 24 Toffoli, 608 CSwap, 14,143 Clifford, 2,152 measurements, 73 peak qubits |

## Tutorial and acceptance

The [native tutorial](../../examples/notebooks/register_elliptic_curve_addition.ipynb)
preserves the two source declarations, graphical signatures, one-level call
graph and leaf counts. It also binds the verified small example and reports
emitted operations. The six executable cells and both SVGs pass in a fresh
kernel with reference packages forbidden. This closes the tutorial row.
The implementation and test rows stay open for the observed source defects, wider-window physical arithmetic, independent symbolic aggregate resources and external execution boundaries. The original slow matrix is now covered as described below.


## Complete classical callback file closure — 2026-11-21

The top-level operation and all six stage owners now provide `classical_values`
in source right-port order. Their Montgomery conversion, slope selections,
coordinate overwrites, consumed slope and final flag overrides follow the frozen
`on_classical_vals` methods. The callbacks reuse the native point and Montgomery
owners. Physical `make_on`, signatures and workspace requirements retain their
independent execution path and source release promises.

The original slow matrix supplies all 504 input tuples at widths 7 and 8,
modulus 17 and every dividing window. Each tuple compares all seven native
callbacks against the unmodified source methods and the original one-level
composition. A separate example runs with reference imports forbidden. Both
source files are checked by Git blob identity before the matrix runs.
The unchanged original suite reports 1,099 passed, one notebook deselected and
one expected failure. The expected failure is the original serialization fixture
policy for ECPoint, not a demonstrated serialization error. Existing native
complex-state, measurement-history, workspace and primitive/model evidence is
reused. Numerical callbacks emit no physical gates or workspace, so they add no
primitive count and do not certify erasure.

Actual ECAdd, ECAddR and ECWindowAddR factory intake covers forward, adjoint and
additional-control objects. All nine shallow declarations enter the owned
component library. Deeper paths expose the original NumPy uint8 serializer,
controlled-And decomposition and missing scalar physical decomposition. The
controlled window graph also exposes distinct Montgomery Split/Join source
objects that serialize to the same owned declaration. This is a separate
Ecosystem dependency with a recorded reproducer, not accepted physical execution.
Receipts are `g2-ecc-foreign-intake-20261121.json` and
`g2-ecc-live-collisions-20261121.json` in the reference evidence directory.

Q-0457 and Q-0460 remain open. The callback matrix completes the original slow
logical domain. It does not remove the modulus-5 dirty-release and coordinate
counterexamples, wider-window primitive differences, independent symbolic
aggregate model gap or external graph execution boundaries.

## Parameterized source resource declarations, 2026-11-28

The Algorithms-owned `elliptic_curve_addition_model.py` now implements all seven
build_call_graph responsibilities. It preserves concrete versus HasLength control
patterns, Montgomery dtype metadata, separate moduli, multiplication windows,
forward/uncompute identities, exact Adjoint wrappers and complete structured
CtrlSpec array shapes/values. No new resource node class or IR schema is added;
the common SymbolicCallGraph owns traversal and aggregation. The existing
EllipticCurveAddition operation consumes this model through source_call_graph;
the label-only tutorial graph is preserved exactly.

Six concrete/symbolic profiles compare every ordered direct declaration for all
seven owners, including symbolic integer constants. Independent symbolic leaf
prices verify weighted aggregation. All seven QEC categories additionally agree
with original root/stage queries when their leaf price policy is explicitly
supplied. Those reference rates are declared conformance inputs, not a claim of
independent native default Montgomery pricing. Symbolic width/concrete window
one succeeds upstream; symbolic window itself currently raises TypeError in its
QEC query. An available source formula is not relabeled as an absent body.

Two complete existing flat append traces (n3/mod3 and n4/mod5) retain every
instruction and physical binding wire against published f765ed390. Typed
Compiler bodies and Ecosystem constructors are unchanged, so their 175/165 SDK,
145 integration, all-branch complex/cleanup, 27 frozen dependency and nine exact
Program receipts remain applicable. New graph queries emit no physical gates,
measurements or workspace. Existing typed primitive and declared category tables
above remain separate.

The complete controlled ECWindow root has five source calls, but its generic
controlled RIGHT-output QROAM child explicitly refuses decomposition. Scalar
ECAddR has no source body or QEC decomposition. These refusals were re-observed
without reference edits in g2-ecc-complete-refusals-20261128.json. They cannot
certify a clean controlled gate stream. Parent closure still requires individual
review of native invocation breadth and any successful source numerical domain;
this resource module alone does not automatically accept the source parents.
