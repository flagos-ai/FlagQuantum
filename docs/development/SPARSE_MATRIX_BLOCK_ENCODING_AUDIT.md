# General sparse matrix block encoding source files

The shared typed root, scoped constructor and numerical payload paths are
documented in [the constructor audit](RESOLVER_SPARSE_MATRIX_AUDIT.md). Existing
Algorithms execution receipts below retain their scope; symbolic-bound storage,
original private-field loss and original resolver/codec failures are separately
recorded in that audit.

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`.
The frozen files under `qualtran/bloqs/block_encoding/` are Apache-2.0,
copyright Google LLC.

| Item | Source file | Frozen Git blob | Disposition |
| --- | --- | --- | --- |
| Q-0296 | `sparse_matrix.ipynb` | `2a85ed895beeee25d508e68a093af307ca9f1826` | Closed by the full native tutorial. |
| Q-0297 | `sparse_matrix.py` | `cacbdab733f0be4f0f28bf2a749e1bad79987bfd` | Symbolic oracle composition and the original concrete primitive streams verified. Closed: concrete foreign oracle binding, symbolic declaration resolution and complete long-history channel coverage verified below. |
| Q-0301 | `sparse_matrix_test.py` | `afcb0771b405ed26bfce7827d70d81c3d7f7a0d8` | Closed with the original stress, random entry, banded index and Vlasov fixtures. |

## Oracle and block ownership

Row and column oracles act on big-endian ordinal and index words. The
top-left oracle binds the ordinal directly. The banded oracle applies
Add followed by signed AddK and reverses those calls on adjoint. The
entry oracle receives the flag, row and column in that order. Uniform
entries use one Ry with angle twice arccos of the entry. Explicit entries
use the source NumPy angle transform and QROM integer conversion, then
the ordered big-endian controlled Ry sequence, then the self-adjoint
QROM again. Private QROM load and traversal wires return to zero.

The composed block binds an ancillary flag followed by the ordinal word
and has no logical resource port. It prepares the low ordinal bits,
calls the column and entry oracles, swaps the index words, applies the
row adjoint and unprepares the ordinal. The reverse path inverts the
stages and preserves their operands. The caller supplies clean temporary
work and classical slots, reused between sequential stages. Sparse
oracle logic belongs to Algorithms and delegates its gate kernels to the
existing Compiler owners. Width and sparsity checks require a concrete
valid bounded ordinal for physical binding.

## Concrete original variants

| Fixture | Physical emission | Independent locked source model |
| --- | --- | --- |
| Four-by-four uniform 0.3 | Four H, two SWAP, one Ry, no temporary wires or measurements | Six Clifford, one rotation, peak five |
| Two-by-two explicit, 10-bit angles | 16 T/T-dagger, four measurements, 11 temporary wires | Four And, 45 Clifford, ten rotations, four measurements, peak 14 |
| Eight-by-eight banded, half-bandwidth one | 72 T/T-dagger, 18 measurements, five temporary wires | 18 And, 129 Clifford, five rotations, 18 measurements, peak 12 |

Both directions of the three original concrete variants match the
source six-call graph totals, QEC categories, peak and full complex
reference output. The uniform stream checks the explicit SWAP operands,
and the explicit entry fixture checks all measured histories. The
banded fixture checks three representative histories in each direction.
All tested work bits clear. The source primitive stream replay covers
these variants in both directions. The symbolic channel certificate below
covers the complete banded histories.

The unchanged original non-notebook run has 23 passes, five mode skips
and seven expected failures. The unchanged 13-cell original and the
15-cell native tutorial pass independent kernels. The tutorial preserves
four signatures, including the original symbolic shape, and the original
six-call graph. The new append example checks the uniform projected
matrix and exact inverse. The concentrated native run passes 193 cases.

## Original random, banded and Vlasov tests

The native signature declares the same packed system and ancilla order,
including the original symbolic fixture. The symbolic declaration can
be queried without binding wires or constructing data.

The eleven explicit tables, both banded pressure fixtures and the
Vlasov example now execute through append in both directions. For each
case, the locked tensor supplies the whole logical operator. The tests
check the projected input matrix at the original approximation tolerance
and compare all populated complex amplitudes of a coherent input across
the full logical space. All private wires return to zero on the all-zero
and alternating measurement histories. Both directions also match the
independent source QEC category vector and peak. The sparse branch
evaluator uses the exact three-CNOT SWAP identity only for evaluation.
Emitted gate counts are taken before that conversion.

The original three-column top-left example matches the projected matrix
and full coherent outputs at every measurement history in both
directions. The standalone three-bit banded oracle covers both full
operators, all 24 valid index/ordinal pairs and every small measurement
history. The complete enumerations for these bounded cases complement
the numerical long-history samples above; the channel certificate below covers
every long history.

| Vlasov emission | Locked logical model |
| --- | --- |
| 376 T/T adjoint, 94 measurements, seven controlled Ry, 19 reserved wires | 94 And, 94 measurements, 11 rotations, 413 Cliffords, 19 peak qubits |

The seven controlled Ry are the explicit entry rotations. Other
rotations include those from uniform preparation. No T synthesis cost
is inferred from the emitted rotation instructions. Both Vlasov
directions match all 2,096 instructions, including operands, rotation
angles and feedback for each of the 94 measurements. Branch execution
covers representative histories, while the symbolic channel certificate covers
all histories. The unchanged source slow run passes
all four tests.

## Primitive stream replay

The frozen Soquet graph provides stage order and operand wiring for the
original uniform, explicit entry, padded top-left, banded and stress
cases. The source oracle's allocation and free bind the caller's angle
word for both QROM passes. The replay uses the independent uniform
preparation, QROM, Add and signed AddK streams. Swaps and controlled
rotations preserve the original operands and angles. Measured stages
share physical workspace across calls, with classical slots bound to
the owning stage. Each wire is checked in dependency order.

## Symbolic oracle costs

The symbolic estimate composes the existing uniform preparation, index
and entry oracles, and SWAP through the six source calls. Top-left index
oracles contribute zero gates at width `2*n`. Uniform entry contributes
one rotation at width `2*n+1`. Banded index oracles compose the existing
Add and signed AddK formulas and follow the locked symbolic peak fallback.
Callers can supply independent directional formulas for other oracles.

For the original symbolic width `n` and full support, both directions
give `6*n-2` And operations and measurements, five rotations and peak
`2*n+1`. The modern Clifford count is `33*n+6`. The legacy Clifford count
is `45*n+2`. Independent symbolic sparsity also retains the unused
high-order diffusion prefix in its AutoPartition peak. A symbolic index
width with band size one has peak `Max(n+4, 2*n+1)`.

The source singleton diffusion omits its target register, and its
AutoPartition refuses the mismatched ports. Symbolic declaration resolution and concrete foreign oracle binding are verified
below. The source can also reject
some concrete-index counts with unresolved width at oracle construction
when its sparsity bound comparison has no truth value.

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/register_sparse_matrix_block_encoding.ipynb
```


## Ordinary external entry binding

The optional ordinary-entry importer now consumes the original amplitude
oracles through the shared CircuitEntryOracle implementation. The square-root
variant is a semantic subclass and remains independently tested. Original
uniform objects at widths one and two match both directional full complex
matrices, the projected entry 0.3/N, and independent source costs. This also
checks that no square-root convention is substituted and no epsilon field is
required from the ordinary source interface.

Original explicit entries at two-bit angle precision retain the NumPy arccos
integer truncation and the original ordered controlled Ry calls. Both directions
match an independently evaluated quantized matrix on a coherent flag/row/column
and spectator input for every small measurement history, with work restored.
Atomic source wrappers are converted through their original concrete operation;
changed port maps are not silently dropped. Controlled rotations retain their
complete complex phase. Scoped decomposition reservations prevent live-QROM
aliasing and preserve the prior three-work-wire banded fixture.

Concrete binding, the channel certificate and the resolved-declaration checks
below complete Q-0297 within the source and optional-import admission boundaries.

## Complete long-history channel certificate

The test-owned `measurement_certificate.py` checks all 28 original explicit,
banded and Vlasov forward/adjoint streams, including the 94 Vlasov erasures.
It proves every branch has the same coherent operator multiplied by the positive
scalar `2**(-events/2)`, with all declared workspace restored. This is a
compositional channel proof, not enumeration of `2**94` numerical trajectories.

Every non-work input starts as an independent Boolean variable. Exact Boolean
algebra tracks X, Y, CX, CCX and SWAP support; diagonal gates preserve it.
Non-diagonal rotations and H conservatively replace their target with a fresh
independent variable. This enlarges support and can cause a valid stream to be
rejected, but cannot justify a false predicate equality. Each exact four-T And
compute requires a zero target. Each literal measured erase requires its target
to equal the current product of both controls. All declared clean wires must
return to zero. Unsupported instructions, unrecognized feedback, nonliteral fixed
gates and nonfinite angles are rejected. The later LCU closure extends this same
certificate to validated explicit matrices and exact controlled swaps, without
changing the admitted sparse streams or their numeric/resource evidence.

The local compute/erase lemma is independently checked on its complete control
basis, including the positive complex scalar for both outcomes. A coherent
entangled rotation example checks phase preservation. Mutated controls, rotated
predicates, dirty targets, corrupted feedback and leaked work must fail.
Linearity extends the certified identity to arbitrary complex inputs in the
admitted clean-work subspace, including external spectators.

The certificate proves outcome independence and workspace cleanup. It does not
numerically evaluate the common coherent operator. The separate unchanged-source
primitive-stream comparison checks gates, operands, angles and feedback, and the
existing full complex reference checks validate the common action. Numerical
samples remain samples; their coverage is not relabeled exhaustive.

The concentrated batch passes 87 tests; strict typing of the certificate,
architecture boundaries, internal IR and public API checks pass. Primitive
emission and source resource estimates in the tables above are unchanged.

## Symbolic declarations resolved before execution

The unchanged source `SparseMatrix.build_composite_bloq` explicitly raises
`DecomposeTypeError` while width or sparsity is symbolic. Banded index and
explicit precision decomposition have the same concrete-only requirement.
There is no source automatic substitution or binding method to transplant.
FlagQuantum keeps symbolic signatures/formulas independent of concrete physical
operations. The caller evaluates parameter expressions first, then constructs
the concrete oracle and appends its bound operation. No unresolved symbol enters
an emitted circuit, and no new symbolic runtime is needed for source parity.

Six original symbolic configurations are resolved in both directions: full
support, independent count, concrete width/symbolic count, two entries, fixed
band and symbolic band. The tests compare resolved signatures, the complete
unchanged-source primitive stream including operands and feedback, concrete
source models, full complex coherent outputs and clean work. Their channel
certificates cover every history; two numerical histories independently check
the common action. Explicit precision resolution additionally checks the original
NumPy/QROM quantization, both directions and every small measurement branch.

Substituting a number into a symbolic cost formula does not recompute the
concrete decomposition. For example the symbolic full-support declaration keeps
generic uniform preparation calls, whereas a concrete power-of-two preparation
simplifies to H gates. Symbolic fallback peaks likewise remain source formulas.
These are compared to the same source profile separately; the concrete tests
query the concrete source rather than relabeling a substituted estimate as an
emission count.

| Complete source definition | Native owner and evidence |
| --- | --- |
| RowColumnOracle / EntryOracle abstract ports | Algorithms protocols; ordered scalar ports validated at optional Ecosystem import; symbolic block signature tests |
| SparseMatrix constructor, dimensions, alpha, epsilon and signal state | SparseMatrixBlockEncoding and BlackBoxPrepare; original metadata and signature tests |
| diffusion, six calls and physical composition | UniformPreparation/AutoPartition and original source walker; forward/adjoint, stress/Vlasov and complete-history certificate |
| SparseMatrix display | Original `B[SparseMatrix]` label, verified with concrete factories |
| TopLeftRowColumnOracle default count and identity | Native top-left owner, symbolic formula, resolved full/bounded support and source stream |
| SymmetricBandedRowColumnOracle count, classical map, calls and decomposition | Native banded owner and signed AddK/Add dependencies; complete valid index table, both full operators and all small histories |
| UniformEntryOracle Ry | Native uniform entry plus actual external snapshots; full matrices and phase-sensitive outputs |
| ExplicitEntryOracle validation, quantization and load/rotate/unload | Native explicit owner/QROM; original eleven tables, precision resolution and source stream |
| Four example factories and documentation specification | Full native tutorial with four signatures and call graph, independently executed in prior accepted evidence |

Known source/admission boundaries remain explicit: singleton diffusion has a
missing target port; unresolved sparsity comparisons may be undecidable; concrete
native binding requires valid widths, sparsity and clean caller-owned work.
The optional external importer accepts its documented scalar port shapes and
supported concrete primitive decompositions, rejects dirty borrowing and unknown
feedback, and preserves owned output mappings. Closing the source file does not
claim universal import of arbitrary external gate sets or symbolic execution
that the source itself refuses. No frozen source was modified.

Final closure validation: 124 concentrated unit cases and the native sparse
matrix notebook independent kernel pass. Strict typing of the changed
production module, formatting, team scope, architecture, IR and API checks pass.
Prior frozen-source, full tutorial and original slow-suite evidence is reused;
no additional source changes or primitive-count changes were introduced.


The shared support representation now uses a reduced ordered Boolean decision
graph, replacing expanded ANF monomials for nine-bit comparator scalability.
Canonical graph identities retain the same exact Boolean support obligations.
The sparse suite is rerun after this replacement; the original 94-event Vlasov
certificate continues to cover every history, with unchanged emitted resources.
