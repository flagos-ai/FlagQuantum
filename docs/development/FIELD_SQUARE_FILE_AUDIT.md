# Finite-field square whole-file audit

G2-FILE-FIELD-SQUARE audits Q-0521, Q-0522 and Q-0523 against Qualtran v0.7.0
commit `bcb17bb25360ed68ce3bc084645db3e929799389`. All three selected assets and
the direct `gf2_multiplication.py`/`gf_utils.py` dependencies match their frozen
Git blobs. The entire notebook Q-0521 closes. Implementation Q-0522 and original
test Q-0523 remain open for the source matrix/adjoint, bookkeeping counts and
generic admission/autotester/serialization boundaries listed below.

## Ownership, interfaces and work

This is a local single-device correctness workflow. Compiler
`finite_field/field_linear.py` retains the existing Frobenius emitter and exact
mathematical inverse. The shared `_FieldEmitter.linear` and `_field_plu` own
ordered PLU gate emission and virtual output permutation. The new
`linear_resource_model.py` counts the same PLU factors without generating an
instruction stream. Simulation `finite_field_frobenius.py` owns numerical field
matrices and value callbacks. Existing field codecs keep optional galois classes
inside the Simulation boundary. Core retains its existing neutral register
metadata. Examples and Drawer compose these owners.

`gf2_frobenius(target, count, modulus=..., adjoint=...)` receives a nonempty
big-endian word and a nonnegative concrete integer count. The numerical matrix
uses least-significant polynomial coefficient first. The call chain reverses
the word into that coefficient order, applies shared PLU synthesis, then reverses
the logical output order. `FieldDecomposition.outputs` records the resulting
free wire permutation. Fixed-wire consumers explicitly call `bind_field_outputs`
and count its binding SWAPs separately.

The normal path remains `circuit.append(FieldOperation(n, instructions))` through
the existing example adapter. The target is THRU. There are zero additional work
qubits, zero classical slots and zero measurements, giving one coherent branch.
The inverse restores every input amplitude. An extra initially zero spectator
wire remains zero in the full-state tests.

## Complete source definition disposition

The source contains one class with three declared fields, four cached properties,
four operational methods, two example factories and one documentation spec.

| Source definitions | Native owner and disposition |
|---|---|
| `qgf` and converter | Existing `FiniteFieldEncoding` and the fully accepted `binary_field_encoding` converter Q-0524. Native descriptors retain the field degree and polynomial. Source attrs/foreign QGF construction remains within the generic admission boundary. |
| `k`, `uncompute` | Native `count` and `adjoint` emitter parameters. Source defaults map to count one and forward. Numerical callbacks retain the original parameter evaluation. Concrete wire admission uses the existing Compiler checks. |
| `signature`, `bitsize` | Existing neutral `Register`/`Signature` and the field degree. The concrete degree-four and unrestricted symbolic-m factories match the original metadata. |
| `squaring_matrix` | Simulation `field_frobenius_matrix` constructs the mathematical x -> x**(2**count) matrix. Source agreement for forward k=1/2/4 reuses G2-GF06. Source k=0 and non-power-of-two matrix errors retain their original witnesses. The native mathematical matrix is explicitly distinguished from those source results. |
| `synthesize_squaring_matrix` | The shared PLU emitter, exact matrix-based resource declaration, and existing symbolic `source_gf2_square_resource_bound`. Concrete matrices and symbolic shapes retain separate cost accounts. Source adjoint identity and decomposition differ, as recorded below. |
| `build_composite_bloq` | Existing reversal/linear-map/reversal chain with source-order CX gates and free output mapping. Existing literal forward gate-stream checks are reused. Corrected mathematical powers and inverse are separately tested. |
| `build_call_graph` | Native catalogue has one declared binary-linear-map child, matching both original examples. Expanded primitive leaves use the independent concrete PLU count or symbolic m(m-1) bound. |
| `adjoint` | Existing native exact inverse flag and output binding. Source toggles `uncompute`, but the underlying linear source emitter ignores `is_adjoint`. That source failure keeps the parent open. |
| `on_classical_vals` | `field_frobenius_classical_value` retains the field-class assertion, forward exponent 2**count and repeated NumPy square roots for inverse evaluation. All original scalar domains and array/error behavior are checked against unchanged source. |
| `_gf16_square`, `_gf2_square_symbolic` | Native catalogue retains default count one, forward direction, degree four and unrestricted symbolic m. Both signatures and concrete/symbolic models match the original factories. |
| `_GF2_SQUARE_DOC` | Complete native notebook covers field, count, inverse, THRU register, polynomial-basis matrix, examples, signature, call graph and cost topics. The original prose exponent a^(2k) is identified alongside the executed exponent 2**k. |

The mathematical emitter and its existing consumers retain the G2-GF06 behavior.
This batch adds numerical and independent model interfaces around that accepted
implementation. The locked reference files remain unchanged.

## Direct dependency closure

| Dependency | Relevant complete call-chain responsibility | Native disposition |
|---|---|---|
| `gf_utils.qgf_converter` | Native field identity, degree, symbolic degree, polynomial and descending powers | Q-0524 is already closed and its prior evidence is reused. |
| `SynthesizeLRCircuit.matrix`, shape admission and `n` | Concrete coefficient matrix or symbolic square shape | Simulation numerical matrices and existing symbolic width declarations. General foreign array/shape admission stays with Q-0519. |
| `SynthesizeLRCircuit.signature`, `lup`, `build_composite_bloq` | QBit array, first-pivot PLU, upper/lower CX order and final permutation | Existing `_field_plu` and `_FieldEmitter.linear`, with accepted gate-order/output-map evidence. |
| `SynthesizeLRCircuit.build_call_graph` | Concrete off-diagonal L/U ones, or symbolic n*n-n | New `source_binary_linear_resource_formula` shares the authoritative factorization without reading emitted instructions. The existing symbolic formula remains in `field_linear.py`. |
| `SynthesizeLRCircuit.on_classical_vals` and `adjoint` | Matrix multiplication/inverse values and direction flag | Mathematical Frobenius callbacks plus executed native emitter/inverse preserve this consumer's required action. General standalone matrix callback/admission and the source direction/decomposition defect remain with Q-0519 and Q-0522. |

The full multiplication source file Q-0519 contains additional operations and
retains its existing open status. This audit closes the dependency use required
by the selected square workflow without claiming that larger file complete.

## Complete original test mapping

The native AST guard verifies all five original test functions. The unchanged
original run reports **nine passes, three skips and three expected failures** in
1.41 seconds, with one original unregistered slow-marker warning.

| Original function | Entire parameter/assertion disposition |
|---|---|
| `test_gf16_square` | Default concrete factory signature, one-level child graph, concrete PLU cost, original literal forward stream and native notebook execution. Generic count/serialization fixture failures remain below. |
| `test_gf2_square_symbolic` | Unrestricted symbolic factory signature, one linear-map child, T=0 and m(m-1) Clifford model. Symbolic decomposition/count applicability and dtype fixture skips remain below. |
| `test_gf2_square_classical_sim_quick` | Every GF(4) input matches the native scalar callback and mathematical matrix. Independent normal append verifies the executed complex permutation. |
| `test_gf2_square_resource` | Unrestricted m retains the exact symbolic Clifford expression m**2-m and total T zero. Both catalogue models agree with unchanged source. |
| `test_gf2_square_classical_sim` | Every original GF(8), GF(16) and GF(32) scalar is checked. Wider callback checks include degrees one and eight, both directions and counts zero, one, two, three, four and seven. |

The 14 existing G2-GF06 tests retain their original source-defect witnesses,
literal forward gate comparisons, mathematical PLU comparisons, symbolic resource
checks and full-state/inverse acceptance. New whole-file checks reuse these
rather than modifying the reference or replacing its expectations.

## Complete notebook asset disposition

The original notebook has 11 cells, including six code cells. Its entire content,
two example instances, signature and one-level call graph have native counterparts.
The native asset adds an append execution cell and its explanation, giving
13 cells, seven code cells and two SVGs. Source prose and parameter-name differences
are explicitly identified. Native signature table layout, operation names and
linear-map resource labels replace the original presentation.

The complete unchanged source notebook executes in an isolated locked kernel.
Kernel selection and a 60-second cell timeout are test-environment configuration.
The native notebook executes in an independent kernel with imports of Qualtran,
Cirq and Quimb blocked. Its append path verifies all 16 complex amplitudes,
correct output binding and inverse restoration. Both native DOT renders were
visually inspected. This closes Q-0521.

## Separate resource accounts

These are default polynomial fields, with modulus 0x13 at degree four and 0x11D
at degree eight. Alternate polynomial representations can have different PLU
counts while implementing the corresponding field operation.

| Case | Actual emitted PLU gates | Independent concrete model | Shape upper bound | Fixed-wire binding |
|---|---|---|---|---|
| Default GF(16), one square | CX 2 | Clifford 2, T 0 | Clifford 12 | SWAP 1 |
| Default GF(256), one square | CX 12 | Clifford 12, T 0 | Clifford 56 | SWAP 5 |
| Symbolic degree m | Concrete gates unavailable | Concrete PLU factors unavailable | Clifford m(m-1), T 0 | Concrete mapping unavailable |

Actual logical operand width is four/eight. Concrete model logical width is the
same operand width. Symbolic source peak bound is Max(2,m). All rows require zero
additional work and zero measurements. Binding gates remain outside the source
free-permutation PLU count. Runtime statistics are a separate account.

## Central remaining boundaries

| Parents | Existing or reproduced boundary | Owner and resolution condition |
|---|---|---|
| Q-0522, Q-0523 and Q-0519 | G2-GF06 already records k=0 as square instead of identity, k=3 matching k=2, and the linear adjoint emitting its forward stream. Native mathematical powers and exact inverse differ intentionally. | Compiler linear-map and Simulation numerical owners. Preserve immutable original witnesses and the full affected definition mapping. An explicit disposition under the original closure criteria is required before closing the source parents. |
| Q-0522 and Q-0523 | Concrete original count assertion omits QGF Split/Join nodes from its annotation. The linear child and physical CX counts agree. The full unfiltered source diagnostic is retained. | Compiler resource and test owners. Resolve bookkeeping assertion coverage under the original criteria without presenting filtered counts as the original check. |
| Q-0522 and Q-0519 | Native descriptor/word/matrix admission replaces foreign attrs/QGF/NumPy-array object construction. Standalone linear classical callback/serialization and arbitrary foreign shape identity retain their existing parent boundary. | Existing metadata, Simulation and Ecosystem owners. Complete native admission and exact failure/identity mapping while preserving protected schemas. |
| Q-0523 | Three original fixture skips cover symbolic decomposition, count applicability and dtype checks. Two serialization xfails explicitly exclude QGF. The third xfail is the count mismatch above. | Generic autotester and serialization owners. Complete the two factories' exact native assertion mappings and preserve source failure dispositions. |

## Batch validation

- 108 targeted whole-file and reused G2-GF06 checks pass.
- 1,698 concentrated field, polynomial, classical-field and resource checks pass
  in 17.10 seconds with zero skips.
- 91 independent-kernel, CPU and tensor integration checks pass in 19.77 seconds
  with zero skips and two existing tensor API deprecation warnings.
- Strict typing passes on all five changed Python files. Black, Ruff, both domain
  scope checks, architecture, multi-level IR and protected API checks pass.
- The three new production/example modules have 62 statements and 14 branches,
  all covered. The executable numerical path uses existing local CPU execution.

## Complete stored-graph chain — 2026-11-20

The complete `gf2_square.py` definition map above now has a stored graph path.
Compiler `linear_component.py` owns shaped coefficient-bit PLU synthesis and the
exact inverse. It reuses `_FieldEmitter.linear`, with no second PLU algorithm.
`frobenius_component.py` owns the typed MSB word, reversal, one real stored linear
child and typed rejoin. The existing mathematical matrix construction is shared
with `field_linear.py`. Canonical binary-field descriptors retain their original
metadata, with the explicit modulus checked against an encoded polynomial.
There are no new Core contracts, numerical field objects or workspace lifetimes.

`examples.algorithms.register_field_frobenius` binds the stored graph through
`circuit.append(op)`. Consumers follow returned output aliases. Appending its
exact inverse restores the initial physical word. Replacement tests make the
previous public Frobenius physical emitter raise while the append workflow
continues to pass. Source's zero-power/non-power-of-two/adjoint defects above
remain independently reproduced by the unchanged accepted tests. Correct
mathematical behavior remains the native operation contract.

The new tests compare the original unmodified builder calls directly, including
all CNOT operands and output permutations. The Cirq exporter can reschedule
independent CNOTs at degree eight, so its linearized sequence is not used as the
source construction order. Its returned mapping and independent resource model
are still checked. This corrected an initially overstrict total-export-order
assertion. Initial test descriptors omitted required polynomial-field metadata,
which was corrected to the existing canonical descriptor format.

All forward/inverse mathematical powers 0/1/2/3/4 at degrees1/2/3/4/5/8 match
the existing native emitted stream and output aliases exactly. Comparable frozen
forward powers1/2/4 match the original source construction and independent
Clifford estimate. Complete complex amplitudes, entangled spectator and a clean
extra wire are checked for all powers0..4 through degree four, then the exact
inverse restores the initial state. There is one coherent branch, zero
measurements, zero fresh classical SSA, zero extra work and no hidden output
binding SWAPs. Existing immutable source notebook and original-suite evidence,
foreign/autotester/serialization and source model boundaries are reused.

Final concentrated regression, strict checks and resource receipts are recorded
under `ftqc-reference-env/g2-frobenius-stored-*`. Parent Q-0522 and Q-0523 remain
open for the specifically enumerated source/admission boundaries above.

Final validation passed 353 concentrated cases without skips and 17 local
integration cases. Strict typing passed for six files, alongside Black, Ruff,
Compiler scope, architecture, multi-level IR and protected API checks. The
normal append example restores the full complex state. Existing field and
Kickmix binary-linear/Frobenius file evidence is included in this regression.

The final dependency scan has 688 existing whole-repository findings and zero
batch findings. Three direct source blobs match the frozen inventory. New test
NumPy imports use the optional-dependency guard required for dev-only installs.

| Concrete forward case | Emitted CX | Independent source Clifford | Extra work | Measurements / fresh SSA | Binding SWAPs |
| --- | ---: | ---: | ---: | ---: | ---: |
| degree1, polynomial0x3 | 0 | 0 | 0 | 0 / 0 | 0 |
| degree3, polynomial0xB | 1 | 1 | 0 | 0 / 0 | 0 |
| degree4, polynomial0x13 | 2 | 2 | 0 | 0 / 0 | 0 |
| degree8, polynomial0x11B | 12 | 12 | 0 | 0 / 0 | 0 |

The source shape-only bound remains a separate analytical observation. These
counts exclude gates for callers who explicitly materialize output mappings.
No whole-repository success is claimed by the focused validation.
