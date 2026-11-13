# Polynomial addition and coefficient-view whole-file audit

G2-FILE-FIELD-POLY-ADD covers Q-0526 through Q-0534 against Qualtran v0.7.0
`bcb17bb25360ed68ce3bc084645db3e929799389`. All nine files match their frozen
Git blobs. The three complete notebook assets Q-0526, Q-0528 and Q-0532 close.
The three implementation files and three original test files retain the specific
source-count, generic admission and generic test boundaries below. This batch
reuses G2-GFP01 `c7acbcc0`, G2-GFP02 `ede9e65a`, G2-GFP03 `bfc85889`, accepted
field codecs and the existing field-addition audit.

## Interfaces, bit order and ownership

Compiler `finite_field/field_polynomial_addition.py` remains the executable and
aggregate-resource authority. `gf2_polynomial_add_ir(source, target, layout)`
preserves source and XORs corresponding coefficient words into target.
`gf2_polynomial_add_constant_ir(target, coefficients, layout)` consumes descending
encoded coefficients, validates their width and left-pads zeros. Both now accept
keyword-only `adjoint=True` for the literal source adjoint traversal.

`FieldPolynomialLayout` owns the degree-plus-one coefficient view. Coefficients
run from highest power to constant, with big-endian bits within each word.
Noncontiguous wire tuples retain their declared order. Both operations own zero
work wires and zero measurements. All physical wires belong to the caller. There
is one coherent branch, with no measurement feedback to enumerate.

Simulation `finite_field_polynomial_addition.py` owns the original value
callbacks and constant constructor validation. Existing
`FiniteFieldPolynomialEncoding` owns polynomial padding, reconstruction and
nominal field identity. Numerical galois objects stay inside Simulation. Existing
neutral register metadata describes signatures and directions. Examples compose
those owners with the shared call graph and Drawer table owners. Production
imports no Qualtran, Cirq or Quimb modules.

The user workflow remains `circuit.append(FieldOperation(n, gates))`, through the
existing example adapter. The new source-specific register adjoint reverses
coefficient calls, preserving each self-adjoint field child's forward bit order.
The generic `Circuit.adjoint()` reverses the complete primitive stream. Both are
numerically exact. Their literal register-addition streams differ when a
coefficient has more than one bit. Constant addition reverses both coefficient
and bit traversal, matching the full reversed X stream. Neither reference source
nor existing forward emission was changed.

## Complete implementation definition disposition

| Source definitions | Native owner and disposition |
|---|---|
| `GF2PolyAdd.qgf_poly`, `signature`, `is_symbolic` | Existing polynomial codec and neutral metadata. Concrete factory degree 4 over GF(8), symbolic n/m and exact original symbol assumptions are retained. Source symbolic classification inspects polynomial degree. Native wire execution requires a concrete layout. |
| `GF2PolyAdd.build_composite_bloq` | Existing layout split, per-coefficient `gf2_add_ir` and join. Three complete concrete source streams, arbitrary caller binding, preserved output wires and primitive counts reuse accepted G2-GFP02. |
| `GF2PolyAdd.build_call_graph` | Complete native factory graph declares degree-plus-one field-addition children. The original symbolic coefficient width `ceiling(log2(floor(2**m)))` is retained without replacing it by m. Expanded primitive calls reuse `field_addition_calls`. |
| `GF2PolyAdd.on_classical_vals` | `polynomial_add_classical_values` returns the same source object and source-plus-target, including the original permissive plain-number callback behavior. |
| Inherited register adjoint | Explicit source traversal in the existing emitter, compared to unchanged upstream inverse for degree/width 0/1, 1/2 and 4/3. Generic circuit reversal remains independently checked. |
| `_gf2_poly_4_8_add`, `_gf2_poly_add_symbolic`, `_GF2_POLY_ADD_DOC` | Both complete factories, signatures, original parameter/register topics and one-level/expanded graphs in native catalogue and notebook. |
| `GF2PolyAddK.qgf_poly`, `g_x`, `signature`, `is_symbolic`, `_validate_g_x` | Existing codec plus `validate_polynomial_constant`. Checks retain degree-before-field ordering, nominal field identity and symbolic short circuits. Symbolic degree skips both checks. Concrete degree with symbolic field skips field identity. |
| `GF2PolyAddK.build_composite_bloq` | Existing constant emitter and coefficient layout, with left padding and exact bit order. Accepted five constant shapes and three source-specific inverse dimensions are reused or directly exercised. |
| `GF2PolyAddK.build_call_graph` | Concrete graph retains polynomial split/join and all five field-constant children. Symbolic graph retains fresh `_g_x0`, even for the original zero constant polynomial over GF(2). Expanded X edges reuse the existing field declaration owner. |
| `GF2PolyAddK.on_classical_vals` | `polynomial_add_constant_classical_value` executes original addition without introducing constructor checks into the callback. |
| Inherited constant adjoint | Existing emitter's optional reverse stream, with the exact source operand order and output wire mapping. |
| `_gf2_poly_4_8_add_k`, `_gf2_poly_add_k_symbolic`, `_GF2_POLY_ADD_K_DOC` | Concrete coefficients 1/2/3/4/5, symbolic zero factory and both complete graph/signature examples have native counterparts. |
| `GFPolySplit.dtype`, `GFPolyJoin.dtype`, both signatures and `_validate_dtype` methods | Existing concrete layout and codec metadata retain one packed port versus degree-plus-one array entries and opposite directions. Native symbolic-degree rejection reuses layout admission. Lazy symbolic-field metadata stays separate from concrete execution. |
| Both `decompose_bloq`, `adjoint`, `as_cirq_op`, `as_pl_op` methods | Atomic, gate-free views through existing split/join. Wire reshaping and opposite inverse views reuse G2-GFP01. The native view emits no physical operation, matching the source adapters' no-op behavior. Foreign method signatures are covered by the generic boundary below. |
| Both `on_classical_vals` methods | Existing polynomial codec padding and reconstruction. Original degree-four GF(8) values and both callbacks match unchanged source. GF(9) wire reshaping and coefficient evidence reuse G2-GFP01. |
| Both `my_tensors` methods | Existing physical identity view. Both original full 64-by-64 identity tensors match, every packed basis roundtrip is checked and a normal native empty-operation append preserves the complete coherent state. Per-connection foreign tensor wrapper admission remains below. |
| Both `wire_symbol` methods | Native shared table shows `reg` and indexed coefficient lanes with opposite direction and source-equivalent grouping. Native table layout and `reg[i]` spelling explicitly replace the original directional boxes. |
| `_gf_poly_split`, `_gf_poly_join`, both DOC constants | Both complete degree-four GF(8) factories and instructional topics appear in the native view notebook. The source specifies no call-graph example for these atomic views. |

Direct dependencies are disposed through existing owners. Field addition and
constant addition reuse `coefficients.py`, `addition_model.py` and
`FIELD_ADDITION_FILE_AUDIT.md`. The complete converter Q-0524 remains closed.
The polynomial portion of Q-0160 uses the accepted numerical codec and neutral
metadata. Protected executable polynomial Program-port admission remains open.
The ordinary `FieldOperation` append path uses explicit concrete wires.

## Complete original test disposition

The original files contain four, four and five test functions respectively. The
native inventory checks the complete name sets. Unchanged originals report
**18 passed, 12 skipped and seven expected failures** in 1.32 seconds.

| Original function groups | Native assertion mapping |
|---|---|
| Four register/constant concrete and symbolic factory autotests | All four complete signatures, symbolic assumptions, one-level and expanded graph edges/leaves, zero T and exact source Clifford costs. Original generic checks have explicit remaining conditions below. |
| `test_gf2_poly_add_classical_sim` and `test_gf2_poly_add_k_classical_sim` | Original quick values and all four original polynomial-domain entries, tested against both numerical callbacks. The source identity is preserved. Existing gate-level evidence and new coherent append checks supply independent physical validation. |
| Both `test_gf2_poly_add*_resource` functions | Concrete register CX=15 and constant X=7, symbolic exact width expression, T=0 and original register widths. Models and emitted primitives have separate accounts. |
| `test_gf_poly_split` and `test_gf_poly_join` | Both complete factory signatures, opposite directions, unchanged wire binding, zero physical gates and original scalar padding/reconstruction assertions. Generic fixture omission remains explicit. |
| `test_no_symbolic_degree` | Existing native layout rejects symbolic polynomial degree. Source rejection is unchanged. Symbolic field metadata and concrete executable width remain distinct. |
| `test_classical_sim` | Original `[1,2,3,4]` over degree-four GF(8) becomes `[0,1,2,3,4]`; join restores the same polynomial. |
| `test_tensor_sim` | Both entire original 64-by-64 identity tensors, all 64 native packed basis roundtrips and complete complex native append identity. |

Normal append checks use noncontiguous words, a complex spectator and an
initially zero unused wire. All complex amplitudes agree within 1e-12, the unused
wire stays zero and the source-specific inverse restores the full initial state.
Existing degree/width and constant-shape source streams are reused. The new
source inverse checks preserve every operand and output binding.

## Complete notebook asset disposition

The register and constant source notebooks each contain 11 cells and six code
cells. The source split/join notebook contains 14 cells and seven code cells.
Each unchanged original executes independently in the locked reference kernel.
All three files match their frozen blobs before and after execution.

Both native addition notebooks contain 13 cells and seven code cells, adding
normal append, complete coherent-state validation, source-specific adjoint
restoration and separate resource accounts. The native view notebook contains
12 cells and six code cells, combining duplicate imports and adding value/view
roundtrips plus native append. Every original factory, parameter/register topic,
view direction and graphical signature has a corresponding native section.
Both original addition call graphs are retained, including the constant graph's
polynomial view nodes. There are two SVG outputs per notebook.

Independent native kernels block Qualtran, Cirq and Quimb imports. All notebooks
pass and their DOT renders were visually inspected. Shared signature table
rendering now expands array entries and respects input/output directions. Its
existing scalar consumers pass concentrated and notebook regression. Native
names, tuple child identities, table layout and indexed `reg[i]` labels are
explicit presentation adaptations. These complete asset dispositions close
Q-0526, Q-0528 and Q-0532.

## Separate resource accounts

| Operation | Actual emitted primitives | Original source model | Operands and work |
|---|---|---|---|
| Degree-four GF(8) register add | CX 15 | Clifford 15, T 0, measurement 0 | 30 operand qubits, auxiliary 0 |
| Degree-four GF(8), constant coefficients 1/2/3/4/5 | X 7 | Clifford 7, T 0, measurement 0 | 15 operand qubits, auxiliary 0 |
| Degree-four GF(8) split or join | Empty stream | All gate categories 0 | 15 aliased operand qubits, auxiliary 0 |
| Symbolic register add | No executable allocation | `(n+1) * coefficient_width` Clifford | Signature `2*(n+1)*coefficient_width` |
| Symbolic constant add | No executable allocation | Same width upper bound, including original symbolic zero factory | Signature `(n+1)*coefficient_width` |

All inverse primitive counts equal their forward counts. Symbolic coefficient
width is the exact source expression. These operand widths are separate from
recursive graph peak estimates and runtime statistics. Gate-free view nodes
remain visible in declared graphs and emit no primitive gates.

## Central remaining boundaries

| Parents | Explicit remaining behavior | Owner and closure condition |
|---|---|---|
| Q-0527/Q-0531 | Original concrete register count assertion omits two polynomial split and two join nodes. The native/source direct diagnostic preserves the entire mismatch. The constant count assertion passes. | Resource/test owners. Preserve original declarations and complete decomposition evidence until disposition under original closure rules. Physical CX parity does not certify the original count check. |
| Q-0527/Q-0529/Q-0533 | Generic foreign attrs/QGFPoly construction and symbolic coefficient array admission do not yet have complete native executable Program-port replacements. Direct Cirq/PennyLane and foreign per-connection tensor methods are represented by concrete native views, without reproducing foreign object method signatures. Binary executable addition is verified; source-wide nonbinary admission is not claimed. | Existing metadata, Simulation, Compiler and Ecosystem boundaries. Complete native admission, errors and connection-level replacements through approved contracts. Keep protected Program schemas unchanged. |
| Q-0530/Q-0531/Q-0534 | Six original serialization xfails explicitly exclude QGFPoly. Twelve source skips include symbolic decomposition, count applicability, atomic views and dtype-check applicability. Native assertions reproduce physical/value/tensor behavior, but are not a complete generic autotester/serialization replacement. | Generic test/format owners. Supply exact native mappings for all six factories and applicable omitted assertions before closing these parents. Existing metadata codecs do not substitute for these original fixture paths. |

Earlier GF2Square matrix/linear-adjoint and Karatsuba source defects remain in
their existing audits. This batch does not modify or reinvestigate those sources.

## Batch verification

- 50 focused checks, including reused G2-GFP01/02/03, pass.
- 1,723 concentrated field, polynomial and resource tests pass in 17.19 seconds,
  with zero skips.
- 94 independent-kernel, CPU and tensor integration checks pass in 25.08 seconds,
  with zero skips and two existing optional-optimizer/deprecation warnings.
- Strict typing, Black, Ruff, both domain scope checks, architecture, multi-level
  IR and protected API checks pass for the changed surface.
- Both new production/example modules plus the shared signature example owner
  have 99 statements and 34 branches, all covered. Including the existing
  Compiler polynomial-addition module gives 97% across 145 statements and 50
  branches. That existing module is 90% overall, with three pre-existing resource
  admission failures outside this focused coverage selection. Its new adjoint
  paths are fully covered.

Initial diagnostics identified the missing canonical field-child adjoint order,
array signature rendering and a mutable-circuit test expectation. They were
corrected before acceptance. Strict-check namespace/type diagnostics were also
resolved. The frozen reference remains unchanged.
