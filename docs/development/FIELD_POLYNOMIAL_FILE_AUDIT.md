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

The subsequent [polynomial view source acceptance](POLYNOMIAL_VIEW_SOURCE_ACCEPTANCE.md)
closes Q-0534's complete original assertion/protocol mapping. Proposal 125
supersedes the older nonbinary quantum *storage* refusal, and exact per-connection
view tensor factors now cover binary and nonbinary fields. Q-0533 remains open
for generic/symbolic logical declarations and applicable source bindings.
The classic polynomial tensor-index counterexample is in conditional control,
not the Split/Join factors. The historical table below describes the earlier
batch and does not revoke these later results.

| Parents | Explicit remaining behavior | Owner and closure condition |
|---|---|---|
| Q-0527/Q-0531 | Original concrete register count assertion omits two polynomial split and two join nodes. The native/source direct diagnostic preserves the entire mismatch. The constant count assertion passes. | Resource/test owners. Preserve original declarations and complete decomposition evidence until disposition under original closure rules. Physical CX parity does not certify the original count check. |
| Q-0527/Q-0529/Q-0533 | Generic foreign attrs/QGFPoly construction and symbolic coefficient array admission do not yet have complete native executable Program-port replacements. Direct Cirq/PennyLane and foreign per-connection tensor methods are represented by concrete native views, without reproducing foreign object method signatures. Binary executable addition is verified; source-wide nonbinary admission is not claimed. | Existing metadata, Simulation, Compiler and Ecosystem boundaries. Complete native admission, errors and connection-level replacements through approved contracts. Keep protected Program schemas unchanged. |
| Q-0530/Q-0531/Q-0534 | Six original serialization xfails explicitly exclude QGFPoly. Twelve source skips include symbolic decomposition, count applicability, atomic views and dtype-check applicability. Native assertions reproduce physical/value/tensor behavior, but are not a complete generic autotester/serialization replacement. | Generic test/format owners. Supply exact native mappings for all six factories and applicable omitted assertions before closing these parents. Existing metadata codecs do not substitute for these original fixture paths. |

Earlier GF2Square matrix/linear-adjoint and Karatsuba source defects remain in
their existing audits. This batch does not modify or reinvestigate those sources.

## Complete addition original test acceptance (2026-11-25)

Q-0530 and Q-0531 now have complete original assertion and protocol mappings.
This closes the two four-function test files, not implementation parents Q-0527
and Q-0529. The unchanged complete original selection reports 13 passes, six
original skips and five original xfails. No source fixture or assertion changed.
The concentrated regression passes 54 cases and the independent Cirq 1.6
factory/component selection passes 21. Strict typing, Black, Ruff, Integration
scope, language, links, architecture, IR and protected API checks pass.

| Complete original assertion | Owned acceptance evidence |
|---|---|
| Concrete and symbolic factories in both files | All four actual factories and 20 raw protocols in `test_polynomial_addition_source_protocols.py`; complete typed catalog signatures, one-level/full source graphs and costs in the existing file audit. |
| Register-add classical simulation, including all original polynomial pairs | `test_finite_field_polynomial_addition.py` includes every original polynomial and additional values, preserves the source object, and compares actual callbacks. |
| Constant-add classical simulation and leading zero coefficients | The same numerical owner checks all original values against actual source callbacks; coefficient layout and existing exact primitive/component tests preserve padding and degree order. |
| Concrete and symbolic register-add resources | Complete source/native graph leaves and QEC costs in the file audit; measured default factory has 15 CX and no private work or measurement. |
| Concrete and symbolic constant-add resources | Complete graph/model assertions preserve the seven actual X gates, symbolic width bound and original zero-polynomial symbolic example. A symbolic estimate does not claim a physical allocation. |

The four factories have nine PASS, five FAIL, four NA and two UNVERIFIED raw
observations. All four original QGFPoly serialization failures remain visible.
The concrete register-add count failure is proved separately: its actual direct
graph has exactly two GFPolySplit and two GFPolyJoin nodes absent from its
annotation, in addition to five GF2Addition calls. Both view kinds cost zero
Clifford; the primitive and QEC total remains 15. The constant-add count protocol
passes. Removing view nodes from a count comparison is a separately named
generalization, not a repair of the failing original annotation.

Existing actual-factory tests reuse complete source-word sparse complex
coverage for the 30-bit register adder, two target words per input and an
entangled spectator. The constant adder preserves all 15-bit amplitudes and a
spectator. Both directions have exact operand/dependency streams, typed views,
zero private work and SDK-blocked `circuit.append(op)` execution. Small dense
coherent tests and mathematical self-inverse evidence retain their scopes.

The implementation parents remain open for generic/symbolic logical Program
declarations and applicable foreign construction/codec boundaries. Proposal 125
closed concrete nonbinary raw storage; it does not authorize binary polynomial
arithmetic over nonbinary fields. Proposed Proposal 126 is not yet approved.

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

## Package entrance closure (Q-0525)

The pinned initializer blob `44c084de068fdc8e4581a7c9fe6b6815fecc6c49`
re-exports four classes. The package test verifies the entire export set and
the identity of each definition. Their direct native owners are:

| Export | Native owner | Executable evidence |
| --- | --- | --- |
| GF2PolyAdd | `gf2_polynomial_add_ir` | Flattened source CNOT order and both output wire maps agree at (degree, coefficient width) = (0, 1), (1, 2), and (2, 3) with reversed, interleaved wires. |
| GF2PolyAddK | `gf2_polynomial_add_constant_ir` | The package export supplies the ordered source X operands and output map for the five existing constant cases. |
| GFPolySplit | `FieldPolynomialLayout.split` and `FiniteFieldPolynomialEncoding.to_gf_coefficients` | The package export contracts to the 64-dimensional identity with the corresponding native bit view. |
| GFPolyJoin | `FieldPolynomialLayout.join` and `FiniteFieldPolynomialEncoding.from_gf_coefficients` | The inverse view from the package export has the same identity tensor and preserves every complex amplitude through append. |

The unchanged coefficient owners emit one CNOT per coefficient bit for register
addition and one X per set constant bit for constant addition. Views emit zero
primitives. Source model counts and logical widths are covered by the existing
independent QEC comparisons. All four paths use caller-owned wires and have no
measurement branches. The concentrated regression passed 272 tests. The
implementation and original-test boundaries above remain open for their own
file rows.


### Typed coefficient rebinding under inherited controls

The Compiler wraps an opaque packed word and a typed array of admitted
binary-field coefficients using the existing Partition split/join body.
It preserves the source's two directional `reg` ports, descending
coefficient order and big-endian bits. Source and native graphs each
perform one rebinding and return the same inherited control tuple.
Degree-zero and degree-two cases match in both directions with reversed
physical wire order. The existing Split/Join drawer callback reproduces
the source's blank packed label and indexed coefficient boxes.

The bound programs preserve complete complex amplitudes through append,
including coherent controls. They emit zero gates and have no workspace
events. The independent source model counts zero gates and peaks at
three or eight lines including the controls. The physical view keeps
its coefficient dtype. General QGFPoly Program descriptors, nonbinary
typed field execution, and direct foreign object protocols remain open.

## Complete stored coefficient-call chain — 2026-11-20

The full two addition source files now execute as stored Programs through
`finite_field/polynomial_addition_component.py`. Existing typed coefficient
split/join graphs own the packed view. Each original degree-plus-one addition
is a real call to `field_addition_program`, with shared child definitions.
No new polynomial dtype or executable descriptor is introduced. Packed ports
continue to use the existing opaque view and coefficient ports retain their
canonical binary-field type. The specific packed polynomial admission gap above
remains open. Numerical constructor validation, symbolic source graphs and
original-suite dispositions are reused from the complete definition map.

`addition_component.py` now exposes the exact constant-child adjoint through
its internal `adjoint` argument. It reverses the existing X sequence. Quantum
field XOR retains its source-self-adjoint bit order. Polynomial inverse traversal
reverses coefficient calls and split/join order. Descending zero padding and
all-zero coefficient children remain in the stored call graph. Both directions
return the original physical words with no hidden output swaps, workspace,
measurements or classical SSA slots.

`python -m examples.algorithms.register_field_polynomial_addition` executes the
normal `circuit.append(op)` workflow and restores every complex amplitude of
two packed degree-one GF(4) words. Replacement tests make both previous public
polynomial physical emitters raise while the stored append operation executes.
Full complex states, entangled spectator and a clean wire pass for quantum XOR
and zero/single/multiple encoded constants, in both directions. Degree0/1/4 and
coefficient widths1/2/3/5/8 match the entire previously accepted stream exactly.
Original frozen forward and adjoint circuits are also checked at all five widths
for exact gates and operands on each wire, output aliases and independent QEC
Clifford counts. Source DAG scheduling may reorder disjoint operations. The
source-self-adjoint quantum child and reversed constant child retain their
separate accepted literal traversal checks.

The batch passed 246 concentrated cases without skips and 17 local integration
cases. Strict typing passed for four files, with Black, Ruff, Compiler scope,
architecture, multi-level IR and protected API checks. Final evidence and source
inventories are stored under `ftqc-reference-env/g2-polynomial-add-stored-*`.
Optional source-test Cirq access uses the existing pytest dependency guard.

| Degree / coefficient width | Emitted quantum XOR CX | Source Clifford | Emitted one-word all-ones constant X | Source Clifford |
| --- | ---: | ---: | ---: | ---: |
| 0 / 1 | 1 | 1 | 1 | 1 |
| 1 / 2 | 4 | 4 | 2 | 2 |
| 4 / 3 | 15 | 15 | 3 | 3 |
| 4 / 8 | 40 | 40 | 8 | 8 |

All rows use zero extra quantum work, zero measurements and zero classical SSA.
The quantum operand width is twice the packed polynomial width for XOR and one
packed word for constants. Symbolic models stay independent of these emitted
counts. Generic packed-type, foreign-object, nonbinary and original-autotester
boundaries remain open in Q-0527/Q-0529/Q-0533. This closes their stored binary
execution chain without claiming full parent or whole-repository acceptance.

Five direct frozen source blobs match. The final dependency scan reports 688
existing repository findings and zero batch findings.

## Approved typed packed ports and complete view dependency closure

The user approved Proposal 109 on 2026-11-20. Contract commit `652250cdc`
precedes Core commit `b7c8cfcdd`. The complete frozen split/join source still has
two classes with nine methods each, two factories and their DocSpecs. The
persistent inventory in `g2-packed-field-polynomial-evidence.json` retains all
members and six original source/test blobs. The full definition dispositions
above remain applicable. Existing value, tensor, drawing, factory, notebook,
atomic-decomposition and source model evidence is reused.

Core now admits the existing concrete quantum `finite_field_polynomial`
descriptor. Its first parameter is a built-in nonnegative polynomial degree.
All remaining parameters pass the existing canonical executable binary-field
validator, including optional complete nominal polynomial identity. The packed
quantum width equals coefficient width times degree plus one, and classical
width is zero. Canonical scalar, shaped, empty-array, split/join and linear
ownership paths reuse Program's existing verifier. Classical polynomial storage,
nonbinary fields and unresolved symbolic physical widths remain separate domains.

Compiler adds the explicit `packed_dtype` argument to its existing view and
addition graph builders. The descriptor must exactly match the supplied field
and layout. The opaque default retains the exact stored Program representation,
bound CircuitIR JSON, content hash and output aliases captured before the change.
The typed route stores actual calls to the same split/join and field-addition
children. Existing Cast and control callbacks preserve typed arrays, control
handles and output aliases. The caller continues `circuit.append(op)` through
`examples.algorithms.register_field_polynomial_addition`. Numerical codecs and
foreign metadata projection retain their existing owners.

The complete typed and opaque addition suites compare the original forward and
adjoint primitive order, operands, output aliases and resource models. Full
complex states include a populated spectator and a clean unused wire, and the
source-specific inverse restores every amplitude. Register and constant addition
use both directions. Gate-free views create zero primitives, work allocations,
measurements and classical slots. The empty physical program still needs a
caller-provided spare wire at binding, preserving the existing binder contract.

| Concrete account | Emitted primitives | Independent frozen source model | Quantum ownership |
| --- | --- | --- | --- |
| Typed split or join | 0 | All categories 0 | Aliased packed/coefficient bits, zero scratch |
| Degree-four GF(8) register add | 15 CX | 15 Clifford | 30 operand bits, zero scratch |
| Degree-four GF(8) constant 7 | 3 X | 3 Clifford | 15 operand bits, zero scratch |

The concentrated field, polynomial, native Kickmix, classical-word and type
compatibility lane passes 2,357 cases without skips. The final focused lane
passes 226 cases, including two additional typed control callback cases. The
17 local integration cases, strict typing of nine files, Black, Ruff, team
scopes, architecture, multi-level IR and public API checks pass. Qualtran 0.7.0,
Kickmix 0.1 and its native Simulator are verified. The dependency scan retains
687 whole-repository findings with zero changed-path findings.

A supplemental wider diagnostic lane initially reported eight failures. Four
were polynomial admission expectations superseded by the explicitly approved
contract and are updated with positive binary-quantum and negative
classical/nonbinary assertions. Four generic logical/partition expectations also
fail against the exact pre-batch `961c4eee9` verifier functions. The baseline
receipt records four failures and 36 passes. These existing unrelated tests are
retained and excluded from the concentrated acceptance lane. This acceptance
does not certify a green whole-repository test suite.

This closes the concrete binary packed-port dependency of the full view file
and both addition consumers. Generic foreign construction, connection-level
tensor wrappers, symbolic/nonbinary execution and original autotester/serialization
boundaries remain registered under Q-0527/Q-0529/Q-0533 and their test parents.
Original count omissions and source defects remain in their existing accounts.

## Actual complete polynomial factory handoffs — 2026-11-21

All six original factories in the complete addition, constant-addition and
coefficient-view files now have actual-object handoff evidence. This reuses
all whole-definition, original-test, value, tensor, drawing, notebook and approved
packed-Program evidence above. No production interface or source file changes.

Both original directions of the degree-four GF(8) register addition and constant
addition are independently flattened and compared with the named native emitter.
Every primitive operand, per-wire order and returned output alias agrees in both
ordinary and reversed bindings. Register addition has 15 CX on 30 logical wires.
Constant addition has seven X on 15 logical wires. Their independent source QEC
models match these primitive counts. Binding SWAP, private workspace, feedback
and measurements are zero. There is one coherent branch.

The actual 30-bit addition is checked on a sparse coherent subspace containing
all 32,768 source words, two arbitrary target choices and both spectator values.
The exact destination indices and full complex amplitudes agree, and both source
directions restore the input. This does not allocate or certify the complete
31-qubit dense statevector. Dense all-pairs SDK-blocked native `circuit.append(op)`
is separately checked using the same complete source at degree zero over GF(8).
The exact 15-bit constant factory is executed on its entire dense input space
plus a coherent spectator with optional source SDK imports blocked. JSON
roundtrips and exact inverse recovery are checked. Original polynomial callbacks
independently certify selected packed arithmetic values.

Both actual view factories retain their original atomic-decomposition refusal
and the named emitter's through-port refusal. Their successful handoff uses the
existing approved typed packed/coefficient Program owner. Its complete input and
output shape, side, width, big-endian ordering and physical aliases match the
actual source `as_cirq_op` result. Both bindings preserve every complex amplitude
of the complete 15-bit packed word plus spectator through SDK-blocked append.
Views emit zero primitives, workspace events and classical outputs. Their
independent source QubitCount is 15 and all QEC categories are zero.

All six original factory serializers retain the original QGFPoly/QGF TypeError.
Both symbolic additions retain unresolved physical-width rejection while their
exact symbolic Clifford declarations are independently checked. Original count
omissions, generic autotester/codec mappings and unresolved symbolic/nonbinary
execution remain registered under Q-0527/Q-0529/Q-0530/Q-0531/Q-0533/Q-0534.
This batch does not close those complete file parents.

The concentrated polynomial/field/typed-Program regression passes 305 cases
without skips in 6.64 seconds. Local integration passes 17 cases. Cirq 1.6.1
passes all 11 new actual-factory consumers. Strict typing, Black, Ruff, Ecosystem
scope, architecture, multi-level IR and protected API checks pass. Six frozen
source/test blobs, Qualtran 0.7.0 and Kickmix 0.1 Simulator are reverified. Eight
separate primitive/model rows are retained in
`g2-poly-foreign-evidence-20261121.json`. An initial Cirq-profile command selected
a misspelled test path and collected nothing. The corrected accepted log is
`g2-poly-foreign-cirq16-accepted-20261121.log`.

## Complete polynomial catalog dependency closure — 2026-11-22

The complete `gf2_poly_add.py`, `gf2_poly_add_k.py` and
`gf_poly_split_and_join.py` factory groups now use existing native signatures,
packed execution owners and catalog dependency presentation. All six original
factories remain represented. The two concrete addition roots expose their
complete ordered calls. The two symbolic additions retain their original
`DecomposeTypeError`, and both view roots retain the original atomic refusal.

`examples/algorithms/field_polynomial_decomposition.py` owns Program selection,
whole-definition qualification and drawing composition. It resolves existing
polynomial metadata through the Simulation codec, uses the existing
`field_polynomial_addition_program` with the typed packed descriptor, and reuses
field addition and Drawer glyphs. Coefficients remain descending by degree and
field bits remain MSB-first. Core, Compiler, Simulation, binding and existing
execution-test owners are unchanged from `8cbe5ca13`. Users continue with
`circuit.append(op)`.

### Recursive source comparison and native view helpers

Fresh walks through the original composite instance indices compare every
ordered call, named operand, output, register width, shape and side. The two
concrete roots plus six coefficient additions yield eight exact graphs. All
eight fresh comparable drawings agree. Twenty-four source atomic/refusal
contexts retain their original status. Five complete GF2AddK descendants and
one GF2Addition descendant receive qualified class labels only after literal
whole-definition matching. Changed X primitives and reversed CX operands reject
those labels.

The field catalog publishes 333 contextual children, an increase of 54. Thirty
new child contexts correspond to original reachable definitions. Twenty-four
additional native rebinding helpers lie underneath source-atomic GFPolySplit and
GFPolyJoin calls. Their identities are recorded separately. Those helpers keep
the existing typed packed-to-coefficient representation and emit no physical
gates. Source class decomposition equality is not claimed for the original
atomic views. Their existing independent source `as_cirq_op`, typed port/alias
and complete complex-state identity checks are rerun.

### Primitive measurements and independent resource models

| Concrete source factory | Measured atomic primitive stream | Separate original QEC model | Separate original total-T estimate |
| --- | --- | --- | --- |
| Degree-four GF(8) constant addition | 7 X | 7 Clifford | 0 |
| Degree-four GF(8) register addition | 15 CX | 15 Clifford | 0 |

Both streams contain zero CCX, explicit T-family, measurement, classical-output
or allocation/free events. There is one coherent branch. Atomic view operations
retain the previously verified zero-primitive physical identity. The symbolic
models and physical width refusal stay separate.

The accepted actual-factory execution checks cover the entire 15-bit constant
input space with a coherent spectator, and a sparse coherent subspace of the
30-bit register addition containing every source word, two arbitrary targets
and both spectator values. Exact destination indices and full complex
amplitudes are compared. Small complete dense all-pairs append and SDK-blocked
execution checks remain part of the rerun. This batch does not certify a dense
31-qubit allocation. Clean spare wires and inverse recovery remain separate
from logical output ownership.

### Verification and retained complete-file boundaries

Concentrated polynomial execution/reference regression passes 324 cases without
skips or warnings. Focused recursive graph/refusal/mutation checks pass eight
cases. Local integration passes 29 cases. Strict typing covers all four changed
Python files. Black, Ruff, integration scope, architecture, multi-level IR and
public API checks pass. Ten frozen polynomial package/source/test/notebook blobs
match their ledger Git hashes. Qualtran 0.7.0 and Kickmix 0.1 Simulator are
verified. Eleven previously accepted execution-test files are unchanged.

The initial resource receipt incorrectly counted only CX toward Clifford for
the constant factory. The corrected receipt counts X and CX separately and
compares their total against the original Clifford model; its initial diagnostic
is retained. Production execution and gate counts required no change.

Original generic admission/autotester/serialization, count omissions and
symbolic/nonbinary physical-execution boundaries remain registered under
Q-0527/Q-0529/Q-0530/Q-0531/Q-0533/Q-0534. This catalog closure does not close
those file parents or Q-0120. Parent acceptance counts remain unchanged.
Receipts use `g2-field-polynomial-ui-*-20261122` under the reference evidence
directory.

The final complete field/polynomial catalog audit visits all 22 original catalog
factory contexts. Every one of the 14 original source-success roots has a stored
native graph; all eight original symbolic/atomic root refusals remain explicit.
Across those concrete factory contexts the shared original observer records
100 exact recursive graphs, 224 original atomic/refusal contexts and three
explicitly distinct native exact linear inverses. This inventory includes the
duplicate constant-multiplication factory context and does not infer uniqueness
from those context counts. No successful catalog root is omitted.

The complete catalog regression passes 771 cases with three existing selection
exclusions and fourteen existing notebook escape warnings in 390.64 seconds.
Combined inventory is 4158 JSON files, 4157 diagrams, 234 documents, 3804
contextual children, 86 decomposed roots, 267 atomic interfaces and 433 qualified
descendants. The three qualified native exact inverse exceptions across earlier
field groups remain explicitly distinct from source equality. Whitespace
checks pass, and unrelated untracked application work stays outside this commit.
