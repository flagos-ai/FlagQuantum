# Binary field multiplication whole-file audit

The batch covers Q-0518, Q-0519 and Q-0520 at Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. The reference remains immutable.

## Ownership and interfaces

Compiler owns field, binary-polynomial and linear gate emission, output binding,
explicit scratch and source resource declarations. Simulation owns optional
galois values, polynomial callbacks and numerical matrices. Examples compose
these owners through the existing `circuit.append(op)` protocol. These are
single-device correctness paths. No public API or IR schema changes are needed.

Field words are big-endian at the native boundary. Polynomial coefficient arrays
in the original source are little-endian and are reversed explicitly in tests.
The multiplication result is a clean caller-owned word unless quadratic
`plus_equal_prod=True` admits an arbitrary target. Input words are preserved.
Recursive polynomial scratch starts and ends at zero. Output permutations are
free logical relabelings until `bind_field_outputs` materializes them. Classical
measurement/parity slots and product retirement follow the existing erasure
owners, with reset policy recorded independently.

The implementation inventory includes all eleven classes, their fields,
properties and methods, nine example factories, seven document specifications
and the private equality converter. The original test inventory and the entire
60-cell notebook are audited together. Existing accepted gate, full complex
amplitude, measurement-branch and cleanup evidence is reused. Remaining generic
and source-defect boundaries are recorded below at batch acceptance.

## Complete definition disposition

The implementation has 81 methods across eleven classes. All definitions are
included in the following map. Properties are grouped with their dependent
methods so their owner and closure condition remain visible together.

| Source definitions | Native owner and complete disposition |
|---|---|
| `_data_or_shape_to_tuple` | Source equality flattens concrete matrices and retains a symbolic shape object. Native resource keys use immutable tuples. Admission of arbitrary foreign NumPy/Shaped objects remains a generic boundary. |
| `SynthesizeLRCircuit.matrix`, `is_adjoint`, `__attrs_post_init__`, `signature`, `n` | Native concrete square-binary matrix admission is in `linear_resource_model`; field widths and source-order array lanes use existing register metadata. Symbolic shape bounds reuse `source_gf2_square_resource_bound`. Native matrix tuples replace foreign attrs identity. |
| `SynthesizeLRCircuit.lup`, `build_composite_bloq`, `build_call_graph` | Existing `_field_plu` and `_FieldEmitter.linear` retain first-pivot PLU, upper/lower CNOT order and free output permutation. The independent PLU model counts off-diagonal ones. Accepted Q-0519.01/.05 and field-square evidence is reused. |
| `SynthesizeLRCircuit.on_classical_vals`, `adjoint`, `__str__` | Simulation `binary_linear_classical_value` retains forward/inverse finite-field matrix action. Native executable inverses are exact. Source toggles a flag and its display dagger while the gate recipe ignores the flag. That original discrepancy remains open. Native presentation names are an explicit adaptation. |
| `GF2Multiplication.qgf`, `plus_equal_prod`, `signature`, `bitsize` | Existing field encoding/converter and native signatures retain THRU factors and RIGHT or THRU result. Both original factories preserve the unrestricted symbolic degree and its original computed bitsize expression. |
| `GF2Multiplication.build_composite_bloq` | `gf2_multiplication(algorithm="quadratic")` retains the two Toffoli ladders and reduction shift. The new `plus_equal_prod=True` path adds the original inverse shift before those ladders, enabling arbitrary targets. No scratch or measurements are used. |
| `GF2Multiplication.build_call_graph`, `on_classical_vals` | Independent source gate model and Simulation typed product callback, including all small target values. Source declaration omits allocation/split/join. A physical count match does not close an un-generalized bookkeeping-count assertion. |
| `Parity.n`, `signature`, `on_classical_vals` | Existing owned Parity component, executable classical XOR and numerical parity/tensor owner. Accepted Q-0519.04/.07/.10 is reused. Source has no independent gate or tensor recipe. |
| `GF2MulMBUC.qgf`, `signature`, `bitsize`, `reduction_matrix_q` | Existing product-retirement Program owner and numerical `field_scale_matrix` retain factor THRU/result LEFT and the x**(m+i) reduction columns. Numerical tests include all declared concrete widths. |
| `GF2MulMBUC.build_composite_bloq`, `build_call_graph`, `adjoint` | Existing MeasureX, classical-controlled CZ, owned Parity and Discard chain, including measurement order and feedback. Accepted Q-0519.04/.08/.09 reuses exact original builder comparisons and exhaustive small branch checks. Source model declares only m**2 CZ. Explicit reset/retirement policies and fresh/reused classical storage remain separate physical accounts. Automatic generic root-adjoint dispatch remains an admission boundary. |
| `GF2MulK.dtype`, `const`, `galois_field`, `__attrs_post_init__`, `m_x`, `n`, `qgf`, `_const`, `from_polynomials` | Numerical field construction and constant action stay in Simulation. Two native catalogue factories retain the same GF(8), modulus 11 and constant 5. Existing Compiler rejects zero as a noninvertible in-place map, while numerical multiplication by zero is defined. Foreign constructors and invalid symbolic constant admission remain explicit generic boundaries. |
| `GF2MulK.reduction_matrix_q`, `signature`, `on_classical_vals`, `build_composite_bloq`, `build_call_graph` | Numerical constant matrix/value callback, existing `gf2_constant_multiplication`, independent PLU model and one-linear-child native graph. Accepted Q-0519.05 preserves literal ordered CX dependencies, output mapping and all nonzero constants at accepted widths. |
| `MultiplyPolyByOnePlusXk.n`, `k`, `__attrs_post_init__`, `l`, `signature` | Existing concrete polynomial owner admits positive widths and k in [1,n+1], with target width 2k+max(0,2n-k-1). The source constructor only checks k>0. Symbolic executable and wider-k constructor admission are retained boundaries. |
| `MultiplyPolyByOnePlusXk.on_classical_vals`, `build_composite_bloq`, `build_call_graph` | Simulation coefficient callback preserves little-endian arrays, original field polynomial arithmetic and list-valued result. Existing emitter retains reversed upper CNOT ladder, polynomial child and forward restoration. The independent declaration graph matches every recursive source edge at widths 1 through 9 and every declared k. Accepted Q-0519.06 supplies complete complex state and scratch cleanup. |
| `BinaryPolynomialMultiplication.n`, `signature`, `k` | Existing positive concrete polynomial word layout, 2n-1 target and ceiling-half split. Native boundary is big-endian, with explicit reversal against source coefficient arrays. Symbolic direct-call models retain source recurrence identities. |
| `BinaryPolynomialMultiplication.on_classical_vals`, `build_composite_bloq`, `build_call_graph` | Numerical callback, existing `_PolynomialEmitter.polynomial` and `RegisterWorkspace`, plus independent recursive graph. Odd recursive width allocation/free and clean auxiliary ownership reuse Q-0519.06. Powers of two through 1024 match all original Toffoli/Clifford cost assertions without gate construction. |
| `GF2ShiftLeft` and `GF2ShiftRight.qgf`, `k`, `n`, `signature`, `degrees`, `gf`, `_power_2` | Existing concrete monomial emitter and field descriptor retain polynomial degree, nonzero coefficient ordering and inverse direction. Numerical callback preserves degree-one and zero-shift identities, symbolic exponent failure and field-class assertions. |
| Both shifts' `on_classical_vals`, `build_composite_bloq`, `build_call_graph`, `adjoint` | Existing `gf2_monomial_multiplication` retains rotation and CX order in both directions. Two native degree-five/k=3 factories and all direct graph children match source. Source shape-only model assumes a pentanomial. Q-0519.05 provides gates, complex amplitudes and resource evidence. Generic foreign construction remains open. |
| `_GF2MulViaKaratsubaImpl.qgf`, `n`, `signature`, `k`, `m_x`, `build_composite_bloq` | Existing `_FieldEmitter.multiply` composes polynomial products, constant scale/inverse and monomial shift in source order, using an exact inverse for the defective source linear adjoint. The accepted clean-output path and its scratch cleanup are reused. A generic THRU h interface for arbitrary initial values is still open. |
| `GF2MulViaKaratsuba.dtype`, `m_x`, `__attrs_post_init__`, `n`, `gf`, `qgf`, `signature`, `k`, `_GF2MulViaKaratsubamod_impl` | Native field encoding, clean result and recursive emitter. Source rejects concrete degree one. Native degree-one CCX and its model remain an explicitly broader existing operation. |
| `GF2MulViaKaratsuba.__str__`, `build_composite_bloq`, `build_call_graph`, `on_classical_vals`, `adjoint` | Native names, existing clean product, independent exact concrete and shape-only models, typed product callback and explicit measurement erasure. Accepted Q-0519.01/.02/.04/.08/.09 preserve all prior scope. Automatic generic root selection and unchanged-source inverse gate parity remain open. |
| All nine decorated factories and seven `BloqDocSpec` objects | `field_multiplication_catalog` and the complete native notebook retain every original instance, parameter, register topic, signature and one-level edge/leaf count. The original plus-equal prose typo is corrected to z+xy and disclosed. Native tuple graph keys, names and table layout replace foreign presentation objects. |

## Whole original test disposition

All 42 original test functions, their parameter sets and assertions were reviewed.
The following groups map them to native checks and accepted evidence.

| Complete original test group | Disposition |
|---|---|
| Two example autotesters | All factory signatures and complete one-level graphs are tested natively. The original autotester's generic decomposition/count/serialization fixture remains separately open. |
| Three linear synthesis tests, including slow and symbolic resources | Existing literal PLU checks and new full numerical forward/inverse assertions cover widths 2/3/4/5/6/8. Original shape n(n-1) resource evidence is reused. The source gate-adjoint defect is preserved. |
| Four quadratic product/plus-equal classical tests | New typed callbacks cover every product pair at widths 1–5 and every triple at widths 2–3. New normal append checks cover every complex input/target amplitude, entangled spectator, clean unused wire and double restoration at widths 2–3. Literal plus-equal gates and free output maps match widths 1/2/3/4/5/6/8. Existing clean multiplication evidence is reused. |
| Five constant multiplication tests | Numerical callback/matrix checks and existing all-constant literal PLU/decomposition/count/invalid-input checks are reused. Both polynomial/value factories are covered. Native zero-unitary rejection is explicitly distinguished from numerical zero scaling. |
| Notebook test | The immutable original 60-cell notebook passes a separate Jupyter kernel. The native 62-cell notebook passes a kernel blocking reference imports, retains 32 code cells and renders fourteen SVGs. PNG previews of every signature/graph section were visually inspected. |
| Four scaled-polynomial tests | Existing complete source gate/dependency/operands, arbitrary-target complex states, double accumulation and clean work evidence is reused. New recursive graphs and all two-bit value callbacks include zero factors and every target at every admitted k. The raw original wide slow sampling test remains incomplete in the bounded run below. |
| Five binary polynomial tests | Existing full decomposition/gate/state/work evidence is reused. New complete recursive graphs at widths 1–9 and all original powers-of-two cost assertions through 1024 agree. |
| Eight forward/inverse shift tests | Existing original gate/operand/output/count evidence and new typed scalar callbacks cover all original polynomial domains and k=0/1/2/3/4. Normal monomial inverse restoration is reused. |
| Eight Karatsuba tests | Existing clean-output gates under the explicitly repaired reference, all eight-bit input pairs with sparse complex amplitudes, recursive work cleanup and source-defect witness are reused. New independent cost models agree at widths 2/3/4/5/6/8. No unchanged-source mathematical gate agreement is claimed. |
| Two measurement erasure tests | Original non-slow tests execute. Existing stronger exhaustive small measurement histories, complex spectator states, logical retirement/explicit reset variants and eight-bit source instruction comparison are reused. Eight-bit full branch statevector enumeration remains outside the accepted scope. |

The untouched original non-slow/non-notebook selection reports 276 passed,
3 skipped, 3 xfailed and 23 deselected in 6.28 seconds. An additional run including
slow tests was stopped at 113.11 seconds while processing wide polynomial
classical simulation, reporting 79 passed, 3 skipped, 3 xfailed and one notebook
deselected. This run is incomplete. It does not replace the completed native
regression or the separately passing original notebook. No original assertion or
reference source was modified to obtain these results.

## Reused dependencies and retained boundaries

Q-0519.01/.02 (`4565a11c`), .04 (`6309e0fe`), .05 (`26209f50`),
.06 (`403394cf`), .07 (`380ea648`/`5e505764`),
.08 (`fdcd859c`/`8c55471f`), .09 (`61ec15e6`) and
.10 (`dd2065af`) remain the accepted detailed evidence. Historical .03 retains
the superseded 151-CZ witness, while executable classical parity needs 64 CZ.
The field converter Q-0524 is already closed. Register/workspace, Toffoli,
measurement, classical feedback, Parity/Discard and tensor dependencies keep
their existing authoritative owners.

| Parent and remaining gap | Closure condition |
|---|---|
| Q-0519 source linear adjoint and dependent Karatsuba/inverse chains | Resolve the source inconsistency under the pinned-source policy. Preserve the unchanged GF(4) 1*1 -> 0 witness and identify repaired-reference results separately. Equal counts alone cannot close this gate mismatch. |
| Q-0519 generic Program admission, foreign constructors, symbolic execution and the internal arbitrary-target THRU h interface | Complete native endpoint/ownership/admission mapping through existing contracts, including source failure behavior. Concrete clean words and source models do not imply a universal Bloq interface. |
| Q-0519 generic root inverse dispatch and matrix/hash identities | Bind the existing product and measurement erasure owners through the generic executable path, with all adjoint identities and retirement policies validated. Native immutable tuple keys remain an explicit representation adaptation. |
| Q-0520 original autotester and complete slow original suite | Complete native replacements for generic fixture assertions and retained source-specific failures. Reuse existing wide numerical/gate evidence. The stopped slow run remains incomplete and must not be described as passing. |
| Q-0519 symbolic/wider-k polynomial construction and full eight-bit branch enumeration | Preserve the supported concrete range and source exception behavior. Symbolic models are estimates. Eight-bit acceptance currently covers the original instruction/feedback stream and count scope, with exhaustive state branches at small sizes. |

Q-0518 is closed as a complete asset migration. Q-0519 and Q-0520 remain open
under the conditions above. This batch adds no new child progress units.

## Primitive, source-model and runtime accounts

| Operation | Actual free-output primitives | Explicit fixed binding | Source QEC categories |
|---|---|---|---|
| GF(8) quadratic clean | 9 CCX + 3 CX | 0 SWAP | 9 Toffoli + 3 Clifford |
| GF(8) quadratic accumulate | 9 CCX + 6 CX | 0 SWAP | 9 Toffoli + 6 Clifford |
| GF(8) Karatsuba clean | 7 CCX + 26 CX | 2 SWAP | 7 Toffoli + 26 Clifford |
| GF(256), modulus 283, quadratic clean | 64 CCX + 24 CX | 0 SWAP | 64 Toffoli + 24 Clifford |
| GF(256), modulus 283, quadratic accumulate | 64 CCX + 48 CX | 0 SWAP | 64 Toffoli + 48 Clifford |
| GF(256), modulus 283, Karatsuba clean | 27 CCX + 200 CX | 4 SWAP | 27 Toffoli + 200 Clifford |
| Eight-bit lowered product erasure | 64 conditional CZ + 8 H + 8 measurement + 8 conditional X + 8 classical XOR | Fixed output retirement/reset policy | Source only declares 64 CZ |

The listed field products use zero additional quantum scratch at these widths.
Odd recursive polynomial widths can require clean scratch as recorded in the
existing workspace contract. The eight-bit flat erasure uses eight measurement
slots and one reused parity slot. The owned Program variant retains fresh
classical parity slots, a separately accepted storage policy. No T gates appear
at this undecomposed CCX primitive level. Toffoli synthesis costs and sampled
runtime trajectories belong to separate accounts. The notebook's sixteen
trajectories are sampled execution statistics, independent of exhaustive small
branch acceptance.

## Batch acceptance

- Four frozen Git blobs, including the direct field converter, match inventory.
- 76 focused checks pass in 7.03 seconds. The new numerical/model/catalogue scope
  measures 159 statements and 56 branches at 100 percent coverage.
- 1799 concentrated regression checks pass in 20.13 seconds, with zero skips.
- 95 independent notebook/CPU/representation/TN integration checks pass in
  28.90 seconds, with zero skips and two existing integration warnings.
- Strict typing covers all eight touched Python files. Black, Ruff, both domain
  scopes, architecture, multi-level IR and the protected API snapshot pass.
- Source gate/feedback/complex-amplitude/work-zero evidence remains valid through
  the concentrated regression. Source categories, emitted gates and runtime
  samples are reported independently.

The module commits are `cecd0f01` for Simulation, `e15693de` for Compiler and
`03726ce4` for the complete asset/catalogue/source-graph audit.

## Complete stored-call dependency closure — 2026-11-20

The complete eleven-class source file now has owned stored execution chains for
its admitted concrete domains. The original 81-definition, nine-factory and
seven-document inventory above remains authoritative. This batch reuses accepted
numerical, catalogue, notebook, symbolic-model and original-test evidence.
`g2-field-multiplication-stored-inventory.json` records the unchanged file blob
`1338bb4b9f09a0a4b813a1124481e26ee3446cd8` and every class/member disposition.
The source and five direct dependency/test blobs match the frozen intake.

| Complete source owner | Stored native owner and dependency closure |
| --- | --- |
| `SynthesizeLRCircuit` | Existing `linear_component.py`, shared PLU and exact native adjoint |
| `GF2ShiftLeft` and `GF2ShiftRight` | `monomial_component.py`, original sequential rotations, CX dependencies and gate-free returned aliases |
| `GF2MulK` | `constant_multiplication_component.py`, shared `_constant_matrix` authority, one real stored PLU child |
| `BinaryPolynomialMultiplication` | `polynomial_product_component.py`, recursive shared calls, source coefficient order and owned odd-width high padding |
| `MultiplyPolyByOnePlusXk` | Same polynomial owner, overlapping CX ladders and shared recursive child |
| `GF2Multiplication` | `multiplication_component.py`, RIGHT output or arbitrary THRU target, exact LEFT uncompute and both original Toffoli triangles |
| `_GF2MulViaKaratsubaImpl` | `karatsuba_component.py`, real polynomial, scale-adjoint, shift and forward-scale children |
| `GF2MulViaKaratsuba` | Same owner, allocated RIGHT result and original inline implementation shell |
| `GF2MulMBUC` | Existing `product_erasure_program.py`, shared MeasureX/Parity/classical CZ/Discard chain |
| `Parity` | Existing shared classical component, executable parity slots and numerical owner |

`field_word.py` validates existing canonical descriptors against the explicit
binary modulus. `_polynomial_component_calls.py` shares the CX/CCX leaves and
coefficient-to-word view calls across polynomial, quadratic and Karatsuba owners.
All field words remain MSB-first. Polynomial array index zero is the constant
coefficient. A RIGHT result transfers to the caller as logical data. Only the
original recursive padding is freed as clean scratch. There are no physical
routing SWAPs in these stored graphs. The existing fixed-output binding remains
a separate consumer choice.

### Comparable source calls and coherent execution

Unmodified original construction is walked through CX/CCX leaves, bookkeeping
aliases and actual allocation/free connections. Positive and negative shifts,
constant forward scaling, both quadratic modes and their exact inverse, and
both recursive polynomial operations retain every wire's gate dependencies and
operand ordering. Widths 1/2/3/4/5/8 cover field calls. Polynomial widths 1/2/3/5/8
cover shifts 1/n/n+1 and explicit odd-width temporary lifetimes. Disjoint DAG
rescheduling is allowed only when each wire's ordered operations agree.

The original Karatsuba implementation builder also reconstructs the identical
native stored program through an adapter for its actual children. This verifies
root child order, operands, coefficient views and returned aliases. Its scale
adjoint is explicitly mapped to the already accepted exact native inverse.
This root-construction comparison does not establish unchanged-source physical
agreement inside the defective linear-adjoint child. The unmodified GF(4)
1*1 -> 0 witness remains independently asserted. Existing mathematical emitters
match the complete new native instruction streams and output aliases, including
odd-width scratch and the eight-bit case. No reference monkeypatch is used in
the new comparisons.

Every nonzero constant in fields through degree four is checked on complete
complex states, entangled spectator, clean unused wire and exact restoration.
Quadratic clean products and arbitrary-target accumulation cover all inputs
through degree three, with full complex amplitudes and exact inverse restoration.
Recursive polynomial tests include every target value through width three for
the listed small cases and verify padding zero. Karatsuba append covers all
operand pairs through degree three and preserves clean work. Previous wider
sparse complex-state and native Kickmix evidence is included in regression.
The append test replaces the old public physical product emitter with a raising
stub while executing the unchanged consumer.

Both multiplication algorithms compose directly with the existing measured
product erasure. For degrees two and three, every measurement history matches
the complete unnormalized logical complex state with entangled spectator.
Retirement retains the projected physical outcome. The separately requested
reset policy returns those wires to zero. All unmeasured padding remains zero.
The source public Karatsuba adjoint continues to map to product erasure.

### Emitted storage and source estimates

| Width and modulus | Product | Emitted primitives | Peak live quantum | Extra clean padding | Classical SSA | Independent source model |
| --- | --- | --- | --- | --- | --- | --- |
| 2, 7 | Quadratic | 4 CCX + 2 CX | 6 | 0 | 0 | 4 Toffoli + 2 Clifford |
| 2, 7 | Karatsuba | 3 CCX + 9 CX | 6 | 0 | 0 | 3 Toffoli + 9 Clifford |
| 3, 11 | Quadratic | 9 CCX + 3 CX | 9 | 0 | 0 | 9 Toffoli + 3 Clifford |
| 3, 11 | Karatsuba | 7 CCX + 26 CX | 9 | 0 | 0 | 7 Toffoli + 26 Clifford |
| 5, 37 | Karatsuba | 17 CCX + 83 CX | 16 | 1 | 0 | 17 Toffoli + 83 Clifford |
| 8, 283 | Quadratic | 64 CCX + 24 CX | 24 | 0 | 0 | 64 Toffoli + 24 Clifford |
| 8, 283 | Karatsuba | 27 CCX + 200 CX | 24 | 0 | 0 | 27 Toffoli + 200 Clifford |
| 8, 283 | Owned product erasure | 64 guarded CZ + 8 H + 8 measurements + 8 classical XOR | 24 including consumed result | 0 | 16 fresh slots | 64 Clifford, zero declared measurements |

The product has zero measurements and SSA slots. The owned erasure uses eight
measurement and eight fresh parity slots. The existing flat emitter's nine
reusable slots remain a distinct policy. The reset profile adds eight reset
instructions. CCX is the emitted primitive boundary here. No T synthesis count
is inferred from the source Toffoli category. Peak storage excludes the binder's
unused reserve and includes the allocated logical result. Source wrapper
bookkeeping and measurement omissions remain visible.

### Validation and remaining gaps

Final concentrated regression passes 2178 cases without skips. Focused new
stored-chain tests pass 140 cases. Local construction/artifact integration passes
17 cases. Strict typing passes all 14 touched implementation, example and test files. Black, Ruff, team scope, architecture, multi-level IR and public API checks
pass. Qualtran 0.7.0 and Kickmix 0.1 with native Simulator are verified. Dependency
policy retains 688 whole-repository findings with zero findings in this batch.
The first concentrated run found an existing QROAM replacement test patching a
removed module alias. Its replacement now targets the shared measurement owner,
retaining its full instruction equality and call-count assertions. The final
complete concentrated run passes. Prior whole-repository failures stay recorded
in their original audit and are not superseded by this focused acceptance.

Q-0519 and Q-0520 remain open for the unchanged linear-adjoint/Karatsuba defect,
arbitrary internal h, automatic generic inverse/foreign/serialization admission,
unresolved symbolic physical widths, wider-k admission and the incomplete
original slow/autotester scope. The stored-call closure does not resolve those
separately inventoried boundaries. No additional parent acceptance is claimed.

Run the supported consumer and focused verification with the existing reference
environment.

```bash
python -m examples.algorithms.register_field_multiplication
python -m pytest tests/unit/test_field_scaling_component.py tests/unit/test_polynomial_product_component.py tests/unit/test_field_multiplication_component.py -q
```

The independent module commits are e34aef21c for typed shifts, a9a58dca3 for
shared constant scaling, 5f5b21a87 for polynomial recursion, e4f35ef3b for
quadratic ownership, f2467dc05 for Karatsuba/append/coherent verification and
bd9eef5a3 for the shared measurement replacement test.

## Polynomial range and internal target dependency closure — 2026-11-20

The remaining concrete boundaries were rechecked against the unchanged complete
source file and its direct recursive children. This section supersedes the
historical wider-k and arbitrary-internal-h gaps above. Source inventory,
notebook evidence, measured erasure histories, and the frozen linear-adjoint
failure witnesses are reused.

### Original executable shift range

For `MultiplyPolyByOnePlusXk`, degree one supports every positive integer shift.
For n greater than one, the recursive child fits for 1 <= k <= 2n-1.
The source prose only establishes k <= n+1. The additional executable range is
verified directly through the unchanged builder, ordered per-wire primitives,
operands, output aliases and source declarations. The existing flat emitter,
stored Program and concrete model graph now admit that range together. Polynomial
coefficient zero remains first in stored arrays and MSB-first in word consumers.
All f/g/h registers are THRU, with arbitrary initial h preserved by XOR accumulation.

For n > 1 and k >= 2n, original construction succeeds and declaration costs
remain available, but decomposition raises a NumPy broadcasting ValueError.
Native paths reject these shapes before building. Ten committed fixtures retain
the original failure at widths 2/3/4/5/8 and two successive invalid shifts.
The exact exception message differs at this deliberately earlier native boundary.
Width-one arbitrary shifts have finite witnesses through five.

### Internal Karatsuba THRU target

`karatsuba_implementation_program` already accepts arbitrary h. With the accepted
exact scale-adjoint child, its complete action is
`(f,g,h) -> (f,g,x**ceil(n/2)*h + f*g)` in the binary field. The docstring now
states that action explicitly. A zero h produces the public clean product.
The existing immutable-source root-builder comparison retains all actual child
calls, operands, coefficient views and aliases. Its exact-adjoint adapter remains
disclosed. The unmodified source still ignores the linear-adjoint flag, including
the retained GF(4) product failure. No unchanged-source mathematical equivalence
is claimed through that defective child.

Dense states exhaust f/g/h and an entangled spectator at widths 2/3/4, including
clean padding and exact Circuit adjoint restoration through `circuit.append(op)`.
Sparse complex states cover every f/g pair with a nonzero-patterned target at
widths 2/3/4/5/8. The sparse CCX/CX permutation is checked with its amplitudes at
canonical output indices. There are zero measurements or classical slots in
these coherent polynomial and internal product paths.

### Separate resource accounts

| Operation | Actual primitive stream | Independent source QEC model | Clean padding peak |
| --- | --- | --- | --- |
| n=1, k=5 polynomial | 1 CCX + 2 CX | 1 Toffoli + 2 Clifford | 0 |
| n=3, k=5 polynomial | 7 CCX + 38 CX | 7 Toffoli + 38 Clifford | 1 |
| n=5, k=9 polynomial | 17 CCX + 106 CX | 17 Toffoli + 106 Clifford | 1 |
| n=8, k=15 polynomial | 27 CCX + 206 CX | 27 Toffoli + 206 Clifford | 0 |
| n=8 internal field, modulus 283 | 27 CCX + 200 CX | 27 Toffoli + 200 Clifford | 0 |

The polynomial implementation commit is `61a35624c`. Internal target evidence
and corrected interface documentation are `722190750`. All 276 focused checks,
4341 concentrated checks and 17 local integration checks pass. There are no
skips and 30 existing paired source/native overflow warnings. Strict typing
covers all eight touched Python files. Black, Ruff, both scope checks,
architecture, IR, protected API and ledger synchronization pass. The two source
Git blobs, Qualtran 0.7.0 and native Kickmix 0.1 Simulator are verified.
Dependency diagnostics retain 687 whole-repository findings and zero batch
findings. Receipts are `ftqc-reference-env/g2-polynomial-wide-*`.

Q-0519/Q-0520 remain open for the frozen linear-adjoint defect and its dependent
physical mismatch, generic foreign construction and serialization, generic root
inverse dispatch, symbolic physical execution, complete original slow/autotester
scope and full eight-bit coherent measurement-branch enumeration. This batch
closes concrete dependencies within the existing whole-file inventory.


## Complete original slow scope and live product capture

The complete unchanged Q-0520 slow selection now passes all 22 cases in
137.18 seconds, with 283 other items deselected and six original unknown-marker
warnings. This includes every slow linear synthesis, product/accumulation,
wide scaled polynomial, binary polynomial and Karatsuba classical-action case.
The prior interrupted attempt remains diagnostic history. Its incomplete status
is superseded for this specific slow selection. Existing accepted non-slow,
notebook, whole-definition and native execution evidence is reused.

### Actual foreign object handoff

`test_field_product_foreign_snapshot.py` captures actual frozen
GF2Multiplication objects at widths two and three, in clean-product and arbitrary
THRU-accumulation modes. The existing named emitter accepts a RIGHT result only
when explicitly listed as a caller-owned clean local port. It independently
captures the actual source adjoint. Both directions use the existing native
snapshot/append path with no SDK, Galois or reference object retained at execution.

A separate read of the original flattened graph compares every primitive,
ordered operand and per-wire dependency with each captured direction. The source
helper is called with `repair_adjoint=False`. No reference body is modified.
Complete complex states cover every x/y value, every target in accumulation mode
and a coherent spectator under reversed physical binding. Clean-product output
starts at zero and the original inverse consumes its produced value. JSON,
source-adjoint recovery and SDK-blocked native append pass.

### Separate primitive and analytical accounts

| Field width and mode | Actual primitive stream in each direction | Independent original QEC model | Private work / measurements / binding swaps | Declared logical peak |
| --- | --- | --- | --- | --- |
| 2, clean product | 4 CCX + 2 CX | 4 Toffoli + 2 Clifford | 0 / 0 / 0 | 6 |
| 2, arbitrary accumulation | 4 CCX + 4 CX | 4 Toffoli + 4 Clifford | 0 / 0 / 0 | 6 |
| 3, clean product | 9 CCX + 3 CX | 9 Toffoli + 3 Clifford | 0 / 0 / 0 | 9 |
| 3, arbitrary accumulation | 9 CCX + 6 CX | 9 Toffoli + 6 Clifford | 0 / 0 / 0 | 9 |

The result word is logical output or THRU data. Its width is included in the
logical peak and is not counted as private scratch. The retained spectator is
outside the source model. These coherent captures have no measurement branches.

### Remaining whole-file boundaries

The actual source constructor serializer and automatic typed-graph importer
still fail with TypeError before native handoff because QGF is unsupported by the
original codec. Five actual constructor/graph failure-stage witnesses are in
`g2-field-multiplication-foreign-intake-20261121.json`. Successful live capture uses
the existing named boundary and does not claim the original codec works.

The complete source file's other families retain their earlier native owners,
exact-adjoint qualification, declared models and accepted physical dependencies.
Original linear-adjoint defects, dependent Karatsuba mismatch, generic root
inverse/constructor dispatch, symbolic physical execution and full eight-bit
measurement enumeration remain recorded. Q-0519/Q-0520 stay open under their
original scope. This batch closes the previously incomplete original slow
selection and the concrete actual-product snapshot handoff without creating
new parent IDs or changing acceptance denominators.

### Evidence

`g2-field-product-snapshot-evidence-20261121.json` records both unchanged source
blobs, Qualtran 0.7.0, Kickmix 0.1 Simulator and all eight directional resource rows.
The concentrated product/scaling/polynomial/file regression passes 174 cases.
Local integration, strict typing of the new file, Black, Ruff, Ecosystem scope,
architecture, multi-level IR and protected API checks are recorded separately.
The four new cases also run in the independent Cirq 1.6.1 profile. No production
owner changed in this batch, and earlier full-work/measurement evidence is reused.
