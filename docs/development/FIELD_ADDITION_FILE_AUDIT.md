# Finite-field addition whole-file audit

G2-FILE-FIELD-ADD audits Q-0509 through Q-0514 and the direct converter dependency
Q-0524 against Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. All seven original assets match their
frozen Git blobs. Q-0509, Q-0512 and Q-0524 close. Implementation parents Q-0510
and Q-0513 and test parents Q-0511 and Q-0514 remain open under the original
closure criteria for the source bookkeeping/count and generic admission/test
boundaries listed below. Existing source failures remain reproducible.

## Ownership, interfaces and work

Compiler `finite_field/coefficients.py` remains the executable authority.
`gf2_add_ir(source, target)` preserves source and XORs it into target, in ascending
big-endian wire order. `gf2_add_constant_ir(target, value)` emits X on each one bit
of the big-endian constant, in ascending wire order. The existing controlled
extension retains its independently accepted behavior. The separate Kickmix
mixed-coefficient emitter retains its locked reversed traversal.

Compiler `addition_model.py` owns declared CX/X multiplicities and shape formulas.
It imports no numerical field library. Simulation `finite_field_addition.py`
owns conversion to the existing `FiniteFieldEncoding` and the two scalar field
callbacks. Dynamic galois polynomial/array classes remain inside Simulation.
Existing neutral register metadata represents signatures. Drawer and the native
catalogue present these declarations. There is no additional operation class,
IR or reference-package dependency in production.

The ordinary execution path remains `circuit.append(FieldOperation(n, gates))`,
using the existing example append adapter. Both operations require zero auxiliary
qubits and zero measurements. There is one coherent branch. Noncontiguous wire
binding preserves the logical big-endian order. Constant addition's inherited
source adjoint reverses its X stream and matches existing `Circuit.adjoint()`.
Register addition's source-specific adjoint returns the same operation. Repeating
the native register emitter preserves that canonical stream and restores state.

## Complete source definition disposition

| Original definitions | Native owner and disposition |
|---|---|
| `GF2Addition.qgf`, `__init__`, `bitsize`, `signature` | Existing `FiniteFieldEncoding`, `binary_field_encoding`, neutral `Register`/`Signature` and all example metadata. Native construction accepts the native field descriptor in place of a foreign QGF. The deprecated source `bitsize=` constructor/warning and attrs object construction are source API surfaces, recorded within the remaining generic admission boundary. |
| `GF2Addition.build_composite_bloq` | Existing ascending `gf2_add_ir` CX stream, with logical split/join represented by caller word binding. All widths 1 through 8 match unchanged source operands and order. Concrete words are required for execution. |
| `GF2Addition.build_call_graph` | `field_addition_calls` retains one CX edge of multiplicity degree. `field_addition_resource_formula` separates the signature width from physical instruction counts. |
| `GF2Addition.on_classical_vals` | Simulation `field_add_classical_values` checks both field scalar classes, preserves source identity and performs original field addition. All original domains at degrees 2 through 5 match. |
| `GF2Addition.adjoint` | The same self-inverse register emitter. Coherent append repetition restores the entire state, including spectator phase. |
| `_gf16_addition`, `_gf2_addition_symbolic`, `_GF2_ADDITION_DOC` | Native catalogue and complete notebook preserve concrete degree 4 and unrestricted symbolic m, parameter/register topics, signatures and declared graph. |
| `GF2AddK.bitsize`, `k`, `qgf`, `signature`, `_bits_k`, `is_symbolic` | Existing field codec, native catalogue and shared symbolic classification. All constants at degrees 1, 2, 3, 4, 5 and 8 match exact source X operands. Symbolic integer literals remain symbolic as in source. |
| `GF2AddK.build_composite_bloq` | Existing `gf2_add_constant_ir` provides exact concrete gate order and zero-constant empty emission. Native executable admission requires an exact Python integer fitting the word. Foreign numerical scalar construction remains a separate source API boundary. |
| `GF2AddK.build_call_graph` | `field_add_constant_calls` retains the zero X edge and the width bound for any symbolic width/constant. The known Python zero gives count zero, while SymPy zero follows the unchanged symbolic width bound. |
| `GF2AddK.on_classical_vals` | Simulation `field_add_constant_classical_value` preserves the source assertion and field constructor validation. All original degree 2 through 5 constants and inputs match. |
| Inherited `GF2AddK` adjoint | Existing circuit inverse preserves the exact reversed source X stream. No second inverse emitter is introduced. |
| `_gf16_add_k`, `_gf2_add_k_symbolic`, `_GF2_ADD_K_DOC` | Native catalogue and complete notebook preserve degree 4, k=1, and the source `positive=True, integers=True` symbolic assumptions. The source spelling is retained. |
| Entire direct `gf_utils.qgf_converter` | `binary_field_encoding` preserves native descriptor identity, integer/symbolic degree dispatch, polynomial identity and descending-power construction. Success and exception outcomes match, including accepted bool degree and empty polynomial powers. This closes Q-0524. |

The existing codec and source both support lazy numerical construction beyond
executable binary words. This batch preserves that boundary. General foreign
QGF/attrs admission and protected executable descriptor admission are not expanded.

## Complete original test disposition

There are five original functions in `gf2_addition_test.py` and four in
`gf2_add_k_test.py`. The native AST guard checks the whole sets. The unchanged
source suite reports **17 passes, six skips and six expected failures** in
2.64 seconds, with two original unregistered slow-marker warnings.

| Original test functions | Native assertion mapping |
|---|---|
| `test_gf16_addition`, `test_gf2_addition_symbolic` | Both entire factory signatures and model values match. Literal concrete source streams, symbolic costs, native SVG and independent notebook execution pass. Original generic autotester omissions remain below. |
| `test_gf2_addition_classical_sim_quick`, `test_gf2_addition_classical_sim` | Every original pair at degrees 2, 3, 4 and 5 matches both native scalar callbacks and XOR. Normal append independently checks the full complex state at widths 1, 2, 3 and 5. |
| `test_gf2_addition_resource` | Unrestricted symbolic degree preserves Clifford=m and T=0. Concrete streams contain exactly m CX gates. |
| `test_gf16_add_k`, `test_gf2_add_k_symbolic` | Both entire factory signatures, constants and declared calls match. All original symbolic assumptions and concrete/symbolic-zero dispatch are preserved. |
| `test_gf2_add_k_classical_sim_quick`, `test_gf2_add_k_classical_sim` | Every original constant and input at degrees 2, 3, 4 and 5 matches the native callback and XOR. Every constant's literal X stream also matches at degrees 1, 2, 3, 4, 5 and 8. |

Native coherent checks include one complex spectator, one initially zero unused
wire, two noncontiguous register words and repeated execution. Full complex
amplitudes agree within 1e-12. The unused wire remains zero. The implementation
introduces no measured branches, feedback, reset or auxiliary allocation.

## Complete notebook asset disposition

Each original notebook has 11 cells, including six code cells. All original
cells, documentation topics, two examples, graphical signature, one-level call
graph and leaf counts have native counterparts. Each native notebook adds an
append execution cell and explanatory Markdown, giving 13 cells and seven code
cells with two SVGs. Both unchanged originals execute in an isolated locked
kernel with a 60-second cell timeout. Kernel selection and timeout are test-only
environment configuration.

The native kernels block Qualtran, Cirq and Quimb imports. Both native notebooks
execute successfully, including normal append and inverse restoration. Native
signature table layout, operation names and resource labels replace the source
drawing layout. Actual native DOT renders were inspected as PNGs. This explicit
presentation adaptation closes Q-0509 and Q-0512.

## Separate resource accounts

| Operation | Actual emitted primitives | Source declared model | Width and workspace |
|---|---|---|---|
| Eight-bit register addition | CX 8 | Clifford 8, T 0, measurement 0 | 16 logical operand wires, auxiliary 0 |
| Eight-bit constant 0xA5 | X 4 | Clifford 4, T 0, measurement 0 | 8 logical operand wires, auxiliary 0 |
| Eight-bit Python constant zero | Empty stream | Explicit X edge with multiplicity 0 | 8 operand wires, auxiliary 0 |
| Symbolic m/k constant | No executable word emitted | X bound m | Signature width m, auxiliary 0 |
| Four-bit full source register decomposition | CX 4 plus two Split and two Join nodes | Declares only CX 4 | Bookkeeping nodes emit no physical gates |
| Four-bit full source constant-one decomposition | X 1 plus one Split and one Join node | Declares only X 1 | Bookkeeping nodes emit no physical gates |

The formula's logical width is the operand signature width. It is separate from
recursive graph peak estimates and runtime statistics.

## Central remaining boundaries

| Parents | Reproduced boundary | Owner and closure condition |
|---|---|---|
| Q-0510, Q-0513, Q-0511, Q-0514 | The two concrete source autotester count checks fail because annotations omit QGF Split/Join nodes. Physical X/CX counts match. Source diagnostic compares the entire declaration and decomposition without editing either. | Compiler resource and test owners. Preserve full source bookkeeping identities and failure evidence. Close only after an explicit disposition under the original criteria, without presenting filtered counts as the original count assertion. |
| Q-0510, Q-0513 | Source attrs/QGF construction, deprecated `bitsize=` warning and generic foreign object admission are not native production constructors. Explicit native field resolution and supported concrete words are accepted. | Existing Core metadata, Simulation conversion and Ecosystem boundary. Establish the remaining native admission/failure mapping without changing protected schemas or introducing a parallel Bloq API. |
| Q-0511, Q-0514 | Six source fixture skips cover symbolic decomposition/count applicability and dtype checks. Four serialization xfails explicitly reject QGF. The two other xfails are the counts diagnostic above. | Native generic test/serialization owners. Complete exact native autotester and serialization assertion mappings for all four factories. Existing numerical metadata roundtrips are reused but do not substitute for these original assertion paths. |

Q-0516/Q-0519 historical linear-adjoint and Karatsuba issues are unchanged. This
batch reuses their accepted dependency evidence and does not reopen them.

## Batch verification

- 43 targeted whole-file assertions pass.
- 1,604 concentrated field, polynomial, classical-field and resource tests pass
  in 17.98 seconds, with zero skips.
- 90 independent-kernel, CPU and tensor integration checks pass in 19.86 seconds,
  with zero skips and two existing tensor API deprecation warnings.
- Strict typing passes for all five changed Python files. Black, Ruff, both
  domain scope checks, architecture, multi-level IR and protected API checks pass.
- The three new production/example modules have 68 statements and 14 branches,
  all covered. Existing protected API and schema checks pass.
