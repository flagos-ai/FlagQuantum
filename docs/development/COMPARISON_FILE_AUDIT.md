# Comparison complete source-file closure

Q-0180 and Q-0181 cover the unmodified `comparison.py` and its tests at Qualtran
v0.7.0 commit `bcb17bb25360ed68ce3bc084645db3e929799389`. Q-0179 already
accepts the complete notebook, factory metadata, drawing and declaration graphs.
This batch adds missing executable stored component bodies and reuses accepted
physical, numerical action and resource owners.

## Interfaces and ownership before implementation

All concrete words use MSB-first order. Compiler comparison owns constant,
equality, tree, linear, controlled and directional half-comparison graphs.
Compiler bits owns sign extension and truncation over shared fanout. Shared
addition, bitwise, Cast, And/MultiAnd and MCX owners remain authoritative.
Simulation retains source numerical callbacks. Declared models remain independent
from emitted primitives. The ordinary consumer remains `circuit.append(op)`.

| Source definitions | Native execution contract and owner |
|---|---|
| LessThanConstant | `constant_component.less_than_constant_program(width, threshold)` preserves the complete seed/prefix scan and reverse And cleanup, including the source always-true upper bound. |
| BiQubitsMixer, SingleQubitCompare | Accepted directional tree flag owners produce RIGHT ancillas and consume their exact LEFT image. |
| _equality_with_zero, LessThanEqual, GreaterThan | Extend the existing tree owner to unequal concrete widths, retaining the modified prefix and its erasure, source target shell and exact GreaterThan order. |
| LinearDepthGreaterThan | `linear_component.linear_greater_program(width, signed=False)` retains source carries and sign handling. The signed flag retains the source's finite-width difference-sign behavior and callback conflict. |
| GreaterThanConstant | Source has signature, drawing and a same-threshold LessThanConstant model, with no executable decomposition or classical action. Keep that declaration and explicit execution refusal. |
| Equals, EqualsAConstant | Shared bitwise Xor/MCX/Xor graph and already accepted constant Split/MCX/Join graph retain typed through words and dirty XOR target. |
| CLinearDepthGreaterThan | `controlled_component.controlled_greater_program(dtype, control_value=1)` composes signed/zero extension, complements, Cast, full out-of-place sum, MCX, and exact cleanup. Both control polarities remain supported. |
| _HalfLinearDepthGreaterThan | `half_component.half_greater_program(dtype, adjoint=False)` owns RIGHT difference/target and temporary sign prefixes, or consumes their correct LEFT image. |
| _HalfComparisonBase, _LinearDepthHalfComparisonBase, four concrete half relations | `half_component.half_comparison_program(dtype, relation, adjoint=False)` retains source operand exchange and inclusive X shell. The original ge/lt adjoints return swapped a/b aliases, which must be retained explicitly. |
| SignExtend/SignTruncate direct dependency | `bits.sign_extension_component.sign_extension_program(input_dtype, output_dtype, adjoint=False)` consumes x and returns y, allocating or clearing/freeing only the redundant prefix. Inputs describe the forward extension in both directions. |
| Factories, documentation constants, wire symbols, Cirq adapters, classical callbacks and declared models | Reuse the complete notebook catalogue and existing class-specific owners. Source-only object wrappers remain adapter boundaries, without a second public package identity. |

Generic symbolic executable admission and foreign construction retain their
existing protected boundary. Proposal 108 remains pending. Zero-cost views keep
source dtype mismatches visible in source strict checks. Reference defects are
kept separately from the native physical-order correction paths already present.
No reference source is edited to claim equality.

## Unmodified source tests

The complete non-notebook source run passes 395 cases, with 15 skips, 6 expected
failures and 2 notebook cases deselected. The skips preserve symbolic/declaration
and strict-dtype policies. Notebook evidence is reused from Q-0179. Concrete
native graph acceptance remains in progress until this batch's full gate,
amplitude, ownership, resource and normal-consumer checks pass.

## Complete definition disposition

Every top-level definition and method in the frozen file is listed below.
Signature/drawing/factory metadata is already accepted by Q-0179 and the original
notebook catalogue. Concrete callbacks and models reuse the existing native
owners. Symbolic construction and foreign object admission retain their declared
boundaries. Empty native qtyping fixture edges do not certify source strict dtype
consistency.

| Definition | Native owner or existing evidence |
|---|---|
| `LessThanConstant`, `LessThanConstant.signature`, `LessThanConstant.wire_symbol`, `LessThanConstant.registers`, `LessThanConstant.with_registers`, `LessThanConstant.apply`, `LessThanConstant.on_classical_vals`, `LessThanConstant._circuit_diagram_info_`, `LessThanConstant.__pow__`, `LessThanConstant.decompose_from_registers`, `LessThanConstant._has_unitary_`, `LessThanConstant.build_call_graph` | constant_component, less_than, classical_comparison, comparison_resource_model |
| `_lt_k` | Q-0179 complete source factory/notebook catalogue, concrete native factory checks where executable |
| `_lt_k_symb` | Q-0179 complete source factory/notebook catalogue, concrete native factory checks where executable |
| `BiQubitsMixer`, `BiQubitsMixer.signature`, `BiQubitsMixer.decompose_from_registers`, `BiQubitsMixer.adjoint`, `BiQubitsMixer.__pow__`, `BiQubitsMixer.build_call_graph`, `BiQubitsMixer._has_unitary_` | tree_component, comparison_tree, comparison_resource_model |
| `_bi_qubits_mixer` | Q-0179 complete source factory/notebook catalogue, concrete native factory checks where executable |
| `SingleQubitCompare`, `SingleQubitCompare.signature`, `SingleQubitCompare.decompose_from_registers`, `SingleQubitCompare.adjoint`, `SingleQubitCompare.__pow__`, `SingleQubitCompare.build_call_graph` | tree_component, comparison_tree, comparison_resource_model |
| `_sq_cmp` | Q-0179 complete source factory/notebook catalogue, concrete native factory checks where executable |
| `_equality_with_zero` | ordering_component source wire calls and shared conjunction, with component_wire_graph lazy typed views |
| `LessThanEqual`, `LessThanEqual.signature`, `LessThanEqual.registers`, `LessThanEqual.with_registers`, `LessThanEqual.apply`, `LessThanEqual.wire_symbol`, `LessThanEqual.on_classical_vals`, `LessThanEqual._circuit_diagram_info_`, `LessThanEqual.__pow__`, `LessThanEqual._decompose_via_tree`, `LessThanEqual.decompose_from_registers`, `LessThanEqual.build_call_graph`, `LessThanEqual._has_unitary_`, `LessThanEqual.adjoint` | ordering_component, less_equal, classical_comparison, comparison_resource_model |
| `_leq_symb` | Q-0179 complete source factory/notebook catalogue, concrete native factory checks where executable |
| `_leq` | Q-0179 complete source factory/notebook catalogue, concrete native factory checks where executable |
| `GreaterThan`, `GreaterThan.signature`, `GreaterThan.wire_symbol`, `GreaterThan.build_composite_bloq`, `GreaterThan.build_call_graph` | ordering_component, ordering, comparison_resource_model |
| `_greater_than` | Q-0179 complete source factory/notebook catalogue, concrete native factory checks where executable |
| `LinearDepthGreaterThan`, `LinearDepthGreaterThan.signature`, `LinearDepthGreaterThan.on_classical_vals`, `LinearDepthGreaterThan.build_composite_bloq`, `LinearDepthGreaterThan.wire_symbol`, `LinearDepthGreaterThan.build_call_graph` | linear_component, linear_comparison, classical_greater_comparison, existing primitive resource owner |
| `GreaterThanConstant`, `GreaterThanConstant.signature`, `GreaterThanConstant.wire_symbol`, `GreaterThanConstant.build_call_graph` | declaration/signature and source_less_than_constant_qec_cost same-threshold model, original no-decomposition boundary |
| `_gt_k` | Q-0179 complete source factory/notebook catalogue, concrete native factory checks where executable |
| `Equals`, `Equals.signature`, `Equals.bitsize`, `Equals.is_symbolic`, `Equals.wire_symbol`, `Equals.build_composite_bloq`, `Equals.build_call_graph`, `Equals.on_classical_vals` | equality_component, equality, classical_comparison, existing MCX model |
| `_equals` | Q-0179 complete source factory/notebook catalogue, concrete native factory checks where executable |
| `EqualsAConstant`, `EqualsAConstant.signature`, `EqualsAConstant.wire_symbol`, `EqualsAConstant.is_symbolic`, `EqualsAConstant.bits_k`, `EqualsAConstant.build_composite_bloq`, `EqualsAConstant.build_call_graph` | equality_component, equality, existing MCX model |
| `_make_equals_a_constant` | Q-0179 complete source factory/notebook catalogue, concrete native factory checks where executable |
| `_eq_k` | Q-0179 complete source factory/notebook catalogue, concrete native factory checks where executable |
| `CLinearDepthGreaterThan`, `CLinearDepthGreaterThan.signature`, `CLinearDepthGreaterThan.wire_symbol`, `CLinearDepthGreaterThan.build_composite_bloq`, `CLinearDepthGreaterThan.on_classical_vals`, `CLinearDepthGreaterThan.build_call_graph` | controlled_component, controlled_comparison, classical_greater_comparison, existing primitive resource owner |
| `_clineardepthgreaterthan_example` | Q-0179 complete source factory/notebook catalogue, concrete native factory checks where executable |
| `_HalfLinearDepthGreaterThan`, `_HalfLinearDepthGreaterThan.signature`, `_HalfLinearDepthGreaterThan.adjoint`, `_HalfLinearDepthGreaterThan._compute`, `_HalfLinearDepthGreaterThan._uncompute`, `_HalfLinearDepthGreaterThan.build_composite_bloq`, `_HalfLinearDepthGreaterThan.build_call_graph` | half_component, half_comparison, classical_half_comparison, existing primitive resource owner |
| `_HalfComparisonBase`, `_HalfComparisonBase.signature`, `_HalfComparisonBase.adjoint`, `_HalfComparisonBase._half_greater_than_bloq`, `_HalfComparisonBase._classical_comparison`, `_HalfComparisonBase.on_classical_vals`, `_HalfComparisonBase._compute`, `_HalfComparisonBase._uncompute`, `_HalfComparisonBase.build_composite_bloq`, `_HalfComparisonBase.build_call_graph` | half_component, half_ordering, classical_half_comparison, existing primitive resource owner |
| `_LinearDepthHalfComparisonBase`, `_LinearDepthHalfComparisonBase._half_greater_than_bloq` | half_component, half_ordering, classical_half_comparison |
| `LinearDepthHalfGreaterThan` | half_component relation gt, half_ordering |
| `LinearDepthHalfGreaterThanEqual` | half_component relation ge, half_ordering |
| `LinearDepthHalfLessThan` | half_component relation lt, half_ordering |
| `LinearDepthHalfLessThanEqual` | half_component relation le, half_ordering |
| `_lineardepthhalfgreaterthan_small` | Q-0179 complete source factory/notebook catalogue, concrete native factory checks where executable |
| `_lineardepthhalflessthan_small` | Q-0179 complete source factory/notebook catalogue, concrete native factory checks where executable |
| `_lineardepthhalfgreaterthanequal_small` | Q-0179 complete source factory/notebook catalogue, concrete native factory checks where executable |
| `_lineardepthhalflessthanequal_small` | Q-0179 complete source factory/notebook catalogue, concrete native factory checks where executable |

Frozen comparison source Git blob `ae1f0275d081eae99fcf7786659dd33873083f18`.

## Concrete execution and resource evidence

The full native comparison closure covers constant/equality/tree/linear/
controlled/half graphs and sign conversion, retaining all shared arithmetic and
control owners. Ten actual concrete source factories pass all five existing
native check modes. Those qtyping fixtures have no explicit source dtype edges.
The original strict-dtype skips remain recorded independently.

Six normal examples now bind stored graphs before `circuit.append(op)`. The
replacement test disables their former handwritten comparison entry points,
then checks the original complete complex-state workflows and restoration.
Source connection traces cover the composite bodies, operand exchanges and
inclusive shells. Complete primitive dependencies, operands and measurement
feedback match through eight-bit concrete integer families. Unequal tree
prefixes use the original generator allocation order, with temporary names
bound independently from native lifetime allocation.

Complete small source tensors cover dirty targets, entangled spectators and
every raw measurement branch. Half relations independently verify their full
complex difference/flag output state and every two-bit erasure history. Sign
extension/truncation covers both signed encodings, including clean prefix
ownership and coherent spectator restoration. Existing signed-linear overflow
witnesses, wider physical tests, classical callbacks and notebook evidence are
reused.

| Concrete case | Emitted primitives | Original analytical model | Peak live quantum / retained outputs |
|---|---|---|---|
| Eight-bit threshold five | 66 CX, 32 T/TDG, 6 X, 8 measurements and resets | 8 And, 32 Clifford, 8 measurements, total 32 T | 18 / no retained work |
| Three-bit versus eight-bit tree | 106 CX, 40 T/TDG, 35 X, 10 measurements and resets | 10 And, 83 Clifford, 10 measurements, total 40 T | 25 / no retained work |
| Eight-bit signed linear | 83 CX, 28 T/TDG, 18 X, 7 measurements and resets | 7 And, 66 Clifford, 7 measurements, total 28 T | 24 / no retained work |
| Eight-bit signed controlled | 155 CX, 40 T/TDG, 38 X, 10 measurements and resets | 10 And, 143 Clifford, 10 measurements, total 40 T | 31 / no retained work |
| Eight-bit signed half forward | 95 CX, 32 T/TDG, 27 X, no measurements | 8 And, 74 Clifford, no measurements, total 32 T | 27 / nine difference bits and one flag |

All listed native peak lifetimes equal the independent source QubitCount model.
Each measured graph uses a fresh SSA classical slot per event. Logical source
Clifford categories remain separate from expanded primitive counts. Half
forward's ten owned outputs remain live until its inverse consumes them.
Scheduling can reuse a cleared prefix for the output flag, so consumers follow
returned bindings instead of assuming fixed allocation names.

## Remaining source and admission boundaries

GreaterThanConstant still has no original decomposition or classical action.
Its same-threshold LessThanConstant model remains a declaration. Source signed
linear overflow/callback disagreement and ge/lt inverse alias disagreement remain
Q-0180.16 and Q-0180.12. Source strict-dtype gaps and unavailable original count
modes remain visible in the unmodified test run. Generic symbolic physical
execution and foreign construction retain their existing admission boundary.
Proposal 108 remains pending. Q-0180/Q-0181 therefore retain their explicit
parent gaps after this complete concrete graph batch.

The default smoke/unit lane retains the existing
`test_every_module_with_examples_is_covered` failure after 434 passes and 51
skips. Its test and nine reported production modules are unchanged from the
batch base. Batch regression and strict checks are recorded separately in
`g2-comparison-native-*` reference receipts.

## Recursive comparison presentation contract, November 22

The next complete comparison.py dependency closure contains 62 already exact
source-success ordered graphs and 344 atomic/refusal contexts across twelve
concrete roots. Twenty-nine non-root contexts still lack qualified source glyphs.
Tree comparison flags remain RIGHT-owned on compute and LEFT-consumed on reverse.
Unsigned words retain MSB-first views. Control arrays remain row-major QBit words
with declared polarities. Predicate junk/target ownership remains with the shared
control owner, including exact measurement feedback. Sign extension consumes a
narrow signed word and returns its wider image; truncation consumes that image
and releases the correct redundant prefix. Existing Compiler constructors own
execution and resources. Separate native example providers own tree, control and
sign presentation. Qualification requires matching every reachable definition,
operand, phase and feedback instruction. Generic fallback remains for unmatched
bodies. The EqualsAConstant nominal unsigned/source QAny Join difference stays
explicitly recorded. Original source graphs and refusals remain frozen.

The three domain providers close all twenty-nine recorded non-root glyph gaps.
The complete twelve-root walk retains 62 exact ordered graphs and 344 original
atomic/refusal contexts. All fifty non-root source-success graph contexts now
match fresh source score payloads and source class labels, including both
constructor directions of BiQubitsMixer and SingleQubitCompare. Source generic
Adjoint wrappers retain the dagger label. The twelve root diagrams reuse their
preceding accepted source evidence. The symbolic/decomposition refusals and the
EqualsAConstant nominal Join-view boundary remain explicit.

The focused presentation checks pass 45 cases. The combined physical regression
passes 857 cases covering complete complex isometries, spectators, applicable
measurement histories, exact erasure, clean work ownership and independent
source models. All twelve actual primitive histograms equal the previous
published execution baseline. Actual T-family/measurement pairs are 32/8 for
LessThanConstant, 44/11 for LessThanEqual, 8/0 for the mixer, 4/0 for the single-bit
comparison, 28/7 for GreaterThan, 12/3 for each equality case, 28/7 for controlled
linear comparison, and 12/0 for each of the four half comparisons. Full primitive
histograms and separately declared analytical counts remain in the resource
receipt. The half-comparison forward graphs expose live sum work, so their zero
measurement figure describes that forward stream, while round-trip erasure is
verified separately. Drawing qualification adds zero primitive operations.

The Arithmetic inventory now qualifies 204 contextual descendants, an increase
of 116 matching definitions across reused comparison/tree/control/sign owners.
The complete catalog retains 3569 JSON files, 234 documentation files, 3568
diagrams, 3215 contextual children, 68 root graphs and 285 atomic interfaces.
Contextual constructor matches outside the comparison file are recorded
separately from this file's fresh source drawing acceptance.

The concentrated catalog regression passes 679 checks with three existing
notebook-related deselections and fourteen existing escape warnings. Strict
checks cover all seven changed Python files. Black, Ruff, integration scope,
architecture, multilevel IR, protected API baseline and whitespace checks pass.
The batch closes the complete comparison-file recursive presentation. Other
Arithmetic source families retain their own recursive presentation acceptance.

## Recursive sorting consumers — 2026-11-22

GreaterThan descendants now qualify through the existing unsigned comparison
owner. Both operand widths select the candidate, and the complete reachable
Program must match before source labels and comparison glyphs are supplied.
This closes eleven contextual GreaterThan drawing gaps across the four concrete
sorting catalog roots. Their recursively reached LessThanEqual, directional
mixer, single-bit comparison and erasure definitions retain prior acceptance.
The execution owner, phases, measurements, work ownership and primitive resource
stream remain unchanged. See SORTING_FILE_AUDIT.md and the durable
`g2-sorting-child-*20261122` receipts for whole-tree evidence.

## Symbolic GreaterThan graph dependency — 2026-11-22

The symbolic sorting consumers now reuse a complete GreaterThan graph with the
original LessThanEqual call followed by X. The child comparison retains its
unresolved unsigned word widths as a typed declaration. The X definition comes
from the existing native comparison bit owner. Full reachable-definition
matching qualifies this graph before source class labels and wire symbols are
used. Every original source-success descendant across all six sorting catalog
factories is traversed and compared, including these symbolic GreaterThan calls.
The original symbolic LessThanEqual decomposition refusal remains explicit.
Physical binding still rejects unresolved widths. Frozen sources are unchanged.
See SORTING_FILE_AUDIT.md and `g2-symbolic-sorting-recursive-20261122.json`.

## Complete original test-file acceptance (2026-11-25)

Q-0181's complete 503-line original test file contains 37 tests and two helpers.
Its frozen Git blob is `9876537e8199b0e6af0ca7bf9f595f94304146d6`.
The complete earlier original run of 395 passes, 15 skips and six xfails is
reused together with both accepted notebook kernels. Source bytes are unchanged.
The following mapping covers every original function and parameter family;
implementation parent Q-0180 retains its separately recorded gaps.

| Original functions | Native owner and complete assertion disposition |
| --- | --- |
| `test_clineardepthgreaterthan_example`, `test_greater_than`, `test_gt_k`, `test_lt_k_symb`, `test_leq_symb`, `test_equals`, `test_eq_k`, `test_lineardepthhalfgreaterthan_small`, `test_lineardepthhalflessthan_small`, `test_lineardepthhalfgreaterthanequal_small`, `test_lineardepthhalflessthanequal_small` | All actual factory protocols observed before source skip policy; existing constructor, catalog, typed graph and physical owners cover their applicable paths. Fifteen factories include four additional concrete graph examples. |
| `identity_map` | Test-only identity oracle, represented by existing full coherent operator and exact inverse assertions. |
| `test_less_than_gate`, `test_decompose_less_than_gate`, `test_less_than_consistent_protocols` | Constant comparison component/emitter, all original thresholds including out-of-range constants, dirty target XOR, inverse, legacy model and diagram/register interface evidence. |
| `test_bi_qubits_mixer_t_complexity`, `test_single_qubit_compare_t_complexity` | Existing tree primitive/component and historical cost owners preserve forward and measured-adjoint resources independently of actual primitive inventories. |
| `test_multi_in_less_equal_than_gate`, `_less_than_equal_expected_t_complexity`, `test_less_than_equal_consistent_protocols` | All original 1..4 operand-width pairs, two dirty target values, self-inverse full operators, unequal-width tree models and context-independent decomposition. Existing owned register/drawing adapters replace source SDK interface forms. |
| `test_greater_than_manual` | Typed GreaterThan root retains LessThanEqual then X with original port order; source model and wire glyphs checked independently. |
| `test_linear_depth_greater_than_decomp`, `test_classical_linear_depth_greater_than` | Signed/unsigned root graphs and all original value cases; Simulation preserves source callback semantics while physical signed-overflow counterexamples remain explicit. |
| `test_greater_than_constant` | Existing signature/drawing/model and declared atomic constructor. The original class has no decomposition; no successful source primitive body is invented. |
| `test_classical_equals`, `test_equals_call_graph`, `test_equals_a_constant` | Existing equality component/constant emitter and callback owners cover bit, unsigned, signed and Montgomery words, all original domains, original graph generalization, symbol and cost assertions. |
| `test_t_complexity_of_comparison_gates_notebook`, `test_comparison_notebook` | Complete accepted notebook assets and original/native kernel evidence under Q-0179, reused without rerunning unchanged notebooks. |
| `test_decomposition_frees_ancilla` | Existing source allocation/return traces and native ownership events; every small measured history restores private work. |
| `test_clineardepthgreaterthan_classical_action_unsigned`, `test_clineardepthgreaterthan_classical_action_signed`, `test_clineardepthgreaterthan_decomposition`, `test_clineardepthgreaterthan_bloq_counts`, `test_clineardepthgreaterthan_tcomplexity` | Both control polarities, full original unsigned/Montgomery/signed widths, target values, exact root calls and generalized counts. Original symbolic T=4(n+2) declaration is separate from physical emission. |
| `test_linear_half_comparison_decomposition`, `test_linear_half_comparison_bloq_counts`, `test_linear_half_comparison_classical_action`, `test_linear_half_comparison_symbolic_complexity` | All four relations, three integer encodings, original 2..4 widths and both directions; existing typed/native callback owners retain RIGHT output versus LEFT erasure and source alias permutations. Declared forward n-Toffoli and zero-cost erasure remain separate from actual feedback gates. |

The test-only `source_example_protocols` observer records all 75 raw modes:
50 PASS, ten FAIL, five NA, nine UNVERIFIED and one MISSING decomposition.
All ten count failures remain failures. SingleQubitCompare loses a control value
in its declaration, in addition to other missing nodes; it is not classified as
only a zero-cost view mismatch. Six strict dtype failures preserve their actual
incompatible source edges. Symbolic counts are unverified; GreaterThanConstant's
missing decomposition is retained. All fifteen source serializers pass.

Existing ten concrete physical factory replacements, tree mixer/comparator,
symbolic catalogs and source-only atomic declaration complete the factory map.
Full complex amplitudes, coherent spectators, all applicable small measurement
histories, exact output ownership, clean work and ordinary append are reused
from their accepted component/consumer tests. Primitive gate totals, binding
overhead and original historical/modern model counts keep their distinct scopes.
This test-file acceptance does not repair source signed-overflow or half-inverse
alias defects, broaden unresolved physical admission or close Q-0180.

The concentrated comparison/component/constructor/resource regression passes
433 cases. The independent Cirq 1.6 raw-protocol and concrete-factory selection
passes 29. Strict typing of the complete audit test, Black, Ruff, Integration
scope, language, links, architecture, IR and protected API checks pass.
Receipts use `g2-comparison-source-closure-*20261125`. Existing original protocol
deprecation warnings remain separate from acceptance failures.
