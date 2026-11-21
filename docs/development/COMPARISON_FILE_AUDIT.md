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
