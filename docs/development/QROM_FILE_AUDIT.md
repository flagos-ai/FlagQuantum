# Unary QROM source audit

The four frozen files in this batch match the Qualtran v0.7.0 inventory
at `bcb17bb25360ed68ce3bc084645db3e929799389`.

| Inventory | Native evidence |
| --- | --- |
| Q-0479 tutorial | The native notebook preserves all four examples, their source datasets and widths, the signatures and the five-row direct call graph. Its executable example uses the append protocol with a nonzero XOR target. |
| Q-0480 implementation | `source_qrom_ir` retains the source traversal. `source_qrom_calls` exposes immediate calls with literal controls and zero-selection constants. `qrom_wire_symbol` supplies the source register labels to the native score renderer. |
| Q-0481 base | `QromTable` owns concrete data, unknown-data shapes, replacement, signature and binding. `lookup_classical_values` owns the per-address classical callback and respects the original control and XOR behavior. |
| Q-0482 tests | The unchanged source suite passed 45 circuit, cost and metadata tests plus the notebook test in a separate kernel. The source reports three strict-type skips and three expected failures for its unenforced count checks. Native parity reuses the direct operand, feedback, branch, signature, classical and resource tests. |

The direct call declaration shares the bounded count traversal with
the concrete resource model. The calls retain negative AND
`(1,0)`, positive AND and adjoint `(1,1)`, MultiAnd operand lists
and the constant and width of zero-selection XOR. The comparison
reads the exact depth-one edges of the frozen QROM, including
zero multiplicities, with the source Split/Join generalizer. It
covers zero through four controls, zero selection, uniform data,
nonuniform one- and two-axis tables, and shaped outputs.

The five-row tutorial starts each address with a nonzero target and
complex amplitude. The emitted circuit acts on six port wires and
two caller-owned work wires. It uses 12 T-family gates, 26 CX and
three measurements. The source direct graph declares two X, three
negative AND, eight CNOT and three positive AND adjoints. The
separate abstract QEC model reports three AND, 13 Cliffords and
three measurements. These are distinct levels of the same call
chain. The executable tutorial checks the complete output vector
and zero work on sampled histories. Existing gate and branch tests
cover exhaustive small histories, nonzero inputs and feedback.

## Open source boundaries

- The Q-0480 implementation retains the locked emitter's failure on
  nonuniform mixed singleton axes and zero-width target words.
  Pruned out-of-range addresses have source-dependent aliasing.
  The native diagram renders the source port glyphs and uses its
  own layout, so Cirq's variable-spacing ASCII strings have no
  literal native counterpart.
- For Q-0481, the native layout validates shapes early. The source
  constructor can defer shape mismatches to its cached shape
  property, and an empty selection prefix acts as a sentinel
  in its dataset consistency check. Both shape and executable
  admission distinctions are retained in the evidence.
- The source strict-type skips compare opaque QAny targets with
  QUInt inputs at controlled XorK. The recovered notebook test
  passed once its kernel could bind local sockets.

The concentrated native suite passed 256 cases, and both lookup notebooks
passed in independent native kernels. Strict typing, formatting, team
scope, architecture, IR contracts and the public API snapshot passed.

See [the gate and branch audit](REGISTER_SOURCE_QROM.md).

## Complete base-file consumer audit, November 21

The complete `qrom_base.py` source and its direct layout, value and storage
consumers have the following dispositions. Q-0481 remains open for the
constructor-admission differences described below.

| Source definition | Native responsibility and evidence |
| --- | --- |
| `_data_or_shape_to_tuple` and attrs identity | Immutable flattened integer tuples in `QromTable`; actual foreign arrays reconstruct equal native tables |
| `is_symbolic`, `has_data`, `data` | `datasets` availability and explicit dimension checks distinguish shape-only, concrete and symbolic-width data; existing shape and data suites cover the three states |
| `_default_target_shapes` | Both native factories default to one scalar target per dataset |
| `data_shape` | The native `shape` field owns the common prefix; malformed source objects have the separate dispositions below |
| `__attrs_post_init__` | Native metadata validation is earlier and stricter for malformed shapes and negative widths; this gap remains open |
| `_build_from_data` | Integer array conversion, original preconversion maximum-width inference, explicit widths and target shapes map to `from_data` |
| `with_data` | Replacement preserves source dimensions and widths, with the original maximum-value overflow check |
| `_build_from_bitsize` | `from_shape` retains concrete and symbolic shape/width expressions and unknown-data formulas |
| `control_registers` | Existing opaque control word, including symbolic control widths |
| `selection_registers` | Existing bounded unsigned types and original singleton filtering and naming |
| `_target_reg_side`, `target_registers` | Existing THRU opaque scalar/array targets; zero-width filtering and symbolic literal behavior |
| `on_classical_vals` | Existing Simulation `lookup_classical_values`; control bypass, XOR, named outputs, erasure checks and original indexing failures |
| `_QROM_BASE_DOC` | Documentation describes the native owners and executable example `python -m examples.algorithms.register_qrom_table` |

The `from_data` annotations now express the already supported symbolic target
width and control count. Direct construction verifies original symbolic
expressions and data availability. Physical binding continues to require
resolved Python integer dimensions. The symbolic comparison reads original
register fields directly because the foreign register serialization drops
symbol assumptions.

Actual concrete QROM libraries at depths 0, 1 and 2 roundtrip exact bytes and
reconstruct in the unchanged upstream factory. An explicit native factory reads
the serialized data, target shapes, selection widths and control count and
reconstructs the existing `QromTable`. The shared component builder composes
Split, the existing QROM leaf and Join, with explicit work allocation/free.
The normal binder produces identical primitive multisets and complete per-wire
ordered operands and feedback. Every input has a nonzero complex amplitude,
covering all control states, addresses and nonzero scalar/array target values.
Every small measurement history matches the source, including unnormalized
branch weights and zero work. Returned ports retain the input ownership and
CircuitIR roundtrips. The existing append and replacement tests are included
in the concentrated regression.

This consumer audit uses one selection bit, scalar or two-cell one-bit target
arrays, and zero or one control. A controlled two-row lookup allocates one work
wire and one measurement slot. Its T-family count is four and its measurement
count is one, independently matching the source QEC model. Existing larger
lookup and all-branch tests supply the broader physical evidence described
above. The opaque source/library declarations themselves emit zero primitives.

### Remaining base-file gaps

The complete constructor intake reproduces four distinct source/native
admission differences without changing reference objects.

- Different row counts construct a source object and fail only when its cached
  common shape is read. Native construction rejects those rows immediately.
- A declared target shape `(3,)` on a `(3, 2)` array constructs source metadata
  with that shape. Native construction rejects the inconsistent suffix.
- A scalar first dataset followed by a vector lets the source's empty-prefix
  sentinel select the vector shape, while its selector width still comes from
  the scalar. Native construction rejects that inconsistent common prefix.
- A negative control width constructs an original opaque negative-width
  declaration. Native metadata rejects it before physical binding.

The first three require a faithful representation of deferred or inconsistent
source metadata before whole-file admission parity can be claimed. They are
not covered by the successful concrete factory executions. Original shape-only
libraries also fail to serialize their `Shaped` object arrays with pickle
disabled at all three tested depths. Source negative-index, omitted-singleton
and scalar classical indexing behavior remains separately recorded in the
existing classical suite. These gaps keep Q-0481 open.

## Complete implementation-file consumers, November 21

All definitions in the frozen `data_loading/qrom.py` have been assigned to
existing native responsibilities. This batch adds executable coverage of the
actual four documentation factories and their direct consumers. Q-0480 remains
open for the physical and presentation differences listed above.

| Source definition | Native owner and executable disposition |
| --- | --- |
| `_to_tuple` | Native immutable dataset tuples; conversion is verified through the actual factories |
| `QROM`, `build_from_data`, `build_from_bitsize` | `QromTable` concrete and shape factories with explicit selection widths and typed signatures |
| `_load_nth_data` | `source_qrom_ir` target-word toggles in dataset, target-cell and MSB bit order; full factory streams are checked |
| `decompose_zero_selection` | Existing zero-selection load and And/MultiAnd reduction, including measurement-assisted erasure; existing source channel tests cover zero through multiple controls |
| `decompose_from_registers` | Existing table binding and typed component builder; unknown data refuses physical binding |
| `_break_early` | Existing `_uniform` traversal predicate compares all flattened cells in the selected slice, separately for every dataset |
| `nth_operation` | Existing bounded unary traversal and retained singleton-axis source failure |
| `_circuit_diagram_info_` | Existing native score rendering consumes source port glyphs; source Cirq ASCII spacing remains a presentation gap |
| `my_static_costs` | Existing concrete resource peak and symbolic formula match source `QubitCount`; unrelated keys return the original `NotImplemented` |
| `__str__` | Source metadata representation checked against native shape, target axes and widths; native dataclass representation remains domain-owned |
| `wire_symbol` | Existing `qrom_wire_symbol` covers valid signature ports and diagram title |
| `nth_operation_callgraph` | Existing immediate calls count loaded set bits; every row is checked for scalar and shaped targets |
| `build_call_graph` | Existing concrete immediate graph and independent symbolic formula; all seven QEC categories and both shim profiles are compared |
| `adjoint` | Original returns itself; source-matched XOR execution and existing all-branch channel evidence retain the self-inverse logical map |
| `_qrom_small`, `_qrom_multi_data`, `_qrom_multi_dim`, `_qrom_symb`, `_QROM_DOC` | All four actual factories are invoked; the three concrete factories also pass original library depths 0/1/2 and typed primitive execution |

The actual concrete factories pass through the existing shared component
builder, explicit work allocation/free, existing table emitter and normal
binder. Every primitive, operand and measurement feedback dependency is
compared against unchanged original decomposition, returned wire ownership is
retained, and CircuitIR roundtrips. The two-axis factory retains the original
eight-bit target declarations rather than minimizing their widths. Full
complex branch and clean-work evidence is reused from the earlier small
scalar/shaped/control suites and the complete base consumer suite. The large
documentation factories receive literal primitive verification, without a
claim of exhaustive statevector simulation at their larger widths.

Pruning checks deliberately include repeated shaped rows whose cells differ.
The source does not prune those rows because it compares flattened target cells
as well as rows. Prefix slicing and multiple datasets are checked directly.
An empty source pruning slice raises `IndexError`; valid native traversal never
requests an empty interval. This failure is retained as a source boundary.

Source QEC declarations and actual emitted primitives remain separate. The
actual factory resource models are compared independently across all seven
categories and logical peak, including the original product-over-datasets
shape-only CNOT formula. These model comparisons do not certify the physical
singleton or zero-width target cases. Whole-file acceptance remains open until
those recorded differences and the direct base-file admission gaps are closed.

The typed binder assigns fresh classical slots to repeated measurements on the
same recycled work wire. The literal comparison checks that each feedback
condition reads that wire's latest measurement before renaming the slot to the
original source key. Gate count, physical operands, per-wire order and feedback
values remain unchanged by this lifetime-checked name mapping.

| Actual source factory | Measured T-family / CX / measurement / clean work | Independent source model AND / Clifford / measurement / logical peak |
| --- | --- | --- |
| `_qrom_small` | 12 / 26 / 3 / 2 | 3 / 13 / 3 / 8 |
| `_qrom_multi_data` | 12 / 33 / 3 / 2 | 3 / 20 / 3 / 12 |
| `_qrom_multi_dim` | 28 / 77 / 7 / 3 | 7 / 44 / 7 / 23 |

The complete lookup dependency regression passed 474 cases, the local
integration lane passed 17, and the separate Cirq 1.6.1 lane passed 186.
Strict typing, Black, Ruff, team scope, architecture, IR and protected API
checks passed. The dependency-policy scan retains 666 repository findings,
with zero findings on this batch's changed paths. Frozen source hashes and
unchanged physical owners are recorded in the external evidence receipt.
