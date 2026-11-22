# Physical operand arrays and strided views

## Ownership and interface

Compiler `artifacts.operand_array.OperandArray` remains the single immutable
operand storage type. Its existing tuple owner and range-backed slices retain
physical order. The additional `reversed`, `skip`, `keep`, `skip_last` and
`keep_last` methods compose views without copying values. `repeat_false` holds
one immutable cell for any count, including zero, with physical stride zero.
The logical indexing range supplies bounds independently of the physical stride.

Private `artifacts._array_views` owns static constructor hints, owned and span
empty defaults, compact copies, metadata recomputation, family checks, checked
casts, scalar payload projections and domain diagnostics. It reuses the existing
`OperandIdentity`, `OperandKind` and `OperandDomain` authority. Recomputing a
view's metadata scans its identities while retaining its original owner.

The native register binders and broadcast emitters consume these same arrays.
Users keep `circuit.append(op)`. No external library objects enter native code.
This is the existing local single-device path.

## Complete source scope

The unchanged Kickmix lock is
`9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`. The complete source group contains
`id/array_z.cc`, `array_z.h`, `array_z.test.cc`, `array_xz.cc`, `array_xz.h`,
`stride_span_x.h`, `stride_span_xz.cc`, `stride_span_xz.h`,
`stride_span_xz.test.cc`, `stride_span_z.cc`, `stride_span_z.h` and
`stride_span_z.test.cc`. These are inventory parents K-0130 through K-0134 and
K-0156 through K-0162.

The generator recursively validates all quoted local dependencies and linked
implementations against the immutable Git blob inventory. All 72 original
GoogleTests passed unchanged in the retained intake receipt. A separate C++
probe links the same original definitions to record complete view methods and
ownership observations. The reference sources remain unchanged.

| Source responsibility | Native disposition |
| --- | --- |
| Typed and union span constructors | Existing array identities with truthful private type hints; preserve mixed hints on slices |
| Dense owning arrays and conversions to spans | Immutable tuple storage with independently retained view metadata |
| Default owning arrays | Explicit empty BOOL context |
| Default X, Z and mixed spans | Explicit empty QUBIT context and zero stride |
| Copy and concatenation | Compact initialized tuple copy; recompute actual types, including different Z and mixed empty rules |
| Move construction, assignment and self-move | Original pointer transfer and moved-from hints are recorded; native immutable references retain storage without a destructive moved-from state |
| Uninitialized aligned allocation and destruction | Original count, tag and allocation success are observed; native storage accepts initialized identities and uses Python-owned lifetime |
| Front, back, indexing, iteration and equality | Existing Sequence behavior and content equality; checked native bounds |
| Reverse and clamped prefix/suffix transforms | Constant-storage views, retaining hints and owner lifetime |
| Zero-stride repeated false | One immutable cell, with independent logical count |
| Type computation and family predicates | One classifier plus existing scalar and union domains |
| Checked casts | Validate type, retain original array, offset, stride and hint |
| Unchecked pointer casts | Raw array words remain available; private payload projection canonicalizes the requested typed value |
| Boolean byte views | Probe valid zero/one bytes and source byte stride; native values have no byte-addressing API |
| Diagnostics | Preserve contents and scalar spelling, using FlagQuantum array/view names |

`cast_operand_payloads` explicitly projects a scalar value, with dense canonical
output. The source raw pointer reinterpretation retains the original tag word
and byte stride. The raw input array and source observations retain that
information separately. No equivalence between these two storage layouts is
claimed.

## Source defects and excluded observations

The original `stride_span_z::has_any_qubits` returns true unconditionally,
including for an empty or wholly classical view. The corpus preserves this
result. Native family queries inspect the admitted identities.

The original Z-only `is_all_classical` returns false for a nonempty view whose
retained hint is 64 or 192, even when all actual values are ordinary classical
operands. This occurs when a generic mixed view is checked into a Z view.
The original result and the native content-based result are asserted separately.
Other recorded family predicates and checked-cast outcomes compare directly.

The native immutable owner deliberately omits raw pointer mutation, manual free,
uninitialized reads, and a destructive moved-from state. The original allocation
and move observations are retained instead. No undefined Boolean object
representation, dereference beyond storage, or dangling pointer is used as
numerical evidence. Empty view pointer offsets are excluded from comparisons.
C++ constructor type restrictions correspond to native eager family validation.

## Execution and resources

The locked native extension and `Simulator` are required by fixture generation.
The fixture records its SHA256. Sixteen original builder consumers cover the
transforms above and both classical control values. Each includes every five-bit
basis input, exact source phase and actual per-lane operation counters.

Native tests compare ordered physical records and operands, KMX round trips,
full normalized complex superpositions, classical control results and all 32
basis lanes per consumer. Counted and uncounted execution paths produce equal
states. The logical five-qubit state retains its exact dimension, with zero
work qubits. These consumers have no measurement instructions. Existing mixed
control and measurement-assisted erasure branch evidence is rerun in the batch
regression, without claiming new measurement branches for value-only methods.

| Quantity | Evidence |
| --- | --- |
| Array/view primitive gates | 0 |
| Array/view work qubits and measurements | 0 |
| Consumer static raw physical counts | Exact original and native opcode histograms compared independently of execution |
| Consumer actual primitive counts | Exact lane counters from the original Simulator and native observer, including inactive classical guards |
| Analytical estimate | None for storage and views; no estimate substituted for actual counts |

Fixture generation uses
`tests/unit/fixtures/generate_operand_arrays.py`, with `--source`, `--inventory`
and `--output` arguments. The source root contains `kickmix/src`. Focused tests
are `tests/unit/test_operand_array_views.py` and
`tests/unit/test_kickmix_operand_array.py`. Batch receipts retain strict typing,
formatting, architecture, API, dependency and broader execution checks.
