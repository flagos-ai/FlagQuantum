# Swap network source audit

The locked reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The eight source and asset blobs
are pinned in `tests/unit/test_swap_network_file.py`.

| Source export | Native ownership and execution |
| --- | --- |
| CSwapApprox | Compiler `relative_controlled_swap_ir` keeps the literal relative-phase word exchange and shared fanout. The Simulation raw controlled swap owns its basis permutation. The resource declaration retains T and one- and two-wire Cliffords. |
| MultiplexedCSwap | Compiler `multiplexed_cswap_ir` uses bounded unary traversal with exact Fredkin leaves. Selection words are big-endian and targets are row-major. The output is an arbitrary through word. Coherent leaves use the live traversal predicate. |
| SwapWithZero | Compiler `swap_with_zero_ir` uses the source multi-axis schedule, traversing the innermost axis first. The inverse reverses the ordered stages. The leaf is the same relative-phase exchange as CSwapApprox. |

The exported names have named native owners. The examples bind them without
introducing a second package-level API. The [native
notebook](../../examples/notebooks/swap_network.ipynb) retains all three source
sections and their original factory dimensions, including both multidimensional
SwapWithZero shapes and the symbolic declaration. It shows the signatures and
depth-one call graphs and appends each concrete operation through
`circuit.append(op)`.

## Verification

- The approximate swap checks the four- and 64-bit factories, the bare symbolic
  factory, and the original small and large count grids. The four-bit coherent
  test compares all complex amplitudes with a spectator and verifies the inverse
  roundtrip. An unconstrained width retains the source fallback
  `Max(2, 2*n + 1)` through the shared resource dispatcher.
- The multiplexed swap checks all original size triples and the five-row
  factory. The gate comparison retains source per-wire dependencies, exact
  operands and measured feedback in both directions. The sparse execution
  compares the complex selection and data output with clean traversal work.
  Each large case uses three histories. The existing small tests enumerate
  every measurement history and include padded addresses.
- SwapWithZero checks the original basis, diagram, resource and factory
  dimensions, including the seeded multi-axis resource fixture. Both directions
  match the source stage traversal and leaf streams. Smaller cases compare the
  entire complex map and roundtrip. The native in-place callback reuses that
  stage order and is compared against every address in both original data grids.
  The symbolic model retains one fewer exchange than the product of axis
  lengths.
- The unchanged upstream suite, including its notebook, passed 64 cases with
  five skips and two expected failures. The skips are symbolic decomposition
  or dtype checks. The expected failures are upstream counts enforcement flags.

| Original fixture | Emitted T/T† | Emitted CX | Emitted measures | Reserved wires | Source declared categories | Source logical wires |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Four-bit CSwapApprox | 16 | 23 | 0 | 9 | 16 T, 87 Clifford | 9 |
| Four 32-bit rows, SwapWithZero | 384 | 573 | 0 | 136 | 384 T, 2109 Clifford | 136 |
| Five two-bit rows, MultiplexedCSwap | 82 | 101 | 3 | 17 | 10 CSwap, 3 And, 8 Clifford, 3 measurements | 17 |

MultiplexedCSwap's emitted T count uses seven T-family gates per exact Fredkin.
The source model retains the CSwap count and can convert it using its separate
default equivalence. Reserved work and live-width estimates are checked
independently in padded cases.

## Additional control on the relative swap

A scalar positive outer control directly declares controlled T and
controlled one- and two-wire ArbitraryClifford children. The counts are
`4*n`, `16*n` and `6*n - 1`. The adjoint retains the source directions
on all three children. A negative scalar control adds two X calls around
the positive-controlled swap. A shaped single-bit control declares the
positive child even when its value is one. With multiple outer controls,
the wrapper declares predicate compute, the positive child and predicate
erase. The outer signature preserves the scalar versus shaped distinction.

The append example uses the existing controlled emitter. Width-one and
width-two fixtures compare operands, gate dependencies, feedback slots
and peak width in both directions. The broader controlled exchange suite
compares full complex amplitudes and all small measurement histories,
including coherent inactive sectors and clean predicate work. The source
aggregate QEC query still fails on controlled ArbitraryClifford. Concrete
emitted counts remain available independently. For a two-bit swap with
two outer controls they include 11 CCX, 16 CH, 24 controlled phases,
four T/T† in the predicate and one measurement.

## Multiplexed swap register layout

The typed layout preserves each ordered bounded unsigned selector and
derives the target array axes from its iteration limit. Control registers
retain their shape. Target and output words stay opaque at width one,
matching the source. Selectors are big endian, and the target wires are
partitioned in row-major order. The layout supplies the concrete append
binding and the resource model with the same dimensions.

The forward and adjoint grid verifies selectors with padding, multiple
axes, several controls and singleton axes with a live control. It compares
the typed ports, primitive dependencies, all small measurement histories,
full complex output and clean work against the locked source. The
symbolic fixture verifies the declared target axes and shaped controls.
Physical binding requires concrete dimensions. Reserved wires and the
source live width remain distinct when selectors include padding.

## SwapWithZero typed and classical bindings

The compiler declares every live selector as bounded unsigned with its
original axis extent. The target port retains the full array shape,
including omitted zero-bit selection axes. Both execution routes use
the same stage traversal. The append route uses physical MSB-first
selection wires and relative-phase exchanges. The classical callback
uses integer selection bits and mutates the caller's array directly,
without changing selection objects. The adjoint reverses the stage order.

The array checks include strided views and preserve the locked callback's
keyword insertion order. A trailing zero-bit axis works. When a leading
zero-bit axis precedes an exchange, the raw source and native callback
raise `IndexError`. The compiler reports that boundary before emitting
instructions.

## Remaining boundaries

- Q-0707 now has executable resolution of the original symbolic factory,
  including actual imported controlled objects. The upstream abstract Clifford
  aggregate cost query for generic controls remains open.
- Q-0709 now has actual resolved symbolic-register bindings and native emitter
  replacement for multiple axes and shaped declared controls. Source singleton
  and zero-width selection failures, plus specific generic outer-control cases
  requiring undecomposable controlled And leaves, remain open.
- Q-0712 now has executable symbolic-field resolution and complete external
  binding. The earlier leading zero-bit axis failure remains. The compiler
  owns the stage permutation and quantum emitter; Simulation owns the in-place
  callback.

## Complete CSwapApprox source binding

The complete pinned `cswap_approx.py` definition maps as follows:

| Original member or asset | Native ownership and evidence |
| --- | --- |
| Width and signature | `relative_swap_signature` and `RelativeSwapExample`; original opaque words and extra scalar/shaped controls remain ordered. |
| `decompose_from_registers` and nested G/G-inverse helper | Literal `relative_controlled_swap_ir`, shared fanout, and accepted per-wire source gate comparisons. |
| `on_classical_vals` | Simulation controlled swap values; this basis permutation is explicitly separate from the executable relative phase. |
| Wire and circuit-diagram methods | Native catalogue/drawer presentation of the named approximate swap and its typed control/x/y ports; foreign diagram protocol objects are not retained. |
| `build_call_graph` | Shared `ControlledSwapCall(approximate=True)` resource declaration, including the bare-symbol width fallback. |
| All three factories and DOC list | Existing four-/64-bit and symbolic catalogue sections; actual execution now resolves the original symbolic factory through its immutable constructor. |

`test_relative_swap_source_binding.py` binds widths one and two from the actual
original symbolic factory. Each width covers no additional control, scalar
positive and negative controls, and a shaped two-bit mixed control. The actual
source controlled object is captured in both directions. Its owned IR snapshots
are JSON-restored and bound through the existing named emitter, then compared
against `circuit.append(RelativeSwapExample(...))`.

For every logical basis input, both directions preserve complete complex
amplitudes, including the original negative relative phases. The actual source
adjoint restores the complete initial state. Every private output is zero and
the shared erasure certificate proves the same channel for all raw histories.
This includes inactive sectors, arbitrary occupied targets and coherent inputs
by linearity; it is not only a basis permutation or sampled-state comparison.
Existing literal gate, operand, classical-value and independent resource tests
are reused, rather than duplicating the implementation or changing the source.

For width two, either direction, the actual captured and standalone native
histograms agree:

| Extra control | Instructions | Physical wires | Emitted categories |
| --- | ---: | ---: | --- |
| None | 51 | 5 | 8 T/T-dagger, 11 CX, 16 H, 16 S/S-dagger, no measurements |
| Positive scalar | 51 | 6 | 11 CCX, 16 CH, 24 controlled phases |
| Negative scalar | 53 | 6 | Same controlled categories plus 2 X |
| Mixed shaped `(0,1)` | 72 | 9 | Same controlled categories plus predicate 4 T/T-dagger, 6 CX, 3 H, 1 S, 4 X, 1 measure, 1 CZ, 1 reset |

The uncontrolled source model declares 8 T and 43 Cliffords at five logical
wires. Controlled CH/CCX/phase leaves are not priced as synthesized Clifford+T;
the original generic-control QEC aggregate fails on abstract controlled
`ArbitraryClifford` and remains an explicit upstream boundary. No reference
changes, native resource approximation or hidden replacement resolves that
query. This batch changes acceptance and audit only, and adds no parent or child
progress record.

The current bound-source batch passed eight complete-operator cases in 4.33
seconds and 508 concentrated exchange/classical/erasure regression cases in
11.20 seconds. Strict mypy on the added test, Black, Ruff, Algorithms scope,
architecture, multilevel IR and protected API checks passed. Existing source
notebook and CPU evidence is reused because production behavior is unchanged.

## Complete SwapWithZero source binding

| Complete original member | Native ownership and acceptance |
| --- | --- |
| `_to_tuple`, constructor and axis-count assertion | Immutable compiler stage inputs and typed shape declaration. Scalars, arrays and tuples are covered by existing source grids. |
| Recursive `_swap_with_zero_swap_sequence` | `swap_with_zero_stages`, including inner-axis-first traversal and little-endian stage index converted to physical MSB-first selectors. |
| Selection/target/signature properties | `swap_with_zero_signature` preserves bounded selectors, filtered zero-bit ports, original names, opaque word type and row-major target array. |
| `cswap_n`, sequence property and composite body | Shared literal relative swap and source-ordered compiler emitter. |
| Call graph | Independent source declaration retains product-of-extents minus one relative swaps; abstract source Cliffords remain distinct from primitive gates. |
| Diagram, wire symbols and string | Native source catalogue shows direction, indexed targets, selectors and graph labels rather than retaining foreign drawing-protocol objects. |
| `on_classical_vals` | Existing in-place Simulation callback, including strided arrays, insertion-order selectors and inherited leading-zero-axis failure. |
| `adjoint` | Reverse stage order through existing native adjoint flag; the actual original source adjoint is captured independently. |
| Three factories and DOC list | Existing catalogue and locked factory/resource grids. All three factories are concrete. Explicit symbolic constructor fields are tested separately, matching the original symbolic-count fixture. |

`test_swap_zero_source_binding.py` resolves the original symbolic selector,
word and extent fields with `attrs.evolve`, then imports the actual source in
both directions. Only native IR passes through JSON. The existing named emitter
binds the restored snapshots and the standalone comparison uses
`circuit.append(SwapWithZeroExample(...))`.

The four complete logical operators cover a padded three-row selector, a full
two-dimensional array, a trailing zero-bit axis and the original small example
with padded high selector bits and two-bit words. Every basis column is checked
in full complex amplitude, including arbitrary target data and padded selector
sectors; true inverse composition is the identity. Per-wire gate dependencies,
operands and exact counts agree with the captured source after binding. No
workspace or classical slot is allocated, and no measurement or reset appears;
there is no measurement-branch acceptance claim for this coherent operation.

| Selector widths / word / shape | Physical wires | Instructions | Explicit T/T-dagger | CX | Source declared T / Clifford / peak |
| --- | ---: | ---: | ---: | ---: | --- |
| `(2,)` / 1 / `(3,)` | 5 | 50 | 8 | 10 | 8 / 42 / 5 |
| `(1,1)` / 1 / `(2,2)` | 6 | 75 | 12 | 15 | 12 / 63 / 6 |
| `(1,0)` / 1 / `(2,1)` | 3 | 25 | 4 | 5 | 4 / 21 / 3 |
| `(3,)` / 2 / `(2,)` | 7 | 51 | 8 | 11 | 8 / 43 / 7 |

Both directions have these counts. Source independent QEC and peak queries are
asserted separately. The preceding in-place classical callback evidence is
reused; a basis-value callback does not replace the relative-phase unitary.

The original leading-zero-axis fixture `(0,1)` over `(1,2)` still raises
`IndexError('list index out of range')` in both direct source decomposition and
actual capture. Native compiler admission raises `ValueError` before emission.
No reference repair, axis reassignment or hidden port renaming is performed.
Q-0712 remains open for this source failure; no new child or parent completion
is counted in this batch.

The current SwapWithZero batch passed five focused cases in 2.14 seconds and
532 concentrated swap/callback/erasure regression cases in 8.95 seconds.
Strict mypy on the new test, Black, Ruff, Algorithms scope, architecture,
multilevel IR and protected API checks passed. Existing source notebook and CPU
acceptance is reused because this batch changes no production behavior.

## Complete MultiplexedCSwap source binding

| Complete original member | Native ownership and acceptance |
| --- | --- |
| Selection/word/control fields and tuple converters | `MultiplexedSwapLayout` retains named bounded selectors, opaque target words, scalar selector requirement and scalar/shaped controls. |
| Control and selection properties | Ordered native layout registers; actual custom `address0`/`address1` and shaped `enable` ports are bound without renaming. |
| Target-register property | Native layout derives row-major array axes from selection iteration bounds and retains the arbitrary through `output` word. |
| Circuit diagram method | Native catalogue/drawer conveys controls, indexed input and output words; foreign diagram objects do not enter native algorithm state. |
| `nth_operation` | Existing exact controlled swap; shared unary traversal owns compute, leaf selection and measured predicate erasure. |
| Original factory and DOC list | Existing locked five-row/two-bit example, graph, notebook and original test-grid evidence. The original factory is concrete; symbolic register resolution is an explicit constructor case. |
| Inherited signature, decomposition and call graph | Existing shared unary-iteration owner and independent resource model; existing stage tests preserve gate dependencies, operands and feedback. |

`test_multiplexed_source_binding.py` resolves actual original symbolic selector,
extent and target-word fields before importing the source. Four complete
operator cases cover a padded three-row selector with zero, one and two shaped
control bits, and two selection axes with a shaped control. Both source
directions are captured independently and restored only through owned native
IR JSON. The native replacement uses `MultiplexedSwapLayout.bind` and
`CircuitRegisterEmitter` at the same named `BlackBoxPrepare` consumer.
`circuit.append(op)` executes either implementation without changing consumers.
No new traversal, allocator, conversion registry or algorithm implementation is
introduced.

All logical columns, including arbitrary targets/output, inactive controls and
padded selectors, agree in full complex amplitude. Actual source inverse
composition is the identity. A shared raw-history certificate proves every
measurement history implements the same channel; separate complete-column
checks prove every private output zero. Both directional instruction histograms
match. Existing source-model comparisons retain every declared category and
logical peak separately from primitive counts.

For a three-row, one-bit-word table with a two-bit selector:

| Shaped control count | Instructions, either direction | Explicit T/T-dagger | Measures | Captured wires | Native reserved wires |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 73 | 25 | 1 | 7 | 7 |
| 1 | 91 | 29 | 2 | 9 | 9 |
| 2 | 108 | 33 | 3 | 11 | 11 |

The source model retains three CSwap plus one/two/three And and measurements.
Its default T-equivalence prices CSwap differently from the emitted seven-T
exact Fredkin. Tests compare the native independent declaration to the complete
source model; explicit T counts above are physical instructions, not the default
model T-equivalence.

Generic `.controlled(...)` is a different source call chain from declared
`control_regs`. On a two-row, one-bit selector, positive/negative scalar and
mixed shaped outer controls can be captured; all three now have full logical
operator, raw-history, inverse, cleanup and JSON checks against the existing
native generic-control emitter. Both directional histograms match (70, 72 and
91 instructions for positive, negative and mixed controls respectively).

On the three-row base, the same generic wrappers produce controlled And and
controlled And-adjoint leaves. Source decomposition of those non-through
controlled leaves is unavailable. Actual import therefore retains the explicit
`Foreign oracle primitive cannot be decomposed` error for all three outer
activation patterns. Declared two-bit controls do work and must not be described
as this failure. The existing native compiler likewise rejects exactly the
undecomposable generic-control cases; original singleton/zero-bit-axis failures
remain covered by the previous source tests.

Q-0709 remains open for those original source boundaries. The complete
MultiplexedCSwap class/factory is mapped and its previously missing actual
symbolic-register execution is now covered. No parent or child progress count is
added merely for the extra evidence.

The current MultiplexedCSwap batch passed ten focused cases in 7.91 seconds and
138 concentrated typed-layout, source-import, controlled-exchange and erasure
regression cases in 13.22 seconds. Strict mypy on the new test, Black, Ruff,
Algorithms scope, architecture, multilevel IR and protected API checks passed.
Existing notebook and CPU acceptance is reused because production code and
public contracts are unchanged.


## MultiplexedCSwap complete source acceptance (2026-11-28)

The approved faithful-migration closure rule supersedes historical open status
for Q-0709. Complete source review covers scalar/tuple register converters,
constructor defaults, control/selection/target properties, row-major layout,
diagram, nth_operation, original example and DocSpec plus inherited unary
signature/decomposition/model dependencies. Existing source-ordered typed calls
and specialized exact Adjoint provide the implementation; this is acceptance
of those existing owners, not a new multiplexing algorithm.

All caller control, selection, targets and output words remain live THRU ports.
Unary predicate work is private and clean; logical output is arbitrary, not a
zero-work assumption. Complete small complex operators, all-history channel
certificates, separate final-zero checks, padded addresses, multi-axis selectors,
shaped controls, forward/inverse and native replacement evidence are reused.

Source uncontrolled singleton/zero-width-axis failures and generic control of
undecomposable non-through And leaves remain explicit. Declared control_regs
and generic outer controls have different source call chains. Native bounded
Join descriptors differ nominally from source generic QAny at a recorded return
boundary; physical bit ordering and complex action are checked independently.
The exact Fredkin primitive inventory uses seven T-family gates per bit; source
CSwap model equivalences are separately reported. Large fixture samples are
not relabeled exhaustive measurement-branch simulations.


The complete three-file closure passes 335 concentrated source/primitive/
complex-execution cases and 51 independent Cirq 1.6 constructor cases. All 54
frozen source/dependency/asset blobs are unchanged. Algorithms scope, language,
links, architecture, IR, protected API baseline and whitespace pass. Existing
strict Python, 32 local integration and real Kickmix Simulator evidence from
the accepted preceding batches is reused: no Python code changes here. Original
source tests/notebook outcomes and separately recorded defects remain intact.
