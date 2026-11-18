# Chebyshev block encoding source files

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`.
The frozen files under `qualtran/bloqs/block_encoding/` are Apache-2.0,
copyright Google LLC.

| Item | Source file | Frozen Git blob |
| --- | --- | --- |
| Q-0279 | `chebyshev_polynomial.ipynb` | `30fa0a9688a8704c7ed9cc32619ca7ed96c93ada` |
| Q-0280 | `chebyshev_polynomial.py` | `327d0ace4b411c9d8290d59d496e717b0c9e3915` |
| Q-0281 | `chebyshev_polynomial_test.py` | `2bc29ea862924b8e213d94b9f3f15ff1c53414b9` |

## Unscaled block

`ChebyshevBlockEncoding` retains the child's separately ordered system,
ancilla and resource ports, uses identity signal preparation, returns alpha
one and scales the child error by the polynomial order. The binding follows
the original reflected forward/adjoint pairs and its optional forward tail.
Degree zero emits no child calls. The zero-width ancilla omits reflections.
The exact adjoint reverses the call order and switches each child direction.
A positive outer control binds the reflection on the same predicate as the
supplied controlled child.

The reflection compiler owns its phase target, predicate workspace and
classical bits. Each reflection releases its work before the child call.
Separate child and reflection work reservations can overlap between these
sequential stages. The source forward and reverse call order and complete
complex output agree for orders 0, 1, 2, 4 and 5 across the tested layouts,
including all histories of the two-reflection, three-ancilla-bit case. The
unmodified original order-four matrix fixture gives the same projected block.
Independent reflection tests cover the primitive mark, feedback, phases,
work cleanup and source operand order.

The direct source graph omits the final odd reflection count when ancilla
width is nonzero. The executable source decomposition includes that
reflection. The model therefore exposes both routes. Physical emissions
retain the decomposition sequence. For order two with three ancilla bits
and a T/T-dagger child, the physical stream contains two child T/T-dagger
instructions plus 16 T/T-dagger instructions for the four measured
conjunctions. Four measurements are emitted, and three reflection work
qubits are reused. The model counts the two child T gates, four And
operations and four measurements separately.

## Scaled block

The scaled block retains the source power-basis conversion, coefficient-wise
multiplication by the child normalization, and conversion back to Chebyshev
basis. Its terms include zero coefficients and preserve degree order. Orders
zero and one directly expose the unscaled child and retain its normalization
one, including when the original input normalization differs. The model
delegates to existing LCU metadata and signal preparation. The caller binds
controlled terms through the shared LCU SELECT callback and appends the
resulting operation to the circuit.

The original fourth- and fifth-order fixtures match the native coefficients,
normalization and packed widths. The native second-order, one-bit
coefficient-precision fixture with one child ancilla matches complete complex
outputs for forward and adjoint across all 32 measurement histories per
direction. The native state has zero in every temporary work bit at the end
of each branch. Its direct gate model matches the locked source, including
58 Cliffords, four Fredkin gates, five And calls, five measurements and peak
width 11. The decomposition model includes one additional controlled
reflection at degree one, adding four modeled Cliffords. The executable
stream emits 48 T/T-dagger instructions from four seven-T Fredkins and five
four-T conjunctions, plus five measurements. The default source
T-equivalent rates assign four T equivalents to each Fredkin.

## Test coverage and remaining scope

The unchanged original non-notebook suite passes 29 cases with two mode
skips and three expected serialization failures. The original notebook
executes all 20 cells independently. The native 22-cell tutorial preserves
the four original fixtures, two signature figures, two call graphs, and the
unscaled/scaled derivations. It adds an executable append and inverse example
covering the full odd-block matrix.

Q-0279 closes with the native tutorial. Both native constructors now expose
the packed through signature, including zero-width omission. The scaled
constructor also declares the target, selection and junk registers in source
order. Q-0280 closes with the external binding, symbolic model and complete
history evidence below. Source export and serialization failures are explicit
admission boundaries, not successful physical executions.

Q-0281 closes with the original concrete tests and native counterparts.
All 20 matrices from the original seeded random-Hermitian family compare
both the projected block and the full eight-dimensional logical operator.
The full comparison starts with clean scratch for every logical column
and checks that scratch is zero at the output.

The original unscaled order-four and order-five fixtures compare complete
complex columns for both system basis inputs with zero ancilla and resource.
The order-four circuit covers every measurement history; the order-five
circuit has no measurement. For the original large scaled cases, the source
contracts a vector for each system basis input and keeps every logical
output. Sparse execution compares every populated amplitude and its index,
including any nonzero ancilla and resource outputs. The all-zero and
alternating measurement histories each agree, and every temporary work bit
ends at zero. This covers two histories per scaled case, where exhaustive
history counts would be exponential.

| Scaled order | Reserved native width | Emitted T/T adjoint | Emitted measurements | Source peak | Source And | Source Fredkin |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 39 | 426 | 61 | 37 | 61 | 26 |
| 5 | 32 | 178 | 34 | 32 | 34 | 6 |

Reserved width is the explicitly bound native register pool. The source
peak is the independent QubitCount graph result. Native T emission charges
four per measured And and seven per exact Fredkin. The source QEC model
reports those operations as distinct categories. The original non-notebook
suite includes four slow tests, two upstream skips and three expected
serialization failures.

## External recipe intake

The shared optional component snapshot adapter now invokes the unchanged
`TwoBitCSwap.to_clifford_t_circuit()` recipe with its explicit `ctrl`, `x`,
`y` named-wire mapping. This atomic source exposes no standard decomposition;
the adapter uses its own exact public recipe, not a synthesized matrix or a
replacement algorithm. Forward and adjoint captures preserve all 17 literal
instructions, including seven T/T-dagger gates, and the full complex Fredkin
matrix through native IR JSON roundtrip. The entire source swap file is pinned
to blob `5c2481f7a1ee0a592d1192999ddf7e6dba8facc3`.

Initial Chebyshev ordinary-direction intake failed at this atomic leaf. With
the source recipe available, the original unscaled even and odd factories
capture both directions: respectively 520 instructions/eight reserved wires,
and five instructions/one wire per direction. Their private workspace and
classical slot requirements are `(2, 8)` and `(0, 0)`. The original scaled odd
factory captures 989 instructions/32 wires in each direction with `(14, 34)`
requirements. At recipe intake these were capture statistics. The subsequent
consumer replacement, numerical and all-history checks are recorded below.

The original scaled even capture now reaches
`Controlled(SingleQubitCompare)`. Its compute/adjoint partial registers have
no source-controlled graph or Cirq decomposition. Direct probes test that
failure unchanged. The adapter does not fabricate a controlled isometry.
The original even factory protobuf check fails on `SpecializeOnCtrlBit.ONE`;
both scaled factory deserializations cannot resolve the original class name;
the unscaled odd factory serialization passes. These distinct outcomes are
recorded separately from native IR serialization.

The recipe dependency passes 119 concentrated external-block, shaped-port,
feedback and sparse-oracle regressions; its final exact recipe and original
comparator-boundary tests pass separately. Strict mypy, Black, Ruff, owning
team scope, architecture, multilevel IR and protected API checks pass.

## Complete definition closure

| Original definition | Native owner and acceptance |
| --- | --- |
| Both constructors, metadata, signatures and signal preparation | Existing unscaled/scaled classes, identity-only validation and typed packed signature owners. All four original factories, both order-zero/one shortcuts and the original coefficient transform remain covered. |
| `reflection_bloq` | Shared Compiler zero-reflection lowering and separate concrete/symbolic models. Source odd-order direct-graph undercount remains separate from executable reflection multiplicity. |
| `build_call_graph` | Concrete directional child models and `source_symbolic_chebyshev_resource_formula`. The symbolic fallback composes independent child/reflection declarations; no circuits are emitted for estimates. |
| Both `build_composite_bloq` methods | Native alternating child/reflection binding and the accepted LCU owner. Actual external child snapshots replace the unchanged native consumer. |
| Scaled `linear_combination` and register selectors | Existing coefficient basis transform, preserved zero coefficients, nested terms and original port order, followed by the shared native LCU binder. |
| Display and four factories | Original teaching counterparts, four signatures and call graphs in the accepted complete notebook; source/native frozen asset hashes remain checked. |

The actual original even and odd child objects replace the native unscaled
consumer in both directions. A coherent input occupies every logical flag,
signal and resource configuration. The whole output vector agrees with the
unchanged original tensor; all 256 histories of the eight-measurement even
case are evaluated, with every private work bit zero. The zero-measurement odd
case and native IR roundtrip are checked through the same append interface.

The source expansion adapter preserves its expanded primitive order using
one operation per inserted moment. It no longer performs an additional greedy
scheduling pass during conversion. The reference source is unchanged. This
retains contiguous source And recipes and permits the existing channel proof
to certify every history of both captured unscaled roots and the scaled odd
root directly. Earlier failures of the contiguous-macro certificate on the
rescheduled stream are not counted as passing evidence.

The scaled odd whole-root external snapshot replaces the native operation in
both directions. Full sparse complex output on a coherent system input agrees
in both numerical histories, including all occupied logical auxiliaries and
zero private work. Every opcode count matches: both contain 989 instructions,
including 178 T/T-dagger gates and 34 measurements. Private allocation and
classical slot names are adapter-owned; identical gate totals alone are not
used as proof of operand or phase equivalence. Reused primitive/call checks,
the numerical comparison and independent channel proof supply that evidence.

Both original scaled fixtures now compare complete outputs for each system
basis input in forward and adjoint directions. The native four- and five-order
streams also pass the channel certificate for all `2**61` and `2**34` histories
per direction, including IR JSON roundtrip. Their 16 and 14 private work bits
are proved clean. Logical auxiliaries are retained outputs, not declared clean
scratch. This is a compositional proof plus numerical reference columns, not
an exponential numerical enumeration.

## Symbolic declarations and source diagnostics

Eight unscaled/scaled declarations at orders zero through three preserve the
shared packed signature before substitution. Explicit resolution binds actual
external children in both directions and checks the native compiled operation,
all-history cleanup certificate and JSON roundtrip. The source scaled wrapper
may produce a delegating composite; its unresolved child still refuses physical
decomposition. No automatic symbolic circuit execution is claimed.

The shared symbolic reflection model uses the existing predicate and QEC query
owners. The Chebyshev fallback retains both child directions, the source's
direct reflection multiplicity, and source signature/callee peak maxima.
Even zero-count callees participate in the source peak query; degree zero must
not silently drop them. Both QEC profiles match the unchanged source at orders
zero, one, two, three and five. These are symbolic declarations, not reserved
physical wire counts.

SymPy integer zero and one signal widths expose original source resource-query
failures (zero-port Identity and symbolic one-bit CtrlSpec). Native symbolic
queries explicitly reject them instead of returning negative or invented
budgets. SymPy integers two and three match the source diagnostic gate and
peak formulas. Resolved Python integer widths use the separate concrete model.
The source odd-reflection undercount, unsupported partial controlled comparator
and three original serialization failures remain unchanged and independently
tested. They do not authorize replacing the reference or hiding a native gap.

The concentrated Chebyshev, reflection, certificate, feedback, block snapshot
and Unitary regression passes 163 tests with two workers and no skips. The
shared external/block/sparse regression passes 114 tests. Final source integer
diagnostics and symbolic model checks pass 25 cases; both original large
scaled fixtures pass full-output forward/adjoint comparison in four cases.
All three touched production files pass strict mypy, Black and Ruff; all
three ownership scopes, architecture, multilevel IR and protected API checks
pass. The unchanged complete native notebook passes its independent kernel.
Earlier frozen original-suite and teaching evidence is reused.

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/register_chebyshev.ipynb
```
