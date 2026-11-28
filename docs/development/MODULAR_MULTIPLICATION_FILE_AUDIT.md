# Modular multiplication source-file audit

The reference is unchanged Qualtran v0.7.0 at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| File | Frozen Git blob | Ledger item |
| --- | --- | --- |
| `mod_multiplication.ipynb` | `d8f7d6166fece00ea1ec07759ad8907bf9a41b35` | Q-0575 |
| `mod_multiplication.py` | `c0242f9735bf269564c9480c14c008d5f9621fd7` | Q-0576 |
| `mod_multiplication_test.py` | `d6a1caa6231a0dcf4769b3f8c2b747bd15e8212c` | Q-0577 |

The [native tutorial](../../examples/notebooks/modular_multiplication.ipynb)
covers all 29 original cells, including six example instances, three signature
diagrams and three call graphs. The added append section emits the five
concrete instances. Fixture tests compare every signature, direct child and
multiplicity, including Montgomery modulus metadata and the three
direction-changing ports on the dirty operation's adjoint.

## Executable contracts

- `ModDbl` consumes a big-endian residue word. The old high bit returns as
  clean work and the new low bit takes its place in the result's logical
  binding. `ModDbl†` uses the original inverse arithmetic ordering.
- `CModMulK` preserves the control on both branches. It computes and clears
  the shared second word through scaled additions and uses exact controlled
  swaps. The adjoint reverses the original component order.
- `SingleWindowModMul` takes low-order-first multiplier bits and big-endian
  accumulator wires. It retains the selected lookup index and rotates the
  accumulator binding. The whole multiplication processes windows from the
  lowest order upward and retains a big-endian index for each window.
  The adjoint consumes the result, retained indices and reduction bit in
  their original physical wire locations.

The concentrated regression reuses the unchanged-source component walkers,
including gate operands and measurement feedback in both directions. The
small complete-channel cases cover every valid basis row and measurement
history by composing their certified blocks. Complex-state tests include
spectator amplitudes and zero work. The blocked QROAM cases use complete
complex output comparisons on seeded sparse trajectories. The file audit also
checks every residue and control in the original doubling and controlled
prime-constant grids, using both original register types. The numerical
window and Montgomery suites retain array/integer order, result/history
values, and the distinct behaviors of the source callback and primitive graph.

## Counts on the original concrete examples

| Example | Emitted T/T† | Emitted measure | Emitted CX | Qubits | Source default T | Source measure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Doubling, width 4, modulus 13 | 36 | 9 | 109 | 17 | 36 | 9 |
| Doubling, width 32, modulus 1000000007 | 260 | 65 | 807 | 101 | 260 | 65 |
| Controlled multiply by 123, width 8, modulus 221 | 2744 | 672 | 7792 | 44 | 2720 | 672 |
| Dirty multiply, width 6, window 2, modulus 7 | 464 | 116 | 1125 | 42 | 440 | 110 |
| Dirty multiply, width 16, window 4, modulus 32767 | 2924 | 731 | 8394 | 108 | 2764 | 675 |

Using seven T per exact CSwap reconciles the constant-multiplier model with
the emitted T count. For each Montgomery window, the executable bank
cleanup reverses the forward lookup. Its cost is the forward lookup cost
where the source model counts the measurement-assisted lookup adjoint.
The independent source symbolic checks retain `2n + 1` CCZ equivalents for
doubling and `9n²/4 + 29n/4 - 1` for four-bit windows in both directions.

## Remaining source-file boundaries

Q-0575 and Q-0577 are accepted. The locked test suite passed 1,959 cases
including its independent notebook and both slow grids. Six source
example type checks skipped and two doubling example count checks were
expected failures. The native audit explicitly checks source signature and
direct-call equality on all six examples.

Q-0576 remains open for the recorded exterior-input counterexamples, the
window carry overflow disagreement, and the final gate-stage addition
versus callback subtraction when reduction fires. It retains the
concrete-width symbolic-modulus QROAM construction failure and the source
callback's Fermat-based assumptions. The modular package export Q-0568
has a separate accepted disposition in the ledger.

## Stored graph batch interfaces and ownership

This batch covers the complete frozen source file and its direct
arithmetic, lookup and erasure dependency closure. The existing original
1,959-case and notebook evidence remains reusable.

| Source owner | Native graph ownership and port contract |
| --- | --- |
| `ModDbl` | `modular_doubling_component`, MSB-first uint or Montgomery `x`, owns lower/sign and shared arithmetic work, returns the original logical output alias exchange |
| `CModMulK` | `modular_multiplication_component`, positive through `ctrl` and MSB-first uint or Montgomery `x`, owns a result word, two shared scale-add calls followed by the original exact controlled exchange and release |
| `SingleWindowModMul` | Window owner composes shared CAdd, XOR, clean QROAM, Add and lookup adjoint, `x` is a little-endian bit array, target and index are MSB-first, target rotation changes aliases |
| `_DirtyOutOfPlaceMontgomeryModMulImpl` | Window traversal owner handles reversed input slices, owned carry prefix, retained index words and final reduction in the source order |
| `DirtyOutOfPlaceMontgomeryModMul` | Output allocation/release shell retains RIGHT/LEFT product, index and reduced ports, with exact specialized dependency adjoints |

The source QROAM adjoint is a model-only leaf. Existing executable lookup
cleanup reverses the forward circuit and has a different resource count.
The separately owned measurement-assisted erase implementation reports its
decomposition separately from the source abstract cost. The stored window graph
records an explicit dependency disposition at this boundary.
The existing carry-overflow, exterior-input and final-reduction callback/gate
counterexamples remain recorded. Concrete graphs and foreign symbolic
admission retain separate acceptance decisions.

## Complete definition disposition

| Source definitions | Native ownership or retained boundary |
| --- | --- |
| `ModDbl`, `ModDbl.signature`, `ModDbl.build_composite_bloq` | `modular_doubling_component`, shared signed/controlled constants and original output/work exchange |
| `ModDbl._validate_mod` | Concrete odd-modulus admission in the native graph and existing physical owner, symbolic admission remains separate |
| `ModDbl.on_classical_vals` | Existing `simulation.classical_modular_multiplication`, exterior identity remains distinct from circuit domain |
| `ModDbl.wire_symbol`, `ModDbl.build_call_graph` | Existing tutorial labels and independent declarations/QEC models |
| `_moddbl_small`, `_moddbl_large` | Existing concrete catalog fixtures and notebook, normal doubling append now binds the stored graph |
| `CModMulK`, `CModMulK.signature`, `CModMulK.build_composite_bloq`, `CModMulK._Add` | `modular_multiplication_component`, original two scale calls, negative inverse representative and exact controlled exchange |
| `CModMulK.__attrs_post_init__` | Native concrete positive coprime-residue validation, symbolic construction remains declarative |
| `CModMulK.build_call_graph`, `CModMulK.wire_symbol`, `_generalize_k` | Existing symbolic declarations, tutorial labels and placeholder constant generalization, separate from concrete primitive accounting |
| `CModMulK.on_classical_vals` | Existing classical modular multiplication owner, preserve inactive and exterior identity |
| `_modmul`, `_modmul_symb` | Existing concrete/symbolic fixtures and notebook, normal concrete multiply append binds the stored graph |
| `SingleWindowModMul`, `SingleWindowModMul.signature`, `SingleWindowModMul.build_composite_bloq` | `single_window_component`, shared CAdd/XOR/lookup/Add/erase dependencies and explicit target rotation, specialized inverse uses exact arithmetic adjoints |
| `SingleWindowModMul.__attrs_post_init__`, `SingleWindowModMul.qrom` | Existing `single_window_qroam_table`, concrete divisible windows and original correction dataset/block choice |
| `SingleWindowModMul.on_classical_vals` | Existing `simulation.classical_montgomery_window`, original callback overflow boundary remains recorded |
| `SingleWindowModMul.build_call_graph` | Existing source-shaped symbolic window model, model-only lookup erasure stays separate from emitted measurements |
| `_DirtyOutOfPlaceMontgomeryModMulImpl`, `_DirtyOutOfPlaceMontgomeryModMulImpl.signature`, `_DirtyOutOfPlaceMontgomeryModMulImpl.build_composite_bloq`, `_DirtyOutOfPlaceMontgomeryModMulImpl._window` | `windowed_multiplication_component`, reversed input slices, shared window calls, carry lifetime, retained indices and original final additive reduction |
| `_DirtyOutOfPlaceMontgomeryModMulImpl.__attrs_post_init__`, `_DirtyOutOfPlaceMontgomeryModMulImpl.build_call_graph` | Concrete carry-safe admission and existing independent source model, symbolic physical decomposition remains refused |
| `DirtyOutOfPlaceMontgomeryModMul`, `DirtyOutOfPlaceMontgomeryModMul.signature`, `DirtyOutOfPlaceMontgomeryModMul.build_composite_bloq`, `DirtyOutOfPlaceMontgomeryModMul._mod_mul_impl`, `DirtyOutOfPlaceMontgomeryModMul.adjoint` | `montgomery_multiplication_component`, RIGHT/LEFT output ownership and source inlining of the shared window traversal, specialized inverse frees matching outputs |
| `DirtyOutOfPlaceMontgomeryModMul.__attrs_post_init__` | Delegated shared carry-safe, odd-modulus and window admission |
| `DirtyOutOfPlaceMontgomeryModMul._inversion_data`, `DirtyOutOfPlaceMontgomeryModMul._classical_action_window`, `DirtyOutOfPlaceMontgomeryModMul.on_classical_vals` | Existing Simulation Montgomery recurrence/value owners, preserve original Fermat assumption and callback/decomposed-circuit counterexamples |
| `DirtyOutOfPlaceMontgomeryModMul.build_call_graph` | Existing source symbolic model and direction-specific declaration evidence |
| `_dirtyoutofplacemontgomerymodmul_small`, `_dirtyoutofplacemontgomerymodmul_medium` | Existing concrete fixtures and notebook, native windowed append binds the shared stored traversal |

All 45 original definitions have assigned owners. No source reference file is
changed. The complete source-shaped forward root connections are checked for
all five classes, including the wrapper's inlined traversal. The multiply root
compares every call occurrence separately because distinct original scale
constants can reduce to an identical shared native body. Full source physical
streams, operands and fresh per-event feedback are checked for doubling and
controlled constant multiplication. Previously accepted arithmetic dependencies
retain their own complete source-stream evidence.

## Executable lookup dependency and channel evidence

`lookup.clean_lookup_component` owns scalar power-of-two clean bank allocation,
RIGHT outputs, consumed LEFT banks and shared scratch lifetimes. It reuses the
existing forward and measurement-assisted erase emitters. Generic multidataset,
multiaxis and symbolic graph admission remain separate boundaries. Forward
primitives and feedback are compared with the unchanged QROAM source for both
unblocked and blocked tables. Every raw history of these small lookup/erase
instruments preserves complete complex spectator states and returns all work
zero. Existing broader lookup/erasure evidence remains reusable.

Window and Montgomery graphs compose that measured dependency. The upstream
adjoint is model-only, so the complete window primitive stream through that
leaf has no upstream physical reference. The comparison records exact source
root connections, accepted arithmetic primitive dependencies, independently
verified forward lookup and native measured-erasure channel evidence at their
respective boundaries. Physical parity with a missing upstream decomposition
remains an explicit gap.

All valid two-input residues on the recorded three-, four-, five- and six-bit
Montgomery grids preserve the original decomposed circuit's complete complex
amplitudes, spectator, retained index/reduction outputs and clean temporary
work under three selected primitive measurement histories. Specialized inverse
execution restores x/y and releases the matching product/history outputs.
Single-window tests exhaust every raw history in the one-bit case and use
three selected histories for the larger coherent grids. These trajectories
remain distinct from an exhaustive enumeration of the full product's histories.
The original five-bit reduction and composite-modulus counterexamples are
checked directly against both the original decomposed and high-level callbacks.

The normal doubling, controlled multiply and windowed tutorial operations keep
`circuit.append(op)`. Forward doubling and controlled multiply bind the new
graphs and retain the independently verified explicit inverse emitters. Windowed
append uses the stored specialized traversal in both directions. Existing
public logical output/work aliases remain the same in these examples.

## Stored graph resource accounting

Counts below are measured from bound graphs. Source analytical total T uses
its default cost and is reported independently. Exact CSwap substitution uses
seven T per bit for the controlled multiplier. Lookup erasure introduces
physical bank measurements that the source model-only leaf does not emit.
Each difference is reconciled by substituting the measured leaf cost in the
otherwise unchanged source model.

| Graph and parameters | CX | T/T† | Measurement events | Source model total T | Source model measurements | Native/source peak qubits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ModDbl, n=8, p=251 | 213 | 68 | 17 | 68 | 17 | 29/29 |
| CModMulK, n=8, p=221, k=123 | 7792 | 2744 | 672 | 2720, or 2744 with exact CSwap | 672 | 44/44 |
| SingleWindow, n=4, w=2, p=5 | 231 | 100 | 31 | 92 | 23 | 25/25 |
| Montgomery product, n=6, w=2, p=7 | 1098 | 464 | 140 | 440 | 110 | 42/42 |

The single-window measured erase contributes 8 T and 8 events versus source
model zero in this fixture. The six-bit product's three measured erases each
contribute 8 T and 10 events versus source model zero. Quantum work is returned
clean except for the declared logical product/history outputs. Fresh classical
SSA slots equal the measurement-event counts. Earlier explicit emitters' slot
reuse and reverse-forward bank cleanup retain their independent historical
measurements in the earlier table.

## Final verification scope

The final focused snapshot passes 94 cases, including the three normal append
routes, and the complete source-file suite passes 11 cases. The initial
concentrated run collected 10,517 cases and reported 10,516 passes with one
historical reverse-forward erase measurement expectation failing. That
expectation was updated for the requested measured erasure by independently
counting the existing erase emitter and substituting its cost into the unchanged
upstream model. The final concentrated snapshot, including the definition
inventory and corrected cost assertion, passed 10,518 cases with no skips in
481.12 seconds. The 35 existing overflow/remainder warnings remain recorded.

All 14 changed Python files pass strict typing, Black, Ruff, Compiler scope,
architecture, IR and protected API checks. The default suite's established
public docstring coverage failure remains recorded. Its nine implicated modules
and test are unchanged from `8e6eb3613`. Frozen source identities, Kickmix
Simulator availability, resource substitution receipts and dependency findings
are recorded in `g2-modular-mul-native-*` in the external evidence directory.
Q-0576 remains open for the exact source failures, absent physical adjoint leaf
and generic symbolic, strict-type, count-mode and foreign admission boundaries.

## Complete-file acceptance after ordered symbolic word-root closure

The [inverse/product ordered graph review](INVERSE_PRODUCT_ORDERED_GRAPH_AUDIT.md)
completes the missing native symbolic doubling and constant-product libraries.
The literal symbolic reciprocal branch is retained, including its sign difference
from the concrete branch. Source-rejected symbolic windows remain declarations;
no unknown table data or physical leaf is fabricated. Existing five concrete
owners, scalar lookup/measured erasure, exact literal adjoints, root graphs,
complex/work certificates and ordinary append replacement remain authoritative.

Fresh checks prove QROAMCleanAdjoint lacks both Bloq and SDK physical recipes.
Its analytical model and the independent native paper-based erasure retain
separate evidence. The source window overflow, final-reduction and Fermat/
composite-modulus callback differences are preserved. All 45 own definitions
have native responsibility and direct dependency closure. Q-0576 is eligible
after verified publication and final local checks under the approved faithful
rule. Missing generic SDK/serialization providers remain in Q-0866/Q-0876, not
waived or claimed by this source-file acceptance.

## Fixed-width symbolic doubling dependency closure

ModDbl now preserves all fifteen available source calls when only the modulus
is unbound, including its exact generic adjoint. The shared template takes the
actual signed subtract and controlled unsigned correction; its changed output
alias remains the source work-to-data exchange. The constructor catalogue keeps
adjoint provenance per compilation and rejects unknown same-signature children
without an explicit fallback. Unresolved AddK and Controlled(Adjoint(AddK))
declarations reuse the existing constant catalogue; no encoder is invoked.

Widths 3/5, uint/Montgomery storage, both directions and source serialization
depths 0/1 compare all calls and operands. Every stopped child actually refuses
the unmodified source decomposition. Finite complete doubling Programs remain
literally equal to the published baseline, preserving the existing phase,
measurement-history, work-zero and resource evidence. An unbound modulus emits
no certifiable primitive stream; source QEC models remain separate. For width 3
the symbolic source model estimates And=7, Clifford=64, measurements=7,
T-equivalent=28 and peak qubits=14 in either direction. Those estimates are not
measured gates or a successful physical cleanup claim.
