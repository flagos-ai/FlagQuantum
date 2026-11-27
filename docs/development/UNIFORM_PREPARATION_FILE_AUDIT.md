# Uniform preparation source-file audit

The reference is unchanged Qualtran v0.7.0 at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| File | Frozen Git blob | Ledger item |
| --- | --- | --- |
| `state_preparation/__init__.py` | `024a57c221ae73f95645dbaa4247f423b037a932` | Q-0690 |
| `prepare_uniform_superposition.ipynb` | `90f6b5c08b7834551e2a3a6d091a66f43dbdaff4` | Q-0694 |
| `prepare_uniform_superposition.py` | `e567c913bff5777e899f38afae09e4c861c98d0a` | Q-0695 |
| `prepare_uniform_superposition_test.py` | `25f19b8539880ef9f0ce3ebe954cd885562780d4` | Q-0696 |

Run `jupyter notebook examples/notebooks/prepare_uniform_superposition.ipynb`.
The native notebook preserves the original eleven cells, displays the
two five-state signatures and source call graphs, and appends the
concrete bodies. It also displays the third, symbolic factory model.
Both notebooks ran in independent kernels. The native run rendered
three SVGs with Qualtran, Cirq and quimb imports forbidden.

## Original factories and tests

The active control prepares the first five labels in the tutorial
and leaves the clean workspace available for reuse. Both concrete
factories are compared at source-equivalent lowering depth, including
the comparison, rotation angles, measured predicates and feedback.
The gate stream and full complex state checks include the adjoint and
a coherent spectator. Inactive branches of the controlled body are
checked with their relative phase retained.

The native tests cover all 38 combinations from the source state
grid and both variants of each of the 13 T-complexity bounds. The
unchanged original suite passed 65 tests including its notebook.
The native tests also compare the source wire labels and mixed-control
ordering, full direct-call declarations of all three original
factories, and modern gate categories and widths. Earlier small
cases project all native measurement histories. The larger factory
and grid cases check complete complex trajectories and zero scratch.

| Five-state body | T/T† | CX | Native CH | Native Rz | Measurements | Source categories | Wires |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Plain | 32 | 64 | 0 | 2 | 8 | 8 And, 57 Clifford, 2 rotations, 8 measurements | 8 |
| Controlled on one | 36 | 70 | 3 | 2 | 9 | 9 And, 67 Clifford, 8 rotations, 9 measurements | 9 |

The symbolic count uses the source comparison and conjunction calls,
their adjoints and its declared Hadamard and Rz multiplicities. It
requires a concrete count before append.

## Abstract control dispatch

The port declaration retains the full target width and the abstract
control length `c`. For a concrete state count `n = 2**k * L` and
abstract controls, the source graph applies its comparison and phase
model to the odd factor `L` and its high-word width. The predicate is
the source's all-positive declared MultiAnd. The declared calls omit
the abstract controls, while the peak width still includes those ports.

| States | Odd factor and high width | Source And | Source Clifford | Source measurement | Source rotations | Source peak |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 10 | 5 and 3 | 8 | 45 | 8 | 2 | `Max(8, c + 4)` |
| 18 | 9 and 4 | 11 | 55 | 11 | 2 | `Max(10, c + 5)` |

The original direct call keys and both cost profiles were compared.
The source rejects literal factored predicates shorter than three
controls on the abstract path. The native declaration preserves that
failure. A literal control length resolves to positive controls, and
its append stream matches the explicit concrete control tuple. The
existing primitive counts above describe concrete execution.

## Remaining boundaries

Q-0694 and Q-0696 are covered. Q-0695 remains open for the original
symbolic dispatch boundaries. The
source symbolic graph omits control predicates. Its singleton and
one-bit symbolic peak queries exhibit the failures retained by
the existing resource tests. A concrete count with a tuple of symbolic
control values also fails in the locked source. Q-0690 is covered by
the state-preparation package audit.

## Complete source member map

| Source definition | Native owner and evidence |
| --- | --- |
| `signature`, `concrete_cvs` | `source_uniform_preparation_signature` and `normalize_uniform_controls` preserve target width and concrete/abstract control dispatch. Unresolved widths cannot append. |
| `wire_symbol`, `_circuit_diagram_info_` | The tutorial's native diagram and `diagram_symbols` retain the sum caption, control polarity and target labels. Existing tests compare the actual original Cirq caption. |
| `k_l_logL` | `_odd_factor` and uniform preparation retain the power-of-two factor, odd factor and high-word width; abstract declarations retain the source symbolic path. |
| `decompose_from_registers` | `uniform_preparation_ir` uses the existing comparison, threshold rotation, Hadamard and zero-predicate phase owners. The original outer generator and inner lowering recipes have existing exact-stream tests. Actual whole-source import is separately bound through the owned register emitter. |
| `build_call_graph` | Concrete and abstract resource owners preserve original direct calls and both source cost profiles. Symbolic source graphs omit controls; that defect is retained. |
| `_prep_uniform`, `_prep_uniform_symb`, `_c_prep_uniform`, `_PREP_UNIFORM_DOC` | All three factories are mapped in the native catalogue; the two concrete documented examples and the extra symbolic example remain in the tutorial. |

## Explicit symbolic resolution and actual source execution

`test_resolved_symbolic_source_replaces_uniform_execution` begins with the
actual `n`/`HasLength(c)` source declaration and the matching native declaration.
Both reject unresolved execution. Immutable constructor binding resolves the
state count and complete control polarity tuple; no reference implementation is
rewritten. The resolved source is converted to owned directional snapshots,
round-tripped through CircuitIR JSON, and rebound through the existing packed
callback. The user still calls `circuit.append(op)`.

The six configurations include positive/negative single controls, both positive
and mixed two-controls, and a mixed three-control case, each in both directions. Every logical basis column is
compared with the existing native implementation, preserving its complete
complex output and absolute phase. All work amplitudes outside zero are checked.
The shared exact-erasure certificate proves every history has the same channel
with its explicit `2**(-events/2)` scalar, including arbitrary coherent logical
inputs and spectators. The comparison normalizes each route by its own event
count; it does not silently equate different physical lowerings.

| n=5 route | Wires / private work | Instructions | T/T-dagger | CX | Measurements |
| --- | --- | ---: | ---: | ---: | ---: |
| Source capture, control(1) | 10 / 6 | 204 | 36 | 70 | 9 |
| Native recipe, control(1) | 9 / 5 | 204 | 36 | 70 | 9 |
| Source capture, controls(0,1) | 13 / 8 | 375 | 49 | 130 | 10 |
| Native recipe, controls(0,1), either Qualtran direction | 10 / 5 | 375 | 49 | 130 | 10 |

Both source and native two-control streams include 60 local `any` instructions
and 12 phase instructions, with matching opcodes/counts at that same leaf level.
These are not fully synthesized Clifford+T counts. The native initial H now
uses the actual shared source front end rather than measured conjunctions.
Whole-stream tests check operands and feedback with explicit workspace binding;
actual source root import also independently verifies complete consumer operators.
Source capture reserves more private names than the native reusable work layout;
that allocation difference remains explicit in the table.

Independent source declarations for control(1) are 9 And, 67 Clifford,
8 rotations, 9 measurements and peak 9; for controls(0,1) they are 13 And,
75 Clifford, 8 rotations, 13 measurements and peak 10. The native model agrees
with those declarations. The actual source capture's ten measurements differ
from its model's thirteen; those numbers are not combined into one estimate.
The physical control lowering is now aligned at the same source leaf level.
The documented source symbolic model failures remain Q-0695 gaps.

The current dependency closure and validation are recorded in
[Source controlled Hadamard](CONTROLLED_HADAMARD.md). The source-file root label
and SDK-free tutorial remain covered. The original resource-only declaration
is independent of the literal execution path: even a power-of-two target needs
no physical controlled-H workspace, while its source model can retain a larger
logical peak. No checkbox is added or parent closed for that distinction.


## Complete faithful source-file closure — 2026-11-28

Q-0695 is accepted under the approved G2 faithful-migration rule after the complete
member map and direct dependency ownership above were reviewed. No uniform
emitter changes in this batch. Existing original state/cost grids, three example
factories, documentation specification, notebook, actual source bindings and
scoped exact inverses are reused. Ordered comparison, phase, Hadamard and zero
predicate calls retain their private ownership and measured feedback.

Fresh boundary intake confirms that a SymPy Integer count fails original graph,
direct recipe and tensor paths; its k/l/logL assertions require Python integers.
The original singleton fails missing-target lookup through its actual declared
ports. Manually supplying an omitted target is not advertised as successful
whole-operation binding. Literal SymPy control zero/one reaches the original
Cirq not-iterable TypeError. Existing native conversion to concrete binary
controls is a separately admitted path and does not count as successful literal
source comparison. Symbolic control lengths remain declarations/models until
explicitly resolved. Original omitted control terms, small symbolic peak-query
failures and title/SDK boundaries remain explicit source limits.

Five-state stored programs emit 32 T-family/64 CX/eight measurements without
controls, 36/70/nine for one positive control and 52/94/thirteen for mixed
two-controls. Five private bits are touched/live; caller pools reserve twelve.
Source models respectively declare eight/nine/thirteen And, 57/67/75 Clifford,
two/eight/eight rotations, eight/nine/thirteen measurements and peaks eight,
nine and ten. Controlled H remains an unsynthesized operation. Earlier flat
Cirq-default mixed-control inventories (49 T-family, 130 CX, ten measurements
and local matrix leaves) belong to a different accepted lowering policy.
Same-policy source child/operand/feedback comparisons are reused; these policies
are not claimed to have identical global primitive inventories or fully
synthesized T counts. No performance or resource improvement is claimed.

Complete complex operator and all-history certificates, noncontiguous bindings,
spectators, private cleanup, source-free append, exact inverse, copying and IR
roundtrips remain in the existing acceptance suite. Source flaws and unavailable
recipes are retained rather than patched or replaced in the reference. Generic
SDK/workflow/transport parents and pending Proposal 131 remain open.

Final batch acceptance passes 463 concentrated cases, 32 relevant local
integration cases, 59 independent Cirq 1.6.1 constructor/nominal cases and the
final 12-case independent symbolic/nominal grid. Three Python files pass strict
typing, Black and Ruff. Compiler scope, architecture, multi-level IR, required
checks, public API baseline, English, links and whitespace pass. Thirty-eight
complete frozen source/asset/dependency blobs and actual Qualtran 0.7.0/Kickmix
0.1 Simulator are verified. Nine fresh inventory rows retain reserved capacity,
touched/live private work and independent source models separately. Ten actual
optional/required reference-absence scenarios preserve skip versus original
exception behavior. Global dependency findings remain 570 in 362 files, with
none in the added consumer; workflow parents remain open. No source patch,
checker waiver, selection exclusion, CI wait, G3 work or merge occurred.
