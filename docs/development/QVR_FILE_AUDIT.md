# Quantum variable rotation source-file audit

## Current complete-file disposition (2026-11-28)

This review applies the approved
[faithful-migration closure rule](G2_FAITHFUL_MIGRATION_CLOSURE_PROPOSAL.md).
The definition map below covers all 31 inventory identifiers, the nested
phase-size predicate, six concrete constructor fields, two examples and two
documentation specifications. Existing owners retain interfaces, dtype and
signature properties, sizing formulas and the 1..100 binary search, coefficient
quantization, numerical/symbolic models, both bodies and actual constructor,
SDK, catalog and notebook consumers.

A real direct-QVR execution gap was implemented before acceptance. The original
`QvrZPow` can build finite graphs with SymPy integer, rational and floating
coefficients or symbolic precision that yields finite indices. The former
native constructor returned declarations for all those inputs.
`symbolic_qvr_zpow_schedule` now preserves scalar dispatch, fractional count,
offset, sign operation, phase exponents and tolerances. The source iterates its
complete cost width whenever its derived count is a SymPy expression, even a
constant. Converting it to the Python-float schedule changes that decision and
can replace a successful signed body with the numeric route's indexing failure.

Formats translates finite expressions into existing owned parameter nodes.
Numerical and expression schedules share `qvr_phase_power_program`, the
existing typed phase-storage implementation. Named MSB-first cost words retain
their nominal descriptor and remain caller-owned THRU ports. Direct QVR owns
zero work, measurements and feedback. Gradient, scaled-add and measured carry
owners remain unchanged; users still use `circuit.append(op)`.

Forty-five literal profiles cover five unsigned/signed formats, both coefficient
signs and numerical or symbolic constant precision. Actual unmodified source
calls, producer/operand/result edges, exponents and per-gate epsilon match.
Full spectator-entangled complex states, source-derived inverse contracts,
output aliases and JSON pass. Positive symbolic gamma with `eps=pi*gamma/8`
also has finite source indices; the owned parameterized Program binds to the
original phase contracts. Unresolved or out-of-range indices remain declarations
with original refusal witnesses. Symbolic fixed-point metadata, including a
constant SymPy fraction, fails the original Split before scheduling. Proposal
130 raw-storage admission does not replace that source arithmetic refusal.

The original SDK retains constant SymPy Z powers as parameterized and supplies
no unitary for those raw operations, even after its empty parameter resolver.
Tests compare the unchanged source graph and scalar gate contracts, then
independently evaluate those literal Z eigenphases. They do not claim a source
primitive decomposition or SDK unitary that did not run. The first reference
attempt exposed this failure; neither source nor SDK was patched. Native
execution also passes with source SDK imports blocked. Direct metadata retains
scalar identity; generic lossy transport is not advertised as lossless.

Q-0682 is eligible for complete-file acceptance after local checks and
publication. Original precision overruns, signed numeric indexing, zero or
quantized-zero sizing, fixed-point search overflow, mixed-type recursive width,
symbolic indexing and SDK/transport failures remain source follow-ups. Extra
native arithmetic and corrected inverses stay separately specified. Historical
open decisions below remain evidence; they no longer require repairing the
frozen source or declaring a false epsilon success. Generic SDK, transport and
resource-workflow parents are not closed here.

Physical inventories stay separate from models. New direct bodies emit
`width + int(signed)` phase instructions, zero synthesized T gates and zero
work. The representative source Z-power classification is not the individual
emitted angle schedule. Existing gradient/T/measurement/peak tables and every
precision counterexample below remain unchanged.

Local acceptance: 518 concentrated QVR/Hamming/cost-phase/scaled-add cases and
32 integration cases pass. The final constructor run passes 99 cases, including
actual rational-source libraries at depths zero and one. Independent Cirq
1.6.1 verification passes 278 affected cases and the final expression suite.
Four changed Python files pass strict mypy, Black and Ruff. Compiler scope,
architecture, IR, required-check mapping, English, links, public API baseline
and whitespace pass. Eleven complete frozen source/asset/dependency blobs,
Qualtran 0.7.0 and Kickmix 0.1 Simulator are verified. Receipts use
`g2-hamming-qvr-full-closure-*20261128` in the reference workspace. Earlier
complete grids and notebook receipts retain their original scope and limits.

The reference is unchanged Qualtran v0.7.0 at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| File | Frozen Git blob | Ledger item |
| --- | --- | --- |
| `rotations/__init__.py` | `e6c63bdf43262fd6e3771a9eebd5c8e7ad75d4b8` | Q-0666 |
| `quantum_variable_rotation.ipynb` | `e04a9a4f90b717eb8dbd654b445699edc4268773` | Q-0681 |
| `quantum_variable_rotation.py` | `8177ea80e8150c79dac75e79e0868054fa2db8a4` | Q-0682 |
| `quantum_variable_rotation_test.py` | `365027236f79d3723e619f6175654f2b285c3d14` | Q-0683 |

Run `jupyter notebook examples/notebooks/quantum_variable_rotation.ipynb`.
The native notebook binds both original 12-bit factories with
`circuit.append(op)`, reproduces their signatures and direct call graphs,
and evaluates the four original data sweeps. The sizing charts report
both the formula and fixed-point search, including the 54-bit cutoff used
by the executable width selector. The first chart labels each coefficient
as `1 - 2**(-bits)`, matching the source's data. The other two charts use
the source precision-dependent terminal T estimates. Both notebooks ran
in independent kernels. The native run rendered four SVGs and four plots.

## Execution and resources

The cost word is big endian. The gradient route binds a caller-owned
through Fourier gradient and clean measured add carries. The test oracle
walks the source scaled-add children to the bound Add leaves. Direct
Z powers preserve the original retained bits, exponents and tolerance
allocation. The source declares a repeated representative Z power for
costing, while the emitter uses the individual angle on each retained wire.

| Original 12-bit body | Literal T/T† | Literal CX | Literal phase | Literal measurements | Source QEC categories | Reserved / source wires |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Direct Z powers | 0 | 0 | 6 | 0 | 6 rotations | 12 / 12 |
| Gradient, 19-bit through word | 484 | 1244 | 0 | 121 | 119 Toffoli | 49 / 31 |

The source T plots model synthesis and omit the caller's gradient
preparation. The execution tests compare complete complex amplitudes
on the supplied source basis and superposition, with per-history
projection on small fixtures. Larger cases check the measured carry
invariants and sparse coherent executions. The original one-bit direct
and gradient grids, the 12-bit direct precision grid, 48 worst-case
width/tolerance pairs, and the four T-complexity cases are mapped.
Across the four original charts, 1,248 gradient data points and 88
direct T-estimate data points match the locked reference. The source
suite returned 86 passes and three expected failures.

## Source precision and remaining boundaries

The default gradient gives maximum complex phase error
`0.00010785802409596983` at unsigned cost word `4095`, over all 4,096
cost words with requested tolerance `0.0001`. The native gate stream
matches that source fixture. The two nine-bit narrow-gradient fixtures
marked as expected failures in the source's Cirq suite have maximum
errors approximately 0.242 (integer) and 0.233 (fractional) with
requested tolerance 0.1. The tests check translation and work cleanup
against the actual source and record these discrepancies separately
from ideal phase error.

Q-0681 and Q-0683 are covered. Q-0682 remains open for the source
precision overruns, signed direct rotation indexing, zero or quantized
zero sizing, fixed-point search overflow, recursive width on affected
mixed dtypes, and symbolic execution. The symbolic declarations used
by the original tutorial are modeled. Q-0666's export inventory is accepted.

## Whole-file object binding

The source interface, both concrete classes, the complete fixed-point sizing
helper and the original test file have been read. The complete definition map is:

| Source surface | Owned counterpart and evidence |
| --- | --- |
| `QvrInterface.cost_registers`, `extra_registers`, `signature` | Named `CircuitRegisterEmitter` layout and native fixed-point register signatures. External types stop at the optional SDK boundary. |
| `QvrZPow` fields, `from_bitsize`, dtype/port properties | Native `QvrExample`, direct QVR planner, and actual source constructors with a renamed cost port. |
| Fractional/total rotation counts, body, declared call graph | Direct schedule and independent model in `qvr_zpow.py`; representative declared calls remain separate from per-wire emitted angles. |
| `find_optimal_phase_grad_size`, including its nested predicate | Existing sizing owner preserves all-ones cost, binary-search bounds, repeated-add arithmetic and acceptance predicate. Earlier 1,248-point sizing evidence is reused. |
| `QvrPhaseGradient` admission, constructors and port/dtype properties | Native unsigned admission and named cost/gradient bindings, with source signed rejection retained. |
| Phase/formula/search/chosen widths, addition bound, coefficient formats | Existing `qvr_phase_gradient.py` format and symbolic model owners. The 54-bit cutoff, formula and numerical search remain distinct. |
| Gradient body and call graph | Shared scaled-add owner, true separately captured adjoints and one declared scaled-add child. |
| Two factories and documentation specifications | Original twelve-bit objects, native catalogue/notebook and their complete directional snapshots. |

`test_qvr_source_binding.py` first constructs symbolic width/coefficient/epsilon
objects, verifies unresolved physical execution fails, then explicitly resolves
those fields with the original immutable constructor. Both signs of 0.375 are
checked for fractional direct and gradient cases, and mixed direct cost words.
Executable mixed and integer gradient cases with coefficient 1 are included.
The actual port name is `energy`; positional lowering is unchanged and the
named emitter is rebound at the same `BlackBoxPrepare` append consumer.

Full logical operators, all raw-history channel identities and independent
full-column clean-work checks agree with native replacements in both directions.
These columns include every small gradient basis word. JSON snapshots retain
owned IR, and true adjoints restore arbitrary logical states. Primitive opcode
histograms agree for these cases. Earlier per-wire source leaf, operand,
measurement-feedback, callback and phase-error evidence is reused. Direct
fractional/mixed examples also agree with their analytic complex diagonal.

The original twelve-bit direct and gradient factories are captured in both
directions. Their counts match the native catalogue and the independent source
model, including the 484 T-family / 1,244 CX / 121 measurements gradient body.
The earlier sparse/coherent tutorial checks supply numerical evidence for this
large fixture; this extension does not claim a full 31-logical-bit matrix.

Actual source cases `QFxp(3,2)` and `QFxp(3,0)` with coefficient ±0.375 and
epsilon 0.2 still fail at source dtype binding/finalization. The same two
formats at coefficient 1 execute successfully. Tests retain the former errors
and full operators for the latter cases rather than blanket-rejecting mixed
formats. Source canonical Z/S leaves can differ in name from native parameter
phase leaves for other coefficients; matching matrices do not imply identical
opcode spellings in those cases. The source remains unchanged, and Q-0682
stays open for its remaining precision/admission/model defects.

| Actual imported body, epsilon 0.2 | Instructions | T/T† | CX | Measurements | Physical wires |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fractional cost (2,2), coefficient ±0.375, direct | 2 | 0 | 0 | 0 | 2 |
| Mixed cost (3,2), coefficient ±0.375, direct | 3 | 0 | 0 | 0 | 3 |
| Fractional cost (2,2), coefficient ±0.375, gradient | 121 | 24 | 55 | 6 | 9 |
| Mixed cost (3,2), coefficient 1, gradient | 20 | 4 | 9 | 1 | 6 |
| Integer cost (3,0), coefficient 1, gradient | 0 | 0 | 0 | 0 | 4 |

These measurements and physical allocations are observed primitive inventories.
The independent source model continues to describe logical arithmetic costs;
the negative one-bit-gradient estimate and affected recursive widths are not
replaced by these observed counts. No new parent or child progress units are
introduced by this binding extension.

All 14 focused binding cases pass in 4.12 seconds. The batch regression passes
615 cases in 37.65 seconds, including the Hamming precision counterexamples.
Strict typing, Black, Ruff, Algorithms scope and diff whitespace checks pass.
This tests/audit-only extension reuses accepted architecture, multilevel IR,
protected API, CPU and independent notebook evidence. No production behavior,
reference file or public contract changes in this batch.

## Stored catalog dependency interface

The next closure retains complete QvrZPow and QvrPhaseGradient roots as typed
Programs. The cost port preserves its supplied name, unsigned fixed-point dtype
and MSB-first raw bits. The gradient is a caller-owned THRU fractional word.
Direct phases reuse the existing per-wire source schedule. Gradient sizing and
scaled-add calls reuse their authoritative owners. Carries belong to the shared
adder and return to zero. Presentation and source-label qualification belong to
the example provider. Original precision, signed-index and symbolic boundaries
remain independent of concrete catalog graph acceptance.

## Complete stored dependency graphs — 2026-11-22

The four original factory contexts now retain their stored typed Programs.
Fresh recursive comparison with the immutable source accepts eight complete
ordered graphs, including every comparable call, operand and feedback binding.
All eight comparable source drawings agree. The 69 atomic source contexts remain
explicit, including forward and inverse Square and AddIntoPhaseGrad. The 1,064
native-only descendants below those boundaries are execution realizations with
no source gate-decomposition equality claim. Complete definition matching
precedes source-label qualification and rejects changed primitive definitions.

The shared layout owner now follows `_cirq_to_bloq._ensure_in_reg_exists`.
A previously bound whole word is reused. Intersecting words are split only when
needed, then selected bits are cast or joined in original insertion order.
This corrects the twelve-bit scaled child from 37 calls to the original 19.
The primitive stream remains unchanged. Compiler storage uses native integer
position labels and typed bindings throughout.

Both original cost-phasing factories share one root filename even though their
strategies differ. Publication retains the existing first-root-file behavior.
Descendant identities carry the strategy inferred from the actual signature,
so both descendant graphs retain separate files and contextual edges.
The collision regression checks child definitions and outgoing edges separately.

Small original tensors cover direct and gradient QVR with and without Square,
all forward and inverse raw measurement histories, noncontiguous bindings and
an entangled spectator. Complete complex amplitudes agree and private cost and
carry wires return to zero. Dynamic append, stored Program copying and lowered
JSON round trips pass. Existing original-factory stream and feedback evidence
is reused. The large fixtures have primitive inventories and previous bounded
coherent evidence, without a full large-register matrix claim.

| Original factory | Measured T/Tdag | Measured CX | Literal phases | Measurements | Used physical wires | Separate source model | Source width |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| QvrZPow, twelve-bit cost | 0 | 0 | 6 | 0 | 12 | 6 rotations | 12 |
| QvrPhaseGradient, twelve-bit cost | 484 | 1244 | 0 | 121 | 49 | 119 Toffoli | 31 |
| Square, five-bit input, gradient | 1636 | 4062 | 0 | 409 | 78 | 520 Toffoli | 47 |
| Square, five-bit input, Z powers | 160 | 394 | 10 | 40 | 35 | 40 Toffoli, 10 rotations | 15 |

The square-gradient Program reuses released count/compression storage, reducing
used physical wires from the earlier reserved layout's 98 to 78. Minimal
contiguous clean-pool rebinding preserves the exact instruction stream and
allocation events. Source models retain their original formulas and do not
include every realized carry. The default source T-equivalent estimates are
66, 476, 2080 and 270 respectively, separately from measured primitive counts.
Source precision, signed-index, mixed-format and symbolic boundaries remain
recorded under the original parent entries. Reference sources are unchanged.

Batch acceptance passes 851 concentrated catalog checks with three existing
deselections and 14 existing notebook warnings, 465 execution/SDK checks with
one original deprecated-cost warning, and 29 relevant local integration checks.
Strict typing of eight Python files, Black, Ruff, Compiler and integration
scope, architecture, multilevel IR, protected API and whitespace checks pass.
All thirteen frozen source/test/notebook/dependency blobs remain unchanged.
Qualtran 0.7.0 and Kickmix 0.1 Simulator are verified. No source parent is
closed by catalog-only evidence. The recorded source defects remain open.
