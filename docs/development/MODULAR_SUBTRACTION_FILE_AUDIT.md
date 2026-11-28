# Modular subtraction and package source-file audit

The reference is unchanged Qualtran v0.7.0 at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| File | Frozen Git blob | Ledger item |
| --- | --- | --- |
| `mod_arithmetic/__init__.py` | `774aa44d6c868ebc8eaa08386ac9b318b0d898de` | Q-0568 |
| `mod_subtraction.ipynb` | `e925323025d5e221d32d61c6974680b95c759d07` | Q-0578 |
| `mod_subtraction.py` | `e0d0fbf81a81ec2ed55e02b4b31ad3e75ab85324` | Q-0579 |
| `mod_subtraction_test.py` | `3d013fdd1c43b5853681c2920fd7296ba00bdedc` | Q-0580 |

The [native tutorial](../../examples/notebooks/modular_subtraction.ipynb)
covers the four source examples, their signatures, all direct children
and counts, and all eight diagrams. It also binds four concrete operations
and both polarities of the controlled ones via `circuit.append(op)`.
The operation binds control, big-endian residue inputs, and caller-owned
clean work in signature order. Only the negation target or subtraction
right operand changes. The adjoint uses the original inverse components.

The locked source suite passed 428 cases including the independently run
notebook. Two negation fixtures skipped strict type checks, and their
fixture count cases were expected failures. The native audit explicitly
compares these fixture signatures and declared children, including
zero-controlled X, fanout, and the controlled conjunction's release.
The prime residue grids run with unsigned and Montgomery source types
and both control polarities. The large seeded grid checks widths 32
and 33 at both source primes. Existing forward and adjoint suites cover
primitive operands, feedback, independent resources, complex amplitudes,
and work cleanup. The append audit composes every valid basis row through
the feedback certificate, and checks complete complex output and
adjoint restoration at width two. The existing small negation suites
also enumerate every measurement outcome.

## Resources

These are forward primitive counts for width three and modulus five.
Source CCZ equivalents are symbolic declarations evaluated at width three.

| Operation | Control activation | Emitted T/T† | Emitted CX | Emitted measure | Clean work | Source CCZ equivalents |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ModNeg | unconditional | 24 | 56 | 6 | 6 | 6 |
| CModNeg | 0 or 1 | 28 | 62 | 7 | 7 | 7 |
| ModSub | unconditional | 60 | 168 | 15 | 8 | 15 |
| CModSub | 0 or 1 | 80 | 221 | 20 | 9 | 20 |

## Package exports

The pinned source initializer provides the following 13 exports.
The audit checks their original import identities and their respective
FlagQuantum implementation owners under `compiler/registers/modular`.
Q-0568 is accepted at that ownership boundary.

| Source exports | FlagQuantum owner modules in the same order |
| --- | --- |
| `CModAdd`, `CModAddK`, `CtrlScaleModAdd`, `ModAdd`, `ModAddK` | `controlled_modular_addition`, `modular_constant_addition`, `modular_scaled_addition`, `modular_addition`, `bounded_modular_addition` |
| `KaliskiModInverse` | `modular_inverse` |
| `CModMulK`, `DirtyOutOfPlaceMontgomeryModMul`, `ModDbl` | `modular_multiplication`, `windowed_mod_mul`, `modular_doubling` |
| `CModNeg`, `CModSub`, `ModNeg`, `ModSub` | `controlled_modular_negation`, `controlled_modular_subtraction`, `modular_negation`, `modular_subtraction` |

## Remaining source-file boundaries

Q-0578 and Q-0580 are accepted. Q-0579 remains open. All-ones moduli
fail the source constant load because `p + 1` exceeds the word width.
The native emitter reports that boundary explicitly. At width two and
modulus two, the source callback preserves the exterior input two.
The source negation circuit instead produces zero with a dirty work
flag, and its release fails. The active branches of controlled negation
retain the same witness. The clean execution contract is the residue domain.

## Stored graph interfaces and dependency ownership

The complete next source-file batch retains the four composite methods and
already accepted primitive, callback, model and notebook evidence. Words are
MSB first. Concrete uint and Montgomery descriptors follow the original loose
word-view connections. Input states promise residues in `[0, modulus)`.

| Native owner | Ports | Owned work and source sequence |
| --- | --- | --- |
| `modular_negation_component.modular_negation_program` | `x` | Allocate flag, X, zero-controlled word X, fanout, controlled AddK(modulus+1), zero-controlled word X, X and release |
| `controlled_modular_negation_component.controlled_modular_negation_program` | `ctrl, x`, either binary polarity | Explicit Split/MCX/Join zero test, And(control, flag), fanout and controlled AddK, measured And erase, repeated zero test and flag release |
| `modular_subtraction_component.modular_subtraction_program` | `x, y` | BitwiseNot(x), AddK(modulus+1), shared ModAdd graph, exact AddK adjoint, BitwiseNot(x), shared children own their arithmetic work |
| `controlled_modular_subtraction_component.controlled_modular_subtraction_program` | `ctrl, x, y`, either binary polarity | Same source shell around shared CModAdd graph, preserve the unconditional input shell in both sectors |

The all-ones modulus cannot load `modulus+1` in the source width. Out-of-domain
negation can leave a dirty flag. These source failures remain explicit. Shared
work uses Program-owned lifetimes and per-event classical SSA slots. The normal
append workflow and existing explicit inverse evidence retain their ownership
boundaries until the complete source-file acceptance lane closes.

## Complete definition disposition

| Source definitions | Native ownership or retained boundary |
| --- | --- |
| `ModNeg`, `ModNeg.signature`, `ModNeg.build_composite_bloq` | `modular_negation_component`, typed root and whole-word zero-control adapter, shared MCX/fanout/AddK |
| `CModNeg`, `CModNeg.signature`, `CModNeg.build_composite_bloq` | `controlled_modular_negation_component`, original array zero tests, shared positive/negative And and measured inverse |
| `ModSub`, `ModSub.signature`, `ModSub.build_composite_bloq` | `modular_subtraction_component`, original complement/constant/ModAdd/constant-adjoint/complement shell |
| `CModSub`, `CModSub.signature`, `CModSub.build_composite_bloq` | `controlled_modular_subtraction_component`, original unconditional shell around the shared CModAdd with either activation |
| `ModNeg.build_call_graph`, `CModNeg.build_call_graph`, `ModSub.build_call_graph`, `CModSub.build_call_graph` | Existing catalog symbolic declarations and independent QEC resource models, reused against direct source calls |
| `ModNeg.wire_symbol`, `CModNeg.wire_symbol`, `ModSub.wire_symbol`, `CModSub.wire_symbol` | Existing original notebook labels and eight diagrams, declarative ownership remains separate from execution |
| `ModNeg.on_classical_vals`, `CModNeg.on_classical_vals` | Existing `simulation.classical_modular_negation`, retain exterior-value identity and activation order |
| `ModSub.on_classical_vals`, `CModSub.on_classical_vals` | Existing `simulation.classical_modular_subtraction`, retain residue-domain checks and original result types |
| `_mod_neg`, `_cmod_neg`, `_modsub_symb`, `_cmodsub_symb` | Existing four catalog factories, source symbolic modulus remains outside executable graph admission |

All 28 original definitions are assigned. Existing Q-0578 notebook and Q-0580
428-case original suite receipts are reused. Original strict-type fixture skips
and expected count failures retain their recorded disposition. Q-0579 remains
open for source invalid-domain/constant-load failures and generic physical
symbolic, strict type and foreign admission.

## Concrete stored graph acceptance

The normal `ModularSubtractionExample` forward append path now binds these four
stored graphs. The existing exact inverse emitters remain the accepted
`adjoint=True` path. Forward-path replacement tests and forward/inverse complex
state restoration cover all four operations and both controlled polarities.

Unmodified source root connections, primitive names, operands and per-event
measurement feedback agree for uint and Montgomery words through eight bits.
All valid residue columns through three bits certify every And-erasure history.
Small complex states include a spectator and clean work. Every raw history is
projected for up to seven events, with three projected histories above that
bound. This bounded projection evidence is distinct from the all-history
And-erasure certificate.

The original four factories have symbolic moduli. Their accepted metadata,
models and notebook evidence are reused. Concrete graph tests exercise the
native path directly. Symbolic source factories do not provide concrete
five-mode physical admission evidence.

The ModNeg whole-word zero-control adapter retains a typed root and decomposes
through Split, shared zero-MCX and Join. CModNeg follows the source's explicit
array zero tests. Source loose word views remain visible in the graph. Strict
source dtype admission is still an open boundary.

### Separate primitive and analytical resources

For eight-bit uint words, modulus 251, the following counts come from actual
bound forward graphs. Peak qubits include the logical ports and live owned
quantum work. Every owned temporary is released on the promised residue domain.
The source analytical And and Clifford categories describe a different
abstraction level from the emitted primitive columns.

| Graph | CX | T/T† | X | Measurement events | Source model And | Source model Clifford | Peak qubits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ModNeg | 194 | 84 | 66 | 21 | 21 | 141 | 24 |
| CModNeg, cv=1 | 200 | 88 | 66 | 22 | 22 | 142 | 26 |
| CModNeg, cv=0 | 200 | 88 | 70 | 22 | 22 | 142 | 26 |
| ModSub | 538 | 180 | 67 | 45 | 45 | 380 | 34 |
| CModSub, cv=1 | 641 | 220 | 85 | 55 | 55 | 452 | 35 |
| CModSub, cv=0 | 641 | 220 | 121 | 55 | 55 | 452 | 35 |

Source analytical total T, measurement and peak-qubit costs agree with the
corresponding emitted or live counts. Program assigns one fresh classical SSA
slot per measurement event. Earlier explicit emitters reused classical slots,
so their storage count must be reported separately from event count.

The all-ones modulus correction and exterior-input dirty-flag witnesses are
checked against the unchanged source. The audit assigns all 28 definitions,
while Q-0579 retains these source defects and generic symbolic, strict-type,
count-mode and foreign admission gaps.

The final snapshot passes 124 focused and 9,899 concentrated cases with no
skips and 35 existing source overflow/remainder warnings. An initial run had
six test-import NameErrors after loading a stale test module. The import was
fixed and the entire concentrated batch rerun. All six changed Python files
pass strict typing, Black and Ruff. Scope, architecture, IR and protected API
checks pass. Frozen file identities and Kickmix Simulator availability are
recorded under `g2-modular-sub-native-*` in the external evidence directory.
The dependency audit retains 688 pre-existing repository findings and zero
findings in this batch. The established default docstring coverage failure
remains recorded. Its nine implicated modules and test are byte-identical to
the previous accepted head `7929f67b5`.

## Complete-file closure after ordered symbolic-root delivery

The [ordered modular-library audit](SYMBOLIC_MODULAR_GRAPH_AUDIT.md) now covers
all four full source roots, both controlled polarities, Montgomery words and the
source-defined symbolic-width subtraction shells. Its constructor and edge
artifacts supplement the existing numerical owners rather than replacing them.
Fresh known-width symbolic-Montgomery metadata intake matches every source gate
and feedback entry, and complete native circuits/output/work events equal their
accepted concrete counterparts. Independent source model categories and peaks
are retained. No new numerical or Core contract is introduced.

All 28 definitions and their direct dependency responsibilities are accounted
for. Under the approved faithful-migration rule, Q-0579 is eligible after
verified publication and final local checks. Earlier generic symbolic/foreign
notes are superseded by these precise defined/refused paths. Source full-ring
loader and exterior dirty-flag failures, strict/count/protocol witnesses and
unavailable symbolic numerical leaves remain separate limitations. Normal
append, all-history clean work, complex phases and exact adjoint certificates
above are reused at unchanged execution paths. No global SDK/transport promise
is made by this source-file acceptance.

## Fixed-width symbolic negation constructor dependency

The direct ModNeg factory now retains the source eight-call body for symbolic
modulus, instead of returning a whole-root declaration. Its zero-word control
also retains the actual three-call `CtrlSpecAnd` / CNOT / adjoint predicate
hierarchy, not merely an equivalent split/multi-control wrapper. The same
component template serves finite inputs. Widths 3 and 5, uint and Montgomery,
and source serialization depths 0 and 1 are compared at every available layer.
Terminal And and Partition typed views are certified before stopping traversal;
directional predicates and roots are not treated as atomic solely by side.
Every stopped source child must actually refuse decomposition. The symbolic
controlled AddK is one shared unavailable definition, even when called repeatedly.

Unbound modulus supplies no concrete gate stream, amplitude, cleanup or default
resource price. Physical binding refuses its missing constant body. This does
not by itself expand ModSub/CModSub constructor-only symbolic bodies; the
subsequent closure below retains their available fixed-width layers. Finite
source-stream, feedback, coherent-state and all-history proofs remain mandatory.

The same dependency intake now closes the direct CModNeg fixed-width symbolic
constructor: fourteen source calls retain both zero tests, measured conjunction
compute/erase, fanout and explicit constant correction. Uint/Montgomery widths
3/5, both control polarities and original serialization depths 0/1 compare every
available source layer. The correction retains the source dtype, including
Montgomery's modulus-omitted view. Only the actual Controlled(AddK(p+1)) source
refusal remains bodyless. No symbolic gate counts, amplitude or physical cleanup
are fabricated. Twelve complete finite Programs are definition-identical to
the published baseline, so accepted finite phase/work/resource evidence remains
applicable after factoring the shared template.

## Fixed-width symbolic subtraction constructor closure

ModSub and CModSub now retain their complete five-call complement/load/add/
unload/complement shell when the word width is concrete and the modulus is
symbolic. The forward and exact inverse AddK constructors keep modulus-plus-one;
the inner ModAdd/CModAdd retains all seventeen/fourteen available source calls.
Only the four actual unresolved constant definitions remain bodyless. Widths
3/5, uint/Montgomery ports, both control polarities, and original serialization
depths 0/1 traverse every available source layer and certify atomic views.
SDK-free reconstruction consumes the same stored library; physical binding
refuses the missing constant bodies. No phase, zero-work or default cost is
claimed for an unbound modulus. Complete finite Programs are unchanged by
factoring the same shell around explicit children. Symbolic-width roots and
higher generic controlled ECC dependencies remain separately tracked.
