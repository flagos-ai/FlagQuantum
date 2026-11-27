# Ordered modular source libraries

Reference: unchanged Qualtran v0.7.0,
`bcb17bb25360ed68ce3bc084645db3e929799389`, complete
`mod_arithmetic/mod_addition.py` and `mod_subtraction.py`.

## Interface and ownership

`Compiler.formats.symbolic_modular_library.modular_component_library` constructs
an owned `ComponentLibrary` using the existing typed builder and graph exporter.
It imports no reference SDK and adds no Core opcode, descriptor, schema or stable
export. The shared builder pairs typed calls with constructor declarations;
addition and subtraction recipes remain in separate private modules.

The addition and subtraction tutorial operation objects expose
`component_library()`. Their existing `circuit.append(op)` paths still use the
same concrete native component owners. A declaration-only Program is not an
identity or a place to hide constructor parameters: unresolved arithmetic remains
in the library metadata, and physical lowering continues to require bodies.

Words are MSB first. The root owns each sign/junk prefix until the exact source
Free, each loaded constant until IntEffect, and each scaled predicate until its
matching measured inverse. Caller ctrl/x/y aliases and all connection endpoints
are preserved. Constructor arguments remain structured fields, including the
original symbols, their assumptions, negative modulus, modulus-plus-one and each
scaled `Mod(k * 2**i, p)` expression. No numeric sample supplies a symbolic graph.

| Complete source recipe | Direct calls at fixed width n | Source-owned private values |
| --- | ---: | --- |
| ModAdd | 17 | Widening junk and sign prefixes |
| CModAdd, both polarities | 14 | Sign prefix |
| CModAddK | 3 | Constant ket paired with effect |
| CtrlScaleModAdd | 3n+2 | One temporary And result per iteration |
| ModNeg | 8 | Nonzero flag |
| CModNeg, both polarities | 14 | Nonzero flag and controlled predicate |
| ModSub | 5 | Work belongs to called arithmetic children |
| CModSub, both polarities | 5 | Work belongs to called arithmetic children |

CModAddK, ModSub and CModSub also have source-defined symbolic-width shells;
these now retain their finite calls. Roots that require unrolling a symbolic
split width are not fabricated. Atomic ModAddK still has no original ordered
body. Its existing documented permutation and five-Add model remain separate.
The constant ket rejects the same negative or oversized concrete values as the
original. At one bit, IntState and MultiTargetCNOT retain the source QBit port,
while a whole-QUInt zero control remains ControlledViaAnd.

## Verification and reuse

The full root comparisons use the actual unchanged source libraries, not copied
expected edge lists. They compare every occurrence, operand/result endpoint,
array index, direction and recursively referenced constructor. Widths 1/2/3/4,
both polarities, Montgomery words and the symbolic-width shells are covered.
The actual native libraries also feed the existing concrete constructor registry
with reference imports blocked. Their complete bound circuits, feedback slots,
output bindings and workspace events equal the accepted native arithmetic owners.
Their symbolic counterparts still fail physical lowering for missing bodies.

The earlier complete modular component suites provide unchanged-source primitive
and feedback comparisons, complex coherent states, small exhaustive measurement
histories, all-history And-erasure certificates, zero work on promised residue
domains, exact adjoints and normal append replacement. These are reused rather
than replaced with signature or count assertions. Receipts are recorded under
`g2-modular-logical-graphs` in the reference workspace.

## Distinct resources and wire boundaries

This ordered library supplies no analytical `counts` annotation. Counting its
root occurrences is not the original source resource model. Existing catalog
call graphs, QEC categories, peaks and historical models retain their independent
source comparisons and remain separate from actual bound primitive inventories.
The bound circuits are identical to the accepted concrete owners, so no physical
resource improvement or alternative synthesis is claimed.

The source Montgomery wire drops modulus metadata. The owned live library
preserves it; the existing legacy writer correctly refuses to serialize that
lossily. The reference test explicitly projects only this known source wire loss
for its wire-level comparison and independently checks the raw native modulus
and serialization refusal. It neither changes the actual source nor changes the
owned live graph. No new symbolic-library transport contract is claimed.

This delivers the complete ordered modular-library module. The remaining known-
width symbolic-interpretation constructor question is now independently resolved
by the physical intake below; no arithmetic implementation or Core admission
contract needed modification. Q-0570/Q-0579 are eligible for complete-file
acceptance after verified publication and final local checks.
Original full-ring constant-load, exterior dirty-flag, strict typing and source
model/protocol failures remain individually documented in the
[addition](MODULAR_ADDITION_FILE_AUDIT.md) and
[subtraction](MODULAR_SUBTRACTION_FILE_AUDIT.md) audits. Global workflow and SDK
qualification gaps remain owned by their separate parents; none is waived here.

## Known-width symbolic interpretation and complete-file review

A fresh eight-case intake uses QMontgomeryUInt(3,q), concrete modulus 2, all
five dtype-bearing operation families and both controlled polarities. Every
actual source root exists; every native constructor already supplies a physical
body. Source primitive operands and per-event feedback match. Each complete
bound circuit, output binding and workspace event equals its concrete-metadata
counterpart, so the existing complex-state, all-history and clean-work evidence
applies by exact execution-path replacement. The original seven-category model
and source peak also equal the concrete-metadata counterpart. This is reuse of
existing approved storage and arithmetic, not a numerical contract expansion.

| Operation | Actual bound instructions | Source model And / Clifford / measurements | Source model peak |
| --- | ---: | --- | ---: |
| CModAdd, cv=0 | 394 | 16 / 123 / 16 | 16 |
| CModAdd, cv=1 | 378 | 16 / 123 / 16 | 16 |
| ModNeg | 148 | 6 / 48 / 6 | 9 |
| CModNeg, cv=0 | 169 | 7 / 49 / 7 | 11 |
| CModNeg, cv=1 | 165 | 7 / 49 / 7 | 11 |
| ModSub | 362 | 15 / 122 / 15 | 14 |
| CModSub, cv=0 | 494 | 20 / 159 / 20 | 16 |
| CModSub, cv=1 | 478 | 20 / 159 / 20 | 16 |

Ten complete source/dependency blobs match the immutable inventory. Reuse the
unchanged complete original 446-case addition and 428-case subtraction receipts,
including their separately retained source strict/count/type outcomes. The
existing audits assign every original definition (40 and 28 respectively),
fixture, classical action, resource declaration and presentation responsibility.
New ordered libraries fill the missing symbolic-root artifact instead of silently
reclassifying it as an upstream defect. Source-defined finite symbolic shells
are present; source-unavailable numerical leaves and split-width unrolling remain
explicit declaration/refusal boundaries. The literal exterior-input and
full-ring loader failures remain source witnesses, while the native bounded
constant permutation is independently labeled. Generic transport/SDK workflows
remain separately owned and are not certified by this file acceptance.

The approved faithful-migration rule can therefore close these two complete
source parents after publication. No new parent IDs are created for this work.
