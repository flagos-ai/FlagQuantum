# Grid potential file audit

This audit covers Q-0429 and Q-0430 in Qualtran v0.7.0 at `bcb17bb25360ed68ce3bc084645db3e929799389`. The unchanged original test file returned ten passes and three skips. Q-0430 closes with the original cases and the native comparisons. Q-0429 retains the execution gaps below.

The pair takes two opaque through registers, `system_i` and `system_j`, with three equal-width axis words each. Total potential takes one opaque `system` register shaped `(electrons, 3)`, using the same coordinate layout as grid kinetic energy. Tests check source port shapes, order, and row-major indices. The upper-triangular traversal reuses each electron's returned wires in the next incident pair and returns the final wires in their original positions.

At the pair level, the checks follow all three coordinate operands into the forward adders, the sum of squares and the adjoint adders. They also follow the cast and QROM selection, the four coefficient banks in order into the polynomial, the squared-distance and polynomial words into Newton, and its estimate through QVR into release. Every source allocation and release is accounted for. The resource model uses the source's greedy scheduling order and shorter coefficient-prefix table. Its two costing modes are compared per child and in aggregate.

The native dynamic path accepts `circuit.append(op)`, binds big-endian coordinate words and reclaims caller-provided workspace after each pair. It computes signed axis differences and clears the squared distance after applying the phase. Native regression covers all one- and two-bit squared-distance inputs and its measurement histories, and full complex pair-phase states in both directions with feedback, including the cubic option. The one-electron operator is the identity.

| Quantity | Native emitted gates | Locked estimate |
| --- | --- | --- |
| Two-electron baseline fixture | 3,104 T-family instructions, 776 measurements, 177 workspace wires | Peak 81 logical qubits |
| Two-electron cubic fixture | 4,136 T-family instructions, 1,034 measurements, 212 workspace wires | Model peak 81 logical qubits |

The original pair feeds the second axis straight into the out-of-place adder. At equal unit inputs that leaf produces two, while the native difference is zero. The locked comments describe a sign adjustment but the decomposition does not implement it. The locked polynomial and Newton specify costs without primitive decompositions, QVR leaves the angle unspecified, and the original `Free` calls provide no matching native cleanup stream. Arbitrary user-supplied pair coefficient data remains outside the native physical binding.

The shared chemistry drawer preserves pair `U_{label}(dt)_ij` and total
`U_{label}(dt)` labels, with default, custom `Coulomb`, and empty labels.
The whole layouts include ordered `system_i`, `system_j`, and shaped `system`
axis TextBoxes. Source geometry comparisons at three grid sizes, followed by
JSON round trips and rendering, cover these presentation bindings. Existing
physical execution and resource evidence above is reused unchanged.

The grid package initializer Q-0424 exports kinetic, pair potential and total
potential. The source identities and all three imports are checked against
the defining source modules. Their native owners are GridKineticEnergy,
GridPairPotential and GridPotentialEnergy.

GridPairPotential binds the two three-axis ports through the existing
two-electron total-potential core. The coordinate words are unsigned and
big-endian. The caller supplies the phase angle and clean workspace.
Independent numerical tests compare every complex amplitude for all
64 one-bit coordinate inputs, including coincident points, for the C1 seed
and cubic option in both phase directions. The standalone C1 instruction
stream equals the total-potential stream for the same two points. The
one-update fixture emits 3,104 T/T† and 776 measurements, borrowing
177 work wires and 13 feedback slots. The output support certifies work
cleanup.

Q-0424 closes the package's three export mappings. Q-0429 retains its
source addition, angle, arbitrary coefficient and release boundaries.


## Complete constructor follow-up: interface and ownership

PairPotential owns two opaque three-axis THRU words. Its literal recipe keeps
ordinary addition (not the native signed difference), the source square-sum,
adjoint add, explicit bounded-selection Cast, four coefficient targets,
polynomial/Newton declarations, unknown-angle QVR and all seven clean Frees.
Total PotentialEnergy owns the shaped electron/axis system and visits strict
upper-triangle pairs. Original labels are presentation metadata only.
Compiler builds these typed source recipes through existing child owners.
Ecosystem converts the original default coefficient metadata using the already
verified optional real codec and passes owned integer banks into Compiler.
No numerical execution or source framework enters Compiler. Algorithms keeps
its independently specified pair/total numerical phase operations unchanged.
Existing source-stored graph resolution and exact width scheduling are reused.


## Source follow-up coverage and remaining full-file gap

| Original definitions/assets | Current native disposition |
|---|---|
| PairPotential fields/converter and signature | Both opaque three-axis ports share the Compiler signature owner; concrete banks, defaults, widths and labels are preserved from source constructor metadata. |
| PairPotential.wire_symbol | Existing complete source geometry, custom/default/empty labels, JSON and rendering evidence reused unchanged. |
| PairPotential.build_composite_bloq | Concrete shallow construction preserves all 26 source calls and recursively owned helpers; symbol-only target width also retains the complete logical recipe. Dataset and clean-Free mutations change the graph fingerprint. |
| PotentialEnergy fields/signature and wire_symbol | Existing shared grid sizing, ordered electron/axis shape and native source drawing retained. |
| PotentialEnergy.build_composite_bloq | Upper-triangle pair traversal and returned system handles match actual original graphs; default data conversion occurs at Ecosystem before construction. Empty pair loops still perform source default conversion and retain its wide-codec failure. |
| _pair_potential, _potential_energy and both DocSpecs | Both full original examples are checked without reduced data. Class/example associations and both source/test Git blobs are verified. |

Both concrete factories now bind actual source libraries at depths0/1/2.
Comparison recursively includes helper bodies, operands, result/return wiring,
constant gates and measurement conditions. Only names, SSA numbering and the
existing source logical schedule are normalized. Custom all-zero and nonzero
banks, including a five-row prefix, retain original values and QEC/peak resource
results in both source costing profiles. Existing default cost models compare
identically to eight published 0af22e136 cases. Optional qrom_data in the model
helper adds custom-bank estimates while preserving its previous defaults.

Reference SDK imports can be blocked during native construction. Compiler
receives integer banks; it imports neither the numerical codec nor Simulation.
Ecosystem composes the canonical existing coefficient constants and optional
real codec. Missing data ownership and invalid constructor metadata fail
explicitly; a numerical phase is never invented for the source recipe.

**Historical gap at 5b99eceb8.** Original width n or polynomial precision p can build a
26-call in-memory source graph, although its deeper source serialization fails
with the original object-array/pickle refusal. Native construction currently
retains only the logical signature for these two symbolic cases. It does not
yet preserve that complete symbolic lookup body and its coefficient data through
later parameter binding. The source failure is retained but does not discharge
the native logical-graph obligation. The remaining work is to carry static
lookup data with the symbolic logical recipe through existing owned graph/input
representations, or establish an explicitly approved contract if those cannot
express it. Do not disguise this as a complete source-file closure or discard
its data to manufacture an executable-looking declaration. Symbolic target
width alone already has full source/native graph parity.

## Complete admitted native numerical channels

The independently specified numerical operator remains unchanged. Every one-bit
coordinate basis word and a coherent spectator is covered for two electrons
with zero/one C1 updates, three electrons with one update, and two-electron cubic
interpolation. Both forward and adjoint are checked. Each full complex diagonal
column agrees with the independent rational C1 or decimal cubic oracle, every
private work bit is zero, and standalone pair/total instruction streams agree.
The largest system has 512 basis words and 1024 spectator-extended columns.

Actual literal And blocks and their erased conjunctions prove the same positive
local branch scalar in both outcomes. Composition establishes every combined
history, including all 2**2328 histories for the three-pair fixture, without
claiming enumeration or dense simulation. Existing sparse trajectory and small
forced-history evidence remain separate checks. There is no claim that native
signed difference is the original pair's unsigned addition, or that its
approximation is an unspecified original polynomial primitive.

| Native numerical fixture | Measured primitives/storage | Separate original default model |
|---|---|---|
| Two electrons, C1 precision4/updates0 | 224 T + 224 Tdg, 1110 CX, 112 measurements, 16 CCX, 5 phase; 33 clean, 11 slots, 39 bound wires excluding spectator | 1975 Toffoli, 74 And, 683 Clifford, 24 rotations, 74 measurements, peak81 |
| Two electrons, C1 precision4/updates1 | 1552 T + 1552 Tdg, 7824 CX, 776 measurements, 16 CCX, 5 phase; 177 clean, 13 slots, 183 bound wires | Same original default model; it is not a prediction of this independent implementation |
| Three electrons, C1 precision4/updates1 | 4656 T + 4656 Tdg, 23472 CX, 2328 measurements, 48 CCX, 15 phase; 177 clean, 13 slots, 186 bound wires | 5925 Toffoli, 222 And, 2049 Clifford, 72 rotations, 222 measurements, peak84 |
| Two electrons, cubic precision5 | 2068 T + 2068 Tdg, 10458 CX, 1034 measurements, 6 CCX, 6 phase; 212 clean, 13 slots, 218 bound wires | Original default polynomial15/target24 model remains separate |

Concentrated regression:272 passes in each Cirq lane; local integration:145.
Six changed Python files pass strict typing, formatting and lint. Three owner
scopes, language, links, architecture, Program IR and public API checks pass.
The original ten passes/three source skips and source drawing evidence are
reused. Qualtran0.7.0, real Kickmix0.1 Simulator and frozen blobs are verified.
Receipts use g2-grid-potential-source-*-20261126. These results complete concrete
constructor/data/model and admitted numerical-channel work, while the exact
symbolic width/precision logical-body gap above keeps the parent unaccepted.

## Symbolic input graph and binding follow-up

The same pair recipe now projects its complete call layer into the existing
ComponentLibrary input representation. Each child retains its constructor
arguments and exact register signature. All four static banks remain owned
root and QROM metadata; they are not hidden in helper names or a separate
registry. The symbolic QROM is a logical declaration in the temporary recipe,
not a claim that its body can execute before binding.

Comparison with the unmodified in-memory source covers coordinate width n,
polynomial width p, combined n/p/t, and composite dimension expressions.
All 26 instance identities and all 48 port-labelled connections agree exactly,
including allocation/release metadata and QROM datasets. Five- and sixteen-row
custom tables survive partial bindings followed by full bindings. Both binding
orders produce identical concrete library bytes. After serialization roundtrip,
the existing resolver produces the same complete Program fingerprint as the
actual source expansion, retaining operands, constants and feedback. Caller
data mutations do not alter the retained snapshot, and construction/binding
works with reference SDK imports blocked. Empty binding preserves symbolic
integer literals; unknown symbols and boolean dimension values fail explicitly.

Original object-array serialization remains refused by the original protocol.
The native protocol preserves its existing SerializationError wrapper; no
pickle fallback or new serialized vocabulary is introduced. A bound concrete
library uses the existing supported protocol and resolver.

**Q0429 remains open for symbolic resource queries.** New direct reference
intake establishes that symbolic coordinate width has a successful QubitCount
expression, although QECGatesCost raises TypeError for an undecidable selection
bound. Symbolic polynomial width has both a successful gate formula and a
successful peak-width expression. The concrete native resource helper rejects
these symbolic inputs. These missing successful query paths must be implemented
and compared before full-file acceptance; retaining the upstream QEC failure
does not discharge the peak-width obligation. The symbolic logical-body and
static-data-binding gap recorded above is now resolved.

Receipts: g2-grid-potential-symbolic-*-20261126. Existing admitted numerical
channels and concrete/default model evidence remain separate and reusable;
the numerical emitter is unchanged by this follow-up.

Concentrated regression passes 367 cases in each supported Cirq lane. The
default-lane initial run lacked optional QREF and passed 355 with 12 skips;
the separate dependency-restored QREF run passes all 12. The other lane passes
all 367 directly. Local integration passes 145 cases. Strict typing of the
four changed Python files, formatting, lint, Compiler scope, repository
language, documentation links, architecture, IR and public API checks pass.
