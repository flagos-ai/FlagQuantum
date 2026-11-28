# Nested ECC control dependency closure

The frozen source is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. This batch implements the shared
Toffoli-control and third constant-load dependencies needed by nested ECC
arithmetic. It does not close ec_add.py or ec_add_r.py prematurely.

## Actual remaining call chain

The unchanged source permits Controlled(Controlled(ECAdd)) with six calls and
Controlled(Controlled(ECWindowAddR)) with five calls. The former's first,
second, fifth and sixth stage children genuinely refuse decomposition because
of directional ports. Its coordinate and product children are THRU and expose
24 and 28 calls at width three. The window root retains atomic Join/Split,
genuinely refused controlled lookup/erasure, and the same six-call addition.
These available children must not be replaced by refusal declarations.

The source-success native gap is currently the per-compilation curve
constructor control context. Closing it requires additional modular controls,
which in turn require a third control on constant loading. The intake is
`ftqc-reference-env/g2-ecc-control-context-intake-20261128.json`; a fresh literal
source inspection confirms these six/five and 24/28 counts. No source edit or
alternative reference is used.

## Interface and ownership

Compiler control/multi_control_component owns the existing scalar-plus-array
Pauli predicate recipe. Its extracted merged_pauli_control_program is shared
by the previous controlled MultiControlX/Z wrappers and by Toffoli's actual
get_ctrl_system callback. Existing wrapper Programs remain identical.

Compiler bits/bitwise_component_control owns repeated distinct positive scalar
controls on XorK. Word bits are MSB first; each new scalar precedes all
previous controls. For each selected word bit, the source Toffoli callback
calls ControlledViaAnd(X) with a scalar clause followed by the existing two-bit
array. It does not replace the whole word by one shared predicate. Predicate
work is private, starts clean, and is measurement-erased after each loaded bit.

Compiler arithmetic/nested_controlled_constant_component reuses this word
loader twice, preserving allocation, Always(Add/Adjoint(Add)) and free calls.
Always retains its uncontrolled arithmetic body; generic control propagation
must not accidentally control it. Forward and exact inverse retain their
source load ordering and nominal unsigned/signed/Montgomery word types.

The private constant constructor catalogue retains the repeated control count,
including zero and symbolic constants, exact inverse metadata and existing
predicate callbacks. After Toffoli's first merged callback, each subsequent
source ControlledViaAnd.get_ctrl_system prepends one scalar clause to the same
ordered array clause. The shared numerical control recipe implements that
recursion without a fixed two/three-control cap. The stored previous merged
owner is matched by its complete definition before selecting its callback.
Internal predicate implementations are not mistaken for outer bookkeeping;
only the caller's actual Split/Join callees receive unchanged controls.
The user continues to append the resolved operation through the existing entry.
No Core opcode, public export, backend policy or numerical execution is added
to Compiler. Simulation and Runtime retain actual state execution ownership.

## Evidence and resource conventions

Frozen controlled.py, toffoli.py, controlled_via_and.py and bitwise.py blobs
are verified. Qualtran 0.7.0 and a real Kickmix 0.1 Simulator are required before
reference tests. No reference lane may silently skip. The third-load cases
check all available call layers, operands and complete primitive order,
including measurement slot identities and feedback. The shared predicate's
internal source bodies are available; they are not misclassified as atomic
refusals. Every small raw measurement history is compared as an unnormalized
complex vector, retaining its branch scalar and all zero-work amplitudes.

Both exact directions and unsigned/signed/Montgomery constants are covered.
Serialization at original depth zero/one retains the same native Program and
SDK-blocked resolution. Existing actual DynamicCircuit.append execution and
all prior two-control cases remain part of regression. New XOR constructor
resolution is separately compared with its authoritative shared numerical path.

For a two-bit word and one loaded bit, the three-control XOR emits H6, T4,
Tdg4, CX13, S2, CZ2, measurement2 and reset2. The independent source model
reports And2, Clifford3, measurement2, explicit-T0 and no Toffoli. Expanded
primitive T8 equals four per modeled And; the sparse Clifford field is not the
complete expanded Clifford count. Five logical qubits touch seven wire slots.

Three-control AddK(1) and its exact inverse each emit H15, T10, Tdg10, CX35,
S5, CZ5, measurement5 and reset5. Their source models report And5, Clifford10,
measurement5, explicit-T0 and no Toffoli. Five logical qubits touch nine wire
slots, matching the independent source peak nine. Resource receipts also
report unused caller-reserved storage separately; it is not actual touched work.
No source-model Clifford category is presented as an emitted primitive count.

## Remaining obligations

The curve constructor context and next controls of modular addition,
subtraction and doubling remain open, including original-source available
nested coordinate/product stages. This dependency delivery does not mark the
ECC parent files accepted. Original directional, dirty multiplier, symbolic
AddK and lookup/erasure refusals remain explicit. Additional scalar prefixes reuse the literal upstream callback rather than a
new whole-word control synthesis.
Proposals 119, 131 and 132 remain unapproved. No G3, PR merge, ReadTheDocs or
large dependency installation is included.

## Replacement proof and intake follow-up

All 60 existing controlled Pauli wrapper Programs, 42 two-control word-load
Programs and 36 two-control constant-addition/inverse Programs are exactly
equal to the complete stored Programs at 58e1a2ab2. This equality covers all
helpers and operand bindings, not just gates or summary counts, and justifies
reusing their previous numerical certificates. The three-control constructor
roundtrips run both original serialization depths without SDK dependencies;
fourth and fifth controls also resolve and execute rather than being
represented as empty executable bodies. Symbolic constants stay unencoded declarations at both two
and three controls, and physical binding fails explicitly.

The fresh nested modular intake is recorded separately in
`ftqc-reference-env/g2-ecc-nested-modular-intake-20261128.json`. The forward
double-Controlled ModAdd root has 17 calls; ModDbl has 15. Forward
double-Controlled CModAdd fails in unchanged source for both original control
polarities with a scalar-array conversion TypeError. Double-Controlled
CModSub's five-call shell is available and retains that actual failing child.
These root outcomes must not be confused with their inverse direction; intake
records each direction independently before implementation.

The inverse CModAdd roots are different: both polarities expose 14 calls, while
ordinary inverse ModAdd retains 17 and inverse ModDbl retains 15. Both ModSub
and CModSub keep five-call inverse shells. An implementation must preserve
these directional differences rather than classifying the inverse from the
forward failure or substituting an inverse of an alternative forward recipe.

## Recursive prefix validation

The concrete word and constant tests cover two through five scalar controls.
Forward and exact-inverse one-bit AddK exercise all 64 and 256 raw histories
for four and five controls. Complete vectors retain complex branch scale and
zero private storage. XOR covers all eight and sixteen raw histories at those
depths; the zero-pattern identity also retains five controls. Original depth
zero/one serialization and SDK-blocked resolution cover zero through five
controls. The catalogue's existing negative/grouped predicate tests remain in
the concentrated suite. These are executable source comparisons, not a claim
that a limited three-control declaration suffices for larger source contexts.

An initial recursive attempt classified every retained directional helper as
bookkeeping, including an unreachable predicate implementation whose separate
closure was missing. The corrected selection only inspects the actual word
callers' rebindings and stops at the fully matched merged predicate owner. No
error suppression or reference change was used. An initial runner invocation
omitted reference dependency paths and was not acceptance evidence; the final
required lanes use g2_execution_python with real dependencies and preloaded
Torch. The oversized expanded old-ECC run hit its 600-second budget; a separate
185-case per-SDK run completed the changed dependencies and the remaining full
old ECC context file. Its old-path certificates are retained with the complete
Program equality proof, while final recursive checks exercise the final code.

## Final batch checks

The final recursive code passes 168 concentrated cases independently in Cirq
1.7 and 1.6, plus 82 local integration cases, with no skipped reference checks.
The 24 warnings per reference lane are retained upstream protobuf deprecations.
Strict typing covers six Python files. Black, Ruff, Compiler scope, architecture,
IR/API, required checks, Cirq interoperability, English, local links and
whitespace pass. The 138 complete previous Programs still match after the
recursive generalization. Nine resource rows separately record emitted sites,
source model categories, source total T, touched wires and reserved capacity.
Conditional feedback sites are counted in the emitted stream; branch execution
counts may differ. All measured branch comparisons retain exact feedback.
