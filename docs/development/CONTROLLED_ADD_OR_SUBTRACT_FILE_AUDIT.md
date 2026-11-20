# Conditional addition/subtraction whole-file audit

The complete `arithmetic/controlled_add_or_subtract.py` and its original test file
are audited at Qualtran v0.7.0
`bcb17bb25360ed68ce3bc084645db3e929799389`. Both original Git blobs are asserted.
The source defines one class, its default b dtype, signature, composite body and
adjoint, three factories and one documentation specification.

## Interface and direct dependency closure

Compiler `registers/arithmetic/addition.py` owns
`measurement_add_or_subtract_ir`. The authoritative Add owner supplies its
middle operation. Shared controlled constant XOR supplies the two controlled
BitwiseNot shells, with the all-ones target pattern. X supplies the optional
outer control flips. Existing bitwise and addition owners remain replaceable by
the same consumers. The separate unitary ripple path keeps its own contract.

Physical binding is MSB-first ctrl/a/b with explicit carry qubits and classical
slots. All logical ports are through. Carries start and finish in zero, and each
has a distinct feedback slot. Original b dtype defaults to a dtype. A true
`add_when_ctrl_is_on` adds on control one. False reverses those sectors. The
source argument prose describes False inconsistently, so the actual body and
adjoint define the preserved behavior. The adjoint flips the polarity while
retaining an ordinary forward Add in the middle.

| Complete original definition | Native disposition and evidence |
| --- | --- |
| Fields, default, signature | Existing native physical binding, typed register owners and actual source capture retain original ctrl/a/b order and width. Both complete unsigned/signed eight-bit factories and their DocSpec membership are checked. |
| Composite body | Both complete factories are captured independently in each original direction. Every gate, operand and measurement feedback is compared with the original reference traversal. Existing direct native lowering tests cover eight equal/unequal width pairs, all three integer dtypes and both polarities. |
| Adjoint | Actual captured roots retain opposite polarity, original Add and exact source stream. Small coherent tests cover both initial polarities, both directions, all four first histories and all sixteen compute/inverse history pairs, retaining unnormalized branch amplitude and zero carries. |
| Concrete factories | Both eight-bit factories pass named capture, workspace/slot ownership, exact stream, independent model and CircuitIR JSON through ordinary append. Full eight-bit dense state enumeration is not claimed. Existing eight-bit boundary witnesses and branch certificates are reused. |
| Symbolic factory | Original signed n-bit metadata and model are retained. The source decomposes one level to five subbloqs. Physical capture rejects the symbolic port width. This is separate from symbolic resource estimation. |
| Documentation specification | All three factories keep their original source identity. Existing native example `examples.algorithms.register_controlled_add_or_subtract` demonstrates coherent append, measured execution, exact inverse and zero work without a new API. |

The new captured small tests use unsigned, signed, Montgomery and mixed
unsigned/signed words, reordered physical ports and a coherent spectator. They
compare complete complex output states against the original tensor, all raw
measurement histories, clean work and inverse restoration. The wider signed
input remains subject to the source Add contract described below.

## Complete original test disposition

The original file has six test functions and one naive reference class. The
class is intentionally a Bloq with an attrs constructor and is not itself a
pytest test class. `test_examples` covers all three factories. Classical
simulation checks all unsigned words at widths two, four and five.
`test_caddsub_mixed_dtype` covers Add and conditional Add at an occupied signed
result. `test_tensor` compares the complete source circuit with the naive
controlled Add/Subtract/Negate construction at widths three and four.
`test_t_complexity` checks the symbolic n-1 CCZ estimate.
`test_controlled_t_complexity` evaluates both orders of outer control and
adjoint at width ten. These original assertions run without reference edits.
Native exact stream, dtype, coherent branch and resource consumers provide the
separate execution evidence.

## Resource accounts and remaining boundaries

Existing eight-bit native execution reports 28 T/TDG, 97 CX, zero or two X,
seven measurements and seven clean carry qubits, with 24 total physical wires
and seven slots. Independent original models report 28 T-equivalent, seven
measurements and peak 24. The one-bit original Clifford model undercounts its
executed stream by four gates. That diagnostic remains recorded and is not
corrected in the reference. The existing independent formula is compared over concrete and symbolic equal/unequal widths,
all category fields and peak width, separately from the measured inventory.

Q-0183.03 remains open. Original QInt(1)/QInt(2) with control one, a=-1 and b=0
has a classical result -1, whereas its gate circuit and native lowering return
01. The source Add zero-extends that narrow signed input. This batch preserves
the actual quantum behavior and does not claim mathematical signed extension.
Generic symbolic typed Program admission and outer controlled/static foreign
conversion retain their existing boundaries. Q-0183 and Q-0184 remain open for
these source/admission issues and the original unimplemented test mode.
Logs are `ftqc-reference-env/g2-controlled-add-or-subtract-file-*.log`.

The concentrated batch passed 222 cases without warnings or skips. Strict typing
of the new source-capture test, formatting, lint, Compiler scope, architecture,
multilevel IR and protected API checks passed. The untouched original file
reports 12 passed and one skipped count-comparison fixture mode. Its two warnings
retain the original unknown slow marker and the attrs reference class that
pytest cannot collect. The original test class remains a reference Bloq.

The arithmetic and conversion export responsibility tables were corrected in
this batch to point SignedIntegerToTwosComplement's declared one's-complement
input to `ones_complement_to_twos_ir`. They had incorrectly named the separate
sign-magnitude path. The existing signed-encoding code and tests already separate
those contracts and remain unchanged by this documentation correction.

## Native stored-call execution closure

The complete `arithmetic/conditional_addition_component.py` now implements the
original build-composite body in native typed Program calls. The signature is
ctrl/a/b in source order, with caller-owned through values. The omitted b dtype
defaults to a's descriptor. Source-supported concrete unsigned, two's-complement
and Montgomery words retain their exact neutral descriptors. Symbolic physical
execution remains outside this finite binder.

The direct dependency `addition_component_program` gains optional integer dtype
descriptors while preserving its existing unsigned graph exactly when omitted.
It retains its authoritative `addition_components` sequence and shared
compute/erase owners. No signed-value codec or sign-extension circuit is added.
The one-bit case and unequal widths retain the source's actual quantum behavior.
Explicit Montgomery modulus metadata is preserved at the same boundary.

Each controlled complement starts from the native bitwise NOT Program and
uses `control_component_hierarchy`. X specializes to the existing CX primitive
function. Split and Join pass the control through unchanged. Thus the actual
Controlled(BitwiseNot), Controlled(OnEach), Split, ordered CNOT and Join levels
remain visible and compare against the original call graph, named operands and
SSA predecessor connections. The Add child has independent full stream and
tensor/branch tests. The root retains three or five original child occurrences
according to polarity, with one shared complement definition used twice.

The original declared controlled-OnEach resource recipe omits zero-cost
Split/Join views even though its actual decomposition includes them. The native
stored graph retains these occurrences. Tests compare every actual view and
separately compare the declared non-view edges and terminal multiplicities.
Measured primitives and independently queried source QEC totals remain separate.
The one-bit source model's four-Clifford undercount remains explicitly asserted.

Acceptance covers all four unsigned/signed/Montgomery/mixed dtype combinations,
both control polarities and widths (1,1), (1,2), (2,2), (2,3), (3,5), (8,8).
Logical input wires are reversed for primitive/feedback and output-ownership
checks. Full coherent source tensors with an entangled spectator cover the
first four width pairs. Every raw measurement history and every composed
forward/inverse history pair is projected, retaining unnormalized complex
amplitudes and checking all carries are zero. The two eight-bit concrete
factories pass all five existing native make/decompose/counts/serialize/qtyping
modes. No new example-testing framework is introduced.

The existing `register_controlled_add_or_subtract` example now binds the native
typed graph through its unchanged `ConditionalArithmetic` append operation.
The ordinary DynamicCircuit execution path verifies the complete source tensor
and opposite-polarity inverse. Extra circuit wires remain spectators as before.
The narrow signed classical/quantum disagreement recorded above is preserved.
Broader symbolic object conversion/roundtrip and outer generic controls remain
shared admission gaps. Q-0183/Q-0184 stay open for those explicit boundaries.
Batch receipts use `ftqc-reference-env/g2-conditional-component-*`.

The final focused batch passes 128 cases and the concentrated dependency
regression passes 1,355 cases without warnings or skips. Five files pass strict
typing, Black, Ruff, Compiler ownership, architecture and multilevel IR checks.
The unchanged user example executes both coherent control sectors and its inverse.
Its width-three measured stream contains 2 X, 27 CX, 6 H, 4 T, 4 T-dagger, 2 S,
2 measurements, 2 conditional CZ and 2 resets. The independent original model
reports 8 T, 2 measurements and 9 peak qubits.

Default verification reaches 434 passed and 51 skipped before the established
docstring-example coverage failure. All nine reported modules and the failed
policy test are byte-identical to the batch baseline e4acfb741. The global
integration attempt reaches 11 passed and 52 skipped before a QEC retirement
subprocess imports a different editable checkout. Its 94 relevant files are
baseline-identical. Explicitly pinning PYTHONPATH to this worktree passes that
retirement gate and the native conditional append/inverse integration test
(2 passed, 127 deselected). These scoped results do not certify the complete
global suites. Dependency-policy findings are recorded separately for the
whole repository and the changed paths.

## Complete foreign consumer disposition (2026-11-20)

This section supersedes the preceding foreign-object open disposition.
All eight recursive source definitions, three factories, one DocSpec and nine
recursive test definitions map to existing owners. Each actual signed,
unsigned and symbolic factory preserves deterministic source library bytes at
depths zero, one and two. Concrete factory declarations resolve through caller
factories into the existing conditional_addition_program, with exact physical
circuits and aliases against direct binding. The symbolic factory resolves a
native declaration. Unresolved physical widths retain the existing refusal.
The source encodes its boolean constructor argument as integer one in the wire
format. The native library preserves that representation exactly.

The existing typed SDK importer accepts both signed and unsigned original
objects, both addition polarities and widths 1/1, 1/2, 2/3 and 3/3. SDK qubit names
are reversed. Complete ordered instructions, feedback, returned aliases and JSON
match direct native binding. Width-one inputs include complete complex
spectator-entangled vectors and every carry measurement history with zero work.
Existing stored hierarchy, precise inverse, full small branch and inverse-history
coverage, both actual eight-bit five-mode factory checks, normal append and
independent resource evidence is reused and rerun.

The inherited positive outer-control SDK path also executes through the existing
register importer. Its source direct register-generator hook is absent, so the
independent reference reads the unchanged source composite. The comparison
retains original atomic Toffoli leaves. Every wire's complete ordered operations,
operands, measurements and feedback agree, with identical primitive counts.
All four raw measurement histories preserve the full complex source tensor,
including an entangled spectator, and clear both work qubits. Actual counts are
eight T/T-dagger, four Toffoli, two measurements and two resets. The independent
source model reports four Toffoli, two And, two measurements, five or seven
Clifford according to polarity and seven peak qubits. The execution fixture has
eight physical qubits including its spectator. No matrix atoms are introduced.

The inherited negative outer-control SDK path remains an explicit failure.
Its incremental decomposition reaches Controlled(And(uncompute=True)) whose
original unitary protocol recurses between Qualtran and Cirq wrappers. Calling
that original source object alone reproduces RecursionError. The native SDK
import currently propagates the same failure. Merely obtaining a list from
cirq.decompose does not certify execution of those leaves. This path is not
claimed supported and remains part of the open generic SDK consumer work
Q-0937. The source reference is unchanged.

The root's narrow signed callback-versus-zero-extending gate conflict
Q-0183.03 and its one-bit source model undercount remain explicit. Acceptance of
this complete source file records the disposition of its own definitions and
supported consumers. The inherited negative-control failure remains an open
shared boundary. Evidence is stored under ftqc-reference-env/g2-conditional-foreign-*.

Final verification passes 630 concentrated cases without warnings or skips and
17 local integration cases. The unchanged original file reports 12 passed,
one source counts-mode skip and two original marker/collection warnings.
The consumer file passes 23 cases on Cirq 1.7 and 1.6.1. Cirq 1.6.1 retains
1,508 NumPy 2.5 array-shape deprecation warnings, including repeated calls while
reproducing the original recursion failure. Strict mypy, Black, Ruff, scope,
architecture, multilevel IR and public API checks pass. Three frozen blobs,
eight unchanged owners, locked versions and Kickmix Simulator are verified.
The whole-repository dependency baseline remains separate from zero batch
findings. Ten byte-identical files retain the established default docstring
policy failure. Production owners receive no changes in this batch.
