# Complete phase-gradient constructor family and retained storage boundaries

## Frozen source and ownership

The complete rotations/phase_gradient.py and original tests remain frozen at
Qualtran v0.7.0 bcb17bb25360ed68ce3bc084645db3e929799389. Intake verifies
seven exact source/dependency Git blobs and actual locked Kickmix 0.1 Simulator.

| Source definition | Native owner |
|---|---|
| PhaseGradientUnitary fields, signature and decomposition | Compiler formats/legacy_phase_gradient_factories.py reuses phase_gradient_unitary_program and existing Z/CZ power helpers. Unsigned fractional phase_grad and optional ctrl are caller-owned through ports. |
| PhaseGradientState fields, signature and decomposition | Existing phase_gradient_state_program retains Allocate, OnEach/H and gradient calls. Its RIGHT fractional output transfers to the caller. |
| Both phase precision fields | Precision stays on the stored library and independent source model, including eps/width leaf declarations. Numerical execution uses existing phase primitives. |
| AddIntoPhaseGrad fields and callback | Existing phase_add_component_program uses shared integer Add/CAdd and exact measured adjoints. It now forwards control_value to the existing CAdd owner for both polarities. The source leaf has no primitive decomposition. |
| AddScaledValIntoPhaseReg fields and decomposition | Existing scaled_phase_add_program retains coefficient-bit iteration, exact retained slices, right shifts and signed addition direction for admitted unsigned formats. Source callbacks and quantization remain in their existing owners. |
| _fxp, _mul_via_repeated_add and from_bitsize | Existing coefficient scheduling, codec and source-file acceptance retain truncation, saturation and the five original fixtures. |
| Source call graphs, arithmetic and tensors | Existing phase_gradient, phase_addition and scaled_phase_addition owners retain independent analytical models and arithmetic permutations. |
| Five examples, drawings and DocSpecs | Original stored examples retain exact signatures, default metadata and the native helper closure. Existing catalog/drawing evidence is reused. |

The shared gradient Program now accepts already-owned Parameter expressions
through the same validated rotation-angle boundary as Z/CZ power components.
Division uses the approved expression vocabulary. Concrete exponents retain
the original division order. Symbolic forward and exact adjoint execution use
the existing Circuit binding path and return private work to zero.

## Source alignment and physical ownership

Concrete gradient constructors compare source producer/operand traces, full
helper Programs, stored depths zero/one/two and exact complex joint operators.
State preparation compares the complete complex output tensor and transfers
allocated output ownership. The atomic addition is compared against its
original arithmetic operator, while its native helper graph is compared
against the existing Add/CAdd owner. No unavailable source primitive recipe
is claimed for that atomic leaf.

Both control polarities, both signs and two right shifts compare complete
joint operators. Every raw small measurement history additionally compares
absolute complex amplitudes with a coherent spectator and explicit zero work.
Scaled addition compares the original ordered slices and complete arithmetic
operators, including zero, positive and negative coefficients. Existing full
source grids, inverse, saturation, noncontiguous bindings and append evidence
are reused after exact native Program equality.

## Accounting and remaining boundaries

Literal bound primitives and private clean storage are measured independently
of source QEC gate estimates. Original negative small-width Toffoli values,
zero-exponent call-key coalescing and differing controlled-qubit estimates
remain queryable without being treated as emitted instruction counts.

Original empty and symbolic gradient decomposition refusals remain explicit
bodyless declarations. Concrete Sympy Integer values normalize on the source
wire. Symbolic phase exponents are executable through existing owned expressions.
Source symbolic scaled coefficients retain a declaration and physical refusal.

The original 2026-11-23 batch retained signed x and signed coefficients as
bodyless declarations. Approved Proposal 116 and the closure below supersede
that specific signed-storage refusal. Unsupported fractional formats and
permissive nonstandard AddIntoPhaseGrad sign/shift metadata remain separate
gaps. The scaled identifier remains partial under those complete-file criteria.

## Acceptance receipts

The concentrated affected regression passes 321 cases. Local integration
passes 145 cases. Eight freshly rerun original model/symbolic cases pass with
three original skips and 49 deselections. The previously accepted complete
original 57-pass/3-skip source-file receipt is reused after exact blob
verification. Strict checking covers all four changed Python files. Black,
Ruff, Compiler scope, language, architecture, multilevel IR and public API
checks pass. Dependency findings remain exactly the existing 720.

An independent comparison with published ecdf16a01 proves all 108 sampled
concrete forward/inverse gradient, state and addition Programs unchanged.
Literal resource receipts show width-three controlled gradient with twelve
T/T-dagger, eighteen CX, three measurements/resets and three phase instructions
on five wires. Two-bit addition into a three-bit gradient has twelve
T/T-dagger, twenty-one CX and three measurements/resets on nine wires for
both polarities. Open control additionally emits eight X, while positive
control emits none. Its independent source model estimates two Toffoli and
six qubits. The negative-half scaled case has eight T/T-dagger, twenty CX
and two measurements/resets on seven wires, versus one modeled Toffoli
and five qubits. Receipts use prefix g2-phase-constructor and date 20261123.


## Signed coefficient and input storage dependency closure, 2026-11-25

Compiler's existing scaled-phase scheduler now represents signed coefficient
formats explicitly. It saturates the nonnegative magnitude at the signed
positive maximum and preserves the original Fxp.bin()/Fxp.n_int loop literally:
n_int excludes the sign bit, so each retained coefficient bit has one extra
right shift compared with the same-width unsigned format. This source rule is
not replaced by a mathematical signed-multiplication interpretation. The source
input dtype's num_int still equals bitsize minus num_frac, including its sign
position, and raw input slicing preserves that rule.

The existing Program owner retains the input sign flag, calls existing unsigned
fractional adders on the original ordered slices, and restores nominal storage
through shared Split/Join and Cast views. The existing resolver passes both
sign flags to this owner instead of returning a declaration for every signed
case. Coefficient, Program and literal-IR scheduling share the same terms.
Compiler owns no numerical execution or backend selection. Approved Proposal
116 supplies positive-width signed quantum storage; classical casts, signed
controls, signed-zero storage and numerical fixed-point arithmetic contracts
are unchanged. User code retains circuit.append(op).

Actual original serialized depths zero, one and two retain identical root
calls, operands and nominal descriptors. Positive, negative, zero and saturated
coefficients are covered with signed and unsigned input storage. Every complete
small logical operator matches the unchanged source decomposed graph without
phase alignment. Every small raw measurement history agrees on a complex
input entangled with a spectator; every forward/inverse history pair restores
all amplitudes and clean carry storage. Outputs alias caller data wires.
A separate direct original-generator grid covers every signed coefficient
width two through four, with the original one-bit signed constructor refusal, every nonnegative valid fractional width, and five
coefficients, checking term order, operand slices, shifts and signs.

Source model, original classical model and decomposed execution remain separate.
The source signed-input scaled_val(0) raises AssertionError even when its graph
successfully decomposes; this is reproduced, not corrected. Full-word
QFxp(3,2) input with unsigned gamma still fails the original nominal child
QFxp(3,3) binding; native resolution retains its corresponding dtype refusal.
Symbolic coefficients remain bodyless. Negative coefficient fractional widths
and other nonstandard metadata remain outside this closure. No locked source,
protected opcode or serialization vocabulary changes.


The batch passes the 204-case affected regression, 95 direct dependency cases,
79 Cirq 1.6 cases and the final 80-case signed/scheduler scope after adding the
small-format generator grid and explicit one-bit refusal. Local integration
passes 145 cases. Strict typing covers four Python files; Black, Ruff, Compiler
scope, language, links, architecture, IR and protected API checks pass. The
source suite checks all four frozen phase-gradient assets and the real Kickmix
reference. Original protobuf boolean deprecation warnings remain visible.
Receipts use `g2-signed-scaled-*20261125` in the reference workspace.

Ten independent resource rows retain the source declared full-width estimate:
for coefficient 0.5 the native primitive stream has 8 T/Tdag, 20 CX, two
measurement/CZ/reset events and peak eight, while the original model has one
Toffoli, no measurements and peak six. For coefficient 0.75, actual counts are
16 T/Tdag, 37 CX, four measurement/CZ/reset events and peak eight; the original
model has two Toffoli, no measurements and peak six. Neither model is rewritten
to match the native adders' measured decomposition. Source scaled model formulas
use the same literal signed-coefficient bit schedule but retain full-width
atomic leaf estimates.
