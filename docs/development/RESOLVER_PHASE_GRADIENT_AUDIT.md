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

The scaled constructor retains the exact signed fixed-point x signature as a
bodyless declaration. Physical signed fixed-point admission still requires the
separately pending Proposal 116. Signed coefficient formats, unsupported
fractional formats and permissive nonstandard AddIntoPhaseGrad sign/shift
metadata remain explicit gaps. Three constructor identifiers close. The scaled
identifier remains partial and is excluded from the complete-constructor count.

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
