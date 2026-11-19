# Proposal 107 — Real nonlinear parameter expressions

## Review status

Approved by the user on 2026-11-19 with the instruction “批准，按提案实施”.
The approved scope is the six additional expression operators and symbolic
real-angle global phase described below. Implementation proceeds in separate
contract, Core and Ecosystem commits. This approval supersedes Proposal 103
prerequisite 1.24's nonlinear refusal for precisely this scope.

The machine contract is `contracts/parameter-expression-contract.toml`.
The compatibility fixture fixes existing arithmetic payload bytes, content
hashes and binding results before implementation.

## Verified requirement

Frozen Qualtran v0.7.0 composite conversion transports a Cirq Rz parameter with
a nonlinear expression before numeric resolution. Its owned native counterpart
currently rejects such an expression in the Ecosystem translator. Full composite
source acceptance therefore requires a shared expression representation and
execution path, in addition to the previously accepted arithmetic subset.

The reference revision is `bcb17bb25360ed68ce3bc084645db3e929799389`.
Existing `composite_bloq.py` has Git blob
`b0317f675c96f9cce1581fcbb7030c48ab32a529`. Intake observations are recorded in
`ftqc-reference-env/g2-nonlinear-parameter-intake.json`. Reference code remains
unchanged. Source transport and resolved numerical execution are separately
recorded because preserving a symbolic object alone cannot establish execution.

## Proposed behavior

Extend the existing immutable ParameterExpression vocabulary with unary sin,
cos, exp and log, and binary pow and div. Existing construction and parameter
binding remain the entry points. Add no new public methods on Parameter or
ParameterExpression. Each new expression has precisely the stated arity.
Existing numeric, name, immutable-tree and finite-leaf validation still applies.

Binding supports finite real scalars and real Torch scalar tensors. Tensor
operations preserve dtype, device and autograd. Core does not import SymPy or
SDK objects. Invalid real domains raise ValueError before quantum execution.
Division by zero, log of a nonpositive real, a nonreal power result, and
nonfinite resolved values have explicit diagnostic cases. Validation of an
unresolved tree checks its structure without evaluating it at dummy zero values,
since zero is outside several legitimate operator domains. Old arithmetic-tree
behavior and its failure categories require compatibility regression.

Ecosystem translates supported SymPy functions into existing owned nodes and
exports them to matching SDK expressions. Noninteger exponents retain an owned
pow tree. No foreign callable or symbolic object enters Core. Unsupported
functions continue to produce a conversion diagnostic identifying the function.
Distinct SDK symbols sharing one native parameter name remain rejected.

Symbolic global phase uses the existing phase primitive where its angle is a
supported real expression. A generic symbolic complex coefficient requires an
explicit real-angle derivation at the adapter boundary. Expressions whose
coefficient cannot be established as unit modulus remain rejected. Preserve
full global phase, including nested controls and an entangled spectator.

## Serialization and compatibility

CircuitIR retains its existing `$expression` encoding shape with an operator
string and recursively encoded args. The six additional operator strings expand
the accepted serialized vocabulary. Readers supporting only the previous subset
must reject the new operations with their existing unsupported-expression
error. Existing payloads and hashes remain byte-identical. Machine contract
changes and compatibility fixtures must precede implementation commits.

Program scalar literals retain the existing Parameter and ParameterExpression
records. Existing Program scalar validation delegates structural operator and
arity validation to the same Core owner. Compiler owns transport and binding.
Simulation and Runtime receive resolved numeric inputs through their existing
execution boundaries. Ecosystem owns all SDK expression conversion.

## Delivery and acceptance

1. Land the reviewed contract and compatibility fixtures on the integration
   branch, then implement Core evaluation and structural validation independently.
2. Implement Ecosystem import and export through existing register and static
   translators, with no additional graph, registry or deferred foreign callback.
3. Reuse existing component copy, nested calls, physical binding and ordinary
   `circuit.append(op)` consumers. Bind each supported expression at values in
   its real domain and compare complete complex amplitudes with frozen source.
4. Compare each source operation, operand and condition at the same primitive
   boundary. Verify both control polarities, all applicable measurement branches,
   spectator entanglement, clean-work zero, export/reimport and tensor gradients.
5. Verify old arithmetic serialization/hashes and malformed-expression errors.
   Record source protocol defects and admitted expression boundaries separately.
6. Report emitted primitives and source resource estimates in separate fields.
   Expression transport introduces zero quantum primitives. Global phase retains
   the actual existing primitive decomposition and its independent model.
7. Run strict typing, formatting, classified scope, architecture, IR/API contract,
   concentrated regression and local integration. Synchronize the ledger and
   phase overview after complete acceptance. Q-0156 stays open until the whole
   parent audit has evidence for every remaining definition.


## Core implementation evidence

The Core implementation adds the six approved operations behind the existing
ParameterExpression constructor and bind method. Program scalar literals and
physical-control validation reuse one structural validator. Unresolved log and
reciprocal trees are admitted without evaluating them at zero. Old arithmetic
operations retain their binding behavior and byte-identical serialization.

The concentrated Core/Program regression passes 177 cases. Six source files pass
strict typing, Black and Ruff. Architecture and the multi-level IR checker pass.
The seven nested register tests include the six nonlinear operations, copying,
allocated-work phase, complete complex output, gradients and work-zero through
circuit.append(op). Expression transport adds zero primitive operations. These
fixtures still emit exactly two Rz operations, with one clean work qubit.

The native circuit and Rydberg training integration suite passes all 172
cases, with nine existing deprecation warnings. Single-rank Gloo checks run
with local loopback permission. The separate Rydberg legacy-field compatibility suite has one pre-existing
failure. Replaying the pre-change parameters, scalar and physical modules
reproduces the same mismatch between the old allowed Operation tail and the
already present callee/register/workspace/matrix/kraus fields. This evidence
is stored in g2-nonlinear-core-baseline.log. It is not counted as passing.

The Core batch was followed by the adapter delivery recorded below. Q-0156
acceptance still requires the complete source audit and generic eigen-gate
global-shift boundary.

## Adapter delivery boundary

Cirq static, register and measured-feedback import derive an owned real angle
from a coefficient written as exp(i * angle), including equivalent exponential
products reduced by SymPy. The supported real expression domain is enforced at
binding. Numeric coefficients must be finite and unit modulus. An arbitrary
symbolic complex coefficient remains rejected. Static circuits require an
existing declared wire. Register programs may allocate their existing clean
scalar carrier. The existing Compiler scalar_phase_ir primitive emitter accepts
owned Parameter and ParameterExpression angles after structural validation,
retaining the exact X/phase/X/phase order and existing numeric behavior.

Static Cirq phase export/import admits the existing phase opcode. Symbolic
Qiskit global phase becomes the same physical primitive sequence on wire zero,
with no duplicate phase metadata. Numeric Qiskit phase metadata retains its
previous representation. No SDK objects enter Core or Compiler.

## Nonlinear SDK and scalar-phase evidence — 2026-11-20

Cirq and Qiskit import/export now accept the six approved operators through the
existing parameter representation. Commit 6a4f21155 implements expressions;
Compiler b58be7cb4 and Ecosystem a14c013f5 implement scalar phase and consumers.
The exact old expression fixture continues to pass. Assumption-bearing SymPy
symbols retain their original identity during translation. Distinct same-name
symbols in one expression or across supported gates remain rejected.

The concentrated suite passes 398 cases. Two unchanged optional cases skip
because PennyLane and Qiskit Aer are absent in this environment. Cirq 1.6.1
passes its separate 130-case lane; Cirq 1.7.0 is used in the concentrated lane.
Qiskit 2.5.0 passes its separate 94-case lane; 2.0.0 is used in the concentrated
lane. Local native, register-export and Rydberg training integration passes 213
cases. Seven files pass strict typing, Black and Ruff. Architecture, IR and
scope checks pass. Dependency inspection records 687 whole-repository findings
and zero changed-path findings. The test harness registers the unchanged source
notebook marker before importing source test helpers; reference files are not
modified.

Whole complex amplitudes, serialized export/reimport, tensor gradients, both
measurement polarities and every two-outcome branch are checked. One- and
two-control scalar phases cover every binary control pattern with an entangled
spectator. Source composite nonlinear Rz transport is checked with both plain
and assumption-bearing symbols. Zero-port scalar programs allocate and release
one clean carrier and return it exactly to zero. No global phase is removed
from comparisons.

| Workflow | Emitted primitives | Independent source model |
| --- | --- | --- |
| Entangled nonlinear rotation | H, CX, Rz; no extra work | No independent model supplied by this SDK fixture |
| Scalar phase | X, phase, X, phase; one existing carrier or one clean allocated carrier | Frozen GlobalPhase QEC cost empty, logical qubit count zero |
| Binary controlled scalar | phase or cphase, plus X pairs for zero controls | Primitive count is kept separate from higher-level source models |
| Conditional scalar with record reset | Four condition-bearing phase primitives, existing measurement and reset | No independent model supplied by this feedback fixture |

Q-0156 stays open for generic symbolic global shifts on eigen-gates and the
final full-definition audit. The diagnostic source probe is retained; no
unchecked parent is marked accepted. The earlier pre-existing Rydberg legacy
field assertion remains recorded separately from passing integration.

## Generic eigen-gate phase contract

The approved real-angle phase path also applies to XPow, YPow, ZPow, ZZPow and
CZPow gates with symbolic exponents and shifts. Translation factors the source
matrix into a canonical gate and exp(i*pi*exponent*shift_difference), preserving
that operation order and each quantum/classical control. Original symbols from
both fields participate in identity checks. Source Cirq only discovers and
resolves the exponent field, so source transport and explicit scalar substitution
must be reported separately. No new Core operations or parameter methods are
introduced. Static CZPow maps to the existing cphase primitive.

## Generic eigen phase acceptance — 2026-11-20

Commit 9f1272200 closes the generic-shift dependency for five declared families.
Both fields retain original symbol identity and bind through owned expressions.
The 426-case concentrated suite, 213-case local integration and both 166-case
SDK lanes pass. Full matrices, feedback branches, controls and work-zero are
checked. Existing sx/sxdg specializations retain their primitive counts.
Composite source Q-0156 is accepted under its final owner audit. Broader control
provider coverage remains assigned to Q-0158. Raw source atoms, model estimates
and emitted primitives are explicitly separated in that audit.
