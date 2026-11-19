# Proposal 107 — Real nonlinear parameter expressions

## Review status

Draft awaiting explicit user approval for the protected parameter behavior and
serialized expression vocabulary. This draft records no approval. Production
continues to admit the existing add, sub, mul and neg expression operations.
Proposal 103 prerequisite 1.24 explicitly retains diagnostics for nonlinear
expressions and symbolic global shifts. This proposal supplies the required
contract for that remaining Q-0156 dependency.

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
