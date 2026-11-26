# Proposal 128 — Inverse cosine and floor parameter expressions

## Review status

Explicitly approved by the user on 2026-11-26. Implement contract, Core,
adapters and the source consumer in independent commits in PR #1084.
Proposal 107's old encodings and behavior remain covered by compatibility
fixtures; this approval adds exactly acos and floor.

## Verified source requirement

Unmodified Qualtran v0.7.0, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, constructs symbolic Hubbard PREPARE
weights with both operations. At dimension two its normalization is
`16*t + floor(4*u)` and its first angle is
`2*acos(sqrt(8*t/(16*t + floor(4*u))))`. The original shallow constructor wire
serializes. Resolving the original symbolic gate at fractional and endpoint
weights produces finite numerical angles.

The native constructor now preserves that logical declaration, but physical
execution refuses it: the existing expression translator reports unsupported
floor and acos. The literal numerical coefficient path already executes and
matches full complex source branches; it is not a substitute for symbolic
binding. Evidence is retained in
`ftqc-reference-env/g2-hubbard-prepare-symbolic-weight-intake-20261126.json`
beside the integration worktree. The source implementation blob is
`cfdbf20b0baf5ef1f5fa5a7e21826d7f51f5aec5`.

## Proposed contract

Extend the existing immutable ParameterExpression vocabulary with exactly two
unary operations, `acos` and `floor`. Existing Parameter names, expression
construction, Circuit binding, append and execution remain the entry points.
No quantum opcode, public export, parameter method or opaque callable is added.

- Acos accepts finite real scalar inputs in [-1, 1], returning the principal
  inverse cosine in [0, pi]. Reject values outside that domain; do not clamp
  them or take a complex branch.
- Floor accepts finite real scalar inputs and returns the greatest integer
  not exceeding the input. Python scalar evaluation uses math.floor; Torch
  scalar evaluation uses torch.floor. Preserve Python's integer result until
  later arithmetic; tensor results follow Torch dtype rules. Do not round
  integer coefficients prematurely to float.
- Reuse existing structural, arity, cycle, finite-leaf and scalar validation.
  Unknown operations, complex values, booleans, vector tensors and nonfinite
  values retain explicit rejection. Unresolved trees are not evaluated at
  dummy zero bindings.
- Floating-point Torch scalar inputs retain device, dtype and autograd through
  the corresponding Torch operations. Integral inputs follow Torch's numeric
  promotion rules. Floor has zero gradient almost everywhere; acos has singular
  derivatives at its endpoints. Do not fabricate finite gradients or smooth
  either operation. Existing training validation remains authoritative.
- Core remains independent of SymPy and external SDKs. Ecosystem and the
  existing symbolic-format translator map these two SymPy functions to owned
  nodes. SDK exports use matching supported expressions or retain their
  explicit unsupported-expression diagnostics.

## Serialization and compatibility

Keep the existing expression encoding shape, Program scalar literal format and
CircuitIR container. The two additional operator strings expand the accepted
serialized vocabulary and must therefore be approved before implementation.
Old readers may reject the new nodes; old payloads, content hashes, arithmetic
results and diagnostics must remain unchanged. No snapshot is regenerated
merely to hide a compatibility failure.

The integration contract change in
`contracts/parameter-expression-contract.toml` and a compatibility fixture land
first. Core implements evaluation and validation next. Compiler/Ecosystem then
transport the new nodes and replace the symbolic Hubbard declaration with its
real source-ordered body. Numerical coefficient validation, normalization floor
order, logical temp ownership and exact erasure stay with their existing owners.

## Acceptance and delivery

1. Freeze old scalar/expression payloads, hashes, valid bindings and failure
   categories before changing the contract. Verify them unchanged afterward.
2. Test both operations on finite constants, negative floor inputs, real scalar
   tensors, valid acos endpoints and malformed/domain-negative cases. Verify
   dtype/device behavior and interior gradients against direct Torch results.
3. Round-trip owned expressions through existing CircuitIR and Program paths,
   nested calls and supported SDK conversions. Preserve missing-binding and
   duplicate-symbol-name diagnostics.
4. Restore the original symbolic Hubbard constructor and its exact adjoint.
   Bind default, fractional and endpoint t/u values; compare actual source
   angles, calls, operands, full complex branch amplitudes and zero private
   work. Keep caller-owned logical temp separate from private work.
5. Keep actual named primitive counts separate from the original QEC model,
   including the first Ry's endpoint Clifford classification. Run affected
   scalar, parameter, constructor and local execution regressions plus strict
   type, format, architecture, IR and API checks.

Contract, Core, adapters and source consumer changes are independent commits
within integration PR #1084 after approval. Pending Proposals 118, 119, 124, 126
and 127 remain outside this scope. No G3 work or automatic merge is authorized.
