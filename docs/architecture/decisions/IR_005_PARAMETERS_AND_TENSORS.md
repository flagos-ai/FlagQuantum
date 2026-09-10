# IR-005: Parameters, Tensor Constants, and Late Binding

Status: Approved
Date: 2026-09-01
Applicable phase: Phase 1 importer, identity, and gradient differential tests.

Approval record: the API owner explicitly approved IR-004 through IR-006 on
2026-09-01. Approval covers internal parameter classification, binding tables, and
identity rules only. Public Parameter, Module, checkpoint, and training APIs do
not change. The training owner must still review the autograd implementation.

## Background

CircuitIR gate parameters can be Python scalars, `Parameter`,
`ParameterExpression`, or torch tensors. Public JSON/hashes encode detached
tensors, but training must retain the autograd graph. Immutable internal IR must
neither sever gradients through tensor copying/serialization nor mistake changing
training values for program structure changes.

## Decision

Phase 1 classifies parameters into three groups:

1. **StaticConstant**: Python bool/int/float/complex values or explicitly static
   scalar tensors.
2. **SymbolicParameter/Expression**: retains names, expression operations, and
   parameter dependencies.
3. **RuntimeBindingRef**: trainable tensors and explicitly late-bound parameters
   have stable slots in the module; actual tensors remain in an external binding table.

Rules:

- Tensors with `requires_grad=True` must use `RuntimeBindingRef`; execution must
  not use detached/copied replacements.
- Binding tables retain original tensor objects so test lowering connects to
  existing PyTorch autograd.
- Internal program identity includes slots, shapes, dtypes, and expression
  structure, but excludes late-bound values.
- Execution identity includes this run's binding identity/value policy. Phase 1
  does not connect this to public identity.
- Canonical Python/static scalar encoding distinguishes bool/int/float/complex.
- Tensor dtypes cannot silently degrade. Nonscalar gate parameters fail closed
  unless explicitly allowed by schema.
- `ParameterExpression` accepts only operations supported by the public parameter system.
- Binders validate missing/extra values, shapes, dtypes, and device policy.
- Parameter binding must not implicitly change control flow, wires, arity, or target layout.

## Rejected Alternatives

- **Detach all tensors into IR**: severs autograd and contaminates program caches
  with training values.
- **Exclude all parameters from identity**: changed static constants could alter
  semantics yet incorrectly hit caches.
- **Store arbitrary Python callables**: not deterministic, serializable, or safely verifiable.

## Compatibility and Rollback

Phase 1 binding tables serve only the internal differential bridge. Public
`Parameter`, CircuitIR JSON, Module, checkpoint, and training APIs remain unchanged.

## Acceptance

- Preserve Parameter/Expression structure and dependencies exactly.
- Define scalar-tensor dtype/shape rules explicitly.
- Requires-grad tensor forward results and gradients match the legacy path.
- Late-bound value changes preserve internal program identity.
- Static constant changes alter internal program identity.
- Missing/extra bindings and shape/dtype errors fail closed.
- [x] API owner approved the architecture decision.
- [ ] Compiler owner confirms canonical parameter encoding during implementation review.
- [ ] Training owner confirms binding table and autograd graph preservation.
