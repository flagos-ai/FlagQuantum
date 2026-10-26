# Declared JAX circuit batches: approved compatibility proposal

Status: explicitly approved by the user on 2026-10-26 ("Approve implementation
according to this proposal"); compatibility assessment under frozen Proposal 005.
Parent:
#918; delivery: #929. The earlier additional-approval request was withdrawn after
checking the authoritative contract. This is backend conformance to existing
explicit Circuit.bsz/broadcast/autograd semantics, not a new API interpretation.
No signature, default, schema or documented error contract changes. This document
does not authorize automatic input-axis inference or a backend-default change.

[Public API protection](PUBLIC_API_PROTECTION.md) explicitly permits compatible
implementation changes when protected semantics stay unchanged. Proposal 005
already states that builders/backends produce compatible broadcastable shapes
and forbids implicit Cartesian batches; its forward/autograd and ownership rules
remain authoritative. Conformance to those existing rules does not require a new
API freeze;
the user also explicitly approved this implementation scope.

## User outcome

The existing Module builder declares its circuit batch explicitly, as it already
does for PyTorch execution. The same builder should work with the existing JAX
policy, without a per-example Python loop or a private bridge:

```python
import torch
import flagquantum as fq

def build(theta, x):
    return (fq.Circuit(2, bsz=x.shape[0])
            .ry(0, theta[0] + x[:, 0])
            .rz(0, theta[1]).cx(0, 1))

model = fq.Module(
    build, 2,
    policy=fq.RuntimePolicy(
        execution_options=fq.ExecutionOptions(backend="jax")),
)
x = torch.tensor([[0.1], [0.3], [-0.2]], requires_grad=True)
prediction = model(x)
assert prediction.shape == (3,)
(prediction * torch.tensor([0.2, -0.7, 1.1])).sum().backward()
```

This example is the existing contract implemented by the JAX backend in this
change. The existing PyTorch route is the serial/broadcasting conformance
reference. JAX remains optional and explicitly selected.

## Existing surface and implementation gap

No exports, public signatures, defaults, enums, policy fields or serialized
schemas change. The protected Module constructor and forward/execute signatures
remain exactly as recorded in contracts/module-training-v1-candidate.json.

Before: JAX scalar-output execution accepts scalar gate angles. Builders that
declare a circuit batch greater than one with vector gate angles fail inside
scalar angle materialization or scalar value_and_grad. PyTorch already handles
declared circuit batches and broadcastable gate parameters.

Implementation: JAX Module execution respects the builder's
explicit positive Circuit.bsz. Each gate parameter is scalar, a length-one
vector or a vector of exactly that declared size, matching the existing Torch
parameter-row rule. No input axis is interpreted implicitly. Builders keep
control of feature indexing and all parameter expressions.

- Existing z (one selected qubit), z_sum and Hamiltonian observables produce
  one real value per declared circuit entry: shape (bsz,). At size one the
  existing Module shape (1,) is unchanged.
- Module-owned parameters are shared across the declared circuit entries.
  An upstream cotangent contracts all output entries into the parameter VJP;
  input VJPs follow the builder's actual indexing and broadcasting. Nonuniform
  cotangents are mandatory acceptance cases.
- Explicit parameters remain call-local and do not replace parameter ownership.
  No input-batch x parameter-batch Cartesian product is introduced. Existing
  parameter-only kernel batching remains separate and unchanged.
- Zero/negative Circuit.bsz remains refused by the existing constructor.
  Batched gate parameters with rank greater than one or an incompatible
  length fail with ValidationError naming the gate, parameter and declared size.
  Existing scalar-path error categories must not be changed incidentally.
- Multi-qubit vector-valued z observables and named JAX parameter groups remain
  refused as before. Statevector, MPS and TN keep their selected representation;
  no hidden backend switch or per-example Python execution loop is admitted.
- Grad mode and all four parameter/input requires_grad masks remain authoritative.
  eval() does not disable differentiation. no_grad/inference skip derivative work.
  Existing first-order-only cross-framework differentiation limits remain.

## Implementation and memory boundary

Reuse the existing Circuit/IR batch declaration, JAX statevector executor and
Torch autograd boundary. Numerical batching belongs to Simulation; Runtime
coordinates calls and retained state. Do not add another batch class, public
Module, registry or semantic IR.

Vector outputs need a true cotangent VJP. Do not materialize a full output-by-
parameter Jacobian or an output-by-input Jacobian. A compiled value path and a
compiled cotangent-weighted derivative path are acceptable for the first slice;
report their recomputation and separate compilation costs honestly. Retained
inputs must own their consumed values so in-place updates cannot change a later
VJP. Storage for retained arguments is linear in parameter/input sizes, apart
from declared simulator state/intermediates. Native precision/device and
explicit fallback policy remain authoritative.

Keep the scalar hot path unless measured replacement evidence justifies changing
it. Cache reuse/invalidation follows existing Module signatures and precision,
device, policy and checkpoint behavior. No cache per parameter value or step.

## Acceptance before promotion

1. Independent small dense values/finite-difference VJPs and Torch serial/batched
   references for batch sizes 1/8/64, changed features and noncontiguous inputs.
2. Parameter-only, input-only, both and neither gradients; no_grad/inference;
   nonuniform cotangents; shared/reused/nonlinear parameter expressions.
3. Three or more Adam updates, objective parity, checkpoint/reload, consumed IR
   hashes and preserved VJPs after in-place changes. No full Jacobian allocation.
4. Batch-shape errors, unsupported modes/observables, missing optional JAX and
   precision/device changes retain explicit failure/compatibility behavior.
5. Compare the public Module workflow against the strongest applicable
   TensorCircuit JAX/Torch integration and fully compiled JAX training separately.
   Freeze precision, circuit, batch, gradient demand and optimizer; retain all
   losses/timeouts, startup costs, repeated process timings and peak memory.
   An internal compiled diagnostic is never presented as public-path performance.

## Alternatives and migration

Keeping JAX scalar-only preserves the current gap and forces serial Python calls.
An implicit vmap of input axis zero is rejected because builders own input shape
meaning. Full-Jacobian bridging is rejected for memory scaling. A separate batch
Module is rejected because the existing Circuit.bsz contract expresses the need.

There is no deprecation or removal version: existing supported behavior and
serialized policy/checkpoint formats stay valid. Ship only after backend conformance and ordinary PR review, not on a
predetermined release date. Existing unsupported batched JAX calls become supported only in that
release. Document the support extension and its limits in known limitations,
the capability record and runnable examples; do not update frozen snapshots just
to approve a change. Numerical execution/selection defaults remain unchanged.

Proposal 005 remains the authoritative frozen contract; the explicit user
approval covers this implementation proposal, not a successor API freeze or
default change. Automatic selection and implicit input-axis inference
still require their own approved proposals before implementation.
