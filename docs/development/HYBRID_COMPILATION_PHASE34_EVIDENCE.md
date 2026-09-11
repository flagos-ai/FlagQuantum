# Phase 34 symbolic circuit-artifact binding evidence

Date: 2026-09-10

## Result

Symbolic `circuit-ir-1.0` artifacts can now be explicitly bound into new fully
bound v2 artifacts. The binding operation preserves the source artifact,
evaluates Core parameter expressions, and returns immutable lineage evidence
covering the source identity, bound identity, parameter names, and scalar
values.

Runtime accepts the lineage result on the existing circuit-artifact execution
path and records source, binding, and bound-artifact identities in the normal
execution result.

## Verified behavior

- binding is deterministic across input mapping order;
- changed values change both binding and bound-artifact identities;
- expressions bind to the expected circuit scalar;
- missing and extra names fail closed;
- booleans, non-finite values, and tensors fail closed;
- already-bound artifacts cannot be rebound through the symbolic path;
- a bound artifact executes locally and retains the complete lineage chain.

The combined hybrid compiler, Core artifact, binding, Runtime/Deployment
adapter, local execution, and private-contract suite passed **266 tests**.

## Gradient boundary

Artifact binding is a serialization and deployment path, not a differentiable
training path. It accepts only Python real scalars. It never detaches a tensor
or claims to preserve an autograd graph. Training continues through the existing
in-memory binding tables, which retain the original tensors.

## Claim boundary

No public export, executable-text parameter profile, provider submission,
default-path change, or performance claim is introduced.
