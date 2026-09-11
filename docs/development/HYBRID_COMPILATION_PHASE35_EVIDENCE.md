# Phase 35 artifact-to-artifact compilation evidence

Date: 2026-09-10

## Result

Compiler now provides one bounded entrypoint from a fully bound circuit
artifact or verified binding result to a verified target-text executable
artifact. The path mandates target capability legalization, optional topology
routing, native-gate legalization, dependency scheduling, deterministic target
emission, strict conformance, and v2 artifact construction.

The result retains the real input object and verifies that its circuit identity
feeds the topology/native legalization chain. It also binds source, optional
parameter binding, target snapshot, all compilation stages, and final artifact
into a deterministic compilation identity.

## Verified behavior

- serialized circuit artifacts compile deterministically;
- nonlocal two-qubit operations pass through bounded topology routing;
- binding lineage survives into the compilation result;
- the executable artifact passes Runtime preflight and Deployment dry-run;
- symbolic inputs fail until explicitly bound;
- embedded shots fail because they belong to the execution request;
- executable artifacts cannot re-enter the circuit compilation path;
- Compiler performs no numerical execution and no provider call occurs.

The combined hybrid compiler, Core artifact, binding, Artifact-to-Artifact
compiler, Runtime/Deployment adapter, local execution, and private-contract
suite passed **271 tests**.

## Claim boundary

This remains an internal Compiler interface. It introduces no public export,
provider submission, default-path change, or performance claim.
