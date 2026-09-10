# ARCH-001: Compute, Remote, and Core Contract Ownership

Status: Approved
Date: 2026-09-03
Revised: 2026-09-08
Scope: FlagQuantum vNext target architecture and staged migration.

## Background

Using Provider as an umbrella term mixes two fundamentally different lifecycles:
CPU/GPU/NPU resources directly controlled by the current process, and QPUs,
GPU/HPC services, or cloud platforms accessed through an external task control
plane. Hardware or vendor categories cannot reliably express this distinction
and can lead to duplicate registration, dependency cycles, and ambiguous APIs.

## Decision

### 1. Divide by Control Boundary

- **Compute** covers resources the current process directly discovers, activates,
  and invokes. It owns device lifecycle, precision, memory, kernels,
  communication, and execution-path facts.
- **Remote** covers resources invoked through an external task control plane. It
  owns target discovery, credentials, submission, status, cancellation, and
  result decoding.
- Simulation implements numerical algorithms without importing Compute. Runtime
  composes Simulation and Compute, or selects Remote, and manages execution
  lifecycles.
- The same GPU model can belong to Compute or Remote depending on deployment.
  Real QPUs usually belong to Remote.

### 2. Core Alone Owns Cross-Domain Contracts

Core defines ProgramArtifact, TargetCapabilities, ExecutionRequest,
ExecutionResult, and Evidence. Compiler, Runtime, Simulation, Compute, and Remote
may consume or implement these contracts but cannot duplicate their definitions.
In the target architecture, Runtime does not depend on Compiler-internal types.

### 3. Every Migration Must Have an Exit

Each migration records its responsible team, target location, completion evidence,
and conditions for retiring the old implementation. A new directory without exit
conditions must not coexist indefinitely with the old authority.

## Alternatives and Reasons for Rejection

- **One Provider interface**: rejected because direct device and remote task
  lifecycles differ. A shared interface would accumulate optional methods and
  type checks.
- **CPU/GPU/QPU categories**: rejected because a GPU can be controlled locally or
  accessed remotely; hardware type does not express the control relationship.
- **Cloud as the name for remote GPU services**: rejected because remote resources
  can be Jiuding compute, internal clusters, or an on-premises control plane,
  rather than a public cloud.

## Compatibility and Migration

- The code has not been formally released. Remove `flagquantum.providers`
  directly, without a forwarding layer.
- Move device platform implementations into `flagquantum.compute`.
- Move external Braket, Quafu, and generic HTTP task adapters into
  `flagquantum.remote`.
- Migrate complete call paths and close old entry points after tests pass.

## Acceptance

- [x] The machine-readable architecture contract distinguishes direct and remote compute.
- [x] Core alone owns cross-domain contracts.
- [x] The source tree no longer contains `flagquantum.providers`.
- [x] The architecture checker prevents that directory from returning.
- [ ] Remove legacy Provider naming from public Compute types.
- [ ] Fully consolidate shared Remote/Deployment task contracts in Core.
