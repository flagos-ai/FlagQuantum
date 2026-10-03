# FlagQuantum Extension SDK

The approved, pre-freeze SDK contract lives under
`flagquantum.ecosystem.extensions`; it does not add root exports. Extensions declare a
versioned manifest, negotiate capabilities before activation, and are installed
into a task-local immutable registry.
The API owner authorized its direct pre-release move from the former
`flagquantum.extensions` namespace on 2026-09-07; no compatibility layer is
provided.
Individual extensions remain experimental by default and require independent
qualification.

Supported extension kinds are execution backends, circuit compilers, compiler
passes, kernels, operators, devices, providers, measurement collectors, and
planners. A circuit compiler accepts and returns FlagQuantum `CircuitIR`;
compiler passes remain the smaller transformation hook.

Installed packages are discovered only through an explicit, kind-specific
`discover_extensions(...)` call. They register zero-argument factories in the
`flagquantum.extensions` Python entry-point group. Discovery validates the
entry-point identity against the manifest and adds the result to the existing
immutable registry; it does not introduce a second plugin registry. See
[`COMPILER_PLUGINS.md`](../guides/COMPILER_PLUGINS.md).

## Compatibility lifecycle

- SDK API mismatches fail during registration with upgrade guidance.
- Individual extensions are experimental by default. Stabilization requires conformance,
  security review, documentation, and a declared compatibility window.
- Deprecations declare a removal version in the manifest and must retain the
  previous contract for that window.
- Capability negotiation is fail-closed: missing dtype, device, gradient, or
  semantic capabilities produce blockers before activation.

Compatibility failures are also `flagquantum.errors.CapabilityError`; lifecycle
failures are `flagquantum.errors.ExecutionError`, so applications can use the
same stable error categories as core execution.

## Isolation and security

Registration returns a new immutable registry and `extension_scope` uses
task-local context. It never mutates root exports, core operator tables, or
another task's registry. Lifecycle wrappers translate extension exceptions and
attempt cleanup after failed startup; cleanup also runs at normal scope exit.

Raw tokens, passwords, API keys, secrets, and credentials are rejected from
`ExtensionConfig`. Providers must obtain credentials through a host-owned
resolver and must not include them in manifests, errors, measurements, or
serialized payloads. Extensions execute with the Python process's authority;
discovery is not a sandbox, and only trusted packages should be installed.
Importing FlagQuantum does not discover, import, or activate extensions.

## Conformance

`flagquantum.ecosystem.extensions` supplies reusable backend, provider, and
circuit-compiler checks covering manifest/payload serialization, capability
honesty, PyTorch gradients, dtype/device preservation, CircuitIR ownership,
determinism, isolated errors, and cleanup. The conformance submodule remains an
equivalent explicit import path. Reference extensions are in
`examples/extensions/reference_extensions.py` and
`examples/extensions/reference_compiler_extension.py`.

## Backend execution admission

An execution backend that has been discovered and negotiated is still only a
declaration. `flagquantum.ecosystem.extensions.admission` is the single crossing
that turns a negotiated `ExecutionBackendExtension` into a backend `fq.run` can
select by name. It validates the extension's `backend_capabilities` declaration,
negotiates capabilities, records the result through the existing runtime
capability registry, and keeps one live execution route for it. No second
registry, entry-point group, or manifest format is created, and the namespace
above is not exported from `flagquantum.ecosystem.extensions`.

The declaration is a mapping of exactly the capability fields the backend owns.
`name`, `executor`, and `accelerators` are host-owned: a backend that declares
them is rejected rather than silently overridden. An unknown key, a missing
required key, a non-backend manifest kind, a built-in backend name, and a refused
negotiation all fail closed before registration.

`fq.run` reaches an admitted backend through the ordinary planning path. A
requested backend that is unregistered, registered without an execution route, or
lacking a required mode, gradient, or distribution capability fails at planning
time with a `CapabilityError` naming the blocker. There is no execution-time
substitution and no silent CPU path. The built-in `pytorch` route acquires no
lookup, negotiation, or version check.

The admitted route is called as
`execute(program, *, options)` where `program` is FlagQuantum IR and `options`
carries the plan facts (backend, device, mode, dtype, batch size, world size,
target, gradient and approximation permissions, runtime config, and distribution
semantics). The returned object is normalized by the same result adapter as any
other output, and an admitted result is never relabelled as a built-in engine
path. Withdrawing an admitted backend unregisters it; a later request for it
fails closed.

See [`API_CHANGE_PROPOSAL_063_BACKEND_EXECUTION_ADMISSION.md`](../development/API_CHANGE_PROPOSAL_063_BACKEND_EXECUTION_ADMISSION.md)
and [`contracts/backend-execution-admission-v1-candidate.json`](../../contracts/backend-execution-admission-v1-candidate.json).
`examples/extensions/reference_backend_extension.py` is a complete reference
implementation that imports only the extension namespace.

## Compiler pass admission

A declared `compiler_pass` extension is admitted the same way, through
`flagquantum.ecosystem.extensions.pass_admission`. The crossing validates the
extension's `pass_name` declaration, negotiates `circuit_ir`, and registers the
extension's `transform` under a name the host builds as
`extension.<manifest-name>.<declared pass_name>`. No second registry, entry-point
group, or manifest format is created, and the namespace above is not exported
from `flagquantum.ecosystem.extensions`.

The built-in optimization pass names are reserved. Because the host builds the
registered name, no declared value can land on one, and the pass registry
independently refuses a reserved name, so a plugin cannot change what
`flagquantum.compiler.compile` does. An admitted pass runs only when a host names
it, through `optimize_with_extension_pass`, which drives the built-in pipeline to
its own fixed point and then runs the admitted pass once. The registered route
closes over the extension handle, so a pass invoked after its extension closed
fails closed rather than running through a lifecycle that already ended.

An unusable declaration, a non-`compiler_pass` kind, a missing `transform`, a
refused negotiation, an extension that declares no `circuit_ir` support, an SDK
version mismatch, and a failed start all fail closed with a `CapabilityError`
before the pass is registered.

See [`API_CHANGE_PROPOSAL_070_COMPILER_PASS_ADMISSION.md`](../development/API_CHANGE_PROPOSAL_070_COMPILER_PASS_ADMISSION.md)
and [`contracts/compiler-pass-admission-v1-candidate.json`](../../contracts/compiler-pass-admission-v1-candidate.json).
`examples/extensions/reference_compiler_extension.py` implements the `compiler`
kind; a pass-level example is not yet published, which the capability matrix
records as an explicit gap.
