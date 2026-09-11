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
