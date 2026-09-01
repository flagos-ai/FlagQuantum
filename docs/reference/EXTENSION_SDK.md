# FlagQuantum Extension SDK

The frozen first-public-alpha SDK contract lives under
`flagquantum.extensions`; it does not add root exports. Extensions declare a
versioned manifest, negotiate capabilities before activation, and are installed
into a task-local immutable registry.
The protocol contract was separately frozen by the API owner on 2026-09-01.
Individual extensions remain experimental by default and require independent
qualification.

Supported extension kinds are execution backends, kernels, operators, compiler
passes, devices, providers, measurement collectors, and planners. Protocols
define the minimum lifecycle and operation surface for each kind.

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
only trusted packages should be installed.

## Conformance

`flagquantum.extensions` supplies reusable backend and provider checks covering
manifest/payload serialization, capability honesty, PyTorch gradients,
dtype/device preservation, isolated errors, and cleanup. The conformance
submodule remains an equivalent explicit import path. Reference extensions are
in `examples/extensions/reference_extensions.py` and import only the public SDK
plus PyTorch.
