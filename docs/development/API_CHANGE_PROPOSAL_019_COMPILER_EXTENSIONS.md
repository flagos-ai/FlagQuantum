# API Change Proposal 019: Installed compiler extensions

## Status

Approved for the first public alpha on 2026-09-09 by explicit repository-owner
direction. The extension SDK remains pre-freeze.

## Problem

The extension SDK can register objects supplied directly by application code,
but it cannot discover an independently installed compiler package. The existing
`compiler_pass` kind describes one transformation pass and does not accurately
describe a complete circuit transpiler such as QSteed.

## Decision

- Add the `compiler` extension kind for complete circuit-level compilers.
- A compiler consumes and returns FlagQuantum `CircuitIR`; vendor objects do not
  cross the extension boundary.
- Discover installed extensions only after an explicit
  `discover_extensions("compiler")` call.
- Use the Python entry-point group `flagquantum.extensions`. Entry-point names
  have the form `compiler.<manifest-name>` and resolve to zero-argument factories.
- Reuse `ExtensionRegistry`, capability negotiation, lifecycle containment, and
  error handling. Do not introduce another plugin registry.
- Provide `compile_with_extension(program, *, extension, target=None)` as the
  expert one-call extension journey. It performs discovery, `circuit_ir`
  capability negotiation, activation, compilation, result-type validation, and
  cleanup without changing the default compiler path.
- Add `fq.compile(program, compiler=None, target=None)` as the stable user
  journey. Omitting `compiler` uses the built-in FlagQuantum compiler. A named
  compiler selects exactly that installed extension without fallback.
- Extend the stable `fq.run` journey with explicit `compiler`, `target`,
  `shots`, and optional `name` keywords. When `compiler` and `target` are
  present, `fq.run` compiles, packages, submits, waits for the result, and
  returns the same stable
  `ExecutionResult` type used by local execution. Provider counts are exposed
  as a `counts` measurement; the provider-native `DeploymentResult` remains
  available through `result.native()`.
- Accept the compact target locator `provider:backend` at the root facade. For
  Quafu, the facade obtains the current chip snapshot and passes it to the
  selected compiler; the plugin returns logical `CircuitIR` plus an ordered
  physical `target_qubits` mapping.
- Let `fq.compile` and remote `fq.run` accept an optional ordered
  `target_qubits` mapping. When omitted, the compiler selects a connected
  physical subgraph. When supplied, the compiler must validate the mapping
  against the current calibration snapshot and restrict routing to that
  subgraph; invalid, unavailable, duplicate, or disconnected selections fail
  before submission.
- Do not model pulse programs as `CircuitIR`. A pulse-level plugin requires an
  approved FlagQuantum-owned pulse artifact before it can join this contract.

## Naming and alternatives

`fq.compile(..., compiler="qsteed")` states the user intent directly and keeps
the built-in default. `compile_with_extension` remains the explicit expert
mechanism beneath it; built-in target-aware `flagquantum.compiler.compile` and
target-independent `flagquantum.compiler.optimize` retain their existing
responsibilities. A second registry or compiler manager remains rejected as
duplicate authority.

## Acceptance

- discovery does not import extensions of unrequested kinds;
- entry-point and manifest identities must match;
- load, compatibility, negotiation, invocation, and cleanup failures fail closed;
- compiler conformance proves `CircuitIR` ownership, input immutability,
  deterministic output, and lifecycle cleanup;
- the one-call journey selects one named compiler, forwards its target, returns
  `CircuitIR`, always closes the extension, and never falls back;
- remote `fq.run` requires an explicit compiler, target, and positive shot
  count; an omitted name uses the deployment default, while an explicitly
  supplied name must be non-empty. The workflow performs no hidden compiler or
  provider fallback and preserves the provider-native result behind the stable
  result boundary;
- an explicit physical mapping is preserved in the compiled IR, deployment
  submission, and stable result provenance without rewriting logical QASM wire
  indices;
- a standalone package can register a compiler without changing FlagQuantum.

## First-public-alpha release note

Installed circuit compilers can now be selected explicitly through
`fq.compile`; the default FlagQuantum compiler remains unchanged.
