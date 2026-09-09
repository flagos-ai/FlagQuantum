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
- Do not model pulse programs as `CircuitIR`. A pulse-level plugin requires an
  approved FlagQuantum-owned pulse artifact before it can join this contract.

## Acceptance

- discovery does not import extensions of unrequested kinds;
- entry-point and manifest identities must match;
- load, compatibility, negotiation, invocation, and cleanup failures fail closed;
- compiler conformance proves `CircuitIR` ownership, input immutability,
  deterministic output, and lifecycle cleanup;
- a standalone package can register a compiler without changing FlagQuantum.
