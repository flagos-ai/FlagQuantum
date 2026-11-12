# ARCH-014: Realtime package ownership beside Runtime

Status: Approved
Date: 2026-11-09
Approval: The user explicitly requested moving the existing realtime
implementation beside Runtime, including the existing code, on this date.
Scope: Package placement and compatible import migration. Supersedes only the
placement decision in [ARCH-013](ARCH_013_REALTIME_MESSAGING_AND_DEVICE_CALL_BOUNDARY.md).
The wire contract and every hardware, timing and execution non-goal remain.

## Decision and reason

`flagquantum.realtime` is the canonical owner of the existing software messaging
implementation. It is a sibling of `flagquantum.runtime` and
`flagquantum.remote`. Remote owns external task submission, status, cancellation
and result retrieval; realtime framing and classical handler calls are a
different responsibility, including when their eventual endpoints are local.

ARCH-013 selected Remote using the then-existing directory whitelist, team
ownership and reverse-import restrictions. Those engineering constraints do not
require a messaging protocol to be an external task system. The directory move
also removes the canonical import's incidental initialization of Remote's QPU
providers, Compiler/Core and PyTorch.

## Exact migration and compatibility

- Move `protocol`, `payload`, `service`, `channels` and `session`, and their
  package exports, into `flagquantum/realtime/`. Preserve callable signatures,
  ordered exports, wire bytes, status values and behavior. No stable `fq` root
  export or execution mode is added.
- Freeze `flagquantum.remote.realtime` as an import adapter with no protocol
  implementation. Re-export the canonical objects and alias all five old
  submodule paths to the same module objects. Existing imports and old pickles
  remain readable; newly written pickles use the canonical module paths.
- The compatibility path still initializes its existing Remote parent. Only
  the canonical path promises import isolation from Remote and numerical
  dependencies. No removal version is introduced by this migration; retiring
  the compatibility adapter requires a separate explicit decision.
- Move the example to `examples/realtime/device_call.py`; retain the old example
  module as a forwarding entry point. Maintained examples, checks, capability
  records and generated documentation use the canonical path.
- The current Remote maintainers retain maintenance responsibility for the
  protocol and its adapter. Team ownership does not imply a package dependency.

## Dependency enforcement and non-goals

The canonical package may import only the standard library and itself. The
architecture checker resolves relative imports and rejects dependencies on QEC,
Runtime, Remote, Compiler, Simulation or third-party libraries. Core, Compiler,
Runtime, Simulation, Kernels, QEC and Services may not import either realtime
path in this change. This preserves the previous isolation and explicitly
requires review before admitting any execution or application integration.

Moving the protocol does not complete a realtime control loop. The
[QEC integration reservation](../REALTIME_QEC_INTEGRATION_BOUNDARY.md)
continues to describe proposed adapter semantics. Stateful decoder lifetime,
physical correction, hardware transport, clocks, deadlines, dispatcher/ring
implementation, device lifecycle and compiler lowering are not admitted here.

## Acceptance evidence

The existing contract checker and protocol tests remain authoritative for
unchanged wire behavior. The same loopback workflow must run through the
canonical package with imports of Remote, Compiler, Runtime, Simulation, QEC and
PyTorch rejected in a fresh interpreter. Compatibility tests verify all export
and submodule identities, old-pickle reading, and the old example entry point.
Architecture tests must reject forbidden dependencies in both directions,
including relative imports and root aliases. Coverage floors move with the
implementation and continue to cover the compatibility adapter separately.
