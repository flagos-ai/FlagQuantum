# ARCH-013: Realtime messaging protocol and device-call boundary

Status: Approved
Date: 2026-10-06
Scope: boundary and contract record for one new package, `flagquantum/remote/realtime/`.
No Stable Core change. No capability availability or maturity claim.
Related: [ARCH-012](ARCH_012_CUDAQ_PARITY_CONTROL_SEQUENCE.md) clause 5 (an owned
gap is recorded, never silence), [ARCH-001](ARCH_001_COMPUTE_REMOTE_AND_CORE_CONTRACTS.md)
(the control boundary between directly controlled compute and an external task system).

## Context

CUDA-Q exposes a realtime control surface, and FlagQuantum's parity requirement
covers the whole surface. Reading that surface closely is what makes this record
necessary, because the honest split between what is software and what is hardware
is narrow and easy to get wrong in the flattering direction.

**The surface is an RPC ABI, not a simulator feature.** It consists of a fixed
24-byte slot header, a typeless payload whose meaning comes from an out-of-band
handler schema, a function table keyed by a hashed handler name, a status
convention, and a runtime channel-selection seam. Alongside that sits a hardware
stack: an FPGA, a ConnectX NIC, RDMA/RoCE transport, GPUDirect, persistent RX and
TX dispatch kernels over shared ring buffers, and a PTP clock.

**Every timing property lives on the hardware side.** The published latency
figures come from the FPGA, the NIC, and the persistent-kernel ring together.
Nothing in the software half measures, guarantees, or bounds time. The software
half carries a 64-bit timestamp field and echoes it verbatim; it never reads a
clock.

**Three earlier records in this repository already falsify realtime claims.**
`contracts/hybrid-compilation-private-v0-candidate.json` sets
`realtime_decoder_integration`, `hard_realtime_claim`, and
`measurement_feedback_latency` to `false`. `docs/roadmap/CUDAQ_PARITY_STRATEGY.md`
§ 6 states the mandate directly: the software side of the realtime surface — a
vendor-neutral messaging protocol and a device-call abstraction — can be owned and
delivered, the hardware side cannot, and the hardware side is represented as an
owned blocked row rather than as a plan. This ADR records the decision that
follows from those statements.

**Two architectural limits bear on the placement.** `architecture.toml` lists
`remote` among the allowed top-level directories and constrains neither
`remote/**` nor `deployment/**` with a forbidden-dependency prefix; it does list
`remote` inside `compiler_forbidden` and `simulation_forbidden`, so neither the
compiler nor the simulator may reach a realtime channel. `team-ownership.toml`
assigns `flagquantum/remote/**` to the `remote` team. A realtime messaging package
therefore belongs under `remote`, and by construction cannot be wired into the
simulator's inner loop.

## Decision

### 1. FlagQuantum owns the framing, the type system, and the seam

`flagquantum/remote/realtime/` owns:

- the slot framing: a 24-byte packed header, a payload, and unused padding;
- the payload type system: the ten type ids, the 12-byte type descriptor, the
  little-endian and IEEE-754 encoding rules, the least-significant-bit-first bit
  packing, and the eight-argument / four-result limits;
- the handler schema and the schema-driven encode and decode of a call;
- the function-id hash and the function table that resolves it;
- the status convention: zero for a served call, a positive handler code for a
  handler-specific failure, and a negative code for a protocol-level failure;
- the channel-selection seam: a channel is chosen by name when a program starts
  and is not compiled in, and one open channel exists per device id;
- the ownership split around a session: a session owns its function table and
  keeps it valid until it stops, while ring allocation, frame leasing, request
  publication, response completion, and transport shutdown belong to the channel.

The layout, the type ids, and the limits are taken from CUDA-Q's published
protocol rather than invented, because an interoperating implementation has to
produce the same bytes. The names, the default slot size, and the Python
behaviour are FlagQuantum's own.

### 2. A hardware partner owns the transport and every timing property

Not owned here, and not claimed here: the FPGA bitstream and its programmable
logic; RDMA/RoCE transport; a ConnectX NIC; GPUDirect; NVQLink; a Holoscan sensor
bridge; PTP clock accuracy; and microsecond latency.

The package therefore adds no latency field, no timeout, no throughput figure,
and no realtime-ness claim of any kind. `ptp_timestamp` is carried and echoed; it
is never produced or interpreted. A per-dispatch timeout is deliberately absent
because this module owns no clock to measure one with, so waiting belongs to the
transport; adding a field nothing could test would be a claim rather than a
behaviour.

The compiler half of the surface is also out of scope: there is no `device_call`
lowering, no realtime dialect, and no flag family for slot count, slot size, or
timeout. The compile-time constants and the runtime flag names are compiler
surface, and this record does not authorize them.

### 3. ARCH-012 clause 2 does not apply, and the reason is stated

ARCH-012 clause 2 admits a new horizontal abstraction only when it **replaces**
an existing implementation behind an existing boundary, proven by a replacement
test in which at least one implementation is swapped without modifying its
consumers. This package replaces nothing, so clause 2 must be addressed rather
than assumed.

Clause 2 governs *internal horizontal abstractions* — a second way of doing
something the repository already does, introduced to generalize. This package is
not that. It defines an externally specified interchange format that no existing
FlagQuantum type expresses, for a capability that currently has no implementation
at all, under clause 5's rule that a capability present in CUDA-Q and absent here
appears as an explicit owned gap rather than as silence. There is no existing
consumer to swap and no second internal representation to compete with.

The claim is falsifiable rather than rhetorical: the package adds no second
registration authority. The function table is an owned type with a collision
check, the channel registry is owned by a driver, and neither shadows
`flagquantum/ecosystem/extensions/sdk.py`, which remains the only plugin-admission
surface. A future round that tries to reach the simulator's inner loop with this
package would have to change `architecture.toml`, and that is a separate decision.

### 4. The contract and its fake land together with a conformance test

`contracts/realtime-messaging-v1-candidate.json` restates the layout, the ids,
the limits, the status convention, and the hash vectors so a tool can read them
without importing the package. `LoopbackTransport` is the contract fake: an
in-process transport that performs no I/O and is how the protocol is exercised
offline. The gate `tools/check_realtime_messaging_contract.py` is the contract's
reader, and it fails if the contract and the implementation disagree. Two of its
checks reach outside this repository on purpose, because a contract that only
agrees with the code it was written from proves nothing:

- the response magic is compared with the leading bytes of the published
  protocol's own worked response example;
- the function-id hash is compared with the FNV specification's published 32-bit
  test vectors.

Every section of the contract names its readers, and each named reader must exist
as a check in the gate or as a test function in
`tests/unit/test_realtime_protocol.py`. A contract section with no consumer is a
defect, per `contracts/README.md`.

The gate also requires the file `approval.record` names to exist. This record was
written after that field had already been pointed at a file that did not exist,
and a contract that claims an approval it cannot produce is the same class of
defect as a section with no reader: a written assertion nobody can check.

### 5. The consumers that are sanctioned, and the one that is not

A realtime call is consumed by a service-side handler: a function whose arguments
and results a registered schema describes. That is the whole sanctioned coupling
today.

The coupling that is explicitly **not** sanctioned is the simulator. The
simulator is written to run without a channel, and this record does not create a
realtime execution mode: `realtime_session` remains vocabulary in
`contracts/long-horizon-architecture-v1.json` and is not executable here. Nothing
in this package is reachable from `fq.run`, and no runtime option, target, or
result field changes.

The role a future round would most plausibly connect this to is the existing
`StreamingDecoder` protocol in `flagquantum/qec/decoders.py`, whose round-by-round
shape matches a decoding round. This record does **not** claim that connection:
`flagquantum/qec/**` is untouched, and `StreamingDecoder` must not be described as
realtime-satisfied because of this package.

### 6. The parity rows this moves, and the rows it does not

The matrix rows for `realtime_host_api`, `realtime_transport`,
`realtime_feedback_loop`, and `realtime_sensor_bridge` live in
`contracts/cudaq-parity-matrix.toml`. Only `realtime_host_api` moves, and only to
`partial`:

- it is the one row whose `dependency_class` is `B_open_neutral`, so closing part
  of it needs no proprietary component;
- it cannot move to `supported`, because the dispatcher, ring, and transport half
  of the host-facing surface is absent by design.

`realtime_transport` stays `unsupported` and `A_nvidia_proprietary`. Its reason
text is rewritten to record that the software half of the device-call channel
abstraction now exists and that only the RoCE/Ethernet transport half remains
blocked on hardware. Flipping it would contradict the dependency-class accounting
in `docs/roadmap/CUDAQ_PARITY_STRATEGY.md` § 5 and the non-goal in § 6.
`realtime_feedback_loop` stays `unsupported` as the owned blocked row § 6 requires.
`realtime_sensor_bridge` stays `unsupported` and `later`.

The domain-level evidence entry for `realtime_control` currently asserts that no
realtime messaging module exists anywhere under `flagquantum/`. That statement
becomes false the moment this package lands, and no gate catches it, because a
`search:` evidence token is only checked for being non-empty. Replacing it with
the real module path and a still-true narrower search is part of this change, not
a nicety.

## Consequences

**What becomes possible.** A hardware partner can implement a channel against a
published slot layout, a published type system, and a published status
convention, without reading FlagQuantum's source and without FlagQuantum
depending on the partner's headers. The protocol is exercised end to end offline,
so a regression in framing or typing is caught without hardware.

**What remains blocked, and is recorded as blocked.** Every timing claim, every
transport, and every hardware-evidence claim. The `realtime_feedback_loop` row is
the owned representation of that blockage. No capability maturity level above
`development_evidence` is claimed for anything in this package, and the package
carries no hardware evidence.

**What this record does not authorize.** No Stable Core change; no new public
export; no new capability availability claim; a `logical` top-level package; a
realtime execution mode; a compiler lowering; or a dependency on a CUDA-Q shared
object or header. Interoperability is at the wire level only, and even that is an
obligation the other side verifies out of band.

**The open question this record leaves.** Whether FlagQuantum ever exposes a
realtime execution mode is a scheduling decision for a later round, and it depends
on hardware that does not exist here. Until then the vocabulary stays vocabulary.
