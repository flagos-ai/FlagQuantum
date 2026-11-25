# Realtime and QEC Integration Reservation

Status: Proposed QEC integration direction; architecture review only. Package
placement is separately approved by ARCH-014. No QEC adapter, wire schema,
dependency exception or execution mode is admitted by this reservation.

## Purpose and current authority

Reserve a path from the existing software messaging boundary to a future QEC
feedback application without making realtime depend on an unfinished decoder
contract. Protocol work can proceed now; a QEC adapter needs a bounded, reviewed
decoder workflow, rather than maturity of every QEC algorithm.

[ARCH-013](decisions/ARCH_013_REALTIME_MESSAGING_AND_DEVICE_CALL_BOUNDARY.md)
remains the protocol authority; its placement is superseded by
[ARCH-014](decisions/ARCH_014_REALTIME_PACKAGE_OWNERSHIP.md). The implementation lives in
[`flagquantum.realtime`](../../flagquantum/realtime/README.md):
headers and payloads, function tables, channel selection, sessions, and an
offline loopback transport. The
[messaging contract](../../contracts/realtime-messaging-v1-candidate.json) and
its existing checker remain the single source for wire conformance.

The current majority-vote example exercises a software request/response. It
does not use `qec.decoders`, execute a circuit, or establish a hardware feedback
loop. Neither its correction bit nor its echoed timestamp establishes QEC
correctness or a deadline guarantee.

## Ownership and dependency direction

| Concern | Existing owner | Reservation for a later integration |
| --- | --- | --- |
| Headers, payload types, handler dispatch, channel and session interfaces | `realtime` | Keep messages domain neutral; accept an application-supplied handler through the existing function table. |
| Code, syndrome meaning, decoding and correction coordinates | `qec` | Preserve existing decoder contracts; define any translation at an explicitly reviewed application boundary. |
| Circuit lowering and execution | Compiler, Runtime and Simulation | Continue through their existing contracts; no simulator-to-realtime import or realtime scheduling inside this package. |
| Transport and physical control | Outside the current software implementation | Require a separate owner, concrete implementation and conformance evidence before admission. Remote does not acquire direct device lifecycle ownership. |
| Composition of decoder and messaging | Candidate application adapter, placement subject to review | Translate approved domain records to an existing `HandlerSchema` and back; do not create another decoder or execution engine. |

The proposed composition consumes QEC and realtime interfaces. Neither domain
imports that composition, and realtime modules must not import QEC. A caller
selects a handler and a channel independently; choosing a decoder must not
require changing slot framing or transport consumers.

The implementation now lives at `flagquantum.realtime`, beside Runtime and
Remote. ARCH-014 preserves the old `flagquantum.remote.realtime` imports through
a frozen adapter. Canonical realtime modules import only the standard library
and themselves; the fresh-process round trip below rejects Remote, Compiler,
Runtime, Simulation, QEC and PyTorch imports. Using the old compatibility path
still executes its Remote parent initializer.

## Contract questions before a QEC adapter

The [bounded repetition adapter proposal](REALTIME_REPETITION_ADAPTER_PROPOSAL.md)
answers these questions for a candidate three-round offline replay. It remains
proposed and admits no implementation, execution mode or dependency change.

1. **Input identity:** fix one code and noise scope, syndrome/detector ordering,
   round numbering, shot identity and code configuration. Specify whether the
   decoder receives a full history or an incremental batch. The existing
   repetition `StreamingDecoder.decode_round` consumes a history; its signature
   is not evidence of a stateful incremental service.
2. **Output meaning:** distinguish a physical correction, a Pauli-frame update
   and a logical-observable prediction. State coordinates and application order;
   do not reinterpret a decoder prediction as a physical gate command.
3. **State lifetime:** specify initialization, shot isolation, reset, window
   advancement, flush/finalization and correction commitment. Reopening a
   messaging channel is not a defined decoder reset. Decide these semantics
   before adding lifecycle methods or choosing an adapter home.
4. **Delivery and failure:** define ordering, duplicate/retried requests, stale
   responses, failed calls and uncertain correction commitment. `request_id`
   correlates a response; it does not supply exactly-once delivery, a shot id,
   cancellation or idempotency. Preserve the existing wire contract; any
   additional envelope or schema requires its own review.
5. **Execution and timing:** identify where an approved correction is consumed
   and which execution combinations are supported or refused. Define the
   deadline endpoints, clock source, queue bounds and late-response policy
   separately for a real transport. `ptp_timestamp` is opaque metadata today.

## Staged delivery and acceptance

| Stage | Concrete deliverable | Gate |
| --- | --- | --- |
| Architecture reservation, this change | Existing ownership map, QEC integration questions and a QEC-independent loopback witness. | Existing wire checks and round-trip/refusal tests pass; public behavior is unchanged. |
| Bounded QEC adapter proposal | One existing decoder, explicit input/output mapping and lifecycle/failure tables, plus a proposed implementation home. | Review and approve the concrete contract before introducing APIs; compare results to an independent reference on multiple rounds and separate shots. |
| Offline adapter implementation | Only the approved mapping, using the existing handler and channel interfaces. | Validate correction meaning, reset/finalization, duplicate/stale/failure behavior and unsupported cases against the approved contract. Swap the handler behind the same consumer where replacement is claimed. |
| Hardware transport proposal and implementation | Named device/control owner, framing-compatible transport, bounded queues and measured end-to-end feedback. | Review transport lifecycle and failure semantics; pass replacement conformance, then collect device evidence and latency distributions including missed deadlines. |

Hardware transport design can advance independently once its own contract is
reviewed. Integrating it with QEC depends on the bounded adapter contract, not
on completing all QEC work. Passing an offline stage never promotes hardware
support or realtime performance claims; capability and parity entries require
their existing evidence process.

## Runnable evidence now

From the checkout with a supported core installation:

```bash
python -m examples.realtime.device_call
python tools/check_realtime_messaging_contract.py
python -m pytest tests/unit/test_realtime_protocol.py tests/integration/test_realtime_device_call_golden_path.py -q
```

The example's stdout includes:

```text
registered channels: ('host-dispatch',)
round 1 correction: 1
slot geometry: 384 bytes, header 24
mismatched schema status: -2
unopened device: device id 1 has no open channel; open device ids are 0
FlagQuantum realtime device call check passed
```

The integration tests also run this same workflow in a fresh interpreter that
rejects imports of Remote, Compiler, Runtime, Simulation, QEC and PyTorch. This witnesses the
current ability to exercise messaging before a QEC adapter exists; it does not
validate the proposed adapter, a hardware transport or a timing budget.
