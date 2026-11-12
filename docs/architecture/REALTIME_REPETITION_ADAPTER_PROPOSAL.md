# Offline repetition decoder over realtime messaging

Status: Proposed; not an approved API, payload schema or dependency exception.
Date: 2026-11-10
Review owners: Integration, Core/QEC and the realtime maintainers.

## Goal and verified starting point

The next implementation would compare one existing QEC decoder called directly
with that same decoder called through a loopback channel. It would process three
syndrome rounds from each of two separate shots, without executing a circuit or
issuing physical correction commands. This is the bounded adapter stage of the
[integration reservation](REALTIME_QEC_INTEGRATION_BOUNDARY.md).

The merged classical [feedback example](../../examples/realtime/feedback_loop.py)
already proves application feedback over messaging, failure refusal and channel
cleanup. Its five observations of one binary state are not QEC syndromes and
cannot serve as the decoder reference for this proposal.

Existing authoritative interfaces are:

- [`StreamingDecoder.decode_round`](../../flagquantum/qec/decoders/repetition.py)
  consumes a complete, dense history beginning at round zero. The immediate
  `RepetitionStreamingLookupDecoder` selects from the latest syndrome; the
  `RepetitionTemporalDecoder` requires consecutive confirmation and consistent
  detection events. Neither is a stateful incremental decoder.
- [`SyndromeRound` and `Correction`](../../flagquantum/qec/types.py) own QEC record
  meaning. Reuse them; add no alternative decoder protocol or correction class.
- [`HandlerSchema`](../../flagquantum/realtime/payload.py),
  [`FunctionTableService`](../../flagquantum/realtime/service.py) and
  [`RealtimeSession`](../../flagquantum/realtime/session.py) already own payload
  encoding, handler dispatch and device-call correlation.

The existing sliding-window decoder accepts detector-model-specific round data
and has different lifetime semantics. It is not interchangeable with this
repetition history interface merely because both expose `decode_round`.

## Placement and dependencies proposed for review

Put the initial composition in `examples/qec/realtime_repetition.py`, with tests
in `tests/integration/test_realtime_repetition.py`. These are proposed paths,
not runnable modules yet. The example may import the existing QEC records and
decoder, and the canonical realtime package. It owns the mapping and caller state.

Keep both libraries unchanged: realtime continues to import only itself and the
standard library, and QEC, Runtime, Compiler and Simulation do not import
realtime. No new top-level package, root export, decoder registry, execution mode
or dependency manifest is needed. Integration review must admit this specific
application composition before implementation. A reusable library adapter needs
a separate demonstrated consumer and ownership decision.

## Bounded input and proposed payload

Fix three data qubits numbered 0, 1, 2. Check 0 is `Z0 Z1`; check 1 is `Z1 Z2`.
Syndrome bit 1 means the adjacent check disagrees. Only bit-flip correction
recommendations are in scope. This is not a general stabilizer or noise model.

One shot has exactly three rounds. Request round `r` carries the full history
`0..r`, never a detector delta. The candidate handler name is
`repetition_history`; it is selected by the existing function-id hash.

| Argument | Existing descriptor | Meaning |
| --- | --- | --- |
| Shot id | `UINT64`, 8 bytes | Application-assigned identity, unique within the open application run; not an RPC request id. |
| History length | `UINT8`, 1 byte | `r + 1`, restricted to 1, 2 or 3. |
| Syndrome history | `BIT_PACKED`, 1 byte, 6 elements | Round-major bits `(s00, s01, s10, s11, s20, s21)`; unused rounds must be zero. |

All scalar encoding and bit order use the existing payload codec. Padding bits
6 and 7 of the packed byte must also be zero; the handler must refuse them from
the original payload before decoded bits hide them. This application validation
does not change the general-purpose codec's accepted wire format. Request
payload length is 10 bytes; with the existing 24-byte header it fits the default
384-byte slot. No framing field or protocol version changes.

For each used round, reconstruct detection events by comparing its two syndrome
bits to the previous round, initially `(0, 0)`. Emit changed check indices in
ascending order, and reuse `DetectionEvent` and `SyndromeRound`. The sender must
verify any supplied event records against this derivation before transmission;
they are not independently encoded or silently discarded. The handler receives
only the validated used prefix and calls the selected existing decoder once.

## Proposed response and action meaning

| Result | Existing descriptor | Meaning |
| --- | --- | --- |
| Shot id | `UINT64`, 8 bytes | Echo the application shot identity. |
| Round index | `UINT8`, 1 byte | `history_length - 1`; must match the returned `Correction.round_index`. |
| Correction selector | `UINT8`, 1 byte | 0 means `qubit=None`; 1, 2, 3 mean X recommendation on data qubit 0, 1, 2 respectively. |

Response payload length is 10 bytes. Reject any decoder result that is not a
`Correction`, has a different round, unsupported Pauli, or invalid qubit. The
caller checks RPC correlation, success status, payload decoding, shot, round
and selector before reconstructing the existing `Correction` record.

The returned value is a **recommendation for the supplied history**. The first
implementation records and compares it; it does not execute X, update an
accumulating Pauli frame or predict a logical observable. A persistent syndrome
can produce the same recommendation on successive calls. XOR-accumulating those
recommendations would cancel them and is not a valid interpretation of this
offline replay. Physical feedback must instead use the existing QEC execution
controller and account for intervening corrections in later observations.

## Application lifetime and failure rules

One application run permits one active shot, one synchronous outstanding call
and three rounds per shot. The decoder and handler are stateless; caller state
enforces progression. A handler alone does not establish exactly-once action.

| Transition or condition | Required application behavior |
| --- | --- |
| Begin shot | Require no active shot, choose a previously unused shot id, start with an empty history and expected round 0. |
| Submit round | Require the next dense round, immutable earlier history and one active shot. Append to a candidate history without committing it. |
| Successful matching response | Record the recommendation and commit that candidate history once; advance expected round. |
| Final round succeeds | Finalize the shot after round 2; no extra decoder flush or repeated terminal call. Preserve the three decisions as evidence. |
| Begin another shot | Initialize empty history and round 0 with a new shot id; reuse of a channel does not reset hidden decoder state. |
| Duplicate or out-of-order local submission | Refuse before sending; do not reuse an earlier decision or execute an action twice. |
| Failure, stale shot/round, changed history or malformed response | Abort the active shot without committing its candidate round; do not return fabricated success or a partial successful-shot result. |
| Uncertain delivery or transport exception | Abort; no automatic retry, resume, cancellation acknowledgement or exactly-once guarantee. Recovery starts a new application run. |
| Shutdown, normal or exceptional | Close application channels, then stop the owned handler session in `finally`; reject further calls. |

RPC request ids increase within a run, never wrap, and are distinct from shot
and round identity. The proposed example uses only six calls. `ptp_timestamp`
stays opaque metadata, with no overload as a shot id, deadline or clock reading.
Directly replaying an RPC may still invoke a stateless handler again; duplicate
prevention is the synchronous caller's local policy, not a protocol property.

For invalid application arguments use a positive handler-specific status 1;
invalid decoder output uses positive status 2. Existing negative schema,
unknown-function and handler-exception statuses remain unchanged. Preserve these
distinctions in caller errors. A peer without this application schema is refused.
No decoder callback may be replaced silently after failure.

## Acceptance before implementation is considered complete

1. Enumerate all four syndrome values for the immediate decoder and compare
   direct and loopback decisions to the independent table:
   `(00 -> None, 10 -> 0, 11 -> 1, 01 -> 2)`.
2. Enumerate all 64 three-round syndrome histories, reconstruct events and
   compare every prefix. Replace the immediate handler with the existing
   temporal decoder without changing payload framing or the caller. Check its
   independent rule: first-round no action; later action only for two identical
   nonzero consecutive syndromes. Include isolated readout faults and terminal
   onset without confirmation; do not claim temporal decoding corrects them all.
3. Use separate shots and unique ids. Check dense progression, unchanged
   prefixes, finalization, no-op correction and bounded encoding. Exercise
   nonzero padding, invalid lengths, inconsistent events and invalid selectors.
4. Inject wrong RPC id, wrong timestamp, wrong shot and wrong round separately.
   Confirm refusal before recommendation commitment. Exercise declared failure,
   decoder exceptions, transport failure, duplicate submission and cleanup.
5. Confirm no edits to package imports, public API snapshots, realtime wire
   contract, capability maturity or hardware parity claims. Run the existing
   protocol/architecture tests alongside the new application tests.

No latency, deadline, throughput, hardware or QEC logical-error improvement is
established by these tests. They prove the bounded translation and lifecycle.

## Runnable baseline today

These existing commands exercise the source interfaces this proposal composes:

```bash
python -m examples.realtime.feedback_loop
python -m pytest tests/qec/test_repetition_memory.py -q
python tools/check_realtime_messaging_contract.py
```

The first prints two shots with `corrected states=(0, 0, 0)` and
`FlagQuantum offline feedback loop check passed`. The second tests existing
repetition decoding and feedback directly; it does not use realtime messaging.
The contract checker reports 159 checks. The proposed adapter is not implemented
by this document, and no new command or `feedback_mode="realtime"` is available.
