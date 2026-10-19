# Realtime messaging and device calls

This package defines the **software half** of a realtime control surface: the slot
framing a call travels in, the payload type system that gives the bytes meaning,
the handler table a call resolves against, and the seam that chooses a channel at
runtime. It is the part of the surface that can be specified, implemented, and
tested without hardware.

It is also the part that makes a hardware partner's job well-defined: an
interoperating implementation — an FPGA image, a NIC driver, a service on another
host — produces the same bytes because the layout is written down here rather
than inferred from a vendor header.

## What this package owns

| Module | Responsibility |
| --- | --- |
| `protocol.py` | The 24-byte packed request and response headers, their magic words, the function-id hash, and slot framing: `header_size + payload_length <= slot_size`. |
| `payload.py` | The ten payload type ids, the 12-byte type descriptor, the eight-argument and four-result limits, little-endian and IEEE-754 encoding, and least-significant-bit-first bit packing. |
| `service.py` | The function table, the schema-driven decode of a request and encode of a response, and the status convention for a call that cannot be served. |
| `channels.py` | The channel lifecycle, the transport protocol, the runtime channel registry, and `LoopbackTransport`, the offline contract fake. |
| `session.py` | One open channel per device id, resolved by name, released together. |

## What this package must not own

- **No transport.** No socket, no RDMA verbs, no shared memory, no NIC, no FPGA,
  no GPU. A channel's transport is the other side of the `RingTransport`
  protocol and does not live here.
- **No clock and no timing claim.** `ptp_timestamp` is carried and echoed
  verbatim; nothing in this package reads a clock, measures a duration, or bounds
  a wait. There is deliberately no per-dispatch timeout, because this module owns
  no clock to measure one with.
- **No dispatcher and no ring buffer.** Ring allocation, frame leasing, request
  publication, response completion, and transport shutdown stay the channel's
  responsibility. A session owns its function table and nothing that carries it.
- **No compiler surface.** There is no `device_call` lowering, no realtime
  dialect, and no slot-count, slot-size, or timeout flag family.
- **No device lifecycle.** Per `flagquantum/remote/AGENTS.md`, this package does
  not implement direct device lifecycle, simulation kernels, compiler passes, or
  Runtime scheduling.
- **No simulator coupling.** `architecture.toml` forbids `simulation` from
  importing `remote`, so a channel is not reachable from the simulator's inner
  loop. This is a boundary, not an oversight.

## Allowed dependencies

Standard library, `flagquantum.remote`'s own modules, and nothing else. The
package imports no third-party library, no `flagquantum.simulation`, no
`flagquantum.compiler`, and no `flagquantum.runtime`.

## Public entry points

The package is reachable by module path and is **not** re-exported from
`flagquantum.remote`, so nothing here is Stable Core:

```python
from flagquantum.remote import realtime as rt
```

| Name | Use |
| --- | --- |
| `rt.RPCHeader`, `rt.RPCResponse` | Build, encode, and read a slot header. |
| `rt.function_id` | Hash a handler name to the id a call carries. |
| `rt.PayloadType`, `rt.TypeDescriptor`, `rt.HandlerSchema` | Declare what a call's bytes are. |
| `rt.encode_arguments`, `rt.decode_arguments` | Encode a call and decode its response against a schema. |
| `rt.FunctionEntry`, `rt.FunctionTableService` | Serve a table of handlers. |
| `rt.DeviceCallChannel`, `rt.DeviceCallChannelDriver`, `rt.loopback_channel` | Open a channel by name. |
| `rt.RealtimeSession` | Resolve device ids to open channels for one run. |

## The shortest path to a typical change

1. Change the implementation in the module that owns it.
2. If the change alters anything the contract states, edit
   `contracts/realtime-messaging-v1-candidate.json` in the same commit.
3. Run the gate, which reconciles the contract with the implementation and
   requires every declared reader to exist:

   ```bash
   python tools/check_realtime_messaging_contract.py
   ```

4. Run the witness tests, which are the contract's named readers:

   ```bash
   python -m pytest tests/unit/test_realtime_protocol.py -q
   ```

5. Run the offline golden path, which is a real round trip over the loopback
   transport:

   ```bash
   python examples/remote/realtime_device_call.py
   ```

## How to exercise the protocol without hardware

`LoopbackTransport` frames a request into a slot, reads it back, dispatches it
through a `DeviceCallSession`, frames the response, and hands that back. A test
that needs a channel registers `loopback_channel(name, session)` with a
`DeviceCallChannelDriver` and opens it by name, exactly as a program would open a
hardware channel. `examples/remote/realtime_device_call.py` is the ten-minute
golden path: it builds a majority-vote QEC handler, dispatches one decoding round,
and prints the request hex next to the response status.

## Evidence status

No hardware was contacted and no hardware evidence exists. A realtime link has
never been established by this package, and no latency, throughput, or scalability
number is claimed or measurable here. The capability maturity entry for this
package is `development_evidence`, and the parity rows for the hardware half of
the surface stay `unsupported` in `contracts/cudaq-parity-matrix.toml`. See
[ARCH-013](../../../docs/architecture/decisions/ARCH_013_REALTIME_MESSAGING_AND_DEVICE_CALL_BOUNDARY.md)
for the boundary this package implements and the boundary it refuses.
