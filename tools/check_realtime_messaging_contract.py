#!/usr/bin/env python3
"""Validate the realtime messaging contract against the shipped implementation.

``contracts/realtime-messaging-v1-candidate.json`` freezes the wire format, the
payload type system, the status vocabulary, the function-id hash, and the
device-call abstraction, so a hardware partner can implement a channel without
reading this repository's source. A contract nobody reads is decoration, so this
gate is its reader: every section of the contract is reconciled with
``flagquantum.remote.realtime`` rather than with a second copy of the same
numbers.

Two claims are checked against sources outside this repository, because a
contract that only agrees with the code it was written from proves nothing:

* the response magic is compared with the leading bytes of the published
  protocol's own worked response example;
* the function-id hash is compared with the FNV specification's published
  32-bit test vectors.

The remaining sections name their readers in the contract's ``readers`` map, and
every one of those readers must exist: either a check in this file or a test
function that is really present in the witness test module.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from flagquantum.remote import realtime as rt
from flagquantum.remote.realtime import protocol as realtime_protocol

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts" / "realtime-messaging-v1-candidate.json"
WITNESS_TESTS = ROOT / "tests" / "unit" / "test_realtime_protocol.py"
PACKAGE_INIT = ROOT / "flagquantum" / "remote" / "realtime" / "__init__.py"

EXPECTED_SCHEMA = "flagquantum.realtime_messaging_contract"
EXPECTED_SCHEMA_VERSION = "1.0"
EXPECTED_STATUS = "candidate"
EXPECTED_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "status",
        "owner",
        "approval",
        "source",
        "wire",
        "payload",
        "status_codes",
        "function_id",
        "device_call",
        "hardware_boundary",
        "readers",
    }
)
EXPECTED_READER_SECTIONS = (
    "wire",
    "payload",
    "status_codes",
    "function_id",
    "device_call",
    "hardware_boundary",
)
REQUIRED_OUT_OF_SCOPE = (
    "fpga_bitstream_and_programmable_logic",
    "rdma_roce_transport",
    "connectx_nic",
    "gpu_direct",
    "nvqlink",
    "holoscan_sensor_bridge",
    "ptp_clock_accuracy",
    "microsecond_latency",
)
REQUIRED_NOT_CLAIMED = (
    "hardware_evidence",
    "latency",
    "throughput",
    "scalability",
    "a working realtime link",
)
# A package that describes the software half of a hardware protocol has to say
# which hardware it does not cover, in the place a reader looks first.
REQUIRED_DISAVOWALS = ("FPGA", "RDMA", "GPUDirect", "NIC")

REFUSALS = (rt.RealtimeProtocolError, rt.RealtimeCallError)

_FAILURES: list[str] = []
_CHECKS = 0


def _require(condition: bool, message: str) -> None:
    global _CHECKS
    _CHECKS += 1
    if not condition:
        _FAILURES.append(message)


def _refused(action: Callable[[], object], message: str) -> None:
    """Require `action` to refuse rather than to answer."""
    global _CHECKS
    _CHECKS += 1
    try:
        action()
    except REFUSALS:
        return
    _FAILURES.append(message)


def _field_bytes(struct: dict[str, Any]) -> int:
    return sum(int(field["bytes"]) for field in struct["fields"])


def check_wire(contract: dict[str, Any]) -> None:
    """Reconcile the framing, the magic words, and the worked examples."""
    wire = contract["wire"]
    _require(wire["byte_order"] == "little", "wire.byte_order must be little-endian")
    _require(wire["float_format"] == "ieee754", "wire.float_format must be ieee754")
    _require(wire["packed"] is True, "wire.packed must be true")
    _require(
        int(wire["header_size_bytes"]) == rt.HEADER_SIZE,
        f"wire.header_size_bytes is {wire['header_size_bytes']} but the "
        f"implementation frames {rt.HEADER_SIZE} bytes",
    )
    for name, struct in (("request", wire["request"]), ("response", wire["response"])):
        _require(
            _field_bytes(struct) == rt.HEADER_SIZE,
            f"wire.{name} fields do not add up to the declared header size",
        )
    for name, struct, magic in (
        ("request", wire["request"], rt.MAGIC_REQUEST),
        ("response", wire["response"], rt.MAGIC_RESPONSE),
    ):
        _require(
            int(struct["magic"]) == magic,
            f"wire.{name}.magic is {struct['magic']} but the implementation "
            f"frames {magic}",
        )
        _require(
            struct["magic_bytes_little_endian"] == magic.to_bytes(4, "little").hex(),
            f"wire.{name}.magic_bytes_little_endian does not encode the declared magic",
        )
    # The published protocol shows a response slot beginning `53 51 55 43`. That
    # worked example is the only byte-level fact the upstream documentation
    # states outright, so it is the one independent check available on framing.
    _require(
        wire["response"]["magic_bytes_little_endian"] == "53515543",
        "wire.response.magic_bytes_little_endian no longer matches the published "
        "response example",
    )
    _check_request_example(wire["request_example"])
    _check_response_example(wire["response_example"])
    _require(bool(wire["slot_rule"]), "wire.slot_rule must state the rule it names")
    _require(
        bool(wire["padding_rule"]), "wire.padding_rule must state the rule it names"
    )
    _check_slot_rule(wire["request_example"])


def _check_request_example(example: dict[str, Any]) -> None:
    header, payload = rt.read_request(bytes.fromhex(_rebuild(example, request=True)))
    expected = (
        (header.function_id, int(example["function_id"]), "function_id"),
        (header.arg_len, int(example["arg_len"]), "arg_len"),
        (header.request_id, int(example["request_id"]), "request_id"),
        (header.ptp_timestamp, int(example["ptp_timestamp"]), "ptp_timestamp"),
    )
    for actual, declared, name in expected:
        _require(
            actual == declared,
            f"wire.request_example.{name} does not survive a round trip",
        )
    _require(
        payload.hex() == example["payload_hex"],
        "wire.request_example.payload_hex is not the payload the slot carries",
    )


def _check_response_example(example: dict[str, Any]) -> None:
    response, payload = rt.read_response(
        bytes.fromhex(_rebuild(example, request=False))
    )
    expected = (
        (response.status, int(example["status"]), "status"),
        (response.result_len, int(example["result_len"]), "result_len"),
        (response.request_id, int(example["request_id"]), "request_id"),
        (response.ptp_timestamp, int(example["ptp_timestamp"]), "ptp_timestamp"),
    )
    for actual, declared, name in expected:
        _require(
            actual == declared,
            f"wire.response_example.{name} does not survive a round trip",
        )
    _require(
        payload.hex() == example["payload_hex"],
        "wire.response_example.payload_hex is not the payload the slot carries",
    )


def _rebuild(example: dict[str, Any], *, request: bool) -> str:
    """Rebuild a worked example and require the contract's own bytes."""
    slot_size = int(example["slot_size"])
    payload = bytes.fromhex(example["payload_hex"])
    if request:
        built = rt.frame_request(
            rt.RPCHeader(
                int(example["function_id"]),
                int(example["arg_len"]),
                int(example["request_id"]),
                int(example["ptp_timestamp"]),
            ),
            payload,
            slot_size=slot_size,
        )
        label = "wire.request_example.slot_hex"
    else:
        built = rt.frame_response(
            rt.RPCResponse(
                int(example["status"]),
                int(example["result_len"]),
                int(example["request_id"]),
                int(example["ptp_timestamp"]),
            ),
            payload,
            slot_size=slot_size,
        )
        label = "wire.response_example.slot_hex"
    _require(
        built.hex() == example["slot_hex"],
        f"{label} is not what the implementation frames",
    )
    _require(
        len(built) == slot_size,
        f"a framed slot is {len(built)} bytes, not the declared {slot_size}",
    )
    return example["slot_hex"]


def _check_slot_rule(example: dict[str, Any]) -> None:
    slot_size = int(example["slot_size"])
    _require(
        rt.HEADER_SIZE + int(example["arg_len"]) <= slot_size,
        "wire.request_example does not satisfy wire.slot_rule",
    )
    _refused(
        lambda: rt.frame_request(
            rt.RPCHeader(int(example["function_id"]), slot_size, 0),
            bytes(slot_size),
            slot_size=slot_size,
        ),
        "a payload that does not satisfy wire.slot_rule was framed instead of "
        "refused",
    )


def check_payload(contract: dict[str, Any]) -> None:
    """Reconcile the type system, the schema, and the bit-packed encoding."""
    declared = contract["payload"]
    shipped = {member.name: int(member) for member in rt.PayloadType}
    _require(
        {name: int(value) for name, value in declared["type_ids"].items()} == shipped,
        f"payload.type_ids disagrees with the implementation's {shipped}",
    )
    _require(
        int(declared["descriptor_size_bytes"]) == rt.TYPE_DESCRIPTOR_SIZE,
        "payload.descriptor_size_bytes does not match the implementation",
    )
    _require(
        sum(_descriptor_field_bytes(field) for field in declared["descriptor_fields"])
        == rt.TYPE_DESCRIPTOR_SIZE,
        "payload.descriptor_fields do not add up to the declared descriptor size",
    )
    _check_widths(declared)
    _require(
        int(declared["max_arguments"]) == rt.MAX_ARGUMENTS,
        "payload.max_arguments does not match the implementation",
    )
    _require(
        int(declared["max_results"]) == rt.MAX_RESULTS,
        "payload.max_results does not match the implementation",
    )
    _require(
        int(declared["schema_size_bytes"]) == rt.SCHEMA_SIZE,
        "payload.schema_size_bytes does not match the implementation",
    )
    _check_schema_image(int(declared["schema_size_bytes"]))
    _check_bit_packing(declared)
    _check_no_delimiters()
    _check_schema_is_the_only_type_source()
    _refused(
        lambda: rt.TypeDescriptor(rt.PayloadType.UINT8, 1, 2),
        "a scalar descriptor declaring more than one element was accepted, "
        "contradicting payload.num_elements_semantics",
    )


def _descriptor_field_bytes(field: dict[str, Any]) -> int:
    sizes = {"uint8": 1, "uint8[3]": 3, "uint32": 4}
    declared = field["type"]
    _require(
        declared in sizes,
        f"payload.descriptor_fields names an unknown field type {declared!r}",
    )
    return sizes.get(declared, 0)


def _check_widths(declared: dict[str, Any]) -> None:
    for name, width in declared["scalar_widths_bytes"].items():
        _require(
            rt.TypeDescriptor(rt.PayloadType[name], int(width)).size_bytes
            == int(width),
            f"payload.scalar_widths_bytes[{name}] is not the width the "
            f"implementation enforces",
        )
    for name, width in declared["array_element_widths_bytes"].items():
        _require(
            rt.TypeDescriptor(rt.PayloadType[name], int(width), 1).size_bytes
            == int(width),
            f"payload.array_element_widths_bytes[{name}] is not the width the "
            f"implementation enforces",
        )


def _check_schema_image(schema_size: int) -> None:
    schema = rt.HandlerSchema(
        args=(rt.TypeDescriptor(rt.PayloadType.UINT8, 1),),
        results=(rt.TypeDescriptor(rt.PayloadType.UINT8, 1),),
    )
    image = schema.encode()
    _require(
        len(image) == schema_size,
        f"a schema image is {len(image)} bytes, not the declared {schema_size}",
    )
    _require(
        rt.HandlerSchema.decode(image) == schema,
        "a schema image does not survive a round trip",
    )
    _require(
        rt.HandlerSchema().encode() == bytes(schema_size),
        "an empty schema image is not all zero",
    )


def _check_bit_packing(declared: dict[str, Any]) -> None:
    example = declared["bit_packed_example"]
    bits = tuple(bool(bit) for bit in example["bits"])
    schema = rt.HandlerSchema(
        args=(rt.TypeDescriptor(rt.PayloadType.BIT_PACKED, 1, len(bits)),)
    )
    encoded = rt.encode_arguments(schema, (bits,))
    _require(
        encoded.hex() == example["bytes_hex"],
        f"payload.bit_packed_example encodes to {encoded.hex()} rather than the "
        f"declared {example['bytes_hex']}",
    )
    _require(
        rt.decode_arguments(schema, encoded) == (bits,),
        "a bit-packed payload does not survive a round trip",
    )
    _require(
        rt.TypeDescriptor(rt.PayloadType.BIT_PACKED, 1, len(bits)).size_bytes
        == -(-len(bits) // 8),
        "payload.bit_packed_size_rule does not hold for the declared example",
    )
    # LSB-first is the whole content of payload.bit_order: the first declared bit
    # is the least significant bit of the first byte.
    _require(
        rt.encode_arguments(
            rt.HandlerSchema(
                args=(rt.TypeDescriptor(rt.PayloadType.BIT_PACKED, 1, 8),)
            ),
            ((True, False, False, False, False, False, False, False),),
        )
        == b"\x01",
        "payload.bit_order is not least-significant-bit first within a byte",
    )
    _refused(
        lambda: rt.encode_arguments(
            rt.HandlerSchema(
                args=(rt.TypeDescriptor(rt.PayloadType.BIT_PACKED, 1, 5),)
            ),
            ((2, 0, 0, 0, 0),),
        ),
        "a bit outside zero and one was encoded into a bit-packed payload",
    )


def _check_no_delimiters() -> None:
    schema = rt.HandlerSchema(
        args=(
            rt.TypeDescriptor(rt.PayloadType.UINT8, 1),
            rt.TypeDescriptor(rt.PayloadType.INT32, 4),
        )
    )
    _require(
        rt.encode_arguments(schema, (7, 8)) == b"\x07\x08\x00\x00\x00",
        "a multi-argument payload is not the concatenation declared by "
        "payload.payload_rule",
    )


def _check_schema_is_the_only_type_source() -> None:
    schema = rt.HandlerSchema(args=(rt.TypeDescriptor(rt.PayloadType.UINT8, 1),))
    _refused(
        lambda: rt.decode_arguments(schema, b"\x01\x02"),
        "payload.typing_note is contradicted: a payload that disagrees with the "
        "schema was decoded anyway",
    )


def check_status(contract: dict[str, Any]) -> None:
    """Reconcile the status vocabulary and prove no path invents success."""
    declared = contract["status_codes"]
    _require(
        int(declared["success"]) == rt.STATUS_SUCCESS == 0,
        "status.success must be zero",
    )
    codes = {
        "unknown_function": rt.STATUS_UNKNOWN_FUNCTION,
        "invalid_arguments": rt.STATUS_INVALID_ARGUMENTS,
        "handler_raised": rt.STATUS_HANDLER_RAISED,
    }
    for name, shipped in codes.items():
        _require(
            int(declared["protocol_codes"][name]) == shipped,
            f"status.protocol_codes[{name}] is {declared['protocol_codes'][name]} "
            f"but the implementation reports {shipped}",
        )
        _require(
            shipped < 0,
            f"status.protocol_codes[{name}] must be a protocol-level negative code",
        )
    _require(
        declared["handler_error"] == "positive",
        "status.handler_error must be positive",
    )
    _require(
        declared["protocol_error"] == "negative",
        "status.protocol_error must be negative",
    )
    _require(bool(declared["echo_rule"]), "status.echo_rule must state the rule")
    _require(
        bool(declared["non_invention_rule"]),
        "status.non_invention_rule must state the rule",
    )
    _refused(
        lambda: rt.RealtimeHandlerError(0, "not a handler failure"),
        "a handler can claim a non-positive status",
    )
    _check_failures_report_and_echo()


def _probe_schema() -> rt.HandlerSchema:
    return rt.HandlerSchema(
        args=(rt.TypeDescriptor(rt.PayloadType.UINT8, 1),),
        results=(rt.TypeDescriptor(rt.PayloadType.UINT8, 1),),
    )


def _echo_probe(arguments: tuple[object, ...]) -> tuple[int]:
    return (int(arguments[0]),)


def _live_session(
    invoke: Callable[[tuple[object, ...]], object] = _echo_probe,
    *,
    name: str = "majority_decode",
) -> rt.DeviceCallSession:
    session = rt.FunctionTableService(
        [rt.FunctionEntry(name, _probe_schema(), invoke)]
    ).open_session("host_call")
    assert session is not None
    return session


def _check_failures_report_and_echo() -> None:
    def explode(_: tuple[object, ...]) -> tuple[int]:
        raise RuntimeError("the handler itself is broken")

    def refuse(_: tuple[object, ...]) -> tuple[int]:
        raise rt.RealtimeHandlerError(7, "the handler refused this round")

    for invoke, expected_status in ((explode, rt.STATUS_HANDLER_RAISED), (refuse, 7)):
        response, result = _live_session(invoke).dispatch(
            rt.RPCHeader(rt.function_id("majority_decode"), 1, 4242, 99), b"\x00"
        )
        _require(
            response.status == expected_status,
            f"a failing handler reported status {response.status} rather than "
            f"{expected_status}",
        )
        _require(
            (response.request_id, response.ptp_timestamp) == (4242, 99),
            "a failed call did not echo the request id and timestamp",
        )
        _require(result == b"", "a failed call returned a result payload")
    response, result = _live_session().dispatch(rt.RPCHeader(1, 0, 5, 11), b"")
    _require(
        response.status == rt.STATUS_UNKNOWN_FUNCTION,
        "an unowned function id did not report the unknown-function status",
    )
    _require(
        (response.request_id, response.ptp_timestamp, result) == (5, 11, b""),
        "an unknown function id did not echo the request with an empty result",
    )


def check_function_id(contract: dict[str, Any]) -> None:
    """Reconcile the hash parameters and both vector sets."""
    declared = contract["function_id"]
    _require(
        declared["algorithm"] == "fnv1a_32",
        "function_id.algorithm must name the implemented hash",
    )
    for name, parameter in (
        ("offset_basis", realtime_protocol._FNV_OFFSET_BASIS),
        ("prime", realtime_protocol._FNV_PRIME),
        ("mask", realtime_protocol._FNV_MASK),
    ):
        _require(
            int(declared[name]) == parameter,
            f"function_id.{name} is {declared[name]} but the implementation "
            f"computes with {parameter}",
        )
    for label in ("published_vectors", "implementation_vectors"):
        vectors = declared[label]
        _require(bool(vectors), f"function_id.{label} must not be empty")
        for name, digest in vectors.items():
            _require(
                rt.function_id(name) == int(digest),
                f"function_id.{label}[{name}] does not reproduce",
            )
    _require(
        bool(declared["published_vectors_source"]),
        "function_id.published_vectors_source must name where the vectors came from",
    )
    _refused(
        lambda: rt.function_id("   "),
        "a blank handler name was hashed instead of refused",
    )


def check_device_call(contract: dict[str, Any]) -> None:
    """Reconcile the device-call abstraction and prove its refusals."""
    declared = contract["device_call"]
    _require(
        tuple(declared["dispatch_modes"]) == rt.DISPATCH_MODES,
        f"device_call.dispatch_modes is {declared['dispatch_modes']} but the "
        f"implementation defines {list(rt.DISPATCH_MODES)}",
    )
    _require(
        tuple(declared["compiled_facts"]) == ("device_id", "function_id"),
        "device_call.compiled_facts must be the device id and the function id",
    )
    for name in (
        "runtime_selection",
        "unsupported_mode_result",
        "function_table_validity",
        "transport_ownership",
        "collision_rule",
        "session_key",
    ):
        _require(bool(declared[name]), f"device_call.{name} must state the rule")
    _check_runtime_channel_selection()
    _check_mode_refusal()
    _check_session_key()
    _check_table_validity()
    _check_collision_refusal()


def _check_runtime_channel_selection() -> None:
    # The channel name is not compiled in: a channel registered under an
    # arbitrary name reaches the same handler table, and no channel name appears
    # in that table.
    driver = rt.DeviceCallChannelDriver(
        {"shared-memory": rt.loopback_channel("shared-memory", _live_session())}
    )
    with rt.RealtimeSession(driver) as realtime:
        realtime.open_channel(0, "shared-memory")
        outcome = realtime.call(
            0,
            rt.function_id("majority_decode"),
            rt.encode_arguments(_probe_schema(), (9,)),
        )
    _require(
        outcome.succeeded
        and rt.decode_results(_probe_schema(), outcome.result) == (9,),
        "a channel registered under an arbitrary name did not reach the handler",
    )
    table = _live_session().function_table()
    _require(
        all(entry.name != "shared-memory" for entry in table.values()),
        "a channel name reached the function table, so it would be compiled in",
    )
    _refused(
        lambda: rt.DeviceCallChannelDriver().open("shared-memory"),
        "a channel name nobody registered was opened anyway",
    )


def _check_mode_refusal() -> None:
    service = rt.FunctionTableService(
        [rt.FunctionEntry("majority_decode", _probe_schema(), _echo_probe)],
        modes=("host_call",),
    )
    _require(
        service.open_session("graph_launch") is None,
        "a service served a dispatch mode it does not declare",
    )
    _require(
        service.modes == ("host_call",),
        "a service does not report the dispatch modes it serves",
    )
    _refused(
        lambda: service.open_session("unified"),  # type: ignore[arg-type]
        "an undeclared dispatch mode was accepted",
    )


def _check_session_key() -> None:
    driver = rt.DeviceCallChannelDriver(
        {"host-dispatch": rt.loopback_channel("host-dispatch", _live_session())}
    )
    with rt.RealtimeSession(driver) as realtime:
        realtime.open_channel(0, "host-dispatch")
        _require(realtime.device_ids == (0,), "a session is not keyed by device id")
        _refused(
            lambda: realtime.open_channel(0, "host-dispatch"),
            "a second channel for one device id was accepted",
        )
        _refused(
            lambda: realtime.call(1, 0),
            "a call reached a device id with no open channel",
        )
        realtime.open_channel(1, "host-dispatch")
        _require(
            realtime.device_ids == (0, 1),
            "a second device id did not get its own channel",
        )


def _check_table_validity() -> None:
    session = _live_session()
    _require(
        bool(session.function_table()),
        "a live session did not expose a function table",
    )
    session.stop()
    _refused(
        session.function_table,
        "a stopped session still handed out its function table",
    )


def _check_collision_refusal() -> None:
    entry = rt.FunctionEntry("majority_decode", _probe_schema(), _echo_probe)
    _refused(
        lambda: rt.FunctionTableService([entry, entry]),
        "two handlers sharing a function id were accepted",
    )


def check_hardware_boundary(contract: dict[str, Any]) -> None:
    """Require the contract and the package to disavow the hardware half."""
    declared = contract["hardware_boundary"]
    out_of_scope = tuple(declared["out_of_scope"])
    not_claimed = tuple(declared["not_claimed"])
    for term in REQUIRED_OUT_OF_SCOPE:
        _require(
            term in out_of_scope,
            f"hardware_boundary.out_of_scope must keep {term!r} out of scope",
        )
    for term in REQUIRED_NOT_CLAIMED:
        _require(
            term in not_claimed,
            f"hardware_boundary.not_claimed must keep {term!r} unclaimed",
        )
    _require(
        "not evidence" in declared["statement"],
        "hardware_boundary.statement must say the contract is not evidence of "
        "working hardware",
    )
    docstring = PACKAGE_INIT.read_text(encoding="utf-8")
    for term in REQUIRED_DISAVOWALS:
        _require(
            term in docstring,
            f"the package docstring must name {term} as out of scope",
        )


#: Keyed by the check's own name, because that is what the contract's `readers`
#: map names. `EXPECTED_READER_SECTIONS` is what keeps every section read.
CHECKERS: dict[str, Callable[[dict[str, Any]], None]] = {
    "check_wire": check_wire,
    "check_payload": check_payload,
    "check_status": check_status,
    "check_function_id": check_function_id,
    "check_device_call": check_device_call,
    "check_hardware_boundary": check_hardware_boundary,
}


def _witness_test_exists(reader: str) -> bool:
    _, _, node = reader.partition("::")
    if not WITNESS_TESTS.exists():
        return False
    tree = ast.parse(WITNESS_TESTS.read_text(encoding="utf-8"))
    return any(
        isinstance(item, ast.FunctionDef) and item.name == node for item in tree.body
    )


def check_readers(contract: dict[str, Any]) -> None:
    """Require every contract section to name a reader that really exists."""
    readers = contract["readers"]
    for section in EXPECTED_READER_SECTIONS:
        declared = readers.get(section)
        _require(bool(declared), f"readers.{section} must name at least one reader")
        for reader in declared or ():
            if reader.startswith("tool:"):
                name = reader.removeprefix("tool:")
                _require(
                    name in CHECKERS,
                    f"readers.{section} names tool:{name}, which this gate does not "
                    f"implement",
                )
            elif reader.startswith("test:"):
                _require(
                    _witness_test_exists(reader),
                    f"readers.{section} names {reader}, which does not exist in "
                    f"{WITNESS_TESTS.relative_to(ROOT).as_posix()}",
                )
            else:
                _require(
                    False,
                    f"readers.{section} names {reader!r}, which is neither a check "
                    f"in this gate nor a witness test",
                )


def main() -> int:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    _require(
        set(contract) == EXPECTED_KEYS,
        f"the contract declares {sorted(set(contract) ^ EXPECTED_KEYS)} outside its "
        f"schema",
    )
    _require(
        contract.get("schema") == EXPECTED_SCHEMA,
        f"the contract schema must be {EXPECTED_SCHEMA!r}",
    )
    _require(
        contract.get("schema_version") == EXPECTED_SCHEMA_VERSION,
        f"the contract schema version must be {EXPECTED_SCHEMA_VERSION!r}",
    )
    _require(
        contract.get("status") == EXPECTED_STATUS,
        f"the contract status must be {EXPECTED_STATUS!r}",
    )
    _require(bool(contract.get("owner")), "the contract must name an owner")
    _require(
        not contract["approval"].get("public_api_change"),
        "the realtime contract must not change the public API",
    )
    _require(
        not contract["approval"].get("capability_claim"),
        "the realtime contract must not carry a capability claim",
    )
    # A contract that cites a record which does not exist is a claim with no
    # approver; the path is checked rather than trusted.
    record = contract["approval"].get("record", "")
    _require(
        isinstance(record, str) and (ROOT / record).is_file(),
        f"the contract names {record!r} as its approval record, which is not a file",
    )
    _require(
        bool(contract["source"]["documents"]),
        "the contract must name the documents it was read from",
    )
    for check in CHECKERS.values():
        check(contract)
    check_readers(contract)
    if _FAILURES:
        for failure in _FAILURES:
            print(f"realtime messaging contract: {failure}")
        return 1
    print(f"realtime messaging contract passed: {_CHECKS} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
