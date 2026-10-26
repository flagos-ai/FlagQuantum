"""Strict source encoding for the native physical ProgramRecord profile.

Containing execution plans remain outside Core. This module only owns the
actual Core graph/context and reuses the established Core scalar JSON codec.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import fields
from typing import Any

from ....errors import SerializationError
from ...geometry import cartesian_pair, finite_real
from .. import _decode_value, _encode_value
from .model import BlockRecord, FunctionRecord, Operation, ProgramRecord
from .physical import (
    IsingContext,
    IsingExchange,
    RydbergContext,
    RydbergPresence,
    RydbergPulse,
    RydbergTransport,
    check_ising_exchange,
    control_scalar,
    identifier,
)
from .verifier import verify_program_record


def _keys(value: object, expected: set[str], field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise SerializationError(f"{field} must have exactly fields {sorted(expected)}")
    return value


def _array(value: object, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise SerializationError(f"{field} must be a JSON array")
    return value


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _checked(record: ProgramRecord) -> None:
    if record.physical_context is None:
        raise SerializationError("a native source requires a physical context")
    diagnostics = verify_program_record(record)
    if diagnostics:
        raise SerializationError(
            "; ".join(f"{item.code}: {item.message}" for item in diagnostics)
        )


def _context_payload(context: RydbergContext | IsingContext) -> dict[str, Any]:
    if isinstance(context, IsingContext):
        return {
            **{
                field.name: _encode_value(getattr(context, field.name))
                for field in fields(context)
            },
            "kind": "ising_braid",
        }
    payload = {
        field.name: _encode_value(getattr(context, field.name))
        for field in fields(context)
    }
    payload.update(
        {
            "kind": "rydberg",
            "sites": [
                [name, list(cartesian_pair(point, "sites"))]
                for name, point in context.sites
            ],
            "c6": finite_real(context.c6, "c6"),
            "loss_rate": finite_real(context.loss_rate, "loss_rate"),
            "minimum_separation": finite_real(
                context.minimum_separation, "minimum_separation"
            ),
        }
    )
    return payload


def _control_payload(
    data: RydbergTransport | RydbergPulse | RydbergPresence | IsingExchange,
) -> dict[str, Any]:
    if isinstance(data, RydbergPresence):
        return {
            "kind": "presence",
            "atom": data.atom,
            "at": finite_real(data.at, "at"),
            "efficiency": finite_real(data.efficiency, "efficiency"),
        }
    if isinstance(data, IsingExchange):
        return {"kind": "exchange", "left": data.left, "clockwise": data.clockwise}
    payload = {
        field.name: _encode_value(getattr(data, field.name)) for field in fields(data)
    }
    payload["start"] = finite_real(data.start, "start")
    payload["duration"] = finite_real(data.duration, "duration")
    payload["kind"] = "transport" if isinstance(data, RydbergTransport) else "pulse"
    if isinstance(data, RydbergPulse):
        for field in ("amplitude", "detuning", "phase"):
            payload[field] = _encode_value(control_scalar(getattr(data, field), field))
    return payload


def native_program_to_dict(record: ProgramRecord) -> dict[str, Any]:
    """Encode a validated native source; never replace it by a circuit payload."""
    _checked(record)
    assert record.physical_context is not None
    version = "1.1" if isinstance(record.physical_context, IsingContext) else "1.0"
    functions = []
    for function in record.functions:
        blocks = []
        for block in function.blocks:
            operations: list[dict[str, Any]] = []
            for operation in block.operations:
                physical_data: dict[str, Any] | None
                if operation.opcode in ("ising.dephase_pair", "ising.measure_pair"):
                    physical_data = None
                    version = (
                        "1.3" if operation.opcode == "ising.measure_pair" else "1.2"
                    )
                else:
                    assert operation.physical_data is not None
                    physical_data = _control_payload(operation.physical_data)
                    if isinstance(operation.physical_data, RydbergPresence):
                        version = "1.4"
                operations.append(
                    {
                        "opcode": operation.opcode,
                        "operands": [],
                        "results": [],
                        "parameters": [],
                        "literal": None,
                        "condition_clauses": [],
                        "physical_data": physical_data,
                    }
                )
            blocks.append(
                {
                    "name": block.name,
                    "arguments": [],
                    "terminator": block.terminator,
                    "edges": [],
                    "operations": operations,
                }
            )
        functions.append(
            {"name": function.name, "entry": function.entry, "blocks": blocks}
        )
    return {
        "schema": "flagquantum.program_ir",
        "version": version,
        "functions": functions,
        "physical_context": _context_payload(record.physical_context),
    }


def _context_from_dict(value: object) -> RydbergContext | IsingContext:
    if isinstance(value, Mapping) and value.get("kind") == "ising_braid":
        payload = _keys(
            value,
            {field.name for field in fields(IsingContext)} | {"kind"},
            "physical_context",
        )
        return IsingContext(
            tuple(_array(payload["anyon_ids"], "anyon_ids")),
            payload["total_charge"],
            payload["fusion_tree"],
            payload["gauge"],
            payload["initial_channel"],
        )
    payload = _keys(
        value,
        {field.name for field in fields(RydbergContext)} | {"kind"},
        "physical_context",
    )
    if payload["kind"] != "rydberg":
        raise SerializationError("unknown physical_context kind")
    atoms = []
    for value in _array(payload["atoms"], "atoms"):
        row = _array(value, "atom row")
        if len(row) != 2:
            raise SerializationError("atom row must be an atom/site pair")
        atoms.append((row[0], row[1]))
    sites = []
    for value in _array(payload["sites"], "sites"):
        row = _array(value, "site row")
        if len(row) != 2:
            raise SerializationError("site row must contain an ID and Cartesian pair")
        sites.append(
            (row[0], cartesian_pair(_array(row[1], "coordinates"), "coordinates"))
        )
    return RydbergContext(
        tuple(atoms),
        tuple(sites),
        payload["length_unit"],
        payload["time_unit"],
        finite_real(payload["c6"], "c6"),
        finite_real(payload["loss_rate"], "loss_rate"),
        finite_real(payload["minimum_separation"], "minimum_separation"),
        payload["coordinate_frame"],
        tuple(_array(payload["transport_channels"], "transport_channels")),
        tuple(_array(payload["drive_channels"], "drive_channels")),
        payload["geometry_sampling"],
    )


def _control_from_dict(
    value: object,
) -> RydbergTransport | RydbergPulse | RydbergPresence | IsingExchange:
    if not isinstance(value, Mapping):
        raise SerializationError("physical_data must be an object")
    kind = value.get("kind")
    if kind == "presence":
        payload = _keys(value, {"kind", "atom", "at", "efficiency"}, "physical_data")
        return RydbergPresence(
            identifier(payload["atom"], "atom"),
            finite_real(payload["at"], "at"),
            finite_real(payload["efficiency"], "efficiency"),
        )
    if kind == "exchange":
        payload = _keys(value, {"kind", "left", "clockwise"}, "physical_data")
        data = IsingExchange(payload["left"], payload["clockwise"])
        check_ising_exchange(data)
        return data
    if kind not in ("transport", "pulse"):
        raise SerializationError("unknown physical_data kind")
    record_type = RydbergTransport if kind == "transport" else RydbergPulse
    payload = _keys(
        value, {field.name for field in fields(record_type)} | {"kind"}, "physical_data"
    )
    atoms = tuple(_array(payload["atoms"], "control atoms"))
    start, duration = finite_real(payload["start"], "start"), finite_real(
        payload["duration"], "duration"
    )
    if kind == "transport":
        return RydbergTransport(
            atoms,
            tuple(_array(payload["source"], "source")),
            tuple(_array(payload["destination"], "destination")),
            start,
            duration,
            payload["channel"],
        )
    for field in ("amplitude", "detuning", "phase"):
        _control_encoding(payload[field])
    amplitude, detuning, phase = (
        control_scalar(_decode_value(payload[field]), field)
        for field in ("amplitude", "detuning", "phase")
    )
    return RydbergPulse(
        atoms, start, duration, amplitude, detuning, phase, payload["channel"]
    )


def _control_encoding(value: object) -> None:
    """Refuse disallowed scalar leaves before the general codec can allocate.

    The existing codec still constructs and validates the expression. This is
    only the physical profile's narrower input admission, not another AST or
    expression evaluator.
    """
    if type(value) in (int, float):
        finite_real(value, "control scalar")
        return
    if not isinstance(value, Mapping):
        raise SerializationError(
            "native control scalar must be real or a Core parameter/expression"
        )
    if "$parameter" in value:
        payload = _keys(value, {"$parameter"}, "parameter")
        if not isinstance(payload["$parameter"], str):
            raise SerializationError("native parameter names must be strings")
    elif "$expression" in value:
        payload = _keys(value, {"$expression"}, "expression")
        expression = _keys(payload["$expression"], {"op", "args"}, "expression node")
        if not isinstance(expression["op"], str):
            raise SerializationError("native expression operators must be strings")
        for argument in _array(expression["args"], "expression args"):
            _control_encoding(argument)
    else:
        raise SerializationError(
            "tensor, complex and container leaves are not native control scalars"
        )


def _block_from_dict(value: object) -> BlockRecord:
    payload = _keys(value, {field.name for field in fields(BlockRecord)}, "block")
    if payload["arguments"] != [] or payload["edges"] != []:
        raise SerializationError(
            "physical blocks do not admit SSA arguments or control-flow edges"
        )
    operations = []
    for value in _array(payload["operations"], "operations"):
        operation = _keys(
            value, {field.name for field in fields(Operation)}, "operation"
        )
        if (
            any(
                operation[field] != []
                for field in ("operands", "results", "parameters", "condition_clauses")
            )
            or operation["literal"] is not None
        ):
            raise SerializationError(
                "physical operations cannot carry circuit/SSA data"
            )
        physical_data: (
            RydbergTransport | RydbergPulse | RydbergPresence | IsingExchange | None
        )
        if operation["opcode"] in ("ising.dephase_pair", "ising.measure_pair"):
            if operation["physical_data"] is not None:
                raise SerializationError(
                    f"{operation['opcode']} physical_data must be null"
                )
            physical_data = None
        else:
            physical_data = _control_from_dict(operation["physical_data"])
        operations.append(
            Operation(
                operation["opcode"],
                physical_data=physical_data,
            )
        )
    return BlockRecord(
        payload["name"], terminator=payload["terminator"], operations=tuple(operations)
    )


def native_program_from_dict(value: Mapping[str, Any]) -> ProgramRecord:
    """Decode strictly without changing legacy scalar or circuit acceptance."""
    try:
        payload = _keys(
            value, {"schema", "version", "functions", "physical_context"}, "program"
        )
        if payload["schema"] != "flagquantum.program_ir" or payload["version"] not in (
            "1.0",
            "1.1",
            "1.2",
            "1.3",
            "1.4",
        ):
            raise SerializationError("unsupported native Program IR schema/version")
        context = _context_from_dict(payload["physical_context"])
        functions = []
        for value in _array(payload["functions"], "functions"):
            function = _keys(
                value, {field.name for field in fields(FunctionRecord)}, "function"
            )
            functions.append(
                FunctionRecord(
                    function["name"],
                    tuple(
                        _block_from_dict(block)
                        for block in _array(function["blocks"], "blocks")
                    ),
                    entry=function["entry"],
                )
            )
        record = ProgramRecord(tuple(functions), physical_context=context)
        canonical = native_program_to_dict(record)
        if _json(canonical) != _json(payload):
            raise SerializationError(
                "native source must round-trip through the canonical Core scalar encoding"
            )
        return record
    except (TypeError, ValueError, KeyError, IndexError, RecursionError) as exc:
        if isinstance(exc, SerializationError):
            raise
        raise SerializationError(f"invalid native Program IR source: {exc}") from exc
