"""Approved unread charge source semantics, independent of numerical admission."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest

from flagquantum.core.ir.program.model import Operation, ProgramRecord
from flagquantum.core.ir.program.physical import IsingContext, IsingExchange
from flagquantum.core.ir.program.physical_codec import (
    native_program_from_dict,
    native_program_to_dict,
)
from flagquantum.core.ir.program.verifier import verify_program_record
from flagquantum.core.ising import IsingBraidProgram
from flagquantum.errors import SerializationError

pytestmark = pytest.mark.unit


def _operations(
    record: ProgramRecord, operations: tuple[Operation, ...]
) -> ProgramRecord:
    function = record.functions[0]
    return replace(
        record,
        functions=(
            replace(
                function, blocks=(replace(function.blocks[0], operations=operations),)
            ),
        ),
    )


@pytest.mark.parametrize("initial", ["1", "psi"])
def test_current_pair_observation_is_an_ordered_null_data_source_operation(
    initial: Literal["1", "psi"],
) -> None:
    source = IsingBraidProgram(initial_channel=initial).exchange(1).exchange(0)
    before = source.to_ir()
    assert source.dephase_pair() is source
    source.exchange(2, clockwise=True)
    record = source.to_ir()
    assert verify_program_record(record) == ()
    assert record.physical_context == before.physical_context
    context = record.physical_context
    assert isinstance(context, IsingContext)
    order = list(context.anyon_ids)
    for operation in record.functions[0].blocks[0].operations:
        data = operation.physical_data
        if isinstance(data, IsingExchange):
            order[data.left], order[data.left + 1] = (
                order[data.left + 1],
                order[data.left],
            )
        else:
            assert order[:2] == ["c", "a"]  # Current positions, not original a/b.
            assert operation == Operation("ising.dephase_pair")
    payload = native_program_to_dict(record)
    assert payload["version"] == "1.2"
    assert native_program_from_dict(json.loads(json.dumps(payload))) == record
    assert (
        payload["functions"][0]["blocks"][0]["operations"][2]["physical_data"] is None
    )
    assert native_program_to_dict(before)["version"] == "1.1"


def test_duplicate_refusal_is_atomic_and_snapshots_keep_observation_order() -> None:
    source = IsingBraidProgram().exchange(1).dephase_pair()
    record = source.to_ir()
    payload = native_program_to_dict(record)
    identity = hash(record)
    with pytest.raises(ValueError, match="one unread"):
        source.dephase_pair()
    assert source.to_ir() == record
    source.exchange(1, clockwise=True)
    assert source.to_ir() != record and hash(record) == identity
    assert native_program_to_dict(record) == payload
    assert record != IsingBraidProgram().dephase_pair().exchange(1).to_ir()
    duplicate = _operations(record, (Operation("ising.dephase_pair"),) * 2)
    assert verify_program_record(duplicate)
    with pytest.raises(SerializationError):
        native_program_to_dict(duplicate)


@pytest.mark.parametrize(
    "args,kwargs",
    [((0,), {}), ((), {"outcome": "psi"}), ((), {"seed": 1}), ((), {"kraus": []})],
)
def test_observation_accepts_no_arguments_before_mutation(
    args: tuple[object, ...], kwargs: dict[str, object]
) -> None:
    source = IsingBraidProgram().exchange(1)
    before = source.to_ir()
    method: Callable[..., IsingBraidProgram] = source.dephase_pair
    with pytest.raises(TypeError):
        method(*args, **kwargs)
    assert source.to_ir() == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("operands", ["q"]),
        ("results", ["q"]),
        ("parameters", ["theta"]),
        ("condition_clauses", [[True]]),
        ("literal", 1.0),
        ("physical_data", {}),
        ("physical_data", {"kind": "exchange", "left": 0, "clockwise": False}),
        ("physical_data", "null"),
        ("unreviewed", True),
    ],
)
def test_strict_observation_payload_refuses_extra_data(
    field: str, value: object
) -> None:
    payload = native_program_to_dict(IsingBraidProgram().dephase_pair().to_ir())
    payload["functions"][0]["blocks"][0]["operations"][0][field] = value
    with pytest.raises(SerializationError):
        native_program_from_dict(payload)


@pytest.mark.parametrize("version", ["1.0", "1.1", 1.2, None])
def test_observation_requires_its_exact_source_version(version: object) -> None:
    payload = native_program_to_dict(IsingBraidProgram().dephase_pair().to_ir())
    payload["version"] = version
    with pytest.raises(SerializationError):
        native_program_from_dict(payload)


def test_null_is_not_admitted_for_exchange_and_other_profiles() -> None:
    exchange = native_program_to_dict(IsingBraidProgram().exchange(1).to_ir())
    exchange["functions"][0]["blocks"][0]["operations"][0]["physical_data"] = None
    with pytest.raises(SerializationError):
        native_program_from_dict(exchange)
    capture = json.loads(
        (
            Path(__file__).parents[1] / "fixtures/ising_braid_legacy_plans.json"
        ).read_text()
    )
    rydberg = capture["plans"]["native_rydberg"]["program"]
    rydberg["version"] = "1.2"
    with pytest.raises(SerializationError):
        native_program_from_dict(rydberg)
    rydberg["version"] = "1.0"
    operation = rydberg["functions"][0]["blocks"][0]["operations"][0]
    operation["physical_data"] = None
    with pytest.raises(SerializationError):
        native_program_from_dict(rydberg)
    operation["opcode"] = "ising.dephase_pair"
    with pytest.raises(SerializationError):
        native_program_from_dict(rydberg)


def test_typed_source_rejects_non_null_observation_data() -> None:
    record = IsingBraidProgram().to_ir()
    broken = _operations(
        record,
        (Operation("ising.dephase_pair", physical_data=IsingExchange(0, False)),),
    )
    assert verify_program_record(broken)
    with pytest.raises(SerializationError):
        native_program_to_dict(broken)


@pytest.mark.parametrize(
    "field", ["operands", "results", "parameters", "condition_clauses"]
)
def test_empty_mutable_fields_are_not_immutable_source_records(field: str) -> None:
    constructor: Callable[..., Operation] = Operation
    operation = constructor("ising.dephase_pair", **{field: []})
    broken = _operations(IsingBraidProgram().to_ir(), (operation,))
    assert verify_program_record(broken)
    with pytest.raises(SerializationError):
        native_program_to_dict(broken)
