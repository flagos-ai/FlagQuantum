"""Approved sampled source construction, with numerical admission still closed."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest

import flagquantum as fq
from flagquantum.core.ir.program.model import Operation
from flagquantum.core.ir.program.physical import IsingContext, IsingExchange
from flagquantum.core.ir.program.physical_codec import (
    native_program_from_dict,
    native_program_to_dict,
)
from flagquantum.core.ir.program.verifier import verify_program_record
from flagquantum.core.ising import IsingBraidProgram
from flagquantum.errors import CapabilityError, SerializationError
from flagquantum.runtime.execution_plan_contract import (
    ExecutionPlanContractError,
    canonical_hash,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("initial", ["1", "psi"])
def test_sampled_boundary_roundtrips_as_one_null_operation(
    initial: Literal["1", "psi"],
) -> None:
    source = IsingBraidProgram(initial_channel=initial).exchange(1).exchange(0)
    before = source.to_ir()
    assert source.measure_pair() is source
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
            assert order[:2] == ["c", "a"]
            assert operation == Operation("ising.measure_pair")
            assert operation.physical_data is None
            assert operation.literal is None
            assert all(
                type(v) is tuple and not v
                for v in (
                    operation.operands,
                    operation.results,
                    operation.parameters,
                    operation.condition_clauses,
                )
            )
    payload = native_program_to_dict(record)
    assert payload["version"] == "1.3"
    assert native_program_from_dict(json.loads(json.dumps(payload))) == record
    assert native_program_to_dict(before)["version"] == "1.1"
    assert (
        record
        != IsingBraidProgram(initial_channel=initial)
        .measure_pair()
        .exchange(1)
        .exchange(0)
        .exchange(2, clockwise=True)
        .to_ir()
    )


@pytest.mark.parametrize("first", ["measure_pair", "dephase_pair"])
@pytest.mark.parametrize("second", ["measure_pair", "dephase_pair"])
def test_all_second_observations_fail_atomically(first: str, second: str) -> None:
    source = IsingBraidProgram().exchange(1)
    getattr(source, first)()
    before = source.to_ir()
    payload, identity = native_program_to_dict(before), hash(before)
    with pytest.raises(ValueError):
        getattr(source, second)()
    assert source.to_ir() == before and hash(before) == identity
    assert native_program_to_dict(before) == payload


@pytest.mark.parametrize(
    "args,kwargs",
    [((0,), {}), ((), {"seed": 1}), ((), {"outcome": "psi"}), ((), {"kraus": []})],
)
def test_measure_pair_accepts_no_arguments_before_mutating(
    args: tuple[object, ...], kwargs: dict[str, object]
) -> None:
    source = IsingBraidProgram().exchange(1)
    before = source.to_ir()
    method: Callable[..., IsingBraidProgram] = source.measure_pair
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
        ("literal", 0),
        ("physical_data", {}),
        ("physical_data", False),
        ("physical_data", "null"),
        ("physical_data", {"kind": "exchange", "left": 0, "clockwise": False}),
        ("unreviewed", True),
    ],
)
def test_sampled_json_refuses_extra_or_non_null_data(field: str, value: object) -> None:
    payload = native_program_to_dict(IsingBraidProgram().measure_pair().to_ir())
    payload["functions"][0]["blocks"][0]["operations"][0][field] = value
    with pytest.raises(SerializationError):
        native_program_from_dict(payload)


@pytest.mark.parametrize("version", ["1.0", "1.1", "1.2", "2.0", 1.3, None])
def test_sampled_source_requires_exact_version(version: object) -> None:
    payload = native_program_to_dict(IsingBraidProgram().measure_pair().to_ir())
    payload["version"] = version
    with pytest.raises(SerializationError):
        native_program_from_dict(payload)


@pytest.mark.parametrize("unread", [False, True])
def test_old_sources_cannot_be_relabelled_as_sampled(unread: bool) -> None:
    source = IsingBraidProgram().exchange(1)
    if unread:
        source.dephase_pair()
    payload = native_program_to_dict(source.to_ir())
    payload["version"] = "1.3"
    with pytest.raises(SerializationError):
        native_program_from_dict(payload)


@pytest.mark.parametrize(
    "field", ["operands", "results", "parameters", "condition_clauses"]
)
def test_empty_mutable_typed_fields_are_not_valid_source(field: str) -> None:
    constructor: Callable[..., Operation] = Operation
    operation = constructor("ising.measure_pair", **{field: []})
    record = IsingBraidProgram().to_ir()
    function = record.functions[0]
    broken = replace(
        record,
        functions=(
            replace(
                function, blocks=(replace(function.blocks[0], operations=(operation,)),)
            ),
        ),
    )
    assert verify_program_record(broken)
    with pytest.raises(SerializationError):
        native_program_to_dict(broken)


@pytest.mark.parametrize(
    "operations",
    [
        (Operation("ising.measure_pair"), Operation("ising.measure_pair")),
        (Operation("ising.measure_pair"), Operation("ising.dephase_pair")),
        (Operation("ising.dephase_pair"), Operation("ising.measure_pair")),
        (Operation("ising.measure_pair", physical_data=IsingExchange(0, False)),),
    ],
)
def test_typed_sources_cannot_bypass_observation_rules(
    operations: tuple[Operation, ...],
) -> None:
    record = IsingBraidProgram().to_ir()
    function = record.functions[0]
    broken = replace(
        record,
        functions=(
            replace(
                function, blocks=(replace(function.blocks[0], operations=operations),)
            ),
        ),
    )
    assert verify_program_record(broken)
    with pytest.raises(SerializationError):
        native_program_to_dict(broken)


def test_sampled_operation_cannot_borrow_the_rydberg_profile() -> None:
    captures = json.loads(
        (
            Path(__file__).parents[1] / "fixtures/ising_braid_legacy_plans.json"
        ).read_text()
    )
    source = captures["plans"]["native_rydberg"]["program"]
    source["version"] = "1.3"
    source["functions"][0]["blocks"][0]["operations"] = [
        {
            "opcode": "ising.measure_pair",
            "operands": [],
            "results": [],
            "parameters": [],
            "literal": None,
            "condition_clauses": [],
            "physical_data": None,
        }
    ]
    with pytest.raises(SerializationError):
        native_program_from_dict(source)


@pytest.mark.parametrize("plan_version", ["1.1", "1.2", "1.3"])
def test_runtime_refuses_the_real_new_source_before_numerical_execution(
    plan_version: str,
) -> None:
    source = IsingBraidProgram().exchange(1).measure_pair().exchange(0)
    options = fq.ExecutionOptions(shots=1, seed=17)
    with pytest.raises(CapabilityError, match="source 1.1 or 1.2"):
        fq.plan(source, options=options)
    with pytest.raises(CapabilityError, match="source 1.1 or 1.2"):
        fq.run(source, options=options)
    old = fq.plan(IsingBraidProgram().exchange(1).dephase_pair()).to_dict()
    old["program"] = native_program_to_dict(source.to_ir())
    fingerprints = old["fingerprints"]
    assert isinstance(fingerprints, dict)
    fingerprints["program"] = canonical_hash(old["program"])
    old["identity"] = canonical_hash({k: v for k, v in old.items() if k != "identity"})
    old["version"] = plan_version
    old["identity"] = canonical_hash({k: v for k, v in old.items() if k != "identity"})
    with pytest.raises(ExecutionPlanContractError):
        fq.ExecutionPlan.from_dict(old)
