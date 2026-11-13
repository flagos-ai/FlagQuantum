"""Compare single-control AddK and XorK lifetime caches to unchanged source."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, get_args

import pytest
from sympy import Integer

from flagquantum.compiler._component_trace.arithmetic_call import (
    ArithmeticCall,
    ArithmeticEncoding,
)
from flagquantum.compiler._component_trace.resource_generalization import (
    ResourceCall,
)
from flagquantum.compiler._component_trace.resource_width import (
    query_resource_logical_width,
)
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.compiler.registers.arithmetic.arithmetic_declarations import (
    arithmetic_call_graph,
)
from flagquantum.compiler.registers.bits.controlled_constant_width import (
    controlled_constant_width,
)
from flagquantum.drawer.call_graph import (
    GateAggregation,
    export_resource_call_graph_dot,
)
from tests.unit.test_addition_resource_width import ENCODINGS
from tests.unit.test_constant_addition_resource_width import (
    CASES,
    k,
)
from tests.unit.test_constant_addition_resource_width import (
    source_key as plain_source_key,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
Operation = Literal["controlled_constant_add", "controlled_xor_constant"]


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    reference = Path(__file__).resolve().parents[3] / "ftqc-reference-env"
    with (reference.parent / "FlagQuantum-Qualtran-Kickmix-移植进度.csv").open(
        encoding="utf-8-sig"
    ) as stream:
        blobs = {row["源文件"]: row["源对象"] for row in csv.DictReader(stream)}
    for name in (
        "qualtran/resource_counting/_qubit_counts.py",
        "qualtran/bloqs/arithmetic/addition.py",
        "qualtran/bloqs/arithmetic/bitwise.py",
        "qualtran/bloqs/bookkeeping/always.py",
        "qualtran/bloqs/bookkeeping/allocate.py",
        "qualtran/bloqs/bookkeeping/free.py",
        "qualtran/_infra/controlled.py",
        "qualtran/bloqs/mcmt/controlled_via_and.py",
    ):
        raw = (reference / name).read_bytes()
        assert (
            hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blobs[name]
        )


def source_for(call: ArithmeticCall) -> Any:
    from qualtran import CtrlSpec, QInt, QMontgomeryUInt, QUInt
    from qualtran.bloqs.arithmetic import AddK
    from qualtran.bloqs.arithmetic.bitwise import XorK

    dtype = {"unsigned": QUInt, "twos_complement": QInt, "montgomery": QMontgomeryUInt}[
        call.encodings[0]
    ](call.widths[0])
    source = (AddK if call.operation == "controlled_constant_add" else XorK)(
        dtype, call.constant
    )
    source = source.controlled(CtrlSpec(cvs=call.control_value))
    return source.adjoint() if call.adjoint else source


def source_key(source: Any, root: ArithmeticCall | None = None) -> ResourceCall:
    from qualtran import Controlled
    from qualtran.bloqs.mcmt import ControlledViaAnd

    if root is not None and source == source_for(root):
        return root
    if isinstance(source, (Controlled, ControlledViaAnd)):
        call = plain_source_key(source.subbloq)
        assert isinstance(call, ArithmeticCall)
        assert call.operation in {"constant_add", "xor_constant"}
        return replace(
            call,
            operation=(
                "controlled_constant_add"
                if call.operation == "constant_add"
                else "controlled_xor_constant"
            ),
            control_value=int(source.ctrl_spec.cvs[0]),
        )
    return plain_source_key(source)


def compare(call: ArithmeticCall, seed: Any = None) -> dict[ResourceCall, SymbolicInt]:
    from qualtran.resource_counting import QubitCount, get_cost_value

    expected: dict[Any, SymbolicInt] = {} if seed is None else {seed: 20}
    actual: dict[ResourceCall, SymbolicInt] = (
        {} if seed is None else {source_key(seed, call): 20}
    )
    width = get_cost_value(source_for(call), QubitCount(), costs_cache=expected)
    assert query_resource_logical_width(call, cache=actual) == width
    normalized = {source_key(key, call): value for key, value in expected.items()}
    assert actual == normalized
    assert list(actual) == list(normalized)
    return actual


@pytest.mark.parametrize("width,value", CASES)
@pytest.mark.parametrize("encoding", ENCODINGS)
@pytest.mark.parametrize("cv", [0, 1])
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize(
    "operation", ["controlled_constant_add", "controlled_xor_constant"]
)
def test_full_cache_and_order(
    width: SymbolicInt,
    value: SymbolicInt,
    encoding: ArithmeticEncoding,
    cv: int,
    adjoint: bool,
    operation: Operation,
) -> None:
    from qualtran.resource_counting import QubitCount, get_cost_value

    call = ArithmeticCall(
        operation,
        (width,),
        (encoding,),
        constant=value,
        control_value=cv,
        adjoint=adjoint,
    )
    if type(width) is int and type(value) is int:
        invalid = (
            not -(1 << (width - 1)) <= value < (1 << (width - 1))
            if encoding == "twos_complement"
            else operation == "controlled_xor_constant"
            and not 0 <= value < (1 << width)
        )
        if invalid:
            source_cache: dict[Any, SymbolicInt] = {}
            cache: dict[ResourceCall, SymbolicInt] = {}
            with pytest.raises(RuntimeError):
                get_cost_value(source_for(call), QubitCount(), costs_cache=source_cache)
            with pytest.raises(ValueError, match="range"):
                query_resource_logical_width(call, cache=cache)
            assert call not in cache
            assert cache == {
                source_key(key, call): value for key, value in source_cache.items()
            }
            return
    compare(call)


@pytest.mark.parametrize(
    "operation", ["controlled_constant_add", "controlled_xor_constant"]
)
@pytest.mark.parametrize("cv", [0, 1])
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("seed_name", ["cx", "x", "split", "join", "positive"])
def test_caller_words_and_controls_survive_child_peaks(
    operation: Operation, cv: int, adjoint: bool, seed_name: str
) -> None:
    from qualtran import QUInt
    from qualtran.bloqs.basic_gates import CNOT, XGate
    from qualtran.bloqs.bookkeeping import Join, Split

    call = ArithmeticCall(
        operation, (3,), constant=2, control_value=cv, adjoint=adjoint
    )
    positive = source_for(replace(call, control_value=1))
    seeds = {
        "cx": CNOT(),
        "x": XGate(),
        "split": Split(QUInt(3)),
        "join": Join(QUInt(3)),
        "positive": positive,
    }
    result = compare(call, seeds[seed_name])[call]
    if seed_name == "x" and cv == 1:
        assert result == (9 if operation == "controlled_constant_add" else 4)
    else:
        assert result >= 20


@pytest.mark.parametrize("cv", [0, 1])
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize(
    "seed_name", ["load", "allocate", "free", "add", "always", "and", "erase"]
)
def test_controlled_add_allocation_and_carry_peaks(
    cv: int, adjoint: bool, seed_name: str
) -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add
    from qualtran.bloqs.bookkeeping import Allocate, Always, Free
    from qualtran.bloqs.mcmt import And

    add = Add(QUInt(3), QUInt(3))
    add = add.adjoint() if adjoint else add
    load = source_for(ArithmeticCall("controlled_xor_constant", (3,), constant=2))
    seeds = {
        "load": load,
        "allocate": Allocate(QUInt(3)),
        "free": Free(QUInt(3)),
        "add": add,
        "always": Always(add),
        "and": And(),
        "erase": And().adjoint(),
    }
    call = ArithmeticCall(
        "controlled_constant_add", (3,), constant=2, control_value=cv, adjoint=adjoint
    )
    assert compare(call, seeds[seed_name])[call] >= 20


@pytest.mark.parametrize("width,value", [(1, 0), (3, 2), (3, k), (Integer(3), 2)])
@pytest.mark.parametrize(
    "operation", ["controlled_constant_add", "controlled_xor_constant"]
)
@pytest.mark.parametrize("cv", [0, 1])
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("depth", [0, 1, None])
@pytest.mark.parametrize("aggregation", get_args(GateAggregation))
def test_automatic_dot_matches_source_widths(
    width: SymbolicInt,
    value: SymbolicInt,
    operation: Operation,
    cv: int,
    adjoint: bool,
    depth: int | None,
    aggregation: GateAggregation,
) -> None:
    from qualtran.resource_counting import QubitCount, get_cost_value

    call = ArithmeticCall(
        operation, (width,), constant=value, control_value=cv, adjoint=adjoint
    )
    cache: dict[Any, SymbolicInt] = {}
    get_cost_value(source_for(call), QubitCount(), costs_cache=cache)
    widths_by_call = {source_key(key, call): value for key, value in cache.items()}
    declarations = arithmetic_call_graph(call)
    assert export_resource_call_graph_dot(
        call, declared_calls=declarations, max_depth=depth, aggregation=aggregation
    ) == export_resource_call_graph_dot(
        call,
        declared_calls=declarations,
        logical_width=widths_by_call.get,
        max_depth=depth,
        aggregation=aggregation,
    )


@pytest.mark.parametrize(
    "operation", ["controlled_constant_add", "controlled_xor_constant"]
)
@pytest.mark.parametrize("cv", [0, 1])
def test_zero_width_partial_cache_and_retry(operation: Operation, cv: int) -> None:
    from qualtran.resource_counting import QubitCount, get_cost_value

    call = ArithmeticCall(operation, (0,), constant=0, control_value=cv)
    source_cache: dict[Any, SymbolicInt] = {}
    cache: dict[ResourceCall, SymbolicInt] = {}
    with pytest.raises(RuntimeError):
        get_cost_value(source_for(call), QubitCount(), costs_cache=source_cache)
    with pytest.raises(ValueError, match="positive"):
        query_resource_logical_width(call, cache=cache)
    assert cache == {
        source_key(key, call): value for key, value in source_cache.items()
    }
    valid = replace(call, widths=(3,), constant=2)
    assert query_resource_logical_width(valid, cache=cache) == (
        9 if operation == "controlled_constant_add" else 4
    )


def test_owner_rejects_unrelated_operation() -> None:
    with pytest.raises(NotImplementedError):
        controlled_constant_width(
            ArithmeticCall("xor_constant", (3,), constant=1), lambda key: 0
        )


@pytest.mark.parametrize(
    "operation", ["controlled_constant_add", "controlled_xor_constant"]
)
@pytest.mark.parametrize("cv", [0, 1])
def test_gate_override_keeps_available_lifetimes(operation: Operation, cv: int) -> None:
    call = ArithmeticCall(operation, (3,), constant=2, control_value=cv)
    assert query_resource_logical_width(call, declared_calls={call: {}}) == (
        9 if operation == "controlled_constant_add" else 4
    )
