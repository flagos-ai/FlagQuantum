"""Compare sum-output lifetimes and complete caches with the frozen source."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any, get_args

import pytest
from sympy import Integer, Symbol

from flagquantum.compiler._component_trace.arithmetic_call import ArithmeticCall
from flagquantum.compiler._component_trace.resource_generalization import (
    BookkeepingCall,
    ConjunctionCall,
    ResourceCall,
)
from flagquantum.compiler._component_trace.resource_width import (
    query_resource_logical_width,
)
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.compiler.registers.arithmetic.arithmetic_declarations import (
    arithmetic_call_graph,
)
from flagquantum.compiler.registers.arithmetic.out_of_place_addition_width import (
    out_of_place_addition_width,
)
from flagquantum.drawer.call_graph import (
    GateAggregation,
    export_resource_call_graph_dot,
)
from tests.unit import test_addition_resource_width as addition_tests

addition_source_key = addition_tests.source_key

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
n = Symbol("n", positive=True, integer=True)
p = Symbol("p", nonnegative=True, integer=True)
WIDTHS = (1, 2, 3, 5, 8, Integer(0), Integer(1), Integer(3), n, p)


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
        "qualtran/cirq_interop/_cirq_to_bloq.py",
        "qualtran/_infra/gate_with_registers.py",
        "qualtran/_infra/binst_graph_iterators.py",
        "qualtran/bloqs/bookkeeping/allocate.py",
        "qualtran/bloqs/bookkeeping/free.py",
        "qualtran/bloqs/mcmt/and_bloq.py",
    ):
        raw = (reference / name).read_bytes()
        assert (
            hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blobs[name]
        )


def source_for(call: ArithmeticCall) -> Any:
    from qualtran.bloqs.arithmetic.addition import OutOfPlaceAdder

    return OutOfPlaceAdder(
        call.widths[0],
        is_adjoint=call.adjoint,
        include_most_significant_bit=call.include_most_significant_bit,
    )


def source_key(source: Any) -> ResourceCall:
    from qualtran.bloqs.arithmetic.addition import OutOfPlaceAdder
    from qualtran.bloqs.bookkeeping import Allocate, Free

    if isinstance(source, OutOfPlaceAdder):
        return ArithmeticCall(
            "out_of_place_add",
            (source.bitsize,),
            adjoint=source.is_adjoint,
            include_most_significant_bit=source.include_most_significant_bit,
        )
    if isinstance(source, (Allocate, Free)):
        return BookkeepingCall(
            "allocate" if isinstance(source, Allocate) else "free",
            source.dtype.num_qubits,
        )
    return addition_source_key(source)


def compare(call: ArithmeticCall, seed: Any = None) -> dict[ResourceCall, SymbolicInt]:
    from qualtran.resource_counting import QubitCount, get_cost_value

    expected: dict[Any, SymbolicInt] = {} if seed is None else {seed: 20}
    actual: dict[ResourceCall, SymbolicInt] = (
        {} if seed is None else {source_key(seed): 20}
    )
    assert query_resource_logical_width(call, cache=actual) == get_cost_value(
        source_for(call), QubitCount(), costs_cache=expected
    )
    normalized = {source_key(key): value for key, value in expected.items()}
    assert actual == normalized
    assert list(actual) == list(normalized)
    return actual


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("msb", [False, True])
@pytest.mark.parametrize("adjoint", [False, True])
def test_full_cache_and_order(width: SymbolicInt, msb: bool, adjoint: bool) -> None:
    compare(
        ArithmeticCall(
            "out_of_place_add",
            (width,),
            adjoint=adjoint,
            include_most_significant_bit=msb,
        )
    )


@pytest.mark.parametrize("width", [1, 2, 3, 5])
@pytest.mark.parametrize("msb", [False, True])
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize(
    "seed_name",
    [
        "cx",
        "and",
        "erase",
        "allocate",
        "free",
        "split-input",
        "split-output",
        "join-input",
        "join-output",
    ],
)
def test_created_and_consumed_outputs_survive_local_peaks(
    width: int, msb: bool, adjoint: bool, seed_name: str
) -> None:
    from qualtran import QAny, QBit, QUInt
    from qualtran.bloqs.basic_gates import CNOT
    from qualtran.bloqs.bookkeeping import Allocate, Free, Join, Split
    from qualtran.bloqs.mcmt import And

    seeds = {
        "cx": CNOT(),
        "and": And(),
        "erase": And().adjoint(),
        "allocate": Allocate(QAny(1)),
        "free": Free(QBit()),
        "split-input": Split(QUInt(width)),
        "split-output": Split(QUInt(width + int(msb))),
        "join-input": Join(QUInt(width)),
        "join-output": Join(QUInt(width + int(msb))),
    }
    compare(
        ArithmeticCall(
            "out_of_place_add",
            (width,),
            adjoint=adjoint,
            include_most_significant_bit=msb,
        ),
        seeds[seed_name],
    )


@pytest.mark.parametrize("width", [1, 3, n, Integer(3)])
@pytest.mark.parametrize("msb", [False, True])
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("depth", [0, 1, None])
@pytest.mark.parametrize("aggregation", get_args(GateAggregation))
def test_full_dot_uses_source_width_cache(
    width: SymbolicInt,
    msb: bool,
    adjoint: bool,
    depth: int | None,
    aggregation: GateAggregation,
) -> None:
    call = ArithmeticCall(
        "out_of_place_add", (width,), adjoint=adjoint, include_most_significant_bit=msb
    )
    widths = compare(call)
    declarations = arithmetic_call_graph(call)
    assert export_resource_call_graph_dot(
        call, declared_calls=declarations, max_depth=depth, aggregation=aggregation
    ) == export_resource_call_graph_dot(
        call,
        declared_calls=declarations,
        logical_width=widths.get,
        max_depth=depth,
        aggregation=aggregation,
    )


@pytest.mark.parametrize("msb", [False, True])
@pytest.mark.parametrize("adjoint", [False, True])
def test_zero_failure_and_cache_retry(msb: bool, adjoint: bool) -> None:
    from qualtran.resource_counting import QubitCount, get_cost_value

    call = ArithmeticCall(
        "out_of_place_add", (0,), adjoint=adjoint, include_most_significant_bit=msb
    )
    with pytest.raises(RuntimeError):
        get_cost_value(source_for(call), QubitCount())
    cache: dict[ResourceCall, SymbolicInt] = {}
    with pytest.raises(ValueError, match="positive"):
        query_resource_logical_width(call, cache=cache)
    assert cache == {}
    assert query_resource_logical_width(
        replace(call, widths=(3,)), cache=cache
    ) == 9 + int(msb)


def test_owner_boundary_and_decomposition_priority() -> None:
    with pytest.raises(NotImplementedError):
        out_of_place_addition_width(ArithmeticCall("xor", (3,)), lambda call: 0)
    call = ArithmeticCall("out_of_place_add", (3,))
    cache: dict[ResourceCall, SymbolicInt] = {}
    assert (
        query_resource_logical_width(call, declared_calls={call: {}}, cache=cache) == 10
    )
    assert BookkeepingCall("allocate", 1) in cache
    assert ConjunctionCall((1, 1)) in cache


def test_source_power_drops_truncation_but_explicit_adjoint_retains_it() -> None:
    source = source_for(
        ArithmeticCall("out_of_place_add", (3,), include_most_significant_bit=False)
    )
    assert source.adjoint().include_most_significant_bit is False
    assert (source**-1).include_most_significant_bit is True
