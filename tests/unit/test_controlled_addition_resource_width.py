"""Compare CAdd's controlled-load ownership and complete normalized width cache."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any, get_args

import pytest
from sympy import Integer

from flagquantum.compiler._component_trace.arithmetic_call import (
    ArithmeticCall,
    ArithmeticEncoding,
)
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
from flagquantum.compiler.registers.arithmetic.controlled_addition_width import (
    controlled_addition_width,
)
from flagquantum.drawer.call_graph import (
    GateAggregation,
    export_resource_call_graph_dot,
)
from tests.unit.test_addition_resource_width import (
    ENCODINGS,
    WIDTHS,
)
from tests.unit.test_addition_resource_width import (
    source_key as add_source_key,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    reference = Path(__file__).resolve().parents[3] / "ftqc-reference-env"
    with (reference.parent / "FlagQuantum-Qualtran-Kickmix-移植进度.csv").open(
        encoding="utf-8-sig"
    ) as stream:
        blobs = {row["源文件"]: row["源对象"] for row in csv.DictReader(stream)}
    for name in (
        "qualtran/resource_counting/_qubit_counts.py",
        "qualtran/bloqs/arithmetic/controlled_addition.py",
        "qualtran/bloqs/bookkeeping/cast.py",
        "qualtran/bloqs/arithmetic/addition.py",
    ):
        raw = (reference / name).read_bytes()
        assert (
            hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blobs[name]
        )


def source_for(call: ArithmeticCall) -> Any:
    from qualtran import QInt, QMontgomeryUInt, QUInt
    from qualtran.bloqs.arithmetic import CAdd

    types = {"unsigned": QUInt, "twos_complement": QInt, "montgomery": QMontgomeryUInt}
    source = CAdd(
        *(types[e](w) for e, w in zip(call.encodings, call.widths, strict=True)),
        cv=call.control_value,
    )
    return source.adjoint() if call.adjoint else source


def source_key(source: Any) -> ResourceCall:
    from qualtran import QInt, QMontgomeryUInt
    from qualtran._infra.adjoint import Adjoint
    from qualtran.bloqs.arithmetic import CAdd
    from qualtran.bloqs.bookkeeping import Cast

    def encoding(dtype: Any) -> ArithmeticEncoding:
        return (
            "twos_complement"
            if isinstance(dtype, QInt)
            else "montgomery" if isinstance(dtype, QMontgomeryUInt) else "unsigned"
        )

    if isinstance(source, Adjoint) and isinstance(source.subbloq, CAdd):
        call = source_key(source.subbloq)
        assert isinstance(call, ArithmeticCall)
        return replace(call, adjoint=True)
    if isinstance(source, CAdd):
        return ArithmeticCall(
            "controlled_add",
            (source.a_dtype.bitsize, source.b_dtype.bitsize),
            (encoding(source.a_dtype), encoding(source.b_dtype)),
            control_value=source.cv,
        )
    if isinstance(source, Cast):
        return BookkeepingCall(
            "cast",
            (
                source.inp_dtype.num_qubits,
                encoding(source.inp_dtype),
                encoding(source.out_dtype),
            ),
        )
    return add_source_key(source)


def compare(call: ArithmeticCall, seed: Any = None) -> dict[ResourceCall, SymbolicInt]:
    from qualtran.resource_counting import QubitCount, get_cost_value

    expected: dict[Any, SymbolicInt] = {} if seed is None else {seed: 20}
    actual: dict[ResourceCall, SymbolicInt] = (
        {} if seed is None else {source_key(seed): 20}
    )
    width = get_cost_value(source_for(call), QubitCount(), costs_cache=expected)
    assert query_resource_logical_width(call, cache=actual) == width
    # Existing split/join keys normalize dtype to width. Normalize before
    # comparing insertion order, since QBit/QAny/QUInt one-bit keys coalesce.
    normalized = {source_key(key): value for key, value in expected.items()}
    assert actual == normalized
    assert list(actual) == list(normalized)
    return actual


@pytest.mark.parametrize("widths", WIDTHS)
@pytest.mark.parametrize("encoding", ENCODINGS)
@pytest.mark.parametrize("cv", [0, 1])
@pytest.mark.parametrize("adjoint", [False, True])
def test_full_cache_and_order(
    widths: tuple[SymbolicInt, SymbolicInt],
    encoding: ArithmeticEncoding,
    cv: int,
    adjoint: bool,
) -> None:
    compare(
        ArithmeticCall(
            "controlled_add",
            widths,
            (encoding, encoding),
            control_value=cv,
            adjoint=adjoint,
        )
    )


@pytest.mark.parametrize("encodings", tuple(product(ENCODINGS, repeat=2)))
@pytest.mark.parametrize("cv", [0, 1])
@pytest.mark.parametrize("adjoint", [False, True])
def test_mixed_encoding_words(
    encodings: tuple[ArithmeticEncoding, ArithmeticEncoding], cv: int, adjoint: bool
) -> None:
    compare(
        ArithmeticCall(
            "controlled_add", (2, 3), encodings, control_value=cv, adjoint=adjoint
        )
    )


@pytest.mark.parametrize("cv", [0, 1])
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("seed_name", ["load", "erase", "add", "cast", "cx"])
def test_child_peaks_include_idle_caller_words(
    seed_name: str, cv: int, adjoint: bool
) -> None:
    from qualtran import QInt, QUInt
    from qualtran.bloqs.arithmetic import Add
    from qualtran.bloqs.basic_gates import CNOT
    from qualtran.bloqs.bookkeeping import Cast
    from qualtran.bloqs.mcmt import And

    cast = Cast(QUInt(2), QInt(2))
    add = Add(QInt(2), QUInt(3))
    seeds = {
        "load": And(cv, 1),
        "erase": And(cv, 1).adjoint(),
        "add": add.adjoint() if adjoint else add,
        "cast": cast.adjoint() if adjoint else cast,
        "cx": CNOT(),
    }
    call = ArithmeticCall(
        "controlled_add",
        (2, 3),
        ("twos_complement", "unsigned"),
        control_value=cv,
        adjoint=adjoint,
    )
    assert compare(call, seeds[seed_name])[call] >= 20


@pytest.mark.parametrize("widths", [(1, 1), (2, 3), (Integer(3), Integer(3))])
@pytest.mark.parametrize("cv", [0, 1])
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("depth", [0, 1, None])
@pytest.mark.parametrize("aggregation", get_args(GateAggregation))
def test_automatic_dot_matches_source_widths(
    widths: tuple[SymbolicInt, SymbolicInt],
    cv: int,
    adjoint: bool,
    depth: int | None,
    aggregation: GateAggregation,
) -> None:
    from qualtran.resource_counting import QubitCount, get_cost_value

    call = ArithmeticCall("controlled_add", widths, control_value=cv, adjoint=adjoint)
    cache: dict[Any, SymbolicInt] = {}
    get_cost_value(source_for(call), QubitCount(), costs_cache=cache)
    widths_by_call = {source_key(key): value for key, value in cache.items()}
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


@pytest.mark.parametrize("widths", [(0, 0), (0, 1), (3, 2)])
@pytest.mark.parametrize("adjoint", [False, True])
def test_invalid_widths_fail_before_children(
    widths: tuple[int, int], adjoint: bool
) -> None:
    cache: dict[ResourceCall, SymbolicInt] = {}
    with pytest.raises(ValueError, match="positive"):
        query_resource_logical_width(
            ArithmeticCall("controlled_add", widths, adjoint=adjoint), cache=cache
        )
    assert cache == {}
    assert (
        query_resource_logical_width(
            ArithmeticCall("controlled_add", (2, 3), adjoint=adjoint), cache=cache
        )
        == 10
    )


def test_owner_rejects_unrelated_operation() -> None:
    with pytest.raises(NotImplementedError):
        controlled_addition_width(ArithmeticCall("add", (2, 3)), lambda key: 0)


def test_gate_override_keeps_loaded_word_lifetime() -> None:
    call = ArithmeticCall("controlled_add", (2, 3), control_value=0)
    cache: dict[ResourceCall, SymbolicInt] = {}
    assert (
        query_resource_logical_width(call, declared_calls={call: {}}, cache=cache) == 10
    )
    assert ConjunctionCall((0, 1), True) in cache


@pytest.mark.parametrize(
    "layout",
    [
        1,
        (1,),
        (0, "unsigned", "unsigned"),
        (True, "unsigned", "unsigned"),
        (2, "bad", "unsigned"),
    ],
)
def test_unsupported_cast_layouts_keep_boundary(layout: Any) -> None:
    with pytest.raises(NotImplementedError):
        query_resource_logical_width(BookkeepingCall("cast", layout))
