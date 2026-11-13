"""Compare AddK's allocation and constant-word width lifetimes to frozen source."""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Hashable
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, get_args

import pytest
from sympy import Integer, Symbol

from flagquantum.compiler._component_trace.arithmetic_call import (
    ArithmeticCall,
    ArithmeticEncoding,
)
from flagquantum.compiler._component_trace.resource_generalization import (
    BookkeepingCall,
    CliffordCall,
    ResourceCall,
)
from flagquantum.compiler._component_trace.resource_width import (
    query_resource_logical_width,
)
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.compiler.registers.arithmetic.arithmetic_declarations import (
    arithmetic_call_graph,
)
from flagquantum.compiler.registers.arithmetic.constant_addition_width import (
    constant_addition_width,
)
from flagquantum.drawer.call_graph import (
    GateAggregation,
    export_resource_call_graph_dot,
)
from tests.unit.test_addition_resource_width import (
    ENCODINGS,
)
from tests.unit.test_addition_resource_width import (
    source_key as add_source_key,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
n = Symbol("n", positive=True, integer=True)
k = Symbol("k", integer=True)
CASES = [
    (width, value) for width in range(1, 6) for value in (0, 1, -1, -(1 << width) - 3)
] + [
    (8, 27),
    (Integer(3), 2),
    (3, Integer(2)),
    (Integer(3), -1),
    (3, Integer(-1)),
    (Integer(3), Integer(-1)),
    (n, k),
    (3, k),
    (n, 0),
    (Integer(1), Integer(0)),
]


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
    ):
        raw = (reference / name).read_bytes()
        assert (
            hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blobs[name]
        )


def source_for(call: ArithmeticCall) -> Any:
    from qualtran import QInt, QMontgomeryUInt, QUInt
    from qualtran.bloqs.arithmetic import AddK

    types = {"unsigned": QUInt, "twos_complement": QInt, "montgomery": QMontgomeryUInt}
    source = AddK(types[call.encodings[0]](call.widths[0]), call.constant)
    return source.adjoint() if call.adjoint else source


def source_key(source: Any) -> ResourceCall:
    from qualtran import QInt, QMontgomeryUInt
    from qualtran._infra.adjoint import Adjoint
    from qualtran.bloqs.arithmetic import AddK
    from qualtran.bloqs.arithmetic.bitwise import XorK
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.bookkeeping import Allocate, Free

    if isinstance(source, Adjoint) and isinstance(source.subbloq, AddK):
        call = source_key(source.subbloq)
        assert isinstance(call, ArithmeticCall)
        return replace(call, adjoint=True)
    if isinstance(source, (AddK, XorK)):
        encoding: ArithmeticEncoding = (
            "twos_complement"
            if isinstance(source.dtype, QInt)
            else (
                "montgomery"
                if isinstance(source.dtype, QMontgomeryUInt)
                else "unsigned"
            )
        )
        return ArithmeticCall(
            "constant_add" if isinstance(source, AddK) else "xor_constant",
            (source.dtype.bitsize,),
            (encoding,),
            constant=source.k,
        )
    if isinstance(source, (Allocate, Free)):
        assert not source.dirty
        return BookkeepingCall(
            "allocate" if isinstance(source, Allocate) else "free",
            source.dtype.num_qubits,
        )
    if isinstance(source, XGate):
        return CliffordCall("x", 1)
    return add_source_key(source)


def compare(call: ArithmeticCall, seed: Any = None) -> dict[ResourceCall, SymbolicInt]:
    from qualtran.resource_counting import QubitCount, get_cost_value

    expected: dict[Any, SymbolicInt] = {} if seed is None else {seed: 20}
    actual: dict[ResourceCall, SymbolicInt] = (
        {} if seed is None else {source_key(seed): 20}
    )
    width = get_cost_value(source_for(call), QubitCount(), costs_cache=expected)
    assert query_resource_logical_width(call, cache=actual) == width
    normalized = {source_key(key): value for key, value in expected.items()}
    assert actual == normalized
    assert list(actual) == list(normalized)
    return actual


@pytest.mark.parametrize("width,value", CASES)
@pytest.mark.parametrize("encoding", ENCODINGS)
@pytest.mark.parametrize("adjoint", [False, True])
def test_full_cache_and_order(
    width: SymbolicInt, value: SymbolicInt, encoding: ArithmeticEncoding, adjoint: bool
) -> None:
    from qualtran.resource_counting import QubitCount, get_cost_value

    call = ArithmeticCall(
        "constant_add", (width,), (encoding,), constant=value, adjoint=adjoint
    )
    if (
        encoding == "twos_complement"
        and type(width) is int
        and type(value) is int
        and not -(1 << (width - 1)) <= value < (1 << (width - 1))
    ):
        with pytest.raises(RuntimeError):
            get_cost_value(source_for(call), QubitCount())
        with pytest.raises(ValueError, match="range"):
            query_resource_logical_width(call)
        return
    compare(call)


@pytest.mark.parametrize("encoding", ENCODINGS)
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize(
    "seed_name",
    ["allocate", "free", "load", "add", "always", "and", "erase", "cx", "x"],
)
def test_child_peaks_preserve_allocated_and_idle_words(
    seed_name: str, encoding: ArithmeticEncoding, adjoint: bool
) -> None:
    from qualtran import QInt, QMontgomeryUInt, QUInt
    from qualtran.bloqs.arithmetic import Add
    from qualtran.bloqs.arithmetic.bitwise import XorK
    from qualtran.bloqs.basic_gates import CNOT, XGate
    from qualtran.bloqs.bookkeeping import Allocate, Always, Free
    from qualtran.bloqs.mcmt import And

    dtype = {"unsigned": QUInt, "twos_complement": QInt, "montgomery": QMontgomeryUInt}[
        encoding
    ](3)
    add = Add(dtype, dtype)
    add = add.adjoint() if adjoint else add
    seeds = {
        "allocate": Allocate(dtype),
        "free": Free(dtype),
        "load": XorK(dtype, 1),
        "add": add,
        "always": Always(add),
        "and": And(),
        "erase": And().adjoint(),
        "cx": CNOT(),
        "x": XGate(),
    }
    call = ArithmeticCall(
        "constant_add", (3,), (encoding,), constant=1, adjoint=adjoint
    )
    assert compare(call, seeds[seed_name])[call] >= 20


@pytest.mark.parametrize("width,value", [(1, 0), (3, -1), (3, k), (Integer(3), 2)])
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("depth", [0, 1, None])
@pytest.mark.parametrize("aggregation", get_args(GateAggregation))
def test_automatic_dot_matches_source_widths(
    width: SymbolicInt,
    value: SymbolicInt,
    adjoint: bool,
    depth: int | None,
    aggregation: GateAggregation,
) -> None:
    from qualtran.resource_counting import QubitCount, get_cost_value

    call = ArithmeticCall("constant_add", (width,), constant=value, adjoint=adjoint)
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


@pytest.mark.parametrize("width", [0, Integer(0)])
@pytest.mark.parametrize("adjoint", [False, True])
def test_zero_width_rejects_before_child_query(
    width: SymbolicInt, adjoint: bool
) -> None:
    from qualtran.resource_counting import QubitCount, get_cost_value

    call = ArithmeticCall("constant_add", (width,), constant=0, adjoint=adjoint)
    with pytest.raises(RuntimeError):
        get_cost_value(source_for(call), QubitCount())
    cache: dict[ResourceCall, SymbolicInt] = {}
    with pytest.raises(ValueError, match="positive"):
        query_resource_logical_width(call, cache=cache)
    assert cache == {}
    assert query_resource_logical_width(replace(call, widths=(3,)), cache=cache) == 8


@pytest.mark.parametrize("adjoint", [False, True])
def test_invalid_constant_preserves_partial_cache_and_retry(adjoint: bool) -> None:
    from qualtran.resource_counting import QubitCount, get_cost_value

    call = ArithmeticCall("constant_add", (3,), constant=8, adjoint=adjoint)
    source_cache: dict[Any, SymbolicInt] = {}
    cache: dict[ResourceCall, SymbolicInt] = {}
    with pytest.raises(RuntimeError):
        get_cost_value(source_for(call), QubitCount(), costs_cache=source_cache)
    with pytest.raises(ValueError, match="range"):
        query_resource_logical_width(call, cache=cache)
    assert call not in cache
    assert cache == {source_key(key): value for key, value in source_cache.items()}
    assert query_resource_logical_width(replace(call, constant=1), cache=cache) == 8


@pytest.mark.parametrize("opcode", ["allocate", "free"])
@pytest.mark.parametrize("width", [0, 1, 8])
def test_clean_scalar_allocation_atoms(
    opcode: Literal["allocate", "free"], width: int
) -> None:
    from qualtran import QUInt
    from qualtran.bloqs.bookkeeping import Allocate, Free
    from qualtran.resource_counting import QubitCount, get_cost_value

    source = (Allocate if opcode == "allocate" else Free)(QUInt(width))
    assert query_resource_logical_width(
        BookkeepingCall(opcode, width)
    ) == get_cost_value(source, QubitCount())


@pytest.mark.parametrize("opcode", ["allocate", "free"])
@pytest.mark.parametrize("layout", [-1, True, (3,), "3", Integer(3)])
def test_unsupported_allocation_layouts(
    opcode: Literal["allocate", "free"], layout: Hashable
) -> None:
    with pytest.raises(NotImplementedError):
        query_resource_logical_width(BookkeepingCall(opcode, layout))


def test_owner_rejects_unrelated_operation() -> None:
    with pytest.raises(NotImplementedError):
        constant_addition_width(ArithmeticCall("add", (2, 3)), lambda key: 0)


def test_gate_override_keeps_constant_word_lifetime() -> None:
    call = ArithmeticCall("constant_add", (3,), constant=1)
    cache: dict[ResourceCall, SymbolicInt] = {}
    assert (
        query_resource_logical_width(call, declared_calls={call: {}}, cache=cache) == 8
    )
    assert BookkeepingCall("free", 3) in cache
