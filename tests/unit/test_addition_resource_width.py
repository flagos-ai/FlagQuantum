"""Compare Add carry ownership, visitation and source width tables."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any, get_args

import pytest
from sympy import Integer, Symbol

from flagquantum.compiler._component_trace.arithmetic_call import (
    ArithmeticCall,
    ArithmeticEncoding,
)
from flagquantum.compiler._component_trace.resource_generalization import (
    BookkeepingCall,
    CliffordCall,
    ConjunctionCall,
    ResourceCall,
)
from flagquantum.compiler._component_trace.resource_width import (
    query_resource_logical_width,
)
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.compiler.registers.arithmetic.addition_width import addition_width
from flagquantum.compiler.registers.arithmetic.arithmetic_declarations import (
    arithmetic_call_graph,
)
from flagquantum.drawer.call_graph import (
    GateAggregation,
    export_resource_call_graph_dot,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
n = Symbol("n", positive=True, integer=True)
WIDTHS = [(a, b) for b in range(1, 6) for a in range(1, b + 1)] + [
    (3, 8),
    (Integer(3), Integer(3)),
    (1, Integer(3)),
    (Integer(1), 3),
    (n, n),
    (n, n + 1),
    (1, n),
    (Integer(1), Integer(1)),
]
ENCODINGS: tuple[ArithmeticEncoding, ...] = (
    "unsigned",
    "twos_complement",
    "montgomery",
)


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
        "qualtran/bloqs/bookkeeping/always.py",
    ):
        raw = (reference / name).read_bytes()
        assert (
            hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blobs[name]
        )


def source_for(call: ArithmeticCall) -> Any:
    from qualtran import QInt, QMontgomeryUInt, QUInt
    from qualtran.bloqs.arithmetic import Add
    from qualtran.bloqs.bookkeeping import Always

    types = {"unsigned": QUInt, "twos_complement": QInt, "montgomery": QMontgomeryUInt}
    source = Add(
        *(types[e](w) for e, w in zip(call.encodings, call.widths, strict=True))
    )
    if call.adjoint:
        source = source.adjoint()
    return Always(source) if call.always else source


def source_key(source: Any) -> ResourceCall:
    from qualtran import QInt, QMontgomeryUInt
    from qualtran._infra.adjoint import Adjoint
    from qualtran.bloqs.arithmetic import Add
    from qualtran.bloqs.basic_gates import CNOT
    from qualtran.bloqs.bookkeeping import Always, Join, Split
    from qualtran.bloqs.mcmt import And

    if isinstance(source, (Always, Adjoint)):
        call = source_key(source.subbloq)
        assert isinstance(call, ArithmeticCall)
        return (
            replace(call, always=True)
            if isinstance(source, Always)
            else replace(call, adjoint=True)
        )
    if isinstance(source, Add):
        types = (source.a_dtype, source.b_dtype)
        encodings: tuple[ArithmeticEncoding, ...] = tuple(
            (
                "twos_complement"
                if isinstance(t, QInt)
                else "montgomery" if isinstance(t, QMontgomeryUInt) else "unsigned"
            )
            for t in types
        )
        return ArithmeticCall("add", tuple(t.bitsize for t in types), encodings)
    if isinstance(source, (Split, Join)):
        return BookkeepingCall(
            "split" if isinstance(source, Split) else "join", source.dtype.num_qubits
        )
    if isinstance(source, And):
        return ConjunctionCall((source.cv1, source.cv2), source.uncompute)
    assert isinstance(source, CNOT)
    return CliffordCall("cx", 2)


def compare(call: ArithmeticCall, seed: Any = None) -> dict[ResourceCall, SymbolicInt]:
    from qualtran.resource_counting import QubitCount, get_cost_value

    expected: dict[Any, SymbolicInt] = {} if seed is None else {seed: 12}
    actual: dict[ResourceCall, SymbolicInt] = (
        {} if seed is None else {source_key(seed): 12}
    )
    value = get_cost_value(source_for(call), QubitCount(), costs_cache=expected)
    assert query_resource_logical_width(call, cache=actual) == value
    assert actual == {source_key(key): width for key, width in expected.items()}
    assert list(actual) == [source_key(key) for key in expected]
    return actual


@pytest.mark.parametrize("widths", WIDTHS)
@pytest.mark.parametrize("encoding", ENCODINGS)
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("always", [False, True])
def test_complete_width_cache_and_order(
    widths: tuple[SymbolicInt, SymbolicInt],
    encoding: ArithmeticEncoding,
    adjoint: bool,
    always: bool,
) -> None:
    compare(
        ArithmeticCall(
            "add", widths, (encoding, encoding), adjoint=adjoint, always=always
        )
    )


@pytest.mark.parametrize("encodings", tuple(product(ENCODINGS, repeat=2)))
@pytest.mark.parametrize("adjoint", [False, True])
def test_mixed_word_encodings_keep_width_identity(
    encodings: tuple[ArithmeticEncoding, ArithmeticEncoding],
    adjoint: bool,
) -> None:
    # Different widths retain distinct native bookkeeping identities.
    compare(ArithmeticCall("add", (2, 3), encodings, adjoint=adjoint))


@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize(
    "seed_name", ["split-a", "split-b", "join-a", "join-b", "and", "erase", "cx"]
)
def test_carry_local_peaks_and_idle_words_match_source(
    seed_name: str, adjoint: bool
) -> None:
    from qualtran import QUInt
    from qualtran.bloqs.basic_gates import CNOT
    from qualtran.bloqs.bookkeeping import Join, Split
    from qualtran.bloqs.mcmt import And

    seeds = {
        "split-a": Split(QUInt(2)),
        "split-b": Split(QUInt(3)),
        "join-a": Join(QUInt(2)),
        "join-b": Join(QUInt(3)),
        "and": And(),
        "erase": And().adjoint(),
        "cx": CNOT(),
    }
    call = ArithmeticCall("add", (2, 3), adjoint=adjoint)
    actual = compare(call, seeds[seed_name])
    assert actual[call] >= 12


@pytest.mark.parametrize(
    "widths", [(1, 1), (1, 3), (2, 3), (n, n), (Integer(3), Integer(3))]
)
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("depth", [0, 1, None])
@pytest.mark.parametrize("aggregation", get_args(GateAggregation))
def test_full_dot_uses_original_width_cache(
    widths: tuple[SymbolicInt, SymbolicInt],
    adjoint: bool,
    depth: int | None,
    aggregation: GateAggregation,
) -> None:
    from qualtran.resource_counting import QubitCount, get_cost_value

    call = ArithmeticCall("add", widths, adjoint=adjoint)
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


@pytest.mark.parametrize("widths", [(0, 0), (0, 1), (0, 3)])
@pytest.mark.parametrize("adjoint", [False, True])
def test_zero_input_failure_and_retry(widths: tuple[int, int], adjoint: bool) -> None:
    from qualtran.resource_counting import QubitCount, get_cost_value

    call = ArithmeticCall("add", widths, adjoint=adjoint)
    with pytest.raises(RuntimeError):
        get_cost_value(source_for(call), QubitCount())
    cache: dict[ResourceCall, SymbolicInt] = {}
    with pytest.raises(ValueError, match="positive"):
        query_resource_logical_width(call, cache=cache)
    assert cache == {}
    assert query_resource_logical_width(replace(call, widths=(2, 3)), cache=cache) == 7


def test_reversed_widths_reject_before_child_query() -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add

    with pytest.raises(ValueError):
        Add(QUInt(3), QUInt(2))
    cache: dict[ResourceCall, SymbolicInt] = {}
    with pytest.raises(ValueError, match="left <= right"):
        query_resource_logical_width(ArithmeticCall("add", (3, 2)), cache=cache)
    assert cache == {}


def test_owner_rejects_unrelated_operation() -> None:
    with pytest.raises(NotImplementedError):
        addition_width(ArithmeticCall("xor", (3,)), lambda call: 0)


def test_gate_overrides_keep_carry_ownership() -> None:
    call = ArithmeticCall("add", (2, 3))
    cache: dict[ResourceCall, SymbolicInt] = {}
    assert (
        query_resource_logical_width(call, declared_calls={call: {}}, cache=cache) == 7
    )
    assert ConjunctionCall((1, 1), True) in cache
