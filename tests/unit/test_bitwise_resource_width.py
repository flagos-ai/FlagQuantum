"""Compare bitwise width connections and sparse caches with frozen source."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any, get_args

import pytest
from sympy import Integer, Symbol

from flagquantum.compiler._component_trace.arithmetic_call import (
    ArithmeticCall,
    ArithmeticEncoding,
    ArithmeticOperation,
)
from flagquantum.compiler._component_trace.resource_generalization import (
    BookkeepingCall,
    CliffordCall,
    OnEachCall,
    ResourceCall,
)
from flagquantum.compiler._component_trace.resource_width import (
    query_resource_logical_width,
)
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.compiler.registers.arithmetic.arithmetic_declarations import (
    arithmetic_call_graph,
)
from flagquantum.compiler.registers.bits.bitwise_width import bitwise_width
from flagquantum.drawer.call_graph import (
    GateAggregation,
    export_resource_call_graph_dot,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
n = Symbol("n", positive=True, integer=True)
z = Symbol("z", integer=True)
k = Symbol("k", integer=True)
OPERATIONS: tuple[ArithmeticOperation, ...] = ("xor", "bitwise_not")
CASES = [
    ArithmeticCall(operation, (width,))
    for operation in OPERATIONS
    for width in (1, 3, 8, Integer(3), n, z)
] + [
    ArithmeticCall("xor_constant", (width,), constant=constant)
    for width, constant in (
        (1, 0),
        (1, 1),
        (3, 0),
        (3, 3),
        (8, 83),
        (3, Integer(0)),
        (Integer(3), 0),
        (Integer(3), Integer(3)),
        (n, 0),
        (n, k),
        (3, k),
        (z, 0),
    )
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
        "qualtran/bloqs/arithmetic/bitwise.py",
        "qualtran/bloqs/basic_gates/on_each.py",
    ):
        raw = (reference / name).read_bytes()
        assert (
            hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blobs[name]
        )


def source_for(root: ArithmeticCall) -> Any:
    from qualtran import QInt, QMontgomeryUInt, QUInt
    from qualtran.bloqs.arithmetic.bitwise import BitwiseNot, Xor, XorK

    dtype = {"unsigned": QUInt, "twos_complement": QInt, "montgomery": QMontgomeryUInt}[
        root.encodings[0]
    ](root.widths[0])
    source = (
        XorK(dtype, root.constant)
        if root.operation == "xor_constant"
        else Xor(dtype) if root.operation == "xor" else BitwiseNot(dtype)
    )
    return source.adjoint() if root.adjoint else source


def source_key(source: Any, root: ArithmeticCall) -> ResourceCall:
    from qualtran.bloqs.basic_gates import CNOT, OnEach, XGate
    from qualtran.bloqs.bookkeeping import Join, Split

    if source == source_for(root):
        return root
    if isinstance(source, (Split, Join)):
        return BookkeepingCall(
            "split" if isinstance(source, Split) else "join", source.dtype.num_qubits
        )
    if isinstance(source, OnEach):
        return OnEachCall(CliffordCall("x", 1), source.n)
    if isinstance(source, CNOT):
        return CliffordCall("cx", 2)
    assert isinstance(source, XGate)
    return CliffordCall("x", 1)


def compare(root: ArithmeticCall, seed: Any = None) -> dict[ResourceCall, SymbolicInt]:
    from qualtran.resource_counting import QubitCount, get_cost_value

    expected_cache: dict[Any, SymbolicInt] = {} if seed is None else {seed: 12}
    actual: dict[ResourceCall, SymbolicInt] = (
        {} if seed is None else {source_key(seed, root): 12}
    )
    expected = get_cost_value(
        source_for(root), QubitCount(), costs_cache=expected_cache
    )
    assert query_resource_logical_width(root, cache=actual) == expected
    assert actual == {
        source_key(key, root): value for key, value in expected_cache.items()
    }
    assert list(actual) == [source_key(key, root) for key in expected_cache]
    return actual


@pytest.mark.parametrize("root", CASES)
@pytest.mark.parametrize("encoding", ["unsigned", "twos_complement", "montgomery"])
@pytest.mark.parametrize("adjoint", [False, True])
def test_complete_source_caches_and_order(
    root: ArithmeticCall, encoding: ArithmeticEncoding, adjoint: bool
) -> None:
    root = replace(root, encodings=(encoding,), adjoint=adjoint)
    if (
        encoding == "twos_complement"
        and root.operation == "xor_constant"
        and root.widths == (1,)
        and root.constant == 1
    ):
        from qualtran.resource_counting import QubitCount, get_cost_value

        with pytest.raises(RuntimeError):
            get_cost_value(source_for(root), QubitCount())
        with pytest.raises(ValueError, match="range"):
            query_resource_logical_width(root)
        return
    compare(root)


@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("width", [1, 3, 8])
def test_signed_negative_patterns(width: int, adjoint: bool) -> None:
    compare(
        ArithmeticCall(
            "xor_constant", (width,), ("twos_complement",), constant=-1, adjoint=adjoint
        )
    )


@pytest.mark.parametrize("root", [CASES[1], CASES[7], CASES[15]])
@pytest.mark.parametrize("seed_name", ["split", "join", "gate"])
def test_child_local_peaks_preserve_bystanders(
    root: ArithmeticCall, seed_name: str
) -> None:
    from qualtran import QUInt
    from qualtran.bloqs.basic_gates import CNOT, XGate
    from qualtran.bloqs.bookkeeping import Join, Split

    seeds = {
        "split": Split(QUInt(3)),
        "join": Join(QUInt(3)),
        "gate": CNOT() if root.operation == "xor" else XGate(),
    }
    cache = compare(root, seeds[seed_name])
    assert cache[root] >= 12


@pytest.mark.parametrize("operation", ["xor", "xor_constant", "bitwise_not"])
@pytest.mark.parametrize("width", [1, 3, n])
@pytest.mark.parametrize("depth", [0, 1, None])
@pytest.mark.parametrize("aggregation", get_args(GateAggregation))
def test_factory_uses_source_width_rows(
    operation: ArithmeticOperation,
    width: SymbolicInt,
    depth: int | None,
    aggregation: GateAggregation,
) -> None:
    from qualtran.resource_counting import QubitCount, get_cost_value

    root = ArithmeticCall(
        operation, (width,), constant=0 if operation == "xor_constant" else None
    )
    cache: dict[Any, SymbolicInt] = {}
    get_cost_value(source_for(root), QubitCount(), costs_cache=cache)
    widths = {source_key(key, root): value for key, value in cache.items()}
    declarations = arithmetic_call_graph(root)
    assert export_resource_call_graph_dot(
        root, declared_calls=declarations, max_depth=depth, aggregation=aggregation
    ) == export_resource_call_graph_dot(
        root,
        declared_calls=declarations,
        logical_width=widths.get,
        max_depth=depth,
        aggregation=aggregation,
    )


@pytest.mark.parametrize("operation", ["xor", "xor_constant", "bitwise_not"])
@pytest.mark.parametrize("width", [0, Integer(0)])
def test_zero_width_failure_and_retry(
    operation: ArithmeticOperation, width: SymbolicInt
) -> None:
    from qualtran.resource_counting import QubitCount, get_cost_value

    root = ArithmeticCall(
        operation, (width,), constant=0 if operation == "xor_constant" else None
    )
    with pytest.raises(RuntimeError):
        get_cost_value(source_for(root), QubitCount())
    cache: dict[ResourceCall, SymbolicInt] = {}
    with pytest.raises(ValueError, match="nonempty"):
        query_resource_logical_width(root, cache=cache)
    assert cache == {}
    assert query_resource_logical_width(replace(root, widths=(2,)), cache=cache) in (
        2,
        4,
    )


@pytest.mark.parametrize(
    "encoding,constant",
    [
        ("unsigned", -1),
        ("unsigned", 8),
        ("montgomery", -1),
        ("twos_complement", 4),
        ("twos_complement", -5),
    ],
)
def test_invalid_constant_rejected_before_cache_mutation(
    encoding: ArithmeticEncoding, constant: int
) -> None:
    from qualtran.resource_counting import QubitCount, get_cost_value

    root = ArithmeticCall("xor_constant", (3,), (encoding,), constant=constant)
    with pytest.raises(RuntimeError):
        get_cost_value(source_for(root), QubitCount())
    cache: dict[ResourceCall, SymbolicInt] = {}
    with pytest.raises(ValueError, match="range"):
        query_resource_logical_width(root, cache=cache)
    assert cache == {}


def test_unsupported_arithmetic_keeps_explicit_boundary() -> None:
    root = ArithmeticCall("out_of_place_add", (3,))
    with pytest.raises(NotImplementedError):
        bitwise_width(root, lambda child: 0)
    with pytest.raises(NotImplementedError, match="No native logical width binding"):
        query_resource_logical_width(root)


def test_declarations_do_not_replace_available_connections() -> None:
    root = ArithmeticCall("xor_constant", (3,), constant=0)
    cache: dict[ResourceCall, SymbolicInt] = {}
    query_resource_logical_width(
        root, cache=cache, declared_calls={root: {CliffordCall("x", 1): 20}}
    )
    assert CliffordCall("x", 1) not in cache
