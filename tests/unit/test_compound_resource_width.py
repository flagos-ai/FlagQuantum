"""Compare native conjunction and exact swap widths with unchanged source."""

from __future__ import annotations

import csv
import hashlib
from itertools import product
from pathlib import Path
from typing import Any, get_args

import pytest
from sympy import Integer, Symbol

from flagquantum.compiler._component_trace.resource_generalization import (
    BookkeepingCall,
    ConjunctionCall,
    ControlledSwapCall,
    FixedTCall,
    ResourceCall,
)
from flagquantum.compiler._component_trace.resource_width import (
    query_resource_logical_width,
)
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.drawer.call_graph import (
    GateAggregation,
    export_resource_call_graph_dot,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_sources() -> None:
    reference = Path(__file__).resolve().parents[3] / "ftqc-reference-env"
    with (reference.parent / "FlagQuantum-Qualtran-Kickmix-移植进度.csv").open(
        encoding="utf-8-sig"
    ) as stream:
        blobs = {row["源文件"]: row["源对象"] for row in csv.DictReader(stream)}
    for name in [
        "qualtran/resource_counting/_qubit_counts.py",
        "qualtran/bloqs/mcmt/and_bloq.py",
        "qualtran/bloqs/basic_gates/swap.py",
    ]:
        raw = (reference / name).read_bytes()
        assert (
            hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blobs[name]
        )


def native_source_key(source: Any) -> ResourceCall:
    from qualtran._infra.adjoint import Adjoint
    from qualtran.bloqs.basic_gates import CSwap, TwoBitCSwap
    from qualtran.bloqs.bookkeeping import Join, Split
    from qualtran.bloqs.mcmt import And, MultiAnd
    from qualtran.symbolics import HasLength

    if isinstance(source, Adjoint):
        key = native_source_key(source.subbloq)
        assert isinstance(key, (ConjunctionCall, ControlledSwapCall))
        if isinstance(key, ConjunctionCall):
            return ConjunctionCall(key.controls, True)
        return ControlledSwapCall(key.width, key.approximate, True)
    if isinstance(source, And):
        return ConjunctionCall((source.cv1, source.cv2), source.uncompute)
    if isinstance(source, MultiAnd):
        return ConjunctionCall(
            source.cvs.n if isinstance(source.cvs, HasLength) else source.cvs
        )
    if isinstance(source, CSwap):
        return ControlledSwapCall(source.bitsize)
    if isinstance(source, TwoBitCSwap):
        return FixedTCall("fredkin")
    if isinstance(source, (Split, Join)):
        return BookkeepingCall(
            "split" if isinstance(source, Split) else "join", source.dtype.num_qubits
        )
    raise AssertionError(f"Unmapped source cache key {source!r}")


@pytest.mark.parametrize("count", [2, 3, 4, 5])
@pytest.mark.parametrize("adjoint", [False, True])
def test_all_small_polarities_have_exact_source_width_caches(
    count: int, adjoint: bool
) -> None:
    from qualtran.bloqs.mcmt import And, MultiAnd
    from qualtran.resource_counting import QubitCount, get_cost_value

    for values in product((0, 1), repeat=count):
        source: Any = And(*values) if count == 2 else MultiAnd(cvs=values)
        if adjoint:
            source = source.adjoint()
        expected_cache: dict[Any, SymbolicInt] = {}
        expected = get_cost_value(source, QubitCount(), costs_cache=expected_cache)
        cache: dict[ResourceCall, SymbolicInt] = {}
        root = ConjunctionCall(values, adjoint)
        assert (
            query_resource_logical_width(root, cache=cache) == expected == 2 * count - 1
        )
        assert cache == {
            native_source_key(key): value for key, value in expected_cache.items()
        }
        assert list(cache) == [native_source_key(key) for key in expected_cache]


@pytest.mark.parametrize(
    "width", [1, 2, 4, Integer(2), Symbol("n", positive=True, integer=True)]
)
@pytest.mark.parametrize("adjoint", [False, True])
def test_exact_swap_complete_width_cache_matches_source(
    width: SymbolicInt, adjoint: bool
) -> None:
    from qualtran.bloqs.basic_gates import CSwap
    from qualtran.resource_counting import QubitCount, get_cost_value

    source = CSwap(width)
    if adjoint:
        source = source.adjoint()
    expected_cache: dict[Any, SymbolicInt] = {}
    expected = get_cost_value(source, QubitCount(), costs_cache=expected_cache)
    root = ControlledSwapCall(width, adjoint=adjoint)
    cache: dict[ResourceCall, SymbolicInt] = {}
    assert query_resource_logical_width(root, cache=cache) == expected
    mapped = {native_source_key(key): value for key, value in expected_cache.items()}
    # Exact exchange is self-adjoint in the source. Native identity keeps the
    # caller's direction field without changing its decomposition children.
    mapped[root] = mapped.pop(native_source_key(source))
    assert cache == mapped


@pytest.mark.parametrize("count", [3, 5, Symbol("n", positive=True, integer=True)])
@pytest.mark.parametrize("adjoint", [False, True])
def test_unspecified_control_count_preserves_source_fallback(
    count: SymbolicInt, adjoint: bool
) -> None:
    from qualtran.bloqs.mcmt import MultiAnd
    from qualtran.resource_counting import QubitCount, get_cost_value
    from qualtran.symbolics import HasLength

    source: Any = MultiAnd(cvs=HasLength(count))
    if adjoint:
        source = source.adjoint()
    source_cache: dict[Any, SymbolicInt] = {}
    expected = get_cost_value(source, QubitCount(), costs_cache=source_cache)
    root = ConjunctionCall(count, adjoint)
    cache: dict[ResourceCall, SymbolicInt] = {}
    assert query_resource_logical_width(root, cache=cache) == expected
    expected_native = {
        native_source_key(key): value for key, value in source_cache.items()
    }
    # HasLength(concrete int) is normalized to an all-positive source tuple.
    expected_native[root] = expected_native.pop(native_source_key(source))
    assert cache == expected_native


@pytest.mark.parametrize(
    "family",
    [
        "and-positive",
        "and-negative",
        "and-adjoint",
        "symbolic-and",
        "swap",
        "swap-adjoint",
        "symbolic-swap",
    ],
)
@pytest.mark.parametrize("depth", [0, 1, None])
@pytest.mark.parametrize("aggregation", get_args(GateAggregation))
def test_automatic_factory_full_dot_matches_source_width_rows(
    family: str, depth: int | None, aggregation: GateAggregation
) -> None:
    from qualtran.bloqs.basic_gates import CSwap
    from qualtran.bloqs.mcmt import MultiAnd
    from qualtran.resource_counting import QubitCount, get_cost_value
    from qualtran.symbolics import HasLength

    n = Symbol("n", positive=True, integer=True)
    if "and" in family:
        source: Any = MultiAnd(
            cvs=(
                HasLength(n)
                if family == "symbolic-and"
                else (0, 1, 0, 1) if family == "and-negative" else (1, 1, 1, 1)
            )
        )
    else:
        source = CSwap(n if family == "symbolic-swap" else 3)
    if "adjoint" in family:
        source = source.adjoint()
    root = native_source_key(source)
    source_cache: dict[Any, SymbolicInt] = {}
    get_cost_value(source, QubitCount(), costs_cache=source_cache)
    widths = {native_source_key(key): value for key, value in source_cache.items()}
    assert export_resource_call_graph_dot(
        root, max_depth=depth, aggregation=aggregation
    ) == export_resource_call_graph_dot(
        root, logical_width=widths.get, max_depth=depth, aggregation=aggregation
    )


def test_decomposition_width_cache_is_independent_from_gate_rows() -> None:
    root = ConjunctionCall((0, 1, 0, 1))
    unrelated = FixedTCall("fredkin")
    rows: dict[ResourceCall, dict[ResourceCall, SymbolicInt]] = {root: {unrelated: 100}}
    cache: dict[ResourceCall, SymbolicInt] = {}
    assert query_resource_logical_width(root, declared_calls=rows, cache=cache) == 7
    assert unrelated not in cache and ConjunctionCall((0, 1)) in cache
    assert rows == {root: {unrelated: 100}}


@pytest.mark.parametrize("width", [0, -1])
def test_invalid_exact_swap_width_fails_without_cache_entry(width: int) -> None:
    from qualtran.bloqs.basic_gates import CSwap
    from qualtran.resource_counting import QubitCount, get_cost_value

    with pytest.raises((RuntimeError, ValueError)):
        get_cost_value(CSwap(width), QubitCount())
    cache: dict[ResourceCall, SymbolicInt] = {}
    with pytest.raises(ValueError):
        query_resource_logical_width(ControlledSwapCall(width), cache=cache)
    assert cache == {}


@pytest.mark.parametrize("controls", [(), (1,), (0, 2, 1), (None, 1, 1)])
def test_unavailable_control_connections_fail_closed(
    controls: tuple[int | None, ...],
) -> None:
    with pytest.raises((ValueError, NotImplementedError)):
        query_resource_logical_width(ConjunctionCall(controls))


def test_approximate_swap_supports_automatic_and_explicit_provider() -> None:
    root = ControlledSwapCall(3, approximate=True)
    assert query_resource_logical_width(root) == 7
    assert "7" in export_resource_call_graph_dot(root)
    assert "7" in export_resource_call_graph_dot(root, logical_width=lambda node: 7)


@pytest.mark.parametrize("family", ["ladder", "ladder-adjoint", "swap", "swap-split"])
def test_seeded_child_peak_is_combined_with_actual_live_bystanders(family: str) -> None:
    from qualtran import QAny
    from qualtran.bloqs.basic_gates import CSwap, TwoBitCSwap
    from qualtran.bloqs.bookkeeping import Split
    from qualtran.bloqs.mcmt import And, MultiAnd
    from qualtran.resource_counting import QubitCount, get_cost_value

    if family.startswith("ladder"):
        source: Any = MultiAnd(cvs=(0, 1, 0, 1))
        seed: Any = And(0, 1)
        if family == "ladder-adjoint":
            source, seed = source.adjoint(), seed.adjoint()
    else:
        source = CSwap(3)
        seed = Split(QAny(3)) if family == "swap-split" else TwoBitCSwap()
    source_cache: dict[Any, SymbolicInt] = {seed: 8}
    expected = get_cost_value(source, QubitCount(), costs_cache=source_cache)
    native_seed = native_source_key(seed)
    cache: dict[ResourceCall, SymbolicInt] = {native_seed: 8}
    assert (
        query_resource_logical_width(native_source_key(source), cache=cache) == expected
    )
    assert cache == {
        native_source_key(key): value for key, value in source_cache.items()
    }
    assert expected > 7


@pytest.mark.parametrize("count", [0, 1, True])
def test_invalid_unspecified_control_count_is_rejected(count: int) -> None:
    cache: dict[ResourceCall, SymbolicInt] = {}
    with pytest.raises(ValueError, match="at least two"):
        query_resource_logical_width(ConjunctionCall(count), cache=cache)
    assert cache == {}
