"""Compare approximate-swap and fanout width traversal with frozen source."""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any, get_args

import pytest
from sympy import Integer, Symbol

from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import (
    BookkeepingCall,
    CliffordCall,
    ControlledSwapCall,
    FanoutCall,
    PhaseCall,
    ResourceCall,
)
from flagquantum.compiler._component_trace.resource_width import (
    query_resource_logical_width,
)
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.compiler._component_trace.terminal_t_cost import query_terminal_t_cost
from flagquantum.compiler.registers.resources.through_call_width import (
    through_call_width,
)
from flagquantum.drawer.call_graph import (
    GateAggregation,
    export_resource_call_graph_dot,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    reference = Path(__file__).resolve().parents[3] / "ftqc-reference-env"
    with (reference.parent / "FlagQuantum-Qualtran-Kickmix-移植进度.csv").open(
        encoding="utf-8-sig"
    ) as stream:
        blobs = {row["源文件"]: row["源对象"] for row in csv.DictReader(stream)}
    for name in [
        "qualtran/resource_counting/_qubit_counts.py",
        "qualtran/bloqs/mcmt/multi_target_cnot.py",
        "qualtran/bloqs/swap_network/cswap_approx.py",
    ]:
        raw = (reference / name).read_bytes()
        assert (
            hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blobs[name]
        )


def source_key(source: Any) -> ResourceCall:
    from qualtran._infra.adjoint import Adjoint
    from qualtran.bloqs.basic_gates import CNOT, Hadamard, SGate, TGate
    from qualtran.bloqs.bookkeeping import ArbitraryClifford, Join, Split
    from qualtran.bloqs.mcmt import MultiTargetCNOT
    from qualtran.bloqs.swap_network.cswap_approx import CSwapApprox

    if isinstance(source, Adjoint):
        key = source_key(source.subbloq)
        if isinstance(key, FanoutCall):
            return FanoutCall(key.width, True)
        if isinstance(key, ControlledSwapCall):
            return ControlledSwapCall(key.width, approximate=True, adjoint=True)
        assert isinstance(key, CliffordCall)
        return CliffordCall(key.opcode, key.width, True)
    if isinstance(source, CSwapApprox):
        return ControlledSwapCall(source.bitsize, approximate=True)
    if isinstance(source, MultiTargetCNOT):
        return FanoutCall(source.bitsize)
    if isinstance(source, (Split, Join)):
        return BookkeepingCall(
            "split" if isinstance(source, Split) else "join", source.dtype.num_qubits
        )
    if isinstance(source, TGate):
        return PhaseCall("t", source.is_adjoint)
    if isinstance(source, SGate):
        return PhaseCall("s", source.is_adjoint)
    if isinstance(source, CNOT):
        return CliffordCall("cx", 2)
    if isinstance(source, Hadamard):
        return CliffordCall("h", 1)
    assert isinstance(source, ArbitraryClifford)
    return CliffordCall("arbitrary", source.n)


@pytest.mark.parametrize(
    "width", [1, 2, 3, 5, 8, Integer(2), Symbol("n", positive=True, integer=True)]
)
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("approximate", [False, True])
def test_complete_width_cache_matches_source(
    width: SymbolicInt, adjoint: bool, approximate: bool
) -> None:
    from qualtran.bloqs.mcmt import MultiTargetCNOT
    from qualtran.bloqs.swap_network.cswap_approx import CSwapApprox
    from qualtran.resource_counting import QubitCount, get_cost_value

    source: Any = CSwapApprox(width) if approximate else MultiTargetCNOT(width)
    if adjoint:
        source = source.adjoint()
    root = source_key(source)
    expected_cache: dict[Any, SymbolicInt] = {}
    expected = get_cost_value(source, QubitCount(), costs_cache=expected_cache)
    cache: dict[ResourceCall, SymbolicInt] = {}
    assert query_resource_logical_width(root, cache=cache) == expected
    assert cache == {source_key(key): value for key, value in expected_cache.items()}
    assert list(cache) == [source_key(key) for key in expected_cache]
    if approximate and type(width) is int:
        assert FanoutCall(width, adjoint) in cache
        assert CliffordCall("arbitrary", 1) not in cache
    if type(width) is int and width == 1:
        assert BookkeepingCall("split", 1) not in cache


@pytest.mark.parametrize("width", [1, 3, Symbol("n", positive=True, integer=True)])
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("depth", [0, 1, None])
@pytest.mark.parametrize("aggregation", get_args(GateAggregation))
def test_factory_dot_matches_unmodified_source_width_table(
    width: SymbolicInt, adjoint: bool, depth: int | None, aggregation: GateAggregation
) -> None:
    from qualtran.bloqs.swap_network.cswap_approx import CSwapApprox
    from qualtran.resource_counting import QubitCount, get_cost_value

    source: Any = CSwapApprox(width)
    if adjoint:
        source = source.adjoint()
    cache: dict[Any, SymbolicInt] = {}
    get_cost_value(source, QubitCount(), costs_cache=cache)
    widths = {source_key(key): value for key, value in cache.items()}
    root = source_key(source)
    assert export_resource_call_graph_dot(
        root, max_depth=depth, aggregation=aggregation
    ) == export_resource_call_graph_dot(
        root, logical_width=widths.get, max_depth=depth, aggregation=aggregation
    )


@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize(
    "seed_name", ["split", "join", "fanout", "cx", "s", "t-adjoint"]
)
def test_hidden_local_peaks_and_bystanders_match_source(
    seed_name: str, adjoint: bool
) -> None:
    from qualtran import QAny
    from qualtran.bloqs.basic_gates import CNOT, SGate, TGate
    from qualtran.bloqs.bookkeeping import Join, Split
    from qualtran.bloqs.mcmt import MultiTargetCNOT
    from qualtran.bloqs.swap_network.cswap_approx import CSwapApprox
    from qualtran.resource_counting import QubitCount, get_cost_value

    source: Any = CSwapApprox(3)
    fanout: Any = MultiTargetCNOT(3)
    if adjoint:
        source, fanout = source.adjoint(), fanout.adjoint()
    seeds: dict[str, Any] = {
        "split": Split(QAny(3)),
        "join": Join(QAny(3)),
        "fanout": fanout,
        "cx": CNOT(),
        "s": SGate(),
        "t-adjoint": TGate(is_adjoint=True),
    }
    seed = seeds[seed_name]
    expected_cache: dict[Any, SymbolicInt] = {seed: 12}
    expected = get_cost_value(source, QubitCount(), costs_cache=expected_cache)
    actual: dict[ResourceCall, SymbolicInt] = {source_key(seed): 12}
    assert query_resource_logical_width(source_key(source), cache=actual) == expected
    assert actual == {source_key(key): value for key, value in expected_cache.items()}
    assert expected > 7


@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_fanout_adjoint_costs_match_source_and_keep_identity(
    adjoint: bool, legacy: bool
) -> None:
    from qualtran.bloqs.mcmt import MultiTargetCNOT
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source: Any = MultiTargetCNOT(5)
    if adjoint:
        source = source.adjoint()
    root = FanoutCall(5, adjoint)
    expected = get_cost_value(source, QECGatesCost(legacy_shims=legacy))
    counts = asdict(query_qec_gate_cost(root, legacy_shims=legacy))
    assert {key: value for key, value in counts.items() if value} == expected.asdict()
    assert query_terminal_t_cost(root) == 0
    assert FanoutCall(5) != FanoutCall(5, True)
    assert "6" in export_resource_call_graph_dot(root)


@pytest.mark.parametrize("approximate", [False, True])
@pytest.mark.parametrize("adjoint", [False, True])
def test_zero_width_failure_and_retry_match_source(
    approximate: bool, adjoint: bool
) -> None:
    from qualtran.bloqs.mcmt import MultiTargetCNOT
    from qualtran.bloqs.swap_network.cswap_approx import CSwapApprox
    from qualtran.resource_counting import QubitCount, get_cost_value

    source: Any = CSwapApprox(0) if approximate else MultiTargetCNOT(0)
    if adjoint:
        source = source.adjoint()
    with pytest.raises(RuntimeError):
        get_cost_value(source, QubitCount())
    cache: dict[ResourceCall, SymbolicInt] = {}
    with pytest.raises(ValueError, match="nonempty"):
        query_resource_logical_width(source_key(source), cache=cache)
    assert cache == {}
    root: ResourceCall = (
        ControlledSwapCall(2, True, adjoint) if approximate else FanoutCall(2, adjoint)
    )
    assert query_resource_logical_width(root, cache=cache) == (5 if approximate else 3)


@pytest.mark.parametrize("wires", [(), (0, 0), (0, 2), (True,)])
def test_through_projection_rejects_invalid_wire_bindings(
    wires: tuple[int, ...],
) -> None:
    calls: Iterable[tuple[ResourceCall, tuple[int, ...]]] = [
        (CliffordCall("h", 1), wires)
    ]
    with pytest.raises(ValueError):
        through_call_width(2, calls, lambda node: 1)


def test_projection_rejects_unmodeled_emitter_instruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flagquantum.compiler.registers.primitives import (
        relative_controlled_swap_width as owner,
    )
    from flagquantum.core.ir import Instruction

    monkeypatch.setattr(
        owner,
        "_relative_swap_parts",
        lambda left, right: ((Instruction("x", (left[0],)),), ()),
    )
    cache: dict[ResourceCall, SymbolicInt] = {}
    with pytest.raises(ValueError, match="cannot bind gate"):
        query_resource_logical_width(
            ControlledSwapCall(1, approximate=True), cache=cache
        )
    assert cache == {}
