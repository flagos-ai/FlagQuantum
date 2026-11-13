"""Exercise native graph/precision composition against untouched source calls."""

from collections.abc import Callable
from hashlib import sha1
from pathlib import Path
from typing import Any, Literal

import pytest

from flagquantum.compiler._component_trace.resource_generalization import (
    ConjunctionCall,
    ControlledZPowerCall,
    FixedTCall,
    OnEachCall,
    PhaseCall,
    PowerCall,
    ResourceCall,
    RotationCall,
)
from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.compiler._component_trace.terminal_t_cost import query_terminal_t_cost
from flagquantum.drawer.flame_graph import iter_resource_flame_graph_data

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    reference = Path(__file__).resolve().parents[3] / "ftqc-reference-env"
    for path, expected in (
        ("drawing/flame_graph.py", "33e09396e60d1775d7da5d825a6bed040a5ab55e"),
        (
            "resource_counting/_call_graph.py",
            "8c1a6debd8c8a2e06d3a082dc6b6cd6656e4f522",
        ),
        (
            "resource_counting/t_counts_from_sigma.py",
            "eb4578e362849eb3e7719944bb7c01b2dc9adbc4",
        ),
    ):
        raw = (reference / "qualtran" / path).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == expected


def test_native_precision_workflow_multiple_roots_and_depth() -> None:
    root = ComponentCost("Root", 3)
    fredkin = FixedTCall("fredkin")
    rz = RotationCall("rz", 0.17, 1e-4)
    tdg = PhaseCall("t", adjoint=True)
    calls: dict[ResourceCall, dict[ResourceCall, SymbolicInt]] = {
        root: {fredkin: 2, rz: 3, tdg: 5}
    }
    assert query_terminal_t_cost(root, declared_calls=calls) == 94
    rows = list(
        iter_resource_flame_graph_data(root, tdg, declared_calls=calls, max_depth=0)
    )
    assert rows == ["Root(T:94)\t94", "PhaseCall[t][True](T:1)\t1"]
    assert list(iter_resource_flame_graph_data()) == []
    assert (
        sum(
            int(row.split("\t")[1])
            for row in iter_resource_flame_graph_data(root, declared_calls=calls)
        )
        == 94
    )


def test_native_component_calls_and_explicit_exclusive_override() -> None:
    leaf = ComponentCost("Leaf", 1, GateCost(rotation=1))
    root = ComponentCost("Root", 1, calls=(CostCall(leaf, 3),))
    with pytest.raises(ValueError, match="exclusive T-bearing"):
        query_terminal_t_cost(root)
    assert (
        query_terminal_t_cost(
            root, declared_calls={leaf: {RotationCall("rz", 0.1, 1e-4): 1}}
        )
        == 75
    )
    assert query_terminal_t_cost(root, declared_calls={leaf: {}}) == 0
    assert list(
        iter_resource_flame_graph_data(ComponentCost("Zero", 0, GateCost(clifford=1)))
    ) == ["Zero(T:0)\t0"]


@pytest.mark.parametrize(
    "gates",
    [
        GateCost(t=1),
        GateCost(toffoli=1),
        GateCost(cswap=1),
        GateCost(and_bloq=1),
        GateCost(rotation=1),
    ],
)
def test_exclusive_costs_require_explicit_precision_declarations(
    gates: GateCost,
) -> None:
    with pytest.raises(ValueError, match="exclusive T-bearing"):
        list(iter_resource_flame_graph_data(ComponentCost("Root", 3, gates)))


def test_cached_shared_dag_and_wrapped_multiplicities() -> None:
    root, shared = ComponentCost("Root", 1), ComponentCost("Shared", 1)
    t = PhaseCall("t")
    wrapper = PowerCall(OnEachCall(t, 3), 5)
    calls: dict[ResourceCall, dict[ResourceCall, SymbolicInt]] = {
        root: {shared: 2, wrapper: 1},
        shared: {t: 7},
    }
    cache: dict[ResourceCall, SymbolicInt] = {}
    assert query_terminal_t_cost(root, declared_calls=calls, cache=cache) == 29
    assert cache[shared] == 7 and cache[wrapper] == 15
    before = dict(cache)
    assert query_terminal_t_cost(shared, declared_calls=calls, cache=cache) == 7
    assert cache == before
    rows = list(iter_resource_flame_graph_data(root, declared_calls=calls, max_depth=1))
    assert rows[:2] == ["Root(T:29);Shared(T:7)\t7"] * 2
    assert len(rows) == 3


def test_cycles_checked_even_when_display_stops_at_root() -> None:
    root = ComponentCost("Cycle", 0)
    with pytest.raises(ValueError, match="cycle"):
        list(
            iter_resource_flame_graph_data(
                root, declared_calls={root: {root: 1}}, max_depth=0
            )
        )


@pytest.mark.parametrize("count", [-1, True, 1.5])
def test_nonconcrete_display_counts_rejected(count: Any) -> None:
    root = ComponentCost("Root", 0)
    with pytest.raises(ValueError, match="counts"):
        list(
            iter_resource_flame_graph_data(
                root, declared_calls={root: {PhaseCall("t"): count}}
            )
        )


def test_symbolic_query_remains_available_before_rendering() -> None:
    from sympy import symbols

    n = symbols("n", positive=True, integer=True)
    root = PowerCall(PhaseCall("t"), n)
    assert query_terminal_t_cost(root) == n
    with pytest.raises(ValueError, match="concrete"):
        list(iter_resource_flame_graph_data(root))


def native_source_rows(
    bloq: Any, *, depth: int | None, default_keep: bool
) -> list[str]:
    """Translate source identities and declarations, never source cost totals."""
    from qualtran.bloqs.basic_gates import (
        CZPowGate,
        Rx,
        Ry,
        Rz,
        TGate,
        Toffoli,
        TwoBitCSwap,
        XPowGate,
        YPowGate,
        ZPowGate,
    )
    from qualtran.bloqs.mcmt.and_bloq import And
    from qualtran.drawing.flame_graph import _pretty_name

    opcodes: dict[type[Any], Literal["rx", "ry", "rz", "xpow", "ypow", "zpow"]] = {
        Rx: "rx",
        Ry: "ry",
        Rz: "rz",
        XPowGate: "xpow",
        YPowGate: "ypow",
        ZPowGate: "zpow",
    }
    full, _ = bloq.call_graph()
    keys: dict[Any, ResourceCall] = {}
    for index, node in enumerate(full):
        key: ResourceCall = ComponentCost(f"Node{index}", 0)
        if isinstance(node, TGate):
            key = PhaseCall("t", adjoint=node.is_adjoint)
        elif isinstance(node, (Toffoli, TwoBitCSwap)):
            key = FixedTCall("toffoli" if isinstance(node, Toffoli) else "fredkin")
        elif isinstance(node, And):
            key = ConjunctionCall((node.cv1, node.cv2), adjoint=node.uncompute)
        elif isinstance(node, CZPowGate):
            key = ControlledZPowerCall(node.exponent, node.eps)
        elif isinstance(node, (Rx, Ry, Rz)):
            opcode = opcodes[type(node)]
            key = RotationCall(opcode, node.angle, node.eps)
        elif isinstance(node, (XPowGate, YPowGate, ZPowGate)):
            opcode = opcodes[type(node)]
            key = RotationCall(opcode, node.exponent, node.eps)
        keys[node] = key
    calls: dict[ResourceCall, dict[ResourceCall, SymbolicInt]] = {
        keys[node]: {keys[child]: edge["n"] for child, edge in full[node].items()}
        for node in full
    }
    names = {keys[node]: _pretty_name(node) for node in full}
    kwargs: dict[str, Any] = {} if default_keep else {"keep": None}
    return list(
        iter_resource_flame_graph_data(
            keys[bloq],
            declared_calls=calls,
            title=names.__getitem__,
            max_depth=depth,
            **kwargs,
        )
    )


@pytest.mark.parametrize("depth", [None, 0, 1, 2])
@pytest.mark.parametrize("default_keep", [False, True])
@pytest.mark.parametrize("example", ["cswap", "multiand", "uniform", "qft"])
def test_native_facade_matches_source_calls(
    example: str, depth: int | None, default_keep: bool
) -> None:
    from qualtran.bloqs.basic_gates.swap import CSwap
    from qualtran.bloqs.mcmt import MultiAnd
    from qualtran.bloqs.qft.qft_text_book import QFTTextBook
    from qualtran.bloqs.state_preparation import PrepareUniformSuperposition
    from qualtran.drawing.flame_graph import _keep_if_small, get_flame_graph_data

    factories: dict[str, Callable[[], Any]] = {
        "cswap": lambda: CSwap(10),
        "multiand": lambda: MultiAnd((0, 1) * 6),
        "uniform": lambda: PrepareUniformSuperposition(12),
        "qft": lambda: QFTTextBook(5),
    }
    bloq = factories[example]()
    expected = get_flame_graph_data(
        bloq, max_depth=depth, keep=_keep_if_small if default_keep else None
    )
    assert native_source_rows(bloq, depth=depth, default_keep=default_keep) == expected
