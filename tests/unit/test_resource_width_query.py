"""Verify automatic native width binding and drawing against frozen source."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any, get_args

import pytest
from sympy import Integer, Symbol

from flagquantum.compiler._component_trace.resource_generalization import (
    BookkeepingCall,
    CliffordCall,
    FixedTCall,
    OnEachCall,
    PhaseCall,
    PowerCall,
    ResourceCall,
    RotationCall,
    ZeroCostCall,
    identity_cost_key,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost, CostCall
from flagquantum.compiler._component_trace.resource_width import (
    query_resource_logical_width,
)
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.drawer.call_graph import (
    GateAggregation,
    export_resource_call_graph_dot,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def source_atoms() -> dict[ResourceCall, Any]:
    from qualtran.bloqs.basic_gates import (
        CNOT,
        CZ,
        CYGate,
        Hadamard,
        Identity,
        SGate,
        TGate,
        Toffoli,
        TwoBitCSwap,
        TwoBitSwap,
        XGate,
        YGate,
        ZGate,
    )
    from qualtran.bloqs.bookkeeping import ArbitraryClifford

    reference = Path(__file__).resolve().parents[3] / "ftqc-reference-env"
    with (reference.parent / "FlagQuantum-Qualtran-Kickmix-移植进度.csv").open(
        encoding="utf-8-sig"
    ) as stream:
        blobs = {row["源文件"]: row["源对象"] for row in csv.DictReader(stream)}
    for name in [
        "qualtran/resource_counting/_qubit_counts.py",
        "qualtran/bloqs/basic_gates/on_each.py",
        "qualtran/bloqs/basic_gates/power.py",
    ]:
        raw = (reference / name).read_bytes()
        assert (
            hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blobs[name]
        )
    n = Symbol("n", positive=True, integer=True)
    atoms: dict[ResourceCall, Any] = {
        CliffordCall("h", 1): Hadamard(),
        CliffordCall("x", 1): XGate(),
        CliffordCall("y", 1): YGate(),
        CliffordCall("z", 1): ZGate(),
        CliffordCall("cx", 2): CNOT(),
        CliffordCall("cy", 2): CYGate(),
        CliffordCall("cz", 2): CZ(),
        CliffordCall("swap_bit", 2): TwoBitSwap(),
        FixedTCall("toffoli"): Toffoli(),
        FixedTCall("fredkin"): TwoBitCSwap(),
        PhaseCall("s"): SGate(),
        PhaseCall("t"): TGate(),
        PhaseCall("s", True): SGate().adjoint(),
        PhaseCall("t", True): TGate().adjoint(),
    }
    for width in (0, 1, 5, n):
        atoms[identity_cost_key(width)] = Identity(width)
        for adjoint in (False, True):
            source = ArbitraryClifford(width)
            atoms[CliffordCall("arbitrary", width, adjoint)] = (
                source.adjoint() if adjoint else source
            )
    for name in {
        type(source).__module__.replace(".", "/") + ".py" for source in atoms.values()
    }:
        raw = (reference / name).read_bytes()
        assert (
            hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blobs[name]
        )
    return atoms


@pytest.mark.ftqc_reference
def test_all_atomic_widths_and_caches_match_source(
    source_atoms: dict[ResourceCall, Any],
) -> None:
    from qualtran.resource_counting import QubitCount, get_cost_value

    for node, source in source_atoms.items():
        actual: dict[ResourceCall, SymbolicInt] = {}
        expected: dict[Any, SymbolicInt] = {}
        assert query_resource_logical_width(node, cache=actual) == get_cost_value(
            source, QubitCount(), costs_cache=expected
        )
        assert actual == {node: expected[source]}
        assert len(expected) == 1


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "family",
    ["h", "t-adjoint", "identity", "symbolic", "literal-zero", "symbolic-zero", "one"],
)
@pytest.mark.parametrize("depth", [0, 1, None])
@pytest.mark.parametrize("aggregation", get_args(GateAggregation))
def test_automatic_factory_matches_source_width_rows(
    family: str,
    depth: int | None,
    aggregation: GateAggregation,
    source_atoms: dict[ResourceCall, Any],
) -> None:
    from qualtran import QAny
    from qualtran.bloqs.basic_gates import OnEach, Power
    from qualtran.bloqs.bookkeeping import Join, Split
    from qualtran.resource_counting import QubitCount, get_cost_value

    n = Symbol("n", positive=True, integer=True)
    leaves: dict[str, ResourceCall] = {
        "h": CliffordCall("h", 1),
        "t-adjoint": PhaseCall("t", True),
        "identity": identity_cost_key(1),
        "symbolic": CliffordCall("h", 1),
        "literal-zero": CliffordCall("h", 1),
        "symbolic-zero": CliffordCall("h", 1),
        "one": CliffordCall("h", 1),
    }
    leaf = leaves[family]
    widths: dict[str, SymbolicInt] = {
        "symbolic": n,
        "literal-zero": 0,
        "symbolic-zero": Integer(0),
        "one": 1,
    }
    width = widths.get(family, 4)
    each = OnEachCall(leaf, width)
    root = each if family == "symbolic" else PowerCall(each, 3)
    source_each = OnEach(width, source_atoms[leaf])
    source = source_each if family == "symbolic" else Power(source_each, 3)
    native_to_source: dict[ResourceCall, Any] = {
        root: source,
        each: source_each,
        leaf: source_atoms[leaf],
    }
    if type(width) is int:
        native_to_source.update(
            {
                BookkeepingCall("split", width): Split(QAny(width)),
                BookkeepingCall("join", width): Join(QAny(width)),
            }
        )
        if width == 0:
            del native_to_source[leaf]
    source_cache: dict[Any, SymbolicInt] = {}
    peak = get_cost_value(source, QubitCount(), costs_cache=source_cache)
    expected_cache = {
        key: source_cache[value] for key, value in native_to_source.items()
    }
    actual: dict[ResourceCall, SymbolicInt] = {}
    assert query_resource_logical_width(root, cache=actual) == peak
    assert actual == expected_cache
    assert set(source_cache) == set(native_to_source.values())
    automatic = export_resource_call_graph_dot(
        root, max_depth=depth, aggregation=aggregation
    )
    explicit = export_resource_call_graph_dot(
        root, logical_width=expected_cache.get, max_depth=depth, aggregation=aggregation
    )
    assert automatic == explicit


def test_component_bystanders_use_owned_width_despite_gate_override() -> None:
    leaf = ComponentCost("leaf", 8)
    root = ComponentCost("root", 2, calls=(CostCall(leaf, 100, 5),))
    extra = ComponentCost("only gate graph", 99)
    rows: dict[ResourceCall, dict[ResourceCall, SymbolicInt]] = {root: {extra: 1}}
    before = {key: dict(value) for key, value in rows.items()}
    cache: dict[ResourceCall, SymbolicInt] = {}
    assert query_resource_logical_width(root, declared_calls=rows, cache=cache) == 13
    assert cache == {leaf: 8, root: 13}
    assert rows == before
    assert export_resource_call_graph_dot(
        root, declared_calls=rows
    ) == export_resource_call_graph_dot(
        root, declared_calls=rows, logical_width=cache.get
    )


def test_component_zero_calls_remain_inactive_and_cached_values_win() -> None:
    leaf = ComponentCost("leaf", 100)
    root = ComponentCost("root", 2, calls=(CostCall(leaf, 0, 5),))
    cache: dict[ResourceCall, SymbolicInt] = {}
    assert query_resource_logical_width(root, cache=cache) == 2
    assert cache == {root: 2}
    assert query_resource_logical_width(root, cache={root: 9}) == 9


@pytest.mark.ftqc_reference
def test_each_hidden_workspace_counts_all_bystanders() -> None:
    from qualtran import Bloq, DecomposeNotImplementedError, Signature
    from qualtran.bloqs.basic_gates import Identity, OnEach
    from qualtran.resource_counting import QubitCount, get_cost_value

    def signature(self: Any) -> Any:
        return Signature.build(q=1)

    def decompose(self: Any) -> Any:
        raise DecomposeNotImplementedError()

    def calls(self: Any, ssa: Any) -> dict[Any, int]:
        return {Identity(8): 1}

    source_leaf: Any = type(
        "HiddenWorkspace",
        (Bloq,),
        {
            "signature": property(signature),
            "decompose_bloq": decompose,
            "build_call_graph": calls,
        },
    )()
    leaf = ComponentCost("hidden", 1, calls=(CostCall(ComponentCost("work", 8)),))
    root = OnEachCall(leaf, 4)
    assert (
        query_resource_logical_width(root)
        == get_cost_value(OnEach(4, source_leaf), QubitCount())
        == 11
    )


@pytest.mark.ftqc_reference
def test_symbolic_power_failed_width_boundary_matches_source() -> None:
    from qualtran.bloqs.basic_gates import Hadamard, Power
    from qualtran.resource_counting import QubitCount, get_cost_value

    n = Symbol("n", positive=True, integer=True)
    with pytest.raises((RuntimeError, ValueError)):
        get_cost_value(Power(Hadamard(), n), QubitCount())
    with pytest.raises(ValueError):
        query_resource_logical_width(PowerCall(CliffordCall("h", 1), n))


@pytest.mark.parametrize(
    "root",
    [
        RotationCall("rz", 0.125),
        ZeroCostCall("identity"),
        ZeroCostCall("identity", ("bad",)),
        BookkeepingCall("split", (2, 3)),
    ],
)
def test_unsupported_binding_fails_and_explicit_provider_remains_available(
    root: ResourceCall,
) -> None:
    with pytest.raises(NotImplementedError, match="logical width"):
        query_resource_logical_width(root)
    if isinstance(root, RotationCall):
        with pytest.raises(NotImplementedError, match="logical width"):
            export_resource_call_graph_dot(root, max_depth=0)
        assert "7" in export_resource_call_graph_dot(root, logical_width=lambda node: 7)


@pytest.mark.parametrize(
    "node",
    [
        CliffordCall("h", 2),
        ZeroCostCall("identity", (-1,)),
        OnEachCall(CliffordCall("cx", 2), 3),
    ],
)
def test_invalid_native_signature_is_rejected(node: ResourceCall) -> None:
    with pytest.raises(ValueError):
        query_resource_logical_width(node)


def test_fallback_gate_rows_visit_zero_multiplicity_children() -> None:
    root = PhaseCall("t")
    child = CliffordCall("arbitrary", 8)
    cache: dict[ResourceCall, SymbolicInt] = {}
    assert (
        query_resource_logical_width(
            root, declared_calls={root: {child: 0}}, cache=cache
        )
        == 8
    )
    assert cache == {child: 8, root: 8}


@pytest.mark.ftqc_reference
def test_each_of_power_preserves_nested_width_visitation() -> None:
    from qualtran.bloqs.basic_gates import Hadamard, OnEach, Power
    from qualtran.resource_counting import QubitCount, get_cost_value

    leaf = CliffordCall("h", 1)
    power = PowerCall(leaf, 2)
    root = OnEachCall(power, 3)
    actual: dict[ResourceCall, SymbolicInt] = {}
    source_cache: dict[Any, SymbolicInt] = {}
    expected = get_cost_value(
        OnEach(3, Power(Hadamard(), 2)), QubitCount(), costs_cache=source_cache
    )
    assert query_resource_logical_width(root, cache=actual) == expected == 3
    assert actual == {
        BookkeepingCall("split", 3): 3,
        leaf: 1,
        power: 1,
        BookkeepingCall("join", 3): 3,
        root: 3,
    }
    assert len(actual) == len(source_cache)
