"""Compare native presentation rules with the unmodified source label helpers."""

from dataclasses import dataclass, fields, make_dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sympy import Symbol

from flagquantum.compiler._component_trace.resource_generalization import (
    ArithmeticCall,
    BookkeepingCall,
    ConjunctionCall,
    ControlledRotationCall,
    ControlledSwapCall,
    FixedTCall,
    OnEachCall,
    PhaseCall,
    PowerCall,
    ResourceCall,
    RotationCall,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost
from flagquantum.drawer.flame_graph import (
    _resource_arg,
    _resource_title,
    iter_resource_flame_graph_data,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    root = Path(__file__).resolve().parents[3] / "ftqc-reference-env/qualtran"
    for path, blob in (
        ("drawing/flame_graph.py", "33e09396e60d1775d7da5d825a6bed040a5ab55e"),
        ("serialization/bloq.py", "132930b2804536155f5b242dbede88518fcca459"),
    ):
        raw = (root / path).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


@pytest.mark.parametrize(
    "value",
    [
        (),
        (0,),
        (0, 1, 0),
        np.empty(()),
        np.empty((0, 3)),
        np.empty((2, 3, 4)),
        0.0,
        -0.0,
        1e-12,
        -1e-12,
        1e-8,
        -1e-8,
        np.nextafter(1e-8, np.inf),
        np.nextafter(-1e-8, -np.inf),
        0.123456,
        -12.345,
        1e20,
        float("inf"),
        float("-inf"),
        float("nan"),
        np.float64(0.123456),
        3,
        True,
        None,
        "native",
        Symbol("n", positive=True, integer=True),
    ],
)
def test_argument_matches_unmodified_source(value: Any) -> None:
    from qualtran.drawing.flame_graph import _pretty_arg

    assert _resource_arg(value) == _pretty_arg(value)


def source_metadata(call: ResourceCall) -> Any:
    """Exercise source reflection using native field names, without gate aliases.

    These are display-only metadata fixtures. They claim no source decomposition
    or foreign-object execution admission.
    """
    from qualtran import Bloq, Signature

    if isinstance(call, ComponentCost):
        raise AssertionError("Component names have their own native boundary")
    constructor = [field for field in fields(call) if field.init]
    arguments = [getattr(call, field.name) for field in constructor]
    arguments = [
        source_metadata(value) if isinstance(value, ResourceCall) else value
        for value in arguments
    ]
    reference_type = make_dataclass(
        type(call).__name__,
        [(field.name, object) for field in constructor],
        bases=(Bloq,),
        namespace={"signature": property(lambda _: Signature([]))},
        frozen=True,
        kw_only=True,
    )
    return reference_type(
        **dict(zip((field.name for field in constructor), arguments, strict=True))
    )


@pytest.mark.parametrize(
    "call",
    [
        RotationCall("rz", 0.123456, 1e-4),
        RotationCall("zpow", Symbol("theta"), 1e-11),
        PhaseCall("t", True),
        FixedTCall("fredkin"),
        ControlledSwapCall(3),
        ControlledSwapCall(3, approximate=True, adjoint=True),
        ConjunctionCall((0, 1, 0, 1), adjoint=True),
        ControlledRotationCall(RotationCall("rz", -1e-12), (0, 1)),
        OnEachCall(PhaseCall("t"), 4),
        PowerCall(OnEachCall(PhaseCall("t", True), 3), 5),
        BookkeepingCall("partition", ((2, 3), (1,))),
        ArithmeticCall("add", (3, 4)),
        ArithmeticCall("constant_add", (4,), constant=3, adjoint=True),
    ],
)
def test_native_constructor_rules_match_source_reflection(call: ResourceCall) -> None:
    from qualtran.drawing.flame_graph import _pretty_name

    assert _resource_title(call) == _pretty_name(source_metadata(call))
    assert "_python" not in _resource_title(call)
    assert "=" not in _resource_title(call)


@dataclass(frozen=True)
class NamedPhase(PhaseCall):
    def __str__(self) -> str:
        return "Custom phase"


def test_custom_string_and_component_name_precedence() -> None:
    assert _resource_title(NamedPhase("t")) == "Custom phase"
    assert _resource_title(ComponentCost("Declared root", 3)) == "Declared root"
    assert _resource_arg(ComponentCost("Nested root", 3)) == "Nested root"


@pytest.mark.parametrize("width", [1, 2, 3, 10])
@pytest.mark.parametrize("depth", [0, 1, 2, None])
def test_default_native_swap_workflow_matches_source_rows(
    width: int, depth: int | None
) -> None:
    from qualtran.bloqs.basic_gates import CSwap, TwoBitCSwap
    from qualtran.drawing.flame_graph import _pretty_name, get_flame_graph_data

    call = ControlledSwapCall(width)
    reference = get_flame_graph_data(CSwap(width), max_depth=depth)
    # Native resource records retain their class and constructor names.
    expected = [
        row.replace(
            _pretty_name(CSwap(width)), f"ControlledSwapCall[{width}][False][False]"
        ).replace(_pretty_name(TwoBitCSwap()), "FixedTCall[fredkin]")
        for row in reference
    ]
    assert list(iter_resource_flame_graph_data(call, max_depth=depth)) == expected
    assert sum(int(row.split("\t")[1]) for row in expected) == 7 * width


def test_colliding_compact_labels_preserve_distinct_graph_keys() -> None:
    root = ComponentCost("Controls", 5)
    positive = ConjunctionCall((1, 1, 1))
    negative = ConjunctionCall((0, 1, 0))
    assert positive != negative
    assert _resource_title(positive) == _resource_title(negative)
    rows = list(
        iter_resource_flame_graph_data(
            root, declared_calls={root: {positive: 2, negative: 3}}, max_depth=1
        )
    )
    assert rows == ["Controls(T:40);ConjunctionCall[3][False](T:8)\t8"] * 5
    custom = list(
        iter_resource_flame_graph_data(
            root,
            declared_calls={root: {positive: 2, negative: 3}},
            max_depth=1,
            title=str,
        )
    )
    assert "controls=(1, 1, 1)" in custom[0]
    assert "controls=(0, 1, 0)" in custom[-1]


@pytest.mark.parametrize("separator", [";", "\t", "\n", "\r"])
def test_default_and_custom_titles_reject_frame_separators(separator: str) -> None:
    root = ComponentCost("Root" + separator, 0)
    with pytest.raises(ValueError, match="frame separators"):
        list(iter_resource_flame_graph_data(root))
    with pytest.raises(ValueError, match="frame separators"):
        list(iter_resource_flame_graph_data(PhaseCall("t"), title=lambda _: separator))
