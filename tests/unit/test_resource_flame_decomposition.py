"""Compare automatically expanded native resource roots with frozen source graphs."""

import csv
from hashlib import sha1
from math import pi
from pathlib import Path
from typing import Any

import pytest

from flagquantum.compiler._component_trace.call_graph import _build_call_graph
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import (
    ArithmeticCall,
    BookkeepingCall,
    CliffordCall,
    ConjunctionCall,
    ControlledRegisterSwapCall,
    ControlledRotationCall,
    ControlledSwapCall,
    ControlledZPowerCall,
    ControlledZRotationCall,
    ControlPredicateCall,
    MergedFredkinCall,
    PartitionedCall,
    PhaseCall,
    PredicateControlledCall,
    ResourceCall,
    RotationCall,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost
from flagquantum.compiler._component_trace.selected_call_count import (
    count_selected_calls,
)
from flagquantum.compiler._component_trace.terminal_t_cost import (
    _terminal_resource_callees,
    query_terminal_t_cost,
)
from flagquantum.drawer.flame_graph import iter_resource_flame_graph_data

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
REFERENCE = Path(__file__).resolve().parents[3] / "ftqc-reference-env"


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    with (REFERENCE.parent / "FlagQuantum-Qualtran-Kickmix-移植进度.csv").open(
        encoding="utf-8-sig"
    ) as stream:
        blobs = {row["源文件"]: row["源对象"] for row in csv.DictReader(stream)}
    for relative in (
        "drawing/flame_graph.py",
        "resource_counting/_bloq_counts.py",
        "resource_counting/t_counts_from_sigma.py",
        "bloqs/basic_gates/rotation.py",
        "bloqs/basic_gates/swap.py",
        "bloqs/mcmt/and_bloq.py",
        "bloqs/swap_network/cswap_approx.py",
    ):
        raw = (REFERENCE / "qualtran" / relative).read_bytes()
        assert (
            sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
            == blobs["qualtran/" + relative]
        )


def source_call(call: ResourceCall) -> Any:
    """Translate native identities only, never source costs or decomposition rows."""
    from qualtran import CtrlSpec
    from qualtran.bloqs.basic_gates import (
        CRz,
        CSwap,
        CZPowGate,
        TGate,
        Toffoli,
        TwoBitCSwap,
        ZPowGate,
    )
    from qualtran.bloqs.mcmt import And, MultiAnd
    from qualtran.bloqs.mcmt.ctrl_spec_and import CtrlSpecAnd
    from qualtran.bloqs.swap_network import CSwapApprox
    from qualtran.symbolics import HasLength

    if isinstance(call, ControlledSwapCall):
        source = CSwapApprox(call.width) if call.approximate else CSwap(call.width)
        return source.adjoint() if call.adjoint else source
    if isinstance(call, ConjunctionCall):
        if isinstance(call.controls, tuple) and len(call.controls) == 2:
            return And(*call.controls, uncompute=call.adjoint)
        source = MultiAnd(
            call.controls
            if isinstance(call.controls, tuple)
            else HasLength(call.controls)
        )
        return source.adjoint() if call.adjoint else source
    if isinstance(call, ControlledZPowerCall):
        return CZPowGate(call.exponent, eps=call.precision)
    if isinstance(call, ControlledZRotationCall):
        return CRz(call.angle, eps=call.precision)
    if isinstance(call, PhaseCall):
        return TGate(is_adjoint=call.adjoint)
    if isinstance(call, RotationCall):
        assert call.opcode == "zpow"
        return ZPowGate(call.parameter, eps=call.precision)
    if isinstance(call, MergedFredkinCall):
        return TwoBitCSwap().controlled(CtrlSpec(cvs=call.controls[:-1]))
    if isinstance(call, ControlledRegisterSwapCall):
        return CSwap(call.width).controlled()
    if isinstance(call, ControlPredicateCall):
        source = CtrlSpecAnd(CtrlSpec(cvs=call.controls))
        return source.adjoint() if call.adjoint else source
    # FixedTCall is kept separate because the primitive Fredkin has its own name.
    from flagquantum.compiler._component_trace.resource_generalization import FixedTCall

    if isinstance(call, FixedTCall):
        return Toffoli() if call.opcode == "toffoli" else TwoBitCSwap()
    raise AssertionError(f"No source identity for {call!r}")


def source_title(call: ResourceCall) -> str:
    from qualtran.drawing.flame_graph import _pretty_name

    if isinstance(call, (CliffordCall, BookkeepingCall)):
        return str(call)
    return str(_pretty_name(source_call(call)))


ROOTS: tuple[ResourceCall, ...] = (
    ControlledSwapCall(1),
    ControlledSwapCall(3),
    ControlledSwapCall(10),
    ControlledSwapCall(2, approximate=True),
    ControlledSwapCall(3, approximate=True, adjoint=True),
    ConjunctionCall((0, 1, 0)),
    ConjunctionCall((1, 0, 1, 0, 1, 1)),
    ConjunctionCall((0, 1, 0), adjoint=True),
    ControlledZPowerCall(0.125, 1e-4),
    ControlledZPowerCall(-0.25, 1e-8),
    ControlledZPowerCall(1.0),
    ControlledZRotationCall(0.3, 1e-4),
    ControlledZRotationCall(-pi / 2, 1e-8),
    MergedFredkinCall((0, 1)),
    ControlledRegisterSwapCall(2),
    ControlPredicateCall((1, 0, 1)),
    ControlPredicateCall((0, 1), adjoint=True),
)


@pytest.mark.parametrize("call", ROOTS)
@pytest.mark.parametrize("depth", [None, 0, 1, 2])
@pytest.mark.parametrize("default_keep", [False, True])
def test_native_root_without_declared_rows_matches_source(
    call: ResourceCall, depth: int | None, default_keep: bool
) -> None:
    from qualtran.drawing.flame_graph import (
        _keep_if_small,
        _t_counts_for_bloq,
        get_flame_graph_data,
    )

    source = source_call(call)
    full, _ = source.call_graph()
    assert query_terminal_t_cost(call) == _t_counts_for_bloq(source, full)
    kwargs: dict[str, Any] = {} if default_keep else {"keep": None}
    rows = list(
        iter_resource_flame_graph_data(
            call, title=source_title, max_depth=depth, **kwargs
        )
    )
    assert rows == get_flame_graph_data(
        source, keep=_keep_if_small if default_keep else None, max_depth=depth
    )


def test_direct_regression_witnesses_are_nonzero_and_precision_sensitive() -> None:
    assert query_terminal_t_cost(ControlledSwapCall(10)) == 70
    assert query_terminal_t_cost(ControlledSwapCall(3, approximate=True)) == 12
    assert query_terminal_t_cost(ConjunctionCall((0, 1, 0, 1))) == 12
    assert query_terminal_t_cost(ControlledZPowerCall(0.125, 1e-4)) == 29
    assert query_terminal_t_cost(ControlledZPowerCall(0.125, 1e-8)) == 44
    assert query_terminal_t_cost(ControlledZRotationCall(0.3, 1e-4)) == 52
    assert list(iter_resource_flame_graph_data(ControlledSwapCall(10), max_depth=0))[
        0
    ].endswith("(T:70)\t70")


def test_explicit_rows_and_empty_rows_replace_automatic_decompositions() -> None:
    call = ControlledSwapCall(3)
    assert query_terminal_t_cost(call, declared_calls={call: {PhaseCall("t"): 2}}) == 2
    assert query_terminal_t_cost(call, declared_calls={call: {}}) == 0
    root = ComponentCost("Program", 1)
    with pytest.raises(ValueError, match="cycle"):
        query_terminal_t_cost(call, declared_calls={call: {root: 1}, root: {call: 1}})
    assert count_selected_calls(call, (PhaseCall("t"),)) == {}


def test_partition_and_predicate_wrappers_reuse_decompositions() -> None:
    from flagquantum.compiler._component_trace.resource_generalization import FixedTCall

    root = ControlledZPowerCall(0.125, 1e-4)
    wrapper = PartitionedCall(root, (("q", 2),))
    assert query_terminal_t_cost(wrapper) == query_terminal_t_cost(root) == 29
    for controls, expected in (((0,), 4), ((1,), 4), ((0, 1), 8), ((1, 0, 1), 12)):
        call = PredicateControlledCall(FixedTCall("toffoli"), controls)
        assert query_terminal_t_cost(call) == expected


def test_symbolic_costs_preserve_shared_recipe_expressions() -> None:
    from qualtran.drawing.flame_graph import _t_counts_for_bloq
    from sympy import symbols

    n = symbols("n", positive=True, integer=True)
    for call in (
        ControlledSwapCall(n),
        ControlledSwapCall(n, approximate=True),
        ConjunctionCall(n),
    ):
        source = source_call(call)
        graph, _ = source.call_graph()
        assert query_terminal_t_cost(call) == _t_counts_for_bloq(source, graph)


def test_unsupported_compounds_fail_before_display_cutoff() -> None:
    for call in (
        ArithmeticCall("add", (3, 3)),
        ControlledRotationCall(PhaseCall("t"), (1, 1)),
    ):
        with pytest.raises(NotImplementedError):
            list(iter_resource_flame_graph_data(call, max_depth=0))
    with pytest.raises(ValueError, match="precision"):
        query_terminal_t_cost(ControlledRotationCall(PhaseCall("t"), (1,)))


def test_existing_qec_consumer_uses_same_shared_recipe() -> None:
    from flagquantum.compiler._component_trace.resource_decomposition import (
        decomposed_resource_callees,
    )
    from flagquantum.compiler._component_trace.resource_generalization import FixedTCall

    root: ResourceCall = ControlledSwapCall(3)
    assert dict(decomposed_resource_callees(root, None)) == {FixedTCall("fredkin"): 3}
    graph = _build_call_graph(
        root,
        lambda node: _terminal_resource_callees(node, None),
        None,
        None,
        None,
    )
    assert query_qec_gate_cost(graph.root).cswap == 3
    assert query_terminal_t_cost(graph.root) == 21


@pytest.mark.parametrize(
    "call",
    [
        ControlledRegisterSwapCall(0),
        MergedFredkinCall((1,)),
        ControlPredicateCall(1),
        ControlPredicateCall((0, 2)),
        PredicateControlledCall(PhaseCall("t"), (2,)),
    ],
)
def test_invalid_automatic_compound_parameters_are_rejected(call: ResourceCall) -> None:
    with pytest.raises((ValueError, TypeError)):
        query_terminal_t_cost(call)


def test_legacy_fanout_rejects_nonconcrete_width_while_terminal_cost_is_zero() -> None:
    from sympy import symbols

    from flagquantum.compiler._component_trace.resource_decomposition import (
        decomposed_resource_callees,
    )
    from flagquantum.compiler._component_trace.resource_generalization import FanoutCall

    for width in (0, symbols("n", positive=True, integer=True)):
        call = FanoutCall(width)
        with pytest.raises(TypeError, match="positive concrete"):
            tuple(decomposed_resource_callees(call, None))
        assert query_terminal_t_cost(call) == 0
