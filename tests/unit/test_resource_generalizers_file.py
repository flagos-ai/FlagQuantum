"""Verify the frozen public rewrite suite and the Ecosystem conversion step."""

from collections.abc import Callable
from hashlib import sha1
from pathlib import Path
from typing import cast

import pytest

from flagquantum.compiler._component_trace.call_graph import (
    Generalizer,
    compose_cost_generalizer,
)
from flagquantum.compiler._component_trace.resource_classification import (
    ignore_known_cliffords,
)
from flagquantum.compiler._component_trace.resource_generalization import (
    BookkeepingCall,
    CliffordCall,
    ConjunctionCall,
    ControlledSwapCall,
    PartitionedCall,
    PhaseCall,
    ResourceCall,
    RotationCall,
    generalize_approximate_swap,
    generalize_control_values,
    generalize_rotation,
    ignore_allocation,
    ignore_register_partition,
)
from flagquantum.ecosystem.cirq.resource_keys import (
    CirqResourceCall,
    generalize_cirq_resource_call,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
Call = ResourceCall | CirqResourceCall


def test_public_generalizer_source_hashes() -> None:
    root = (
        Path(__file__).resolve().parents[3]
        / "ftqc-reference-env/qualtran/resource_counting"
    )
    for name, expected in (
        ("generalizers.py", "b1d6b2ad8ad7d9b8b857a2e1d3a565b8e5ed060c"),
        ("generalizers_test.py", "295696985aa8e0d594026fb83b35928b9281f1de"),
    ):
        path = root / name
        if not path.is_file():
            pytest.skip("Frozen source files are not installed")
        data = path.read_bytes()
        assert sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == expected


def test_complete_ordered_source_filter() -> None:
    cirq = pytest.importorskip("cirq")
    source = pytest.importorskip("qualtran.resource_counting.generalizers")
    cases = pytest.importorskip("qualtran.resource_counting.generalizers_test")
    from qualtran.bloqs.basic_gates import CSwap, Rx
    from qualtran.bloqs.mcmt.and_bloq import And, MultiAnd
    from qualtran.resource_counting._generalization import _make_composite_generalizer

    native: tuple[Call, ...] = (
        CliffordCall("cx", 2),
        CirqResourceCall(cirq.CNOT),
        ControlledSwapCall(44, approximate=True),
        ControlledSwapCall(3, approximate=True, adjoint=True),
        BookkeepingCall("split", 5),
        BookkeepingCall("join", 5),
        CliffordCall("swap_bit", 2),
        ConjunctionCall((0, 0)),
        ConjunctionCall((1, 0, 1, 0)),
        RotationCall("rx", 0.123),
        BookkeepingCall("allocate", 5),
        BookkeepingCall("free", 5),
        CliffordCall("swap_bit", 2, adjoint=True),
        BookkeepingCall("partition", (2, 3)),
        CirqResourceCall(cirq.S),
        PartitionedCall(PartitionedCall(RotationCall("rx", 0.123), "inner"), "outer"),
    )

    def native_only(rule: Generalizer[ResourceCall]) -> Generalizer[Call]:
        def apply(call: Call) -> Call | None:
            return call if isinstance(call, CirqResourceCall) else rule(call)

        return apply

    native_cliffords = native_only(ignore_known_cliffords)
    for py_gate, wrapped in ((cirq.CNOT, native[1]), (cirq.S, native[14])):
        assert native_cliffords(wrapped) is wrapped
        assert native_cliffords(generalize_cirq_resource_call(wrapped)) is None
        assert py_gate == getattr(wrapped, "gate")

    rules: tuple[Generalizer[Call], ...] = (
        generalize_cirq_resource_call,
        native_cliffords,
        native_only(ignore_allocation),
        native_only(ignore_register_partition),
        native_only(generalize_control_values),
        native_only(generalize_rotation),
        native_only(generalize_approximate_swap),
    )
    make_pipeline = cast(
        Callable[[tuple[Generalizer[Call], ...]], Callable[[Call | None], Call | None]],
        compose_cost_generalizer,
    )
    pipeline = make_pipeline(rules)
    upstream = _make_composite_generalizer(
        source.cirq_to_bloqs,
        source.ignore_cliffords,
        source.ignore_alloc_free,
        source.ignore_split_join,
        source.generalize_cvs,
        source.generalize_rotation_angle,
        source.generalize_cswap_approx,
    )
    mapping: dict[object, ResourceCall | None] = {
        None: None,
        CSwap(44): ControlledSwapCall(44),
        CSwap(3).adjoint(): ControlledSwapCall(3),
        And(source.CV, source.CV): ConjunctionCall((None, None)),
        MultiAnd((1,) * 4): ConjunctionCall(4),
        Rx(source.PHI): RotationCall("rx", None),
    }
    expected = [mapping[upstream(node)] for node in cases._BLOQS_TO_FILTER]
    assert len(expected) == len(native) == 16
    assert [pipeline(call) for call in native] == expected


def test_gate_conversion_preserves_partitions_and_unknown_gates() -> None:
    cirq = pytest.importorskip("cirq")
    source = pytest.importorskip("qualtran.resource_counting.generalizers")
    from qualtran import QAny, Register
    from qualtran.bloqs.basic_gates import SGate
    from qualtran.bloqs.bookkeeping import AutoPartition
    from qualtran.cirq_interop import CirqGateAsBloq

    partition = [(Register("q", QAny(1)), ["q"])]

    def wrap(gate: object) -> object:
        return AutoPartition(AutoPartition(gate, partition), partition)

    assert source.cirq_to_bloqs(wrap(CirqGateAsBloq(cirq.S))) == wrap(SGate())
    assert generalize_cirq_resource_call(
        CirqResourceCall(cirq.S, ("outer", "inner"))
    ) == PartitionedCall(PartitionedCall(PhaseCall("s"), "inner"), "outer")
    opaque = CirqResourceCall(cirq.ISWAP, ("outer", "inner"))
    assert generalize_cirq_resource_call(opaque) is opaque
    assert source.cirq_to_bloqs(CirqGateAsBloq(cirq.ISWAP)) == CirqGateAsBloq(
        cirq.ISWAP
    )
