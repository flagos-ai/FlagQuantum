"""Contextual lane policies retain the frozen score allocation call chain."""

from heapq import heapify
from typing import Any

import pytest

from examples.algorithms.classical_tensor_scores import style, symbols
from flagquantum.compiler.registers.arithmetic.addition_component import (
    addition_component_program,
)
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.register_lanes import LaneLocation, RegisterLaneManager
from flagquantum.drawer.register_score import RegisterStep, layout_register_score
from flagquantum.drawer.register_score_data import (
    PortSide,
    SymbolClass,
    register_score_data,
)
from tests.unit.test_component_tensor_nodes import fixture
from tests.unit.test_drawer_component_score import assert_source_geometry
from tests.unit.test_drawer_register_lanes import TrackedLanes, source_manager

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]

Allocation = tuple[LaneLocation, str, tuple[int, ...] | None, tuple[int, ...], int]


def take_lane(available: list[int], high: bool) -> int:
    """Test policy consumes an actual free lane while retaining heap ordering."""
    lane = max(available) if high else min(available)
    available.remove(lane)
    heapify(available)
    return lane


def high_lane(
    location: LaneLocation, register: str, index: tuple[int, ...] | None
) -> bool:
    """Choose both ends using call identity, register identity and element index."""
    return (
        location == "input"
        or (isinstance(location, int) and location % 2 == 1)
        or register == "idle"
    ) and (index is None or sum(index) % 2 == 0)


class CoordinateLanes(TrackedLanes):
    def __init__(self, reserve: bool) -> None:
        super().__init__(reserve)
        self.allocations: list[Allocation] = []

    def new_y_for(
        self,
        location: LaneLocation,
        register: Register,
        index: tuple[int, ...] | None = None,
    ) -> int:
        before = tuple(sorted(self.available))
        lane = take_lane(self.available, high_lane(location, register.name, index))
        self.allocations.append((location, register.name, index, before, lane))
        return lane


def original_coordinates(reserve: bool) -> Any:
    from qualtran import LeftDangle, RightDangle

    manager = source_manager(reserve)
    manager.allocations = []

    def new_y(
        instance: Any, register: Any, index: tuple[int, ...] | None = None
    ) -> int:
        location: LaneLocation = (
            "input"
            if instance == LeftDangle
            else "output" if instance == RightDangle else instance.i
        )
        before = tuple(sorted(manager.available))
        lane = take_lane(manager.available, high_lane(location, register.name, index))
        manager.allocations.append((location, register.name, index, before, lane))
        return lane

    manager.new_y = new_y
    return manager


@pytest.mark.parametrize("reserve", [False, True])
@pytest.mark.parametrize("width", [1, 2, 3, 5, 16, 32])
def test_add_context_coordinates_hooks_and_full_geometry_match_source(
    width: int, reserve: bool
) -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add
    from qualtran.drawing import get_musical_score_data

    record = addition_component_program(width)
    before = hash(record)
    native = CoordinateLanes(reserve)
    source = original_coordinates(reserve)
    score = component_register_score(record, style=style, lane_manager=native)
    expected = get_musical_score_data(
        Add(QUInt(width)).decompose_bloq(), manager=source
    )
    assert_source_geometry(
        register_score_data(score, wire_symbol=symbols(record, {})), expected
    )
    assert native.allocations == source.allocations
    assert native.events == source.events
    assert sorted(native.available) == sorted(source.available)
    assert hash(record) == before
    plain = component_register_score(record, style=style)
    assert [(col.step, col.ordinal) for col in score.columns] == [
        (col.step, col.ordinal) for col in plain.columns
    ]
    assert sorted(
        (line.register, line.width, line.start, line.end) for line in score.lines
    ) == sorted(
        (line.register, line.width, line.start, line.end) for line in plain.lines
    )
    assert {line.register: line.lane for line in score.lines} != {
        line.register: line.lane for line in plain.lines
    }


@pytest.mark.parametrize("reserve", [False, True])
@pytest.mark.parametrize("shape", [(), (2,), (1, 2), (0,), (0, 2)])
def test_scalar_none_array_indices_and_empty_allocation_match_source(
    shape: tuple[int, ...], reserve: bool
) -> None:
    from qualtran.drawing import get_musical_score_data

    record, original = fixture(2, shape)
    native = CoordinateLanes(reserve)
    source = original_coordinates(reserve)

    def h_symbols(
        step: RegisterStep, register: str | None, side: PortSide | None
    ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        if register is None:
            return "Text", {"text": "", "fontsize": 10}
        return "TextBox", {"text": "H"}

    actual = register_score_data(
        component_register_score(record, lane_manager=native), wire_symbol=h_symbols
    )
    expected = get_musical_score_data(original, manager=source)
    assert_source_geometry(actual, expected)
    assert native.allocations == source.allocations
    assert native.events == source.events
    assert all(event[0] == "input" for event in native.allocations)
    idle_indices = [event[2] for event in native.allocations if event[1] == "idle"]
    idle = Register("idle", RegisterType("bit", 2), shape)
    expected_indices = list(idle.all_idxs()) if shape else [None]
    assert idle_indices == expected_indices
    assert sorted(native.available) == sorted(source.available)


def test_no_argument_policy_and_reservations_keep_their_entry_point() -> None:
    class ExistingPolicy(RegisterLaneManager):
        def __init__(self) -> None:
            super().__init__(8)
            self.calls = 0

        def new_y(self) -> int:
            self.calls += 1
            return take_lane(self.available, True)

    manager = ExistingPolicy()
    manager.reserve_n(1, lambda *_: False)
    score = layout_register_score({"q": 1}, ("q",), (), lane_manager=manager)
    assert manager.calls == 2
    assert [line.lane for line in score.lines] == [6]
    assert sorted(manager.available) == list(range(7))


def test_context_override_exception_propagates_before_allocation_hook() -> None:
    class Failing(CoordinateLanes):
        def new_y_for(
            self,
            location: LaneLocation,
            register: Register,
            index: tuple[int, ...] | None = None,
        ) -> int:
            raise RuntimeError("coordinate policy failed")

    manager = Failing(False)
    with pytest.raises(RuntimeError, match="coordinate policy failed"):
        layout_register_score({"q": 1}, ("q",), (), lane_manager=manager)
    assert manager.events == []
    assert manager.available == list(range(128))
    assert layout_register_score({"q": 1}, ("q",), ()).lines[0].lane == 0
