"""Native reservation callbacks follow the frozen source allocation cadence."""

from dataclasses import replace
from hashlib import sha1
from pathlib import Path
from typing import Any

import pytest

from flagquantum.compiler.registers.arithmetic.addition_component import (
    addition_component_program,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
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

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]

Event = tuple[str, LaneLocation, str, tuple[int, ...], tuple[int, ...]]


@pytest.fixture(scope="module", autouse=True)
def frozen_reference() -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "ftqc-reference-env/qualtran/drawing/musical_score.py"
    )
    raw = source.read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "3863bb66cdba17c4654657e275a8485d94b8da75"
    )


class TrackedLanes(RegisterLaneManager):
    def __init__(self, reserve: bool) -> None:
        super().__init__(128)
        self.events: list[Event] = []
        self.hooks = 0
        self.reserve = reserve

    def maybe_reserve(
        self, location: LaneLocation, register: Register, index: tuple[int, ...]
    ) -> None:
        self.events.append(
            ("hook", location, register.name, index, tuple(sorted(self.available)))
        )
        self.hooks += 1
        if self.reserve and self.hooks in (1, 3, 5):
            number = self.hooks

            def until(where: LaneLocation, reg: Register) -> bool:
                self.events.append(
                    ("until", where, reg.name, (), tuple(sorted(self.available)))
                )
                return isinstance(where, int) and where >= number // 2

            self.reserve_n(number // 2, until)


def source_manager(reserve: bool) -> Any:
    from qualtran import LeftDangle, RightDangle
    from qualtran.drawing.musical_score import LineManager

    manager = LineManager(128)
    manager.events = []
    manager.hooks = 0

    def hook(binst: Any, reg: Any, idx: tuple[int, ...]) -> None:
        location: LaneLocation = (
            "input"
            if binst == LeftDangle
            else "output" if binst == RightDangle else binst.i
        )
        manager.events.append(
            ("hook", location, reg.name, idx, tuple(sorted(manager.available)))
        )
        manager.hooks += 1
        if reserve and manager.hooks in (1, 3, 5):
            number = manager.hooks

            def until(instance: Any, register: Any) -> bool:
                where: LaneLocation = (
                    "input"
                    if instance == LeftDangle
                    else "output" if instance == RightDangle else instance.i
                )
                manager.events.append(
                    (
                        "until",
                        where,
                        register.name,
                        (),
                        tuple(sorted(manager.available)),
                    )
                )
                return isinstance(where, int) and where >= number // 2

            manager.reserve_n(number // 2, until)

    manager.maybe_reserve = hook
    return manager


@pytest.mark.parametrize("reserve", [False, True])
@pytest.mark.parametrize("shape", [(), (2,), (1, 2), (0,), (0, 2)])
def test_typed_array_boundaries_and_through_calls_match_source(
    shape: tuple[int, ...], reserve: bool
) -> None:
    from qualtran.drawing import get_musical_score_data

    record, original = fixture(2, shape)
    before = hash(record)
    native = TrackedLanes(reserve)
    source = source_manager(reserve)

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
    assert native.events == source.events
    assert sorted(native.available) == sorted(source.available)
    assert hash(record) == before
    # THRU calls do not allocate, release, or evaluate reservation predicates.
    assert all(event[1] in ("input", "output") for event in native.events)


@pytest.mark.parametrize("reserve", [False, True])
@pytest.mark.parametrize("width", [1, 2, 3, 5, 16, 32])
def test_full_add_array_split_join_work_and_output_callbacks_match_source(
    width: int, reserve: bool
) -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add
    from qualtran.drawing import get_musical_score_data

    from examples.algorithms.classical_tensor_scores import style, symbols

    record = addition_component_program(width)
    native = TrackedLanes(reserve)
    source = source_manager(reserve)
    score = component_register_score(record, style=style, lane_manager=native)
    actual = register_score_data(score, wire_symbol=symbols(record, {}))
    expected = get_musical_score_data(
        Add(QUInt(width)).decompose_bloq(), manager=source
    )
    assert_source_geometry(actual, expected)
    assert native.events == source.events
    assert sorted(native.available) == sorted(source.available)
    if reserve and width > 1:
        assert any(
            event[0] == "until" and isinstance(event[1], int) for event in native.events
        )
    # Geometry customization preserves logical call identities and storage lifetimes.
    plain = component_register_score(record, style=style)
    if reserve and width > 1:
        customized = {line.register: line.lane for line in score.lines}
        assert any(customized[line.register] != line.lane for line in plain.lines)
    assert [(col.step, col.ordinal) for col in score.columns] == [
        (col.step, col.ordinal) for col in plain.columns
    ]
    assert sorted(
        (line.register, line.width, line.start, line.end) for line in score.lines
    ) == sorted(
        (line.register, line.width, line.start, line.end) for line in plain.lines
    )


def test_right_allocation_before_left_release_and_reuse_are_visible_to_hook() -> None:
    native = TrackedLanes(False)
    score = layout_register_score(
        {"a": 1, "b": 1, "c": 1},
        ("a",),
        (RegisterStep("replace", ("a",), ("b",)), RegisterStep("next", ("b",), ("c",))),
        lane_manager=native,
    )
    assert [col.lanes for col in score.columns] == [(0, 1), (1, 0)]
    assert [(where, name, free[:2]) for _, where, name, _, free in native.events] == [
        ("input", "a", (1, 2)),
        (0, "b", (2, 3)),
        (0, "a", (0, 2)),
        (1, "c", (2, 3)),
        (1, "b", (1, 2)),
        ("output", "c", (0, 1)),
    ]


def test_empty_fresh_array_evaluates_until_once_and_never_calls_hook() -> None:
    native = TrackedLanes(False)
    seen: list[tuple[LaneLocation, str]] = []

    def until(where: LaneLocation, reg: Register) -> bool:
        seen.append((where, reg.name))
        return True

    native.reserve_n(1, until)
    empty = Register("empty", RegisterType("bit", 1), (0, 2), RegisterSide.RIGHT)
    layout_register_score(
        {}, (), (RegisterStep("make", output_registers=(empty,)),), lane_manager=native
    )
    assert seen == [(0, "empty")]
    assert native.events == []
    assert native.available[0] == 0


@pytest.mark.parametrize("capacity", [0, 1, 100, None])
def test_capacity_exhaustion_and_growth(capacity: int | None) -> None:
    manager = RegisterLaneManager(capacity)
    count = 101 if capacity is None else capacity
    assert [manager.new_y() for _ in range(count)] == list(range(count))
    if capacity is not None:
        with pytest.raises(IndexError, match="No display lanes"):
            manager.new_y()
    else:
        manager.reserve_n(2, lambda *_: False)
        assert manager.new_y() == 103


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_invalid_capacity_and_reservation_sizes_leave_pool_intact(value: Any) -> None:
    with pytest.raises(ValueError, match="capacity"):
        RegisterLaneManager(value)
    manager = RegisterLaneManager(3)
    with pytest.raises(ValueError, match="size"):
        manager.reserve_n(value, lambda *_: True)
    assert manager.available == [0, 1, 2]


def test_reservation_failures_are_atomic_and_retry_does_not_duplicate_lanes() -> None:
    manager = RegisterLaneManager(3)
    with pytest.raises(IndexError, match="Insufficient"):
        manager.reserve_n(4, lambda *_: True)
    invalid: Any = None
    with pytest.raises(TypeError, match="callable"):
        manager.reserve_n(1, invalid)
    assert manager.available == [0, 1, 2]
    fail = True

    def until(where: LaneLocation, reg: Register) -> bool:
        if fail:
            raise RuntimeError("release failed")
        return True

    manager.reserve_n(1, lambda *_: True)
    manager.reserve_n(1, until)
    register = Register("q", RegisterType("bit", 1))
    with pytest.raises(RuntimeError, match="release failed"):
        manager.unreserve("input", register)
    assert manager.available == [2]
    fail = False
    manager.unreserve("input", register)
    assert sorted(manager.available) == [0, 1, 2]
    manager.unreserve("input", register)
    assert sorted(manager.available) == [0, 1, 2]


@pytest.mark.parametrize(
    "kind", ["count", "width", "uncovered", "outputs", "duplicate_outputs", "style"]
)
def test_invalid_metadata_fails_before_touching_lane_manager(kind: str) -> None:
    manager = TrackedLanes(False)
    register = Register(
        "q",
        RegisterType("bit", 2 if kind == "width" else 1),
        (2,) if kind == "count" else (),
    )
    if kind == "style":
        with pytest.raises(ValueError, match="metadata"):
            component_register_score(
                addition_component_program(2),
                lane_manager=manager,
                style=lambda _, step: replace(step, input_registers=()),
            )
    else:
        keys = ("q", "r") if kind == "uncovered" else ("q",)
        outputs = (
            ()
            if kind == "outputs"
            else ("q", "q") if kind == "duplicate_outputs" else None
        )
        with pytest.raises(ValueError, match="metadata|outputs"):
            layout_register_score(
                {"q": 1, "r": 1},
                keys,
                (),
                lane_manager=manager,
                input_registers=(register,),
                outputs=outputs,
            )
    assert manager.events == []
    assert manager.available == list(range(128))


@pytest.mark.parametrize("side", ["input", "output"])
def test_partial_array_lifetime_metadata_is_rejected_before_allocation(
    side: str,
) -> None:
    manager = TrackedLanes(False)
    array = Register("array", RegisterType("bit", 1), (2,))
    step = (
        RegisterStep("partial", ("a", "b"), ("a",), input_registers=(array,))
        if side == "input"
        else RegisterStep("partial", ("a",), ("a", "b"), output_registers=(array,))
    )
    with pytest.raises(ValueError, match="partially"):
        layout_register_score(
            {"a": 1, "b": 1}, step.inputs, (step,), lane_manager=manager
        )
    assert manager.events == []


def test_hook_exception_propagates_and_an_independent_layout_can_retry() -> None:
    class Failing(RegisterLaneManager):
        def maybe_reserve(
            self, location: LaneLocation, register: Register, index: tuple[int, ...]
        ) -> None:
            raise RuntimeError("hook failed")

    with pytest.raises(RuntimeError, match="hook failed"):
        layout_register_score({"a": 1}, ("a",), (), lane_manager=Failing())
    score = layout_register_score({"a": 1}, ("a",), ())
    assert [(line.lane, line.start, line.end) for line in score.lines] == [(0, -1, 0)]
