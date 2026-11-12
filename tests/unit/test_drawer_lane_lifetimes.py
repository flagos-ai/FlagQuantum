"""Partial lane observations match the frozen source lifecycle at every hook."""

from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from examples.algorithms.classical_tensor_scores import style, symbols
from flagquantum.compiler.registers.arithmetic.addition_component import (
    addition_component_program,
)
from flagquantum.core.ir.program.register_types import (
    classical_bit_type,
    classical_unsigned_register_type,
    opaque_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.register_lanes import (
    LaneLocation,
    LaneSegment,
    RegisterLaneManager,
)
from flagquantum.drawer.register_score import RegisterStep, layout_register_score
from flagquantum.drawer.register_score_data import (
    PortSide,
    SymbolClass,
    register_score_data,
)
from tests.unit.test_component_tensor_nodes import fixture
from tests.unit.test_drawer_component_score import assert_source_geometry
from tests.unit.test_drawer_lane_coordinates import (
    CoordinateLanes,
    original_coordinates,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]

LineState = tuple[int, int, int | None, str]
Observation = tuple[str, tuple[LineState, ...], tuple[int, ...]]


def snapshot(manager: Any) -> tuple[LineState, ...]:
    return tuple(
        sorted((h.y, h.seq_x_start, h.seq_x_end, str(h.flavor)) for h in manager.hlines)
    )


def observe(manager: Any, event: str) -> None:
    manager.observations.append(
        (event, snapshot(manager), tuple(sorted(manager.available)))
    )


def reserve_from_lines(manager: Any) -> None:
    """Caller-supplied policy uses open lines and releases after a line closes."""
    if (
        manager.use_lines
        and not manager.line_reservation_done
        and sum(h.seq_x_end is None for h in manager.hlines) >= 2
    ):
        manager.line_reservation_done = True

        def until(location: Any, register: Any) -> bool:
            observe(manager, "line-predicate")
            return any(h.seq_x_end is not None for h in manager.hlines)

        manager.reserve_n(1, until)


class ObservedLanes(CoordinateLanes):
    def __init__(self, reserve: bool, use_lines: bool) -> None:
        super().__init__(reserve)
        self.use_lines = use_lines
        self.line_reservation_done = False
        self.observations: list[Observation] = []

    def new_y_for(
        self,
        location: LaneLocation,
        register: Register,
        index: tuple[int, ...] | None = None,
    ) -> int:
        observe(self, "coordinate")
        return super().new_y_for(location, register, index)

    def maybe_reserve(
        self, location: LaneLocation, register: Register, index: tuple[int, ...]
    ) -> None:
        super().maybe_reserve(location, register, index)
        observe(self, "hook")
        reserve_from_lines(self)

    def finish_hline(self, y: int, seq_x_end: int) -> None:
        observe(self, "finish-before")
        super().finish_hline(y, seq_x_end)
        observe(self, "finish-after")


def original_observer(reserve: bool, use_lines: bool) -> Any:
    manager = original_coordinates(reserve)
    manager.use_lines = use_lines
    manager.line_reservation_done = False
    manager.observations = []
    coordinate = manager.new_y
    hook = manager.maybe_reserve
    finish = manager.finish_hline

    def new_y(
        instance: Any, register: Any, index: tuple[int, ...] | None = None
    ) -> int:
        observe(manager, "coordinate")
        return int(coordinate(instance, register, index))

    def maybe_reserve(instance: Any, register: Any, idx: tuple[int, ...]) -> None:
        hook(instance, register, idx)
        observe(manager, "hook")
        reserve_from_lines(manager)

    def finish_hline(y: int, seq_x_end: int) -> None:
        observe(manager, "finish-before")
        finish(y, seq_x_end)
        observe(manager, "finish-after")

    manager.new_y = new_y
    manager.maybe_reserve = maybe_reserve
    manager.finish_hline = finish_hline
    return manager


@pytest.mark.parametrize(
    "reserve,use_lines", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("width", [1, 2, 3, 5, 16, 32])
def test_add_every_open_close_observation_and_line_policy_matches_source(
    width: int, reserve: bool, use_lines: bool
) -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add
    from qualtran.drawing import get_musical_score_data

    record = addition_component_program(width)
    native = ObservedLanes(reserve, use_lines)
    source = original_observer(reserve, use_lines)
    score = component_register_score(record, style=style, lane_manager=native)
    expected = get_musical_score_data(
        Add(QUInt(width)).decompose_bloq(), manager=source
    )
    assert_source_geometry(
        register_score_data(score, wire_symbol=symbols(record, {})), expected
    )
    assert native.observations == source.observations
    assert native.allocations == source.allocations
    assert native.events == source.events
    assert snapshot(native) == snapshot(source)
    assert all(h.seq_x_end is not None for h in native.hlines)
    assert sorted((h.y, h.seq_x_start, h.seq_x_end) for h in native.hlines) == sorted(
        (line.lane, line.start, line.end) for line in score.lines
    )
    if use_lines:
        assert native.line_reservation_done
        # Add(1) has only THRU calls after input allocation, so no release check runs.
        assert any(event[0] == "line-predicate" for event in native.observations) == (
            width > 1
        )


@pytest.mark.parametrize("shape", [(), (2,), (1, 2), (0,), (0, 2)])
def test_array_and_scalar_through_observations_match_source(
    shape: tuple[int, ...],
) -> None:
    from qualtran.drawing import get_musical_score_data

    record, original = fixture(2, shape)
    native = ObservedLanes(True, True)
    source = original_observer(True, True)

    def h_symbols(
        step: RegisterStep, register: str | None, side: PortSide | None
    ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        return (
            ("Text", {"text": "", "fontsize": 10})
            if register is None
            else ("TextBox", {"text": "H"})
        )

    score = component_register_score(record, lane_manager=native)
    expected = get_musical_score_data(original, manager=source)
    assert_source_geometry(register_score_data(score, wire_symbol=h_symbols), expected)
    assert native.observations == source.observations
    assert snapshot(native) == snapshot(source)
    assert native.events == source.events


@pytest.mark.parametrize("kind", ["qbit", "qzero", "cbit", "czero", "cword"])
@pytest.mark.parametrize("shape", [(), (2,), (1, 2), (0,)])
def test_direct_quantum_classical_and_zero_width_lifecycles_match_source(
    kind: str, shape: tuple[int, ...]
) -> None:
    import attrs
    import numpy as np
    from qualtran import CBit, CUInt, LeftDangle, QAny, QBit, RightDangle
    from qualtran import Register as SourceRegister
    from qualtran.drawing.musical_score import LineManager

    native_types = {
        "qbit": quantum_bit_type(),
        "qzero": opaque_register_type(0),
        "cbit": classical_bit_type(),
        "czero": classical_unsigned_register_type(0),
        "cword": classical_unsigned_register_type(3),
    }
    source_types = {
        "qbit": QBit(),
        "qzero": QAny(0),
        "cbit": CBit(),
        "czero": CUInt(0),
        "cword": CUInt(3),
    }
    register = Register("q", native_types[kind], shape)
    original_register = SourceRegister("q", source_types[kind], shape=shape)
    native = RegisterLaneManager(8)
    source = LineManager(8)
    native.seq_x = -3
    lanes = native.new("input", register)
    positions = source.new(LeftDangle, original_register, seq_x=-3, topo_gen=7)
    assert snapshot(native) == snapshot(source)
    assert all(h.seq_x_end is None for h in native.hlines)
    for position in np.asarray(positions, dtype=object).flat:
        source.finish_hline(position.y, 4)
    for lane in lanes:
        native.finish_hline(lane, 4)
    assert snapshot(native) == snapshot(source)
    # A separate pair runs the actual free call after positions advance.
    native = RegisterLaneManager(8)
    source = LineManager(8)
    native.seq_x = -3
    lanes = native.new("input", register)
    positions = source.new(LeftDangle, original_register, seq_x=-3, topo_gen=7)
    if shape:
        for index in np.ndindex(shape):
            positions[index] = attrs.evolve(positions[index], seq_x=4)
    else:
        positions = attrs.evolve(positions, seq_x=4)
    native.seq_x = 4
    native.free("output", register, lanes)
    source.free(RightDangle, original_register, positions)
    assert snapshot(native) == snapshot(source)
    assert sorted(native.available) == sorted(source.available) == list(range(8))


def test_closed_segment_replacement_is_immutable_and_lane_reuse_retains_history() -> (
    None
):
    class Saved(RegisterLaneManager):
        def __init__(self) -> None:
            super().__init__()
            self.history: list[frozenset[LaneSegment]] = []

        def maybe_reserve(
            self, location: LaneLocation, register: Register, index: tuple[int, ...]
        ) -> None:
            self.history.append(frozenset(self.hlines))

    manager = Saved()
    score = layout_register_score(
        {"a": 1, "b": 1, "c": 1},
        ("a",),
        (RegisterStep("replace", ("a",), ("b",)), RegisterStep("next", ("b",), ("c",))),
        lane_manager=manager,
    )
    assert [(line.lane, line.start, line.end) for line in score.lines] == [
        (0, -1, 0),
        (0, 1, 2),
        (1, 0, 1),
    ]
    assert manager.history[0] == frozenset({LaneSegment(0, -1)})
    assert LaneSegment(0, -1) not in manager.hlines
    assert manager.history[-1] == frozenset(manager.hlines)
    line = next(iter(manager.history[0]))
    with pytest.raises(FrozenInstanceError):
        setattr(line, "seq_x_end", 100)


@pytest.mark.parametrize("ambiguous", [False, True])
def test_missing_or_ambiguous_finish_is_atomic_and_matches_source(
    ambiguous: bool,
) -> None:
    from qualtran.drawing.musical_score import HLine, LineManager

    native = RegisterLaneManager(2)
    source = LineManager(2)
    if ambiguous:
        native.hlines.update({LaneSegment(0, -1), LaneSegment(0, 2)})
        source.hlines.update({HLine(0, -1), HLine(0, 2)})
    before = snapshot(native)
    with pytest.raises(ValueError):
        native.finish_hline(0, 3)
    with pytest.raises(ValueError):
        source.finish_hline(0, 3)
    assert snapshot(native) == snapshot(source) == before
    assert native.available == source.available == [0, 1]


@pytest.mark.parametrize("invalid", ["missing", "duplicate", "count"])
def test_invalid_free_preserves_pool_and_open_intervals(invalid: str) -> None:
    import numpy as np
    from qualtran import LeftDangle, QBit, RightDangle
    from qualtran import Register as SourceRegister
    from qualtran.drawing.musical_score import LineManager, RegPosition

    manager = RegisterLaneManager(4)
    register = Register("q", RegisterType("bit", 1), (2,))
    lanes = manager.new("input", register)
    before = snapshot(manager), manager.available.copy()
    bad = {
        "missing": (lanes[0], 99),
        "duplicate": (lanes[0], lanes[0]),
        "count": lanes[:1],
    }[invalid]
    with pytest.raises(ValueError):
        manager.free("output", register, bad)
    assert (snapshot(manager), manager.available) == before
    source = LineManager(4)
    original_register = SourceRegister("q", QBit(), shape=(2,))
    positions = source.new(LeftDangle, original_register, seq_x=0, topo_gen=0)
    bad_positions = {
        "missing": np.array([positions[0], RegPosition(99, 0, 0)], dtype=object),
        "duplicate": np.array([positions[0], positions[0]], dtype=object),
        "count": positions[:1],
    }[invalid]
    original_before = snapshot(source), source.available.copy()
    with pytest.raises(IndexError if invalid == "count" else ValueError):
        source.free(RightDangle, original_register, bad_positions)
    # Source mutates before rejecting these malformed releases. Native preflight
    # deliberately preserves ownership so the valid release can be retried.
    assert (snapshot(source), source.available) != original_before
    manager.seq_x = 3
    manager.free("output", register, lanes)
    assert sorted(manager.available) == list(range(4))
    assert all(h.seq_x_end == 3 for h in manager.hlines)
