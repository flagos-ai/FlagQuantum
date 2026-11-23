"""Compare complete context and iterator sessions with unchanged source streams."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from flagquantum.compiler.artifacts.reversible_phase import (
    ReversiblePhaseOperation as Op,
)
from flagquantum.compiler.artifacts.reversible_phase import ReversiblePhaseProgram
from flagquantum.compiler.formats.kmx_block_marks import encode_kmx_block_marks
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient,
)
from flagquantum.compiler.registers.layout.workspace import RegisterWorkspace
from flagquantum.compiler.registers.lookup._unary_iteration_session import (
    UnaryIterationPass,
)

pytestmark = pytest.mark.unit
_FIXTURE = json.loads(
    (Path(__file__).with_name("data") / "unary_iteration_session.json").read_text()
)
_DATA = _FIXTURE["cases"]


@pytest.mark.parametrize(
    "case", _DATA, ids=lambda case: f"{case['width']}-{case['mode']}-{case['action']}"
)
def test_complete_session_source_streams(case: dict[str, object]) -> None:
    width = case["width"]
    assert isinstance(width, int)
    mode, action = str(case["mode"]), str(case["action"])
    operations: list[Op] = []
    start = width + 2
    pool = RegisterWorkspace(tuple(range(start, start + max(1, width))))
    enable: object = {
        "true": True,
        "false": False,
        "quantum": width,
        "classical": ClassicalFieldCoefficient(0),
    }[mode]
    values = None
    if action == "range":
        values = range(-2, (1 << width) + 2)
    elif action == "empty":
        values = range(0)
    cursor = UnaryIterationPass(
        operations,
        pool,
        tuple(reversed(range(width))),
        values,
        control=enable,
        measurement_bit=1 if mode == "classical" else 0,
    )
    visits: list[int] = []
    positions: list[int] = []
    allocations: list[int] = []
    for _ in range(2 if action == "twopass" else 1):
        try:
            with cursor:
                positions.append(cursor.cur_address_value)
                allocations.append(start + (1 if mode == "false" else max(1, width)))
                if action == "seek":
                    for value in (-(10**30), 0, 1, (1 << width) - 1, 10**30, 0):
                        cursor.move_to(value)
                        positions.append(cursor.cur_address_value)
                        operations.append(Op("cx", (cursor.match_qubit, width + 1)))
                else:
                    for value, match in cursor:
                        visits.append(value)
                        if value % 3 == 0:
                            operations.append(Op("cx", (match, width + 1)))
                        if action == "break":
                            break
                        if action == "exception":
                            raise RuntimeError("body stop")
        except RuntimeError as error:
            assert str(error) == "body stop"
        allocations.append(start)
        # A complete temporary lease can be acquired again only after cleanup.
        lease = pool.allocate(max(1, width))
        pool.free(lease)
    cursor.close()
    assert visits == case["visits"]
    assert positions == case["positions"]
    assert allocations == case["allocations"]
    program = ReversiblePhaseProgram(
        start + max(1, width),
        2 if mode == "classical" else 1,
        tuple(operations),
        (0,) if mode == "classical" else (),
    )
    assert (
        encode_kmx_block_marks(program, cursor.calls).splitlines()[1:]
        == case["records"]
    )


def test_iterator_lifecycle_and_capacity() -> None:
    import copy

    operations: list[Op] = []
    pool = RegisterWorkspace((2, 3))
    cursor = UnaryIterationPass(operations, pool, (1, 0))
    for copier in (copy.copy, copy.deepcopy):
        with pytest.raises(TypeError, match="unique workspace"):
            copier(cursor)
    assert not operations
    for action in (
        lambda: cursor.match_qubit,
        lambda: cursor.cur_address_value,
        lambda: cursor.move_to(0),
        lambda: iter(cursor),
        lambda: next(cursor),
    ):
        with pytest.raises(ValueError, match="not open"):
            action()
    with cursor:
        assert iter(iter(cursor)) is cursor
        assert next(cursor)[0] == 0
        with pytest.raises(ValueError, match="already being iterated"):
            iter(cursor)
        with pytest.raises(ValueError, match="already open"):
            cursor.__enter__()
        while True:
            try:
                next(cursor)
            except StopIteration:
                break
        assert [value for value, _ in cursor] == [0, 1, 2, 3]
    cursor.close()
    with (
        pytest.raises(ValueError, match="Insufficient"),
        UnaryIterationPass(operations, RegisterWorkspace((2,)), (1, 0)),
    ):
        pass


def _session_program(
    n: int, controlled: bool, values: tuple[int, ...]
) -> ReversiblePhaseProgram:
    operations: list[Op] = []
    start = n + 3
    cursor = UnaryIterationPass(
        operations,
        RegisterWorkspace(tuple(range(start, start + max(1, n)))),
        tuple(reversed(range(n))),
        control=n if controlled else True,
    )
    with cursor:
        for value in values:
            cursor.move_to(value)
            operations.append(Op("cx", (cursor.match_qubit, n + 1)))
    return ReversiblePhaseProgram(start + max(1, n), 1, tuple(operations))


@pytest.mark.parametrize("n", [0, 1, 2])
@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("park", [False, True])
def test_replace_accepted_all_branch_complex_consumer(
    n: int, controlled: bool, park: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.unit import test_kickmix_unary_iteration as accepted

    values = (
        ((1 << n) - 1, 0, 0, 1 << n, (1 << n) + 1, 0) if park else (0, (1 << n) - 1)
    )
    assert _session_program(n, controlled, values) == accepted._program(
        n, controlled, values
    )
    monkeypatch.setattr(accepted, "_program", _session_program)
    accepted.test_all_reset_and_erasure_branches_preserve_complex_amplitudes(
        n, controlled, park
    )


@pytest.mark.parametrize("control", [True, False, 2, ClassicalFieldCoefficient(0)])
def test_empty_range_open_seek_and_workspace_reuse(control: object) -> None:
    ops: list[Op] = []
    pool = RegisterWorkspace((3, 4))
    cursor = UnaryIterationPass(
        ops, pool, (1, 0), range(0), control=control, measurement_bit=1
    )
    with cursor:
        assert list(cursor) == []
        cursor.move_to(-(10**30))
        assert cursor.cur_address_value == -(10**30)
        cursor.move_to(2)
        assert cursor.cur_address_value == 2
    lease = pool.allocate(2)
    pool.free(lease)


def test_invalid_inputs_and_failed_open_release_lease() -> None:
    pool = RegisterWorkspace((2,))
    operations: list[Op] = []
    for values in (range(0, 4, 2), range(4, 0, -1)):
        with pytest.raises(ValueError, match="step 1"):
            UnaryIterationPass(operations, pool, (1, 0), values)
    with pytest.raises(ValueError):
        UnaryIterationPass(operations, pool, tuple(range(64)))
    with pytest.raises(ValueError):
        UnaryIterationPass(operations, pool, (0,), control=ClassicalFieldCoefficient(0))
    with pytest.raises(ValueError):
        UnaryIterationPass(operations, pool, (0,), measurement_bit=-1)
    # Invalid overlapping work is rejected before a condition or move is emitted.
    cursor = UnaryIterationPass(operations, pool, (2,))
    with pytest.raises(ValueError):
        cursor.__enter__()
    assert not operations
    lease = pool.allocate(1)
    pool.free(lease)


def test_nested_sessions_share_workspace_and_complete_marked_stream() -> None:
    operations: list[Op] = []
    pool = RegisterWorkspace((5, 6, 7, 8))
    first = UnaryIterationPass(operations, pool, (1, 0), range(3))
    calls = []
    with first:
        for left, match in first:
            second = UnaryIterationPass(
                operations, pool, (3, 2), range(3), control=match
            )
            with second:
                for right, enabled in second:
                    if (left + right) % 2:
                        operations.append(Op("cx", (enabled, 4)))
            calls.extend(second.calls)
    calls.extend(first.calls)
    calls.sort(key=lambda call: call.start)
    program = ReversiblePhaseProgram(9, 1, tuple(operations))
    assert encode_kmx_block_marks(program, calls).splitlines()[1:] == _FIXTURE["nested"]
    lease = pool.allocate(4)
    pool.free(lease)


def test_normal_append_reuses_session_without_caller_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from examples.algorithms import unary_address_traversal as consumer

    def session_operations(
        address: tuple[int, ...],
        workspace: tuple[int, ...],
        values: tuple[int, ...],
        body: Callable[[int, int], tuple[Op, ...]],
    ) -> tuple[Op, ...]:
        operations: list[Op] = []
        cursor = UnaryIterationPass(operations, RegisterWorkspace(workspace), address)
        with cursor:
            for value in values:
                cursor.move_to(value)
                operations.extend(body(value, cursor.match_qubit))
        return tuple(operations)

    monkeypatch.setattr(consumer, "unary_iteration_operations", session_operations)
    consumer.main()


def test_source_exception_and_destructor_ownership() -> None:
    operations: list[Op] = []
    pool = RegisterWorkspace((2, 3))
    cursor = UnaryIterationPass(operations, pool, (1, 0))
    with pytest.raises(RuntimeError, match="body failure"), cursor:
        cursor.move_to(2)
        raise RuntimeError("body failure")
    lease = pool.allocate(2)
    pool.free(lease)
    cursor.__enter__()
    cursor.move_to(3)
    before = tuple(operations)
    del cursor
    assert tuple(operations) == before
    with pytest.raises(ValueError, match="Insufficient"):
        pool.allocate(1)


def test_source_range_and_control_admission() -> None:
    operations: list[Op] = []
    pool = RegisterWorkspace((1,))
    with pytest.raises(ValueError):
        UnaryIterationPass(operations, pool, (0,), control="|+>")
    with pytest.raises(ValueError, match="None or a range"):
        UnaryIterationPass(operations, pool, (0,), address_values=3)
    cursor = UnaryIterationPass(operations, pool, (0,), range(4, 2))
    with cursor:
        assert list(cursor) == []
    assert "UnaryIterationPass" in repr(cursor)
    with cursor:
        cursor.move_to(True)
        assert cursor.cur_address_value == 1
    assert operations


@pytest.mark.parametrize("enabled", [False, True])
def test_classical_session_preserves_each_complex_measurement_branch(
    enabled: bool,
) -> None:
    from itertools import product

    import torch

    from flagquantum.compiler.lowering.reversible_phase_execution import (
        bind_reversible_phase_execution,
    )
    from flagquantum.core.classical_dataflow import ClassicalXor
    from tests.unit.measurement_checks import expand_reset_branches, forced_branch

    operations: list[Op] = []
    cursor = UnaryIterationPass(
        operations,
        RegisterWorkspace((3,)),
        (0,),
        control=ClassicalFieldCoefficient(0),
        measurement_bit=1,
    )
    with cursor:
        for value in (0, 2, 1):
            cursor.move_to(value)
            operations.append(Op("cx", (cursor.match_qubit, 2)))
    program = ReversiblePhaseProgram(4, 2, tuple(operations), (0,))
    gates = expand_reset_branches(
        (
            ClassicalXor((), 0, constant=enabled).instruction(),
            *bind_reversible_phase_execution(program),
        ),
        first_slot=10,
    )
    real = torch.arange(1, 9, dtype=torch.float64)
    psi = torch.complex(real, real.flip(0))
    psi /= psi.norm()
    state = torch.zeros(16, dtype=torch.complex128)
    state[::2] = psi
    expected = torch.zeros_like(state)
    for index, amplitude in enumerate(psi):
        expected[(index ^ int(enabled)) << 1] = amplitude
    events = sum(gate.name == "measure" for gate in gates)
    for outcomes in product((0, 1), repeat=events):
        actual = forced_branch(state, gates, outcomes)
        factor = 2 ** (-events / 2) if enabled else float(not any(outcomes))
        torch.testing.assert_close(actual, expected * factor, atol=2e-14, rtol=2e-14)
        assert torch.count_nonzero(actual.reshape(-1, 2)[:, 1:]) == 0
