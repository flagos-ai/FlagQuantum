"""Unchanged proposal 089 consumers, now using the real native plan path."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Hashable, Sequence
from typing import Literal, Protocol

import pytest
import torch

import flagquantum as fq
from flagquantum.core.ising import IsingBraidProgram
from flagquantum.observables import OutputRequest
from flagquantum.runtime.execution_plan import ExecutionPlan
from flagquantum.runtime.result import ExecutionResult

pytestmark = pytest.mark.unit
Channel = Literal["1", "psi"]
Word = tuple[tuple[int, bool], ...]
DTYPE = torch.complex128


class UnreadProgram(Protocol):
    def exchange(self, left: int, *, clockwise: bool = False) -> UnreadProgram: ...

    def dephase_pair(self) -> UnreadProgram: ...

    def projector(self, channel: Channel) -> torch.Tensor: ...

    def to_ir(self) -> Hashable: ...


class UnreadPlan(Protocol):
    @property
    def identity(self) -> str: ...

    def to_json(self, *, indent: int | None = None) -> str: ...


class SourceFactory(Protocol):
    def __call__(self, *, initial_channel: Channel = "1") -> UnreadProgram: ...


class Planner(Protocol):
    def __call__(
        self, program: UnreadProgram, *, outputs: Sequence[OutputRequest] = ()
    ) -> UnreadPlan: ...


class Runner(Protocol):
    def __call__(
        self,
        program: UnreadProgram | UnreadPlan,
        *,
        outputs: Sequence[OutputRequest] = (),
    ) -> ExecutionResult: ...


@pytest.fixture
def unread_source() -> SourceFactory:
    return IsingBraidProgram


@pytest.fixture
def unread_plan() -> Planner:
    def plan(
        program: UnreadProgram, *, outputs: Sequence[OutputRequest] = ()
    ) -> UnreadPlan:
        assert isinstance(program, IsingBraidProgram)
        selected: ExecutionPlan = fq.plan(program, outputs=outputs or None)
        return selected

    return plan


@pytest.fixture
def unread_run() -> Runner:
    def run(
        program: UnreadProgram | UnreadPlan,
        *,
        outputs: Sequence[OutputRequest] = (),
    ) -> ExecutionResult:
        assert isinstance(program, (IsingBraidProgram, fq.ExecutionPlan))
        result: ExecutionResult = fq.run(program, outputs=outputs or None)
        return result

    return run


@pytest.fixture
def unread_read() -> Callable[[str], UnreadPlan]:
    return ExecutionPlan.from_json


def _append(program: UnreadProgram, word: Word) -> None:
    for left, clockwise in word:
        program.exchange(left, clockwise=clockwise)


def _unitary(word: Word) -> torch.Tensor:
    # Independent fixed F/R oracle; never read production exchange matrices here.
    fusion = torch.tensor([[1, 1], [1, -1]], dtype=DTYPE) / math.sqrt(2)
    phases = torch.exp(torch.tensor([-1j, 3j], dtype=DTYPE) * math.pi / 8)
    outer = torch.diag(phases)
    generators = (outer, fusion.mH @ outer @ fusion, outer)
    product = torch.eye(2, dtype=DTYPE)
    for left, clockwise in word:
        gate = generators[left]
        product = (gate.mH if clockwise else gate) @ product
    return product


@pytest.mark.parametrize("initial", ["1", "psi"])
@pytest.mark.parametrize(
    "prefix,suffix",
    [
        ((), ()),
        (((0, False),), ((2, True),)),
        (((1, False),), ((1, True),)),
        (((1, False), (0, True)), ((1, False), (2, False))),
        (((0, True), (1, True)), ((2, False), (1, False))),
    ],
)
def test_unread_continuation_and_saved_replay(
    unread_source: SourceFactory,
    unread_plan: Planner,
    unread_run: Runner,
    unread_read: Callable[[str], UnreadPlan],
    initial: Channel,
    prefix: Word,
    suffix: Word,
) -> None:
    program = unread_source(initial_channel=initial)
    _append(program, prefix)
    assert program.dephase_pair() is program
    _append(program, suffix)
    outputs = [fq.expectation(program.projector(c), name=c) for c in ("1", "psi")]
    plan = unread_plan(program, outputs=outputs)
    encoded = plan.to_json()
    restored = unread_read(encoded)
    assert restored.identity == plan.identity
    assert restored.to_json() == encoded
    direct = unread_run(program, outputs=outputs)
    planned = unread_run(plan)
    replay = unread_run(restored)
    initial_state = torch.diag(
        torch.tensor([initial == "1", initial == "psi"], dtype=DTYPE)
    )
    before = _unitary(prefix) @ initial_state @ _unitary(prefix).mH
    after = torch.diag(torch.diag(before))
    expected = _unitary(suffix) @ after @ _unitary(suffix).mH
    # Unnormalized zero-weight branches contribute zero without division.
    weighted = sum(
        (
            _unitary(suffix)
            @ program.projector(c)
            @ before
            @ program.projector(c)
            @ _unitary(suffix).mH
            for c in ("1", "psi")
        ),
        torch.zeros((2, 2), dtype=DTYPE),
    )
    torch.testing.assert_close(weighted, expected, atol=1e-12, rtol=0)
    for result in (direct, planned, replay):
        assert result.state is not None
        assert result.state.shape == (1, 2, 2)
        assert result.state.dtype == DTYPE and result.state.device.type == "cpu"
        torch.testing.assert_close(result.state[0], expected, atol=1e-12, rtol=0)
        torch.testing.assert_close(
            torch.trace(result.state[0]),
            torch.tensor(1, dtype=DTYPE),
            atol=1e-12,
            rtol=0,
        )
        assert result.value is not None and result.value.shape == (1, 2)
        for index, name in enumerate(("1", "psi")):
            torch.testing.assert_close(
                result.expectation(name)[0],
                expected[index, index].real,
                atol=1e-12,
                rtol=0,
            )
        assert result.samples is None
        assert all(item.shots is None for item in result.measurements)
    assert direct.state is not None and planned.state is not None
    assert replay.state is not None
    assert torch.equal(direct.state, planned.state)
    assert torch.equal(planned.state, replay.state)


def test_order_snapshots_refusals_and_rng(
    unread_source: SourceFactory,
    unread_plan: Planner,
    unread_run: Runner,
) -> None:
    rng = torch.random.get_rng_state().clone()
    python_rng = random.getstate()
    source = unread_source().exchange(1).dephase_pair()
    frozen = source.to_ir()
    output = source.projector("psi")
    plan = unread_plan(source, outputs=[fq.expectation(output, name="psi")])
    encoded = plan.to_json()
    identity = plan.identity
    with pytest.raises(ValueError):
        source.dephase_pair()
    assert source.to_ir() == frozen
    source.exchange(1, clockwise=True)
    output.zero_()
    assert plan.to_json() == encoded and plan.identity == identity
    result = unread_run(plan)
    torch.testing.assert_close(
        result.expectation("psi"),
        torch.tensor([0.5], dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )
    assert source.to_ir() != frozen
    reordered = unread_source().dephase_pair().exchange(1)
    assert (
        unread_plan(reordered).identity
        != unread_plan(unread_source().exchange(1).dephase_pair()).identity
    )
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert random.getstate() == python_rng


@pytest.mark.parametrize("initial", ["1", "psi"])
def test_observation_changes_echo_from_pure_to_mixed(
    unread_source: SourceFactory, unread_run: Runner, initial: Channel
) -> None:
    observed = unread_source(initial_channel=initial)
    observed.exchange(1).dephase_pair().exchange(1, clockwise=True)
    control = unread_source(initial_channel=initial)
    control.exchange(1).exchange(1, clockwise=True)
    mixed = unread_run(observed).state
    pure = unread_run(control).state
    assert mixed is not None and pure is not None
    torch.testing.assert_close(
        mixed[0], torch.eye(2, dtype=DTYPE) / 2, atol=1e-12, rtol=0
    )
    torch.testing.assert_close(pure[0], control.projector(initial), atol=1e-12, rtol=0)
    torch.testing.assert_close(
        torch.trace(mixed[0] @ mixed[0]),
        torch.tensor(0.5, dtype=DTYPE),
        atol=1e-12,
        rtol=0,
    )
