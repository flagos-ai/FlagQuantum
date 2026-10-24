"""Replaceable numerical consumers for approved proposal 089.

The reference factories below compose existing exchange and Kraus actions. Their
saved recipe is explicitly test-only: it does not implement native source/plan
1.2, source validation or Runtime admission. Replace all four fixtures when the
real vertical path lands, retaining the independent consumer assertions.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import pytest
import torch

import flagquantum as fq
from flagquantum.core.ir import _decode_value, _encode_value
from flagquantum.core.ir.program.physical import IsingExchange
from flagquantum.core.ising import IsingBraidProgram
from flagquantum.observables import OutputRequest
from flagquantum.operators import SuperOperator
from flagquantum.runtime.execution_plan_contract import canonical_hash, canonical_json
from flagquantum.runtime.result import ExecutionResult, MeasurementResult
from flagquantum.simulation.ising_braid import _exchange_unitary

pytestmark = pytest.mark.unit
Channel = Literal["1", "psi"]
Action = tuple[int, bool] | None
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


class _ReferenceSource:
    def __init__(self, *, initial_channel: Channel = "1") -> None:
        self.initial = initial_channel
        self.actions: tuple[Action, ...] = ()
        self.source = IsingBraidProgram(initial_channel=initial_channel)

    def exchange(self, left: int, *, clockwise: bool = False) -> _ReferenceSource:
        self.source.exchange(left, clockwise=clockwise)
        self.actions += ((left, clockwise),)
        return self

    def dephase_pair(self) -> _ReferenceSource:
        if None in self.actions:
            raise ValueError("only one unread pair observation is admitted")
        self.actions += (None,)
        return self

    def projector(self, channel: Channel) -> torch.Tensor:
        return self.source.projector(channel)

    def to_ir(self) -> Hashable:
        # A frozen recipe, deliberately not a production ProgramRecord.
        return self.initial, self.actions


@dataclass(frozen=True)
class _ReferencePlan:
    encoded: str

    @property
    def identity(self) -> str:
        return canonical_hash(json.loads(self.encoded))

    def to_json(self, *, indent: int | None = None) -> str:
        return (
            self.encoded
            if indent is None
            else json.dumps(json.loads(self.encoded), indent=indent)
        )


def _reference_plan(
    program: UnreadProgram, *, outputs: Sequence[OutputRequest] = ()
) -> UnreadPlan:
    assert isinstance(program, _ReferenceSource)
    return _ReferencePlan(
        canonical_json(
            {
                "schema": "test.ising_unread_recipe",
                "initial": program.initial,
                "actions": program.actions,
                "outputs": [
                    {"name": item.name, "matrix": _encode_value(item.observable)}
                    for item in outputs
                ],
            }
        )
    )


def _reference_run(
    program: UnreadProgram | UnreadPlan,
    *,
    outputs: Sequence[OutputRequest] = (),
) -> ExecutionResult:
    if isinstance(program, _ReferenceSource):
        program = _reference_plan(program, outputs=outputs)
    assert isinstance(program, _ReferencePlan)
    recipe = json.loads(program.encoded)
    assert recipe["schema"] == "test.ising_unread_recipe"
    source = IsingBraidProgram(initial_channel=recipe["initial"])
    state = source.projector(recipe["initial"])
    channel = SuperOperator.from_kraus([source.projector(c) for c in ("1", "psi")])
    for action in recipe["actions"]:
        if action is None:
            state = channel.apply(state)
        else:
            unitary = _exchange_unitary(IsingExchange(*action))
            state = SuperOperator.left_right_multiply(unitary, unitary.mH).apply(state)
    measurements = []
    values = []
    for index, item in enumerate(recipe["outputs"]):
        matrix = _decode_value(item["matrix"])
        assert isinstance(matrix, torch.Tensor)
        value = torch.trace(state @ matrix).real.unsqueeze(0)
        values.append(value)
        measurements.append(
            MeasurementResult(
                "expectation",
                (),
                value,
                metadata={"fq_output_index": index, "fq_output_name": item["name"]},
            )
        )
    return ExecutionResult(
        state=state.unsqueeze(0),
        value=torch.stack(values, dim=1) if values else None,
        measurements=tuple(measurements),
    )


@pytest.fixture
def unread_source() -> SourceFactory:
    return _ReferenceSource


@pytest.fixture
def unread_plan() -> Planner:
    return _reference_plan


@pytest.fixture
def unread_run() -> Runner:
    return _reference_run


@pytest.fixture
def unread_read() -> Callable[[str], UnreadPlan]:
    return _ReferencePlan


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
