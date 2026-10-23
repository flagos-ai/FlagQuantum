"""Replaceable bounded consumers for approved proposal 085.

The test-local plans are mathematical fixtures, not executable-plan 1.1
producers or public Core records. Replace the four fixtures with admitted
production entries without changing consumer assertions. Full source/schema,
resource, environment and malformed-input admission remain implementation gates.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import pytest
import torch

import flagquantum as fq
import flagquantum.lindblad as fql
from flagquantum.core.ir import _decode_value, _encode_value
from flagquantum.errors import CapabilityError
from flagquantum.experimental.simulation import RydbergProgram
from flagquantum.observables import OutputRequest
from flagquantum.operators import SuperOperator
from flagquantum.runtime import execution_plan_contract as plan_contract
from flagquantum.runtime import sample_terminal
from flagquantum.runtime.execution_plan import CircuitAnalysis
from flagquantum.runtime.execution_plan_contract import canonical_hash, canonical_json
from flagquantum.runtime.result import ExecutionResult, MeasurementResult

pytestmark = pytest.mark.unit
Channel = Literal["1", "psi"]
Word = tuple[tuple[int, bool], ...]
LABELS = (("1", "sigma", "1"), ("psi", "sigma", "1"))
DTYPE = torch.complex128


class BraidProgram(Protocol):
    @property
    def basis_labels(self) -> tuple[tuple[str, ...], ...]: ...

    def exchange(self, left: int, *, clockwise: bool = False) -> BraidProgram: ...

    def projector(self, channel: Channel) -> torch.Tensor: ...

    def to_ir(self) -> Hashable: ...


class BraidPlan(Protocol):
    @property
    def identity(self) -> str: ...

    @property
    def analysis(self) -> CircuitAnalysis | None: ...

    def to_json(self, *, indent: int | None = None) -> str: ...


class ProgramFactory(Protocol):
    def __call__(self, *, initial_channel: Channel = "1") -> BraidProgram: ...


class Planner(Protocol):
    def __call__(
        self,
        program: BraidProgram,
        *,
        outputs: OutputRequest | Sequence[OutputRequest] | None = None,
        options: fq.ExecutionOptions | None = None,
    ) -> BraidPlan: ...


class Runner(Protocol):
    def __call__(
        self,
        plan: BraidProgram | BraidPlan,
        *,
        outputs: OutputRequest | Sequence[OutputRequest] | None = None,
        options: fq.ExecutionOptions | None = None,
    ) -> ExecutionResult: ...


@dataclass(frozen=True)
class _WordSnapshot:
    initial: Channel
    exchanges: Word


class _ProgramReference:
    """Only the fixed Ising fixture's source construction and tensor helpers."""

    def __init__(self, *, initial_channel: Channel = "1") -> None:
        self._initial = initial_channel
        self._word: list[tuple[int, bool]] = []

    @property
    def basis_labels(self) -> tuple[tuple[str, ...], ...]:
        return LABELS

    def exchange(self, left: int, *, clockwise: bool = False) -> _ProgramReference:
        assert type(left) is int and left in (0, 1, 2)
        assert type(clockwise) is bool
        self._word.append((left, clockwise))
        return self

    def projector(self, channel: Channel) -> torch.Tensor:
        return torch.diag(torch.tensor([channel == "1", channel == "psi"], dtype=DTYPE))

    def to_ir(self) -> _WordSnapshot:
        return _WordSnapshot(self._initial, tuple(self._word))


@dataclass(frozen=True)
class _PlanReference:
    """Valid-fixture capture only; deliberately not a native plan-schema reader."""

    source: _WordSnapshot
    outputs: tuple[tuple[str | None, torch.Tensor], ...]
    analysis: None = None

    @property
    def identity(self) -> str:
        return canonical_hash(json.loads(self.to_json()))

    def to_json(self, *, indent: int | None = None) -> str:
        fixture = [self.source.initial, self.source.exchanges, self.outputs]
        return canonical_json(_encode_value(fixture), indent=indent)


def _plan_reference(
    program: BraidProgram,
    *,
    outputs: OutputRequest | Sequence[OutputRequest] | None = None,
    options: fq.ExecutionOptions | None = None,
) -> BraidPlan:
    assert isinstance(program, _ProgramReference)
    if options is not None and options.require_gradients:
        raise CapabilityError("discrete braid words have no control gradient")
    matrices: list[tuple[str | None, torch.Tensor]] = []
    requests = (
        ()
        if outputs is None
        else (outputs,) if isinstance(outputs, OutputRequest) else tuple(outputs)
    )
    for output in requests:
        if output.kind != "expectation" or not isinstance(
            output.observable, torch.Tensor
        ):
            raise CapabilityError("this fixture only supports matrix expectations")
        matrices.append((output.name, output.observable.detach().clone()))
    return _PlanReference(program.to_ir(), tuple(matrices))


def _read_reference(text: str) -> BraidPlan:
    initial, word, outputs = _decode_value(json.loads(text))
    assert initial in ("1", "psi")
    assert all(
        type(left) is int and type(clockwise) is bool for left, clockwise in word
    )
    assert all(
        (name is None or isinstance(name, str)) and isinstance(matrix, torch.Tensor)
        for name, matrix in outputs
    )
    channel: Channel = "1" if initial == "1" else "psi"
    exchanges: Word = tuple((left, clockwise) for left, clockwise in word)
    matrices = tuple((name, matrix) for name, matrix in outputs)
    return _PlanReference(_WordSnapshot(channel, exchanges), matrices)


def _run_reference(
    plan: BraidProgram | BraidPlan,
    *,
    outputs: OutputRequest | Sequence[OutputRequest] | None = None,
    options: fq.ExecutionOptions | None = None,
) -> ExecutionResult:
    if isinstance(plan, _ProgramReference):
        plan = _plan_reference(plan, outputs=outputs, options=options)
    assert isinstance(plan, _PlanReference)
    # Independent closed-form central exchange; consumers below use F^dag R F.
    phase = complex(math.cos(math.pi / 8), math.sin(math.pi / 8))
    middle = phase / math.sqrt(2) * torch.tensor([[1, -1j], [-1j, 1]], dtype=DTYPE)
    outer = torch.diag(torch.exp(torch.tensor([-1j, 3j], dtype=DTYPE) * math.pi / 8))
    state = torch.diag(
        torch.tensor(
            [plan.source.initial == "1", plan.source.initial == "psi"], dtype=DTYPE
        )
    )
    for position, clockwise in plan.source.exchanges:
        unitary = middle if position == 1 else outer
        state = SuperOperator.from_kraus([unitary.mH if clockwise else unitary]).apply(
            state
        )
    expectations = tuple(
        torch.trace(matrix @ state).real.reshape(1) for _, matrix in plan.outputs
    )
    measurements = tuple(
        MeasurementResult(
            "expectation",
            (),
            value,
            metadata={"name": name, "fq_output_name": name, "fq_output_index": index},
        )
        for index, ((name, _), value) in enumerate(
            zip(plan.outputs, expectations, strict=True)
        )
    )
    value = torch.stack(expectations, dim=-1) if expectations else None
    # The reference does not attach a fabricated public ExecutionPlan.
    return ExecutionResult(
        value=value, state=state.unsqueeze(0), measurements=measurements
    )


@pytest.fixture
def program_factory() -> ProgramFactory:
    return _ProgramReference


@pytest.fixture
def braid_planner() -> Planner:
    return _plan_reference


@pytest.fixture
def braid_runner() -> Runner:
    return _run_reference


@pytest.fixture
def plan_reader() -> Callable[[str], BraidPlan]:
    return _read_reference


def _program(
    factory: ProgramFactory, word: Word, initial: Channel = "1"
) -> BraidProgram:
    program = factory(initial_channel=initial)
    for position, clockwise in word:
        program.exchange(position, clockwise=clockwise)
    return program


def _oracle(word: Word, initial: Channel) -> torch.Tensor:
    f = torch.tensor([[1, 1], [1, -1]], dtype=DTYPE) / math.sqrt(2)
    r = torch.diag(torch.exp(torch.tensor([-1j, 3j], dtype=DTYPE) * math.pi / 8))
    generators = (r, f.mH @ r @ f, r)
    state = torch.tensor([initial == "1", initial == "psi"], dtype=DTYPE)
    for left, clockwise in word:
        matrix = generators[left]
        state = (matrix.mH if clockwise else matrix) @ state
    return torch.outer(state, state.conj())


@pytest.mark.parametrize("initial", ["1", "psi"])
@pytest.mark.parametrize(
    "word",
    [
        (),
        ((0, False),),
        ((1, False),),
        ((2, True),),
        ((0, False), (1, False)),
        ((1, True), (2, False), (0, True)),
    ],
)
def test_full_complex_density_axes_and_numeric_replay(
    program_factory: ProgramFactory,
    braid_planner: Planner,
    braid_runner: Runner,
    plan_reader: Callable[[str], BraidPlan],
    word: Word,
    initial: Channel,
) -> None:
    program = _program(program_factory, word, initial)
    assert program.basis_labels == LABELS
    outputs = [
        fq.expectation(program.projector(label), name=label) for label in ("1", "psi")
    ]
    plan = braid_planner(program, outputs=outputs)
    encoded = plan.to_json()
    result = braid_runner(plan)
    direct = braid_runner(program, outputs=outputs)
    assert direct.state is not None and result.state is not None
    assert torch.equal(direct.state, result.state)
    assert direct.value is not None and result.value is not None
    assert torch.equal(direct.value, result.value)
    replay = plan_reader(encoded)
    repeated = braid_runner(replay)
    assert plan.analysis is None and replay.analysis is None
    assert replay.identity == plan.identity and replay.to_json() == encoded
    assert result.state is not None and repeated.state is not None
    assert result.state.shape == (1, 2, 2)
    assert result.state.dtype == DTYPE and result.state.device.type == "cpu"
    expected = _oracle(word, initial)
    torch.testing.assert_close(result.state[0], expected, atol=1e-12, rtol=0)
    assert torch.equal(result.state, repeated.state)
    assert result.value is not None and result.value.shape == (1, 2)
    for index, name in enumerate(("1", "psi")):
        actual = result.expectation(name)
        assert actual.shape == (1,)
        torch.testing.assert_close(
            actual[0], expected[index, index].real, atol=1e-12, rtol=0
        )
    assert result.samples is None
    assert all(
        item.kind == "expectation" and item.qubits == () for item in result.measurements
    )


@pytest.mark.parametrize(
    "words",
    [
        (((0, False), (1, False), (0, False)), ((1, False), (0, False), (1, False))),
        (((1, False), (2, False), (1, False)), ((2, False), (1, False), (2, False))),
        (((0, False), (2, False)), ((2, False), (0, False))),
        (((0, False), (1, True), (2, False), (2, True), (1, False), (0, True)), ()),
    ],
)
def test_braid_relations_distant_commutation_and_inverse(
    program_factory: ProgramFactory,
    braid_planner: Planner,
    braid_runner: Runner,
    words: tuple[Word, Word],
) -> None:
    states = [
        braid_runner(braid_planner(_program(program_factory, word))).state
        for word in words
    ]
    assert states[0] is not None and states[1] is not None
    torch.testing.assert_close(states[0], states[1], atol=1e-12, rtol=0)


def test_noncommuting_words_preserve_source_identity_and_coherence(
    program_factory: ProgramFactory,
    braid_planner: Planner,
    braid_runner: Runner,
) -> None:
    left = braid_planner(_program(program_factory, ((0, False), (1, False))))
    right = braid_planner(_program(program_factory, ((1, False), (0, False))))
    assert left.identity != right.identity
    a, b = braid_runner(left).state, braid_runner(right).state
    assert a is not None and b is not None
    assert torch.max(torch.abs(a - b)).item() > 0.1
    identity_word = braid_planner(_program(program_factory, ((1, False), (1, True))))
    assert identity_word.identity != braid_planner(program_factory()).identity


def test_source_and_output_snapshots_survive_caller_mutation(
    program_factory: ProgramFactory,
    braid_planner: Planner,
    braid_runner: Runner,
) -> None:
    program = program_factory().exchange(1)
    source = program.to_ir()
    original_hash = hash(source)
    effect = program.projector("psi")
    plan = braid_planner(program, outputs=fq.expectation(effect, name="psi"))
    encoded, identity = plan.to_json(), plan.identity
    program.exchange(1, clockwise=True)
    effect.zero_()
    assert source != program.to_ir() and hash(source) == original_hash
    assert plan.to_json() == encoded and plan.identity == identity
    torch.testing.assert_close(
        braid_runner(plan).expectation("psi"),
        torch.tensor([0.5], dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize("initial", ["1", "psi"])
def test_terminal_qnd_charge_statistics_and_unread_back_action(
    program_factory: ProgramFactory,
    braid_planner: Planner,
    braid_runner: Runner,
    initial: Channel,
) -> None:
    program = program_factory(initial_channel=initial).exchange(1)
    state = braid_runner(braid_planner(program)).state
    assert state is not None
    instrument = {label: (program.projector(label),) for label in ("1", "psi")}
    before_rng = torch.random.get_rng_state().clone()
    sampled = sample_terminal(state[0], instrument, shots=20_000, seed=7)
    repeated = sample_terminal(state[0], instrument, shots=20_000, seed=7)
    assert torch.equal(torch.random.get_rng_state(), before_rng)
    assert isinstance(sampled.value, torch.Tensor) and isinstance(
        repeated.value, torch.Tensor
    )
    assert torch.equal(sampled.value, repeated.value)
    assert sampled.metadata["outcome_labels"] == ["1", "psi"]
    probabilities = torch.tensor(sampled.metadata["probabilities"], dtype=torch.float64)
    torch.testing.assert_close(
        probabilities, torch.full((2,), 0.5, dtype=torch.float64), atol=1e-12, rtol=0
    )
    frequencies = torch.bincount(sampled.value, minlength=2).double() / 20_000
    assert torch.all(
        torch.abs(frequencies - probabilities)
        <= 6 * torch.sqrt(probabilities * (1 - probabilities) / 20_000) + 2 / 20_000
    )
    unread = SuperOperator.from_kraus([item[0] for item in instrument.values()]).apply(
        state[0]
    )
    torch.testing.assert_close(
        unread, torch.eye(2, dtype=DTYPE) / 2, atol=1e-12, rtol=0
    )
    assert torch.abs(state[0, 0, 1]).item() > 0.49


def test_discrete_gradient_request_is_refused(
    program_factory: ProgramFactory,
    braid_planner: Planner,
) -> None:
    with pytest.raises(CapabilityError):
        braid_planner(
            program_factory(), options=fq.ExecutionOptions(require_gradients=True)
        )


def test_qubit_output_descriptor_is_not_a_fusion_coordinate(
    program_factory: ProgramFactory,
    braid_planner: Planner,
) -> None:
    with pytest.raises(CapabilityError):
        braid_planner(program_factory(), outputs=[fq.expectation(fq.X(0))])


def _legacy_plans() -> dict[str, fq.ExecutionPlan | fql.LindbladPlan]:
    program = RydbergProgram(
        atoms=(("a", "trap"),),
        sites={"trap": (0.0, 0.0)},
        length_unit="um",
        time_unit="us",
        c6=0.0,
        loss_rate=0.0,
        minimum_separation=1.0,
        drive_channels=("laser",),
    )
    program.pulse("a", start=0.0, duration=1.0, amplitude=1.0, channel="laser")
    return {
        "circuit_h": fq.plan(fq.Circuit(1).h(0), outputs=fq.probabilities()),
        "circuit_bell": fq.plan(
            fq.Circuit(2).h(0).cx(0, 1), outputs=fq.probabilities()
        ),
        "native_rydberg": fql.plan(
            program,
            program.basis_state(("g",)),
            [0.0, 1.0],
            method="krylov-exponential",
        ),
    }


def test_old_execution_and_native_rydberg_plans_keep_captured_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = Path(__file__).parents[1] / "fixtures/ising_braid_legacy_plans.json"
    captures = json.loads(path.read_text())["plans"]
    # Circuit compiler identity includes the producer release. Replay the
    # archived 0.3.1 environment without rewriting the captured plan bytes.
    monkeypatch.setattr(plan_contract, "__version__", "0.3.1")
    for name, plan in _legacy_plans().items():
        assert plan.to_json() == canonical_json(captures[name])
        reader = (
            fql.LindbladPlan.from_json
            if name == "native_rydberg"
            else fq.ExecutionPlan.from_json
        )
        assert reader(plan.to_json()).to_json() == plan.to_json()
    assert captures["native_rydberg"]["version"] == "1.7"
    assert captures["native_rydberg"]["program"]["version"] == "1.0"


def test_archived_circuit_plans_reject_a_different_compiler_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = Path(__file__).parents[1] / "fixtures/ising_braid_legacy_plans.json"
    captures = json.loads(path.read_text())["plans"]
    monkeypatch.setattr(plan_contract, "__version__", "different-producer-release")
    for name in ("circuit_h", "circuit_bell"):
        with pytest.raises(plan_contract.ExecutionPlanContractError) as caught:
            fq.ExecutionPlan.from_json(canonical_json(captures[name]))
        assert caught.value.reason_code == "compiler_incompatible"
