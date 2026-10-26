"""Replaceable bounded consumers for approved proposal 085.

The four factories now use the real Core source, root planning/execution and
executable-plan reader. The original independent mathematical oracles and
consumer assertions remain unchanged. Source/schema, resource, environment and
malformed-input admission have additional Runtime tests.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Hashable, Sequence
from pathlib import Path
from typing import Literal, Protocol

import pytest
import torch

import flagquantum as fq
import flagquantum.lindblad as fql
from flagquantum._api import plan as plan_native
from flagquantum._api import run as run_native
from flagquantum.core.ising import IsingBraidProgram
from flagquantum.errors import CapabilityError
from flagquantum.experimental.simulation import RydbergProgram
from flagquantum.observables import OutputRequest
from flagquantum.operators import SuperOperator
from flagquantum.runtime import execution_plan_contract as plan_contract
from flagquantum.runtime import sample_terminal
from flagquantum.runtime.execution_plan import CircuitAnalysis, ExecutionPlan
from flagquantum.runtime.execution_plan_contract import canonical_json
from flagquantum.runtime.result import ExecutionResult

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


def _plan_native(
    program: BraidProgram,
    *,
    outputs: OutputRequest | Sequence[OutputRequest] | None = None,
    options: fq.ExecutionOptions | None = None,
) -> BraidPlan:
    assert isinstance(program, IsingBraidProgram)
    return plan_native(program, outputs=outputs, options=options)


def _run_native(
    plan: BraidProgram | BraidPlan,
    *,
    outputs: OutputRequest | Sequence[OutputRequest] | None = None,
    options: fq.ExecutionOptions | None = None,
) -> ExecutionResult:
    assert isinstance(plan, (IsingBraidProgram, ExecutionPlan))
    return run_native(plan, outputs=outputs, options=options)


@pytest.fixture
def program_factory() -> ProgramFactory:
    return IsingBraidProgram


@pytest.fixture
def braid_planner() -> Planner:
    assert fq.plan is plan_native
    return _plan_native


@pytest.fixture
def braid_runner() -> Runner:
    assert fq.run is run_native
    return _run_native


@pytest.fixture
def plan_reader() -> Callable[[str], BraidPlan]:
    return ExecutionPlan.from_json


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
