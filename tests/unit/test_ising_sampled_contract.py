"""Proposal 091 consumers: replace four factories, retain independent assertions."""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from typing import Literal, Protocol

import pytest
import torch

import flagquantum as fq
from flagquantum.errors import CapabilityError
from flagquantum.observables import OutputRequest
from flagquantum.runtime.measurements import _draw_probability_indices
from flagquantum.runtime.result import ExecutionResult

from .ising_sampled_reference import (
    ReferenceRecipe,
    ReferenceSource,
    plan_reference,
    read_reference,
    run_reference,
)

pytestmark = pytest.mark.unit
Channel = Literal["1", "psi"]
Word = tuple[tuple[int, bool], ...]
DTYPE = torch.complex128


class SampledSource(Protocol):
    def exchange(self, left: int, *, clockwise: bool = False) -> SampledSource: ...

    def measure_pair(self) -> SampledSource: ...

    def dephase_pair(self) -> SampledSource: ...

    def projector(self, channel: Channel) -> torch.Tensor: ...

    def to_ir(self) -> object: ...


class SampledPlan(Protocol):
    @property
    def identity(self) -> str: ...

    def to_json(self, *, indent: int | None = None) -> str: ...


class SourceFactory(Protocol):
    def __call__(self, *, initial_channel: Channel = "1") -> SampledSource: ...


class Planner(Protocol):
    def __call__(
        self,
        source: SampledSource,
        *,
        options: fq.ExecutionOptions,
        outputs: tuple[OutputRequest, ...] = (),
    ) -> SampledPlan: ...


class Runner(Protocol):
    def __call__(
        self,
        source: SampledSource | SampledPlan,
        *,
        options: fq.ExecutionOptions | None = None,
        outputs: tuple[OutputRequest, ...] = (),
    ) -> ExecutionResult: ...


@pytest.fixture
def sampled_source() -> SourceFactory:
    return ReferenceSource


@pytest.fixture
def sampled_plan() -> Planner:
    def plan(
        source: SampledSource,
        *,
        options: fq.ExecutionOptions,
        outputs: tuple[OutputRequest, ...] = (),
    ) -> SampledPlan:
        assert isinstance(source, ReferenceSource)
        return plan_reference(source, options=options, outputs=outputs)

    return plan


@pytest.fixture
def sampled_run() -> Runner:
    def run(
        source: SampledSource | SampledPlan,
        *,
        options: fq.ExecutionOptions | None = None,
        outputs: tuple[OutputRequest, ...] = (),
    ) -> ExecutionResult:
        assert isinstance(source, (ReferenceSource, ReferenceRecipe))
        return run_reference(source, options=options, outputs=outputs)

    return run


@pytest.fixture
def sampled_read() -> Callable[[str], SampledPlan]:
    return read_reference


def _append(source: SampledSource, word: Word) -> None:
    for left, clockwise in word:
        source.exchange(left, clockwise=clockwise)


def _unitary(word: Word) -> torch.Tensor:
    # Independent fixed F/R; never obtain a generator from the reference/kernel.
    fusion = torch.tensor([[1, 1], [1, -1]], dtype=DTYPE) / math.sqrt(2)
    outer = torch.diag(torch.exp(torch.tensor([-1j, 3j], dtype=DTYPE) * math.pi / 8))
    generators = (outer, fusion.mH @ outer @ fusion, outer)
    product = torch.eye(2, dtype=DTYPE)
    for left, clockwise in word:
        gate = generators[left]
        product = (gate.mH if clockwise else gate) @ product
    return product


@pytest.mark.parametrize("initial", ["1", "psi"])
@pytest.mark.parametrize("seed", [0, 17, 2**64 - 1])
@pytest.mark.parametrize(
    "prefix,suffix",
    [
        ((), ()),
        (((1, False),), ((0, False),)),
        (((1, False),), ((1, True),)),
        (((1, False), (0, True)), ((1, False), (2, False))),
        (((0, True), (1, True)), ((2, False), (1, False))),
    ],
)
def test_selected_density_outputs_and_exact_saved_replay(
    sampled_source: SourceFactory,
    sampled_plan: Planner,
    sampled_run: Runner,
    sampled_read: Callable[[str], SampledPlan],
    initial: Channel,
    seed: int,
    prefix: Word,
    suffix: Word,
) -> None:
    source = sampled_source(initial_channel=initial)
    _append(source, prefix)
    assert source.measure_pair() is source
    _append(source, suffix)
    options = fq.ExecutionOptions(shots=1, seed=seed)
    outputs = tuple(fq.expectation(source.projector(c), name=c) for c in ("1", "psi"))
    torch_rng, python_rng = torch.random.get_rng_state().clone(), random.getstate()
    cuda_initialized: Callable[[], bool] = torch.cuda.is_initialized
    cuda_rng = torch.cuda.get_rng_state_all() if cuda_initialized() else None
    plan = sampled_plan(source, options=options, outputs=outputs)
    text = plan.to_json()
    restored = sampled_read(text)
    assert restored.identity == plan.identity and restored.to_json() == text
    direct = sampled_run(source, options=options, outputs=outputs)
    planned, replay = sampled_run(plan), sampled_run(restored)
    initial_density = torch.diag(
        torch.tensor([initial == "1", initial == "psi"], dtype=DTYPE)
    )
    prefix_u, suffix_u = _unitary(prefix), _unitary(suffix)
    before = prefix_u @ initial_density @ prefix_u.mH
    probabilities = before.diag().real
    pairs = ["a", "b", "c", "d"]
    for left, _ in prefix:
        pairs[left], pairs[left + 1] = pairs[left + 1], pairs[left]
    weighted = torch.zeros((2, 2), dtype=DTYPE)
    for j, probability in enumerate(probabilities):
        # Zero-weight contributions are zero, without constructing rho/0.
        if float(probability) <= 0:
            continue
        projector = torch.diag(torch.eye(2, dtype=DTYPE)[j])
        branch = projector @ before @ projector / probability
        weighted += probability * (suffix_u @ branch @ suffix_u.mH)
    unread_expected = suffix_u @ torch.diag(before.diag()) @ suffix_u.mH
    torch.testing.assert_close(weighted, unread_expected, atol=1e-12, rtol=0)
    for result in (direct, planned, replay):
        event = result.measurement(-1)
        assert event.kind == "mid_circuit_measurement" and event.qubits == ()
        assert event.shots == 1 and isinstance(event.value, torch.Tensor)
        assert event.value.shape == (1, 1) and event.value.dtype == torch.int64
        index = int(event.value[0, 0])
        assert index in (0, 1) and float(probabilities[index]) > 0
        projector = torch.diag(torch.eye(2, dtype=DTYPE)[index])
        expected = (
            suffix_u
            @ (projector @ before @ projector / probabilities[index])
            @ suffix_u.mH
        )
        assert result.state is not None
        assert result.state.shape == (1, 2, 2) and result.state.dtype == DTYPE
        assert result.state.device.type == "cpu"
        torch.testing.assert_close(result.state[0], expected, atol=1e-12, rtol=0)
        torch.testing.assert_close(
            torch.trace(result.state[0]),
            torch.tensor(1, dtype=DTYPE),
            atol=1e-12,
            rtol=0,
        )
        torch.testing.assert_close(
            torch.trace(result.state[0] @ result.state[0]),
            torch.tensor(1, dtype=DTYPE),
            atol=1e-12,
            rtol=0,
        )
        assert result.value is not None and result.value.shape == (1, 2)
        for j, name in enumerate(("1", "psi")):
            torch.testing.assert_close(
                result.expectation(name)[0], expected[j, j].real, atol=1e-12, rtol=0
            )
        assert result.samples is None and len(result.measurements) == 3
        assert event.metadata["stage"] == "intermediate"
        assert event.metadata["outcome_labels"] == ["1", "psi"]
        assert event.metadata["source_operation_index"] == len(prefix)
        assert event.metadata["pair_anyon_ids"] == pairs[:2]
        torch.testing.assert_close(
            torch.tensor(event.metadata["probabilities"], dtype=torch.float64)[0],
            probabilities,
            atol=1e-12,
            rtol=0,
        )
        assert 0 <= event.metadata["probability_correction"] <= 1e-10
        assert result.runtime["shots"] == 1 and result.runtime["seed"] == seed
        assert result.runtime["mid_circuit_measurements_available"] is True
    assert (
        direct.state is not None
        and planned.state is not None
        and replay.state is not None
    )
    assert torch.equal(direct.state, planned.state) and torch.equal(
        planned.state, replay.state
    )
    direct_index, replay_index = (
        direct.measurement(-1).value,
        replay.measurement(-1).value,
    )
    assert isinstance(direct_index, torch.Tensor)
    assert isinstance(replay_index, torch.Tensor)
    assert torch.equal(direct_index, replay_index)
    assert (
        torch.equal(torch_rng, torch.random.get_rng_state())
        and python_rng == random.getstate()
    )
    if cuda_rng is not None:
        assert all(
            torch.equal(a, b)
            for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True)
        )


def test_snapshots_order_and_name_before_kind(
    sampled_source: SourceFactory, sampled_plan: Planner, sampled_run: Runner
) -> None:
    source = sampled_source().exchange(1).measure_pair()
    output = source.projector("psi")
    options = fq.ExecutionOptions(shots=1, seed=17)
    plan = sampled_plan(
        source,
        options=options,
        outputs=(fq.expectation(output, name="mid_circuit_measurement"),),
    )
    frozen, text = source.to_ir(), plan.to_json()
    with pytest.raises(ValueError):
        source.measure_pair()
    assert source.to_ir() == frozen
    source.exchange(1)
    output.zero_()
    assert plan.to_json() == text
    result = sampled_run(plan)
    assert result.measurement("mid_circuit_measurement").kind.startswith("expectation")
    assert result.measurement(-1).kind == "mid_circuit_measurement"
    other_seed = sampled_plan(
        sampled_source().exchange(1).measure_pair(),
        options=fq.ExecutionOptions(shots=1, seed=18),
    )
    other_order = sampled_plan(
        sampled_source().measure_pair().exchange(1), options=options
    )
    original = sampled_plan(
        sampled_source().exchange(1).measure_pair(), options=options
    )
    assert len({original.identity, other_seed.identity, other_order.identity}) == 3


def test_explicit_sampling_and_numeric_resource_boundary(
    sampled_source: SourceFactory, sampled_plan: Planner
) -> None:
    source = sampled_source().measure_pair()
    for options in (
        fq.ExecutionOptions(seed=17),
        fq.ExecutionOptions(shots=1),
        fq.ExecutionOptions(shots=2, seed=17),
        fq.ExecutionOptions(shots=1, seed=17, memory_limit_bytes=215),
    ):
        with pytest.raises(CapabilityError):
            sampled_plan(source, options=options)
    sampled_plan(
        source, options=fq.ExecutionOptions(shots=1, seed=17, memory_limit_bytes=216)
    )


def test_shared_sampler_single_index_statistical_witness() -> None:
    rng = torch.random.get_rng_state().clone()
    generator = torch.Generator(device="cpu").manual_seed(17)
    probabilities = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    count = sum(
        int(_draw_probability_indices(probabilities, 1, generator=generator)[0, 0])
        for _ in range(4096)
    )
    assert 1856 <= count <= 2240
    assert torch.equal(rng, torch.random.get_rng_state())


@pytest.mark.parametrize("sampled_first", [False, True])
def test_sampled_and_unread_observations_are_mutually_exclusive(
    sampled_source: SourceFactory, sampled_first: bool
) -> None:
    source = sampled_source().exchange(1)
    if sampled_first:
        source.measure_pair()
    else:
        source.dephase_pair()
    frozen = source.to_ir()
    with pytest.raises(ValueError):
        if sampled_first:
            source.dephase_pair()
        else:
            source.measure_pair()
    assert source.to_ir() == frozen


@pytest.mark.parametrize(
    "change",
    [
        {"device": "cuda"},
        {"backend": "jax"},
        {"require_gradients": True},
        {"allow_backend_fallback": True},
        {"allow_approximate": True},
        {"batch_size": 2},
        {"mode": "statevector"},
        {"target": "samples"},
    ],
)
def test_unsupported_profiles_fail_before_execution(
    sampled_source: SourceFactory, sampled_plan: Planner, change: dict[str, object]
) -> None:
    from dataclasses import replace

    options = replace(fq.ExecutionOptions(shots=1, seed=17), **change)
    with pytest.raises(CapabilityError):
        sampled_plan(sampled_source().exchange(1).measure_pair(), options=options)
