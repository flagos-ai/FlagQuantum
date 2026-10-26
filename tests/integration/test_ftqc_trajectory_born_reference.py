"""Approved finite I-L observation, not a frequency or RNG certificate."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from importlib import import_module
from typing import Any

import pytest
import torch

import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.integration.test_ftqc_dynamic_coherent_reference import (
    _conditional,
    _density,
    _RecordReference,
    _weight,
)
from tests.integration.test_ftqc_dynamic_coherent_reference import (
    _one_cpu_thread as _one_cpu_thread,
)
from tests.integration.test_ftqc_dynamic_coherent_reference import (
    records as records,
)

pytestmark = pytest.mark.integration
_ATOL = 1e-12
_CASES = ("E0", "E1", "E2")
_STRATEGIES = ("trajectory", "batched")
_NATURAL = tuple((name, strategy, None) for name in _CASES for strategy in _STRATEGIES)
_CONTROLLED = tuple(
    (name, strategy, record)
    for name in _CASES
    for strategy in _STRATEGIES
    for record in ((0,) if name == "E0" else (0, 1))
)
_STABLE = tuple((name, "stable", None) for name in _CASES)


@dataclass(frozen=True)
class _Draw:
    probabilities: torch.Tensor
    selected: torch.Tensor


@dataclass(frozen=True)
class _Observation:
    name: str
    strategy: str
    controlled_record: int | None
    shots: int
    draws: tuple[_Draw, ...]
    samples: torch.Tensor
    classical: torch.Tensor
    states: torch.Tensor | None


def _program(name: str) -> DynamicCircuit:
    if name not in _CASES:
        raise ValueError("outside the approved I-L descriptions")
    circuit = DynamicCircuit(2, dtype=torch.complex128, device="cpu")
    circuit.gate("h", 0)
    if name != "E0":
        circuit.gate("t", 0)
    circuit.gate("h", 0)
    circuit.gate("cx", (0, 1))
    circuit.measure(0, classical_bit=0)
    if name == "E2":
        circuit.reset(0)
    circuit.conditional("x", 1, classical_bit=0)
    circuit.gate("h", 1)
    circuit.gate("t", 1)
    circuit.gate("h", 1)
    return circuit


def _born(name: str) -> torch.Tensor:
    h = (2 + math.sqrt(2)) / 4
    return torch.tensor((1.0, 0.0) if name == "E0" else (h, 1 - h), dtype=torch.float64)


def _probability(values: torch.Tensor, expected: torch.Tensor) -> None:
    if (
        values.shape != expected.shape
        or values.ndim not in (1, 2)
        or values.dtype != torch.float64
        or values.device.type != "cpu"
        or not bool(torch.isfinite(values).all())
        or bool((values < 0).any())
        or bool((values.sum(dim=-1) <= 0).any())
    ):
        raise ValueError("finite nonnegative CPU float64 draw rows required")
    torch.testing.assert_close(
        values.sum(dim=-1), torch.ones_like(values.sum(dim=-1)), atol=_ATOL, rtol=0
    )
    torch.testing.assert_close(values, expected, atol=_ATOL, rtol=0)


def _positive_selection(
    values: torch.Tensor, expected: torch.Tensor, record: int
) -> torch.Tensor:
    _probability(values, expected)
    if type(record) is not int or record not in (0, 1):
        raise ValueError("invalid controlled record")
    if not bool((expected[..., record] > 0).all()) or not bool(
        (values[..., record] > 0).all()
    ):
        raise ValueError("zero-support outcome cannot be selected")
    return torch.full(values.shape[:-1] + (1,), record, dtype=torch.int64)


@pytest.fixture(scope="module")
def conditional_states(
    records: dict[str, tuple[_RecordReference, ...]],
) -> dict[tuple[str, int], torch.Tensor]:
    result = {}
    for name, outcomes in records.items():
        for record, outcome in enumerate(outcomes):
            state = outcome.coherent()
            expected = float(_born(name)[record])
            torch.testing.assert_close(state, outcome.direct, atol=_ATOL, rtol=0)
            torch.testing.assert_close(state, outcome.analytic, atol=_ATOL, rtol=0)
            if expected > 0:
                result[name, record] = _conditional(
                    state, expected, _weight(outcome.direct)
                )
    return result


def _observer(
    name: str, shots: int, controlled: int | None, draws: list[_Draw]
) -> Callable[..., torch.Tensor]:
    original = torch.multinomial

    def observe(values: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        copied = values.detach().clone()
        if not draws and controlled is not None:
            expected = _born(name)
            if copied.ndim == 2:
                expected = expected.expand(shots, -1)
            selected = _positive_selection(copied, expected, controlled)
        else:
            selected = original(values, *args, **kwargs)
        draws.append(_Draw(copied, selected.detach().clone()))
        return selected

    return observe


@pytest.fixture(scope="module")
def observations(
    conditional_states: dict[tuple[str, int], torch.Tensor],
) -> dict[tuple[str, str, int | None], _Observation]:
    # All imports and construction precede observation; only these 19 calls execute.
    for module in (
        "flagquantum._api",
        "flagquantum.runtime.execution",
        "flagquantum.runtime.plan_execution",
    ):
        import_module(module)
    programs = {name: _program(name) for name in _CASES}
    options = fq.ExecutionOptions(
        shots=8,
        seed=53,
        device="cpu",
        allow_approximate=False,
        allow_backend_fallback=False,
    )
    original = torch.multinomial
    result = {}
    calls = trajectories = 0
    for name, strategy, controlled in (*_NATURAL, *_CONTROLLED, *_STABLE):
        shots = 1 if controlled is not None else 8
        draws: list[_Draw] = []

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                torch, "multinomial", _observer(name, shots, controlled, draws)
            )
            calls += 1
            trajectories += shots
            if strategy == "stable":
                public = fq.run(programs[name], options=options)
                assert (
                    public.state is None
                    and public.value is None
                    and public.plan is not None
                )
                assert public.samples is not None
                assert len(public.measurements) == 2
                final, record = public.measurements
                assert final.kind == "sample" and torch.equal(
                    final.value, public.samples
                )
                assert record.kind == "mid_circuit_measurement"
                assert final.qubits == (0, 1) and record.qubits == (0,)
                assert final.shots == record.shots == 8
                assert not hasattr(public, "final_states")
                samples, classical, states = public.samples, record.value, None
            else:
                native = run_dynamic(
                    programs[name], shots=shots, seed=53, strategy=strategy
                )
                samples, classical, states = (
                    native.final_samples,
                    native.classical_bits,
                    native.final_states,
                )
        assert torch.multinomial is original
        result[name, strategy, controlled] = _Observation(
            name, strategy, controlled, shots, tuple(draws), samples, classical, states
        )
    assert calls == 19 and trajectories == 82 and len(result) == 19
    return result


def _state(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if (
        actual.shape != (4,)
        or actual.dtype != torch.complex128
        or actual.device.type != "cpu"
    ):
        raise ValueError("ordered CPU complex128 conditional state required")
    assert abs(_weight(actual) - 1) <= _ATOL
    torch.testing.assert_close(_density(actual), _density(expected), atol=_ATOL, rtol=0)


def _check(
    observation: _Observation, conditional: dict[tuple[str, int], torch.Tensor]
) -> None:
    name, shots = observation.name, observation.shots
    assert observation.samples.shape == (shots, 2)
    assert observation.classical.shape == (shots, 1)
    for bits in (observation.samples, observation.classical):
        assert bits.dtype == torch.int64 and bits.device.type == "cpu"
        assert bool(((bits == 0) | (bits == 1)).all())
    batched = observation.draws[0].probabilities.ndim == 2
    if observation.strategy != "stable":
        assert batched == (observation.strategy == "batched")
    stages = 3 if name == "E2" else 2
    assert len(observation.draws) == (stages if batched else shots * stages)
    if observation.states is not None:
        assert observation.states.shape == (shots, 4)
    for shot in range(shots):
        row = []
        for stage in range(stages):
            draw = observation.draws[stage if batched else shot * stages + stage]
            assert draw.probabilities.shape == (
                (shots, 4 if stage == stages - 1 else 2)
                if batched
                else (4 if stage == stages - 1 else 2,)
            )
            assert draw.selected.shape == ((shots, 1) if batched else (1,))
            assert (
                draw.selected.dtype == torch.int64
                and draw.selected.device.type == "cpu"
            )
            row.append(
                (
                    draw.probabilities[shot] if batched else draw.probabilities,
                    int(draw.selected.reshape(-1)[shot if batched else 0]),
                )
            )
        first, record = row[0]
        _probability(first, _born(name))
        assert record in (0, 1) and float(_born(name)[record]) > 0
        assert record == int(observation.classical[shot, 0])
        if observation.controlled_record is not None:
            assert record == observation.controlled_record
        expected = conditional[name, record]
        if name == "E2":
            _probability(
                row[1][0], torch.tensor((1 - record, record), dtype=torch.float64)
            )
            assert row[1][1] == record
        terminal, basis = row[-1]
        _probability(terminal, expected.abs().square())
        assert 0 <= basis < 4 and float(terminal[basis]) > 0
        assert observation.samples[shot].tolist() == [basis >> 1, basis & 1]
        assert int(observation.samples[shot, 0]) == (0 if name == "E2" else record)
        if observation.states is not None:
            _state(observation.states[shot], expected)


@pytest.mark.parametrize("key", _NATURAL)
def test_natural_draw_inputs_and_continuations(
    key: tuple[str, str, None],
    observations: dict[tuple[str, str, int | None], _Observation],
    conditional_states: dict[tuple[str, int], torch.Tensor],
) -> None:
    _check(observations[key], conditional_states)


@pytest.mark.parametrize("key", _CONTROLLED)
def test_controlled_positive_branch_coverage(
    key: tuple[str, str, int],
    observations: dict[tuple[str, str, int | None], _Observation],
    conditional_states: dict[tuple[str, int], torch.Tensor],
) -> None:
    _check(observations[key], conditional_states)


@pytest.mark.parametrize("key", _STABLE)
def test_stable_projection_without_native_state_fields(
    key: tuple[str, str, None],
    observations: dict[tuple[str, str, int | None], _Observation],
    conditional_states: dict[tuple[str, int], torch.Tensor],
) -> None:
    _check(observations[key], conditional_states)


@pytest.mark.parametrize(
    "bad",
    [
        torch.tensor([float("nan"), 1.0], dtype=torch.float64),
        torch.tensor([-0.1, 1.1], dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
        torch.ones(3, dtype=torch.float64),
        torch.tensor([1.0, 0.0], dtype=torch.float32),
    ],
)
def test_invalid_draw_data_is_refused_before_reduction(bad: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        _probability(bad, _born("E1"))


def test_zero_branch_and_biased_substitution_are_refused() -> None:
    with pytest.raises(ValueError, match="zero-support"):
        _positive_selection(_born("E0"), _born("E0"), 1)
    with pytest.raises(AssertionError):
        _probability(torch.tensor([0.5, 0.5], dtype=torch.float64), _born("E1"))


def test_record_reset_terminal_order_and_phase_negatives_reuse_observations(
    observations: dict[tuple[str, str, int | None], _Observation],
    conditional_states: dict[tuple[str, int], torch.Tensor],
) -> None:
    for name in ("E1", "E2"):
        observed = observations[name, "batched", None]
        wrong = observed.classical.clone()
        wrong[0, 0] = 1 - wrong[0, 0]
        with pytest.raises(AssertionError):
            _check(replace(observed, classical=wrong), conditional_states)
    reset = observations["E2", "batched", None]
    draws = list(reset.draws)
    draws[1] = replace(draws[1], probabilities=_born("E2").expand(8, -1))
    with pytest.raises(AssertionError):
        _check(replace(reset, draws=tuple(draws)), conditional_states)
    observed = observations["E1", "batched", None]
    draws = list(observed.draws)
    draws[-1] = replace(
        draws[-1], probabilities=draws[-1].probabilities[:, [0, 2, 1, 3]]
    )
    with pytest.raises(AssertionError):
        _check(replace(observed, draws=tuple(draws)), conditional_states)
    assert observed.states is not None
    actual = observed.states[0].clone()
    nonzero = torch.nonzero(actual.abs() > 0).reshape(-1)
    actual[nonzero[-1]] *= -1
    torch.testing.assert_close(
        actual.abs().square(), observed.states[0].abs().square(), atol=0, rtol=0
    )
    record = int(observed.classical[0, 0])
    with pytest.raises(AssertionError):
        _state(actual, conditional_states["E1", record])
    with pytest.raises(AssertionError):
        _state(observed.states[0], conditional_states["E1", 1 - record])
    with pytest.raises(AssertionError):
        _state(observed.states[0] * 0.5, conditional_states["E1", record])
