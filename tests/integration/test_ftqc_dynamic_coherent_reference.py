"""Finite E-H Born and conditional references, not trajectory certification."""

from __future__ import annotations

import cmath
import itertools
import math
from collections.abc import Iterator
from dataclasses import dataclass

import pytest
import torch

from flagquantum.dynamic import DynamicCircuit
from tests.integration.test_ftqc_coherent_reference import _A, _B, _clifford_vector

pytestmark = pytest.mark.integration
_ATOL = 1e-12
_CASES = ("E0", "E1", "E2")


@dataclass(frozen=True)
class _RecordReference:
    coefficients: tuple[complex, ...]
    branches: tuple[torch.Tensor, ...]
    direct: torch.Tensor
    analytic: torch.Tensor

    def coherent(self) -> torch.Tensor:
        return sum(
            (
                coefficient * branch
                for coefficient, branch in zip(
                    self.coefficients, self.branches, strict=True
                )
            ),
            torch.zeros(4, dtype=torch.complex128),
        )


def _projected_terms(
    name: str, record: int
) -> tuple[tuple[complex, ...], tuple[torch.Tensor, ...]]:
    if name not in _CASES or type(record) is not int or record not in (0, 1):
        raise ValueError("outside the approved E-H descriptions")
    coefficients: list[complex] = []
    branches: list[torch.Tensor] = []
    for prefix_choice, tail_choice in itertools.product(
        (0,) if name == "E0" else (0, 1), (0, 1)
    ):
        instructions: list[tuple[str, tuple[int, ...]]] = [("h", (0,))]
        coefficient = 1 + 0j
        if name != "E0":
            coefficient *= _B if prefix_choice else _A
            if prefix_choice:
                instructions.append(("s", (0,)))
        instructions.extend(
            [("h", (0,)), ("cx", (0, 1)), ("p1" if record else "p0", (0,))]
        )
        if name == "E2" and record:
            instructions.append(("x", (0,)))
        if record:
            instructions.append(("x", (1,)))
        instructions.append(("h", (1,)))
        coefficient *= _B if tail_choice else _A
        if tail_choice:
            instructions.append(("s", (1,)))
        instructions.append(("h", (1,)))
        coefficients.append(coefficient)
        branches.append(_clifford_vector(2, tuple(instructions)))
    return tuple(coefficients), tuple(branches)


def _direct_record(name: str, record: int) -> torch.Tensor:
    # Hand-written matrices use direct T phases, not the Clifford term evaluator.
    identity = torch.eye(2, dtype=torch.complex128)
    h = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / math.sqrt(2)
    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    t = torch.diag(
        torch.tensor([1, cmath.exp(1j * math.pi / 4)], dtype=torch.complex128)
    )
    cx = torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=torch.complex128
    )
    state = torch.tensor([1, 0, 0, 0], dtype=torch.complex128)
    state = torch.kron(h, identity) @ state
    if name != "E0":
        state = torch.kron(t, identity) @ state
    state = cx @ torch.kron(h, identity) @ state
    projector = torch.diag(torch.tensor([1 - record, record], dtype=torch.complex128))
    state = torch.kron(projector, identity) @ state
    if name == "E2" and record == 1:
        state = torch.kron(x, identity) @ state
    if record == 1:
        state = torch.kron(identity, x) @ state
    return torch.kron(identity, h @ t @ h) @ state


def _weight(state: torch.Tensor) -> float:
    if (
        state.shape != (4,)
        or state.dtype != torch.complex128
        or state.device.type != "cpu"
        or not bool(torch.isfinite(state).all())
    ):
        raise ValueError("reference requires finite ordered CPU complex128 amplitudes")
    weight = float(state.abs().square().sum())
    if not math.isfinite(weight):
        raise ValueError("reference Born weight must be finite")
    return weight


def _density(state: torch.Tensor) -> torch.Tensor:
    _weight(state)
    return state[:, None] * state.conj()[None, :]


def _conditional(state: torch.Tensor, analytic: float, direct: float) -> torch.Tensor:
    weight = _weight(state)
    if (
        not math.isfinite(analytic)
        or not math.isfinite(direct)
        or analytic < 0
        or direct < 0
        or abs(weight - analytic) > _ATOL
        or abs(weight - direct) > _ATOL
    ):
        raise ValueError(
            "unnormalized Born weight disagrees with independent references"
        )
    if analytic == 0 or weight == 0:
        raise ValueError("zero outcome has no conditional state")
    conditional = state / math.sqrt(weight)
    if abs(_weight(conditional) - 1) > _ATOL:
        raise ValueError("conditional reference lost unit norm")
    return conditional


@pytest.fixture(scope="module", autouse=True)
def _one_cpu_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def records() -> dict[str, tuple[_RecordReference, ...]]:
    w = cmath.exp(1j * math.pi / 4)
    alpha, beta = (1 + w) / 2, (1 - w) / 2
    u = torch.tensor([alpha, beta], dtype=torch.complex128)
    result: dict[str, tuple[_RecordReference, ...]] = {}
    for name in _CASES:
        outcomes = []
        for record in (0, 1):
            coefficients, branches = _projected_terms(name, record)
            analytic = torch.zeros(4, dtype=torch.complex128)
            start = 2 if name == "E1" and record else 0
            amplitude = (
                (1 if record == 0 else 0)
                if name == "E0"
                else (alpha if record == 0 else beta)
            )
            analytic[start : start + 2] = amplitude * u
            outcomes.append(
                _RecordReference(
                    coefficients, branches, _direct_record(name, record), analytic
                )
            )
        result[name] = tuple(outcomes)
    assert (
        sum(len(outcome.branches) for case in result.values() for outcome in case) == 20
    )
    return result


@pytest.mark.parametrize("name", _CASES)
def test_record_born_weights_conditional_states_and_public_density(
    name: str, records: dict[str, tuple[_RecordReference, ...]]
) -> None:
    outcomes = records[name]
    states = [outcome.coherent() for outcome in outcomes]
    weights = [_weight(state) for state in states]
    h, low = (2 + math.sqrt(2)) / 4, (2 - math.sqrt(2)) / 4
    analytic_weights = (1.0, 0.0) if name == "E0" else (h, low)
    assert abs(sum(weights) - 1) <= _ATOL and all(weight >= 0 for weight in weights)
    for state, outcome, expected in zip(
        states, outcomes, analytic_weights, strict=True
    ):
        torch.testing.assert_close(state, outcome.direct, atol=_ATOL, rtol=0)
        torch.testing.assert_close(state, outcome.analytic, atol=_ATOL, rtol=0)
        assert abs(_weight(outcome.direct) - expected) <= _ATOL
        if expected == 0:
            with pytest.raises(ValueError, match="zero outcome"):
                _conditional(state, expected, _weight(outcome.direct))
        else:
            conditional = _conditional(state, expected, _weight(outcome.direct))
            torch.testing.assert_close(
                _density(conditional),
                _density(outcome.analytic / math.sqrt(expected)),
                atol=_ATOL,
                rtol=0,
            )
    aggregate = _density(states[0]) + _density(states[1])
    torch.testing.assert_close(
        aggregate,
        _density(outcomes[0].direct) + _density(outcomes[1].direct),
        atol=_ATOL,
        rtol=0,
    )
    circuit = DynamicCircuit(2, dtype=torch.complex128)
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
    actual = circuit.density_matrix()
    assert actual.shape == (1, 4, 4) and actual.dtype == torch.complex128
    assert actual.device.type == "cpu" and bool(torch.isfinite(actual).all())
    assert abs(float(actual[0].trace().real) - 1) <= _ATOL
    assert abs(float(aggregate.trace().real) - 1) <= _ATOL
    torch.testing.assert_close(actual[0], aggregate, atol=_ATOL, rtol=0)


def test_term_mixture_and_term_normalization_lose_born_interference(
    records: dict[str, tuple[_RecordReference, ...]],
) -> None:
    outcome = records["E1"][0]
    mixture = sum(
        abs(c) ** 2 * _weight(branch)
        for c, branch in zip(outcome.coefficients, outcome.branches, strict=True)
    )
    assert abs(mixture - _weight(outcome.analytic)) > _ATOL
    wrong = torch.zeros(4, dtype=torch.complex128)
    for coefficient, branch in zip(outcome.coefficients, outcome.branches, strict=True):
        weight = _weight(branch)
        if weight != 0:
            wrong += coefficient * branch / math.sqrt(weight)
    assert float((_density(wrong) - _density(outcome.analytic)).abs().max()) > _ATOL


def test_exclusive_records_must_not_keep_inter_record_cross_terms(
    records: dict[str, tuple[_RecordReference, ...]],
) -> None:
    first, second = (outcome.coherent() for outcome in records["E1"])
    correct = _density(first) + _density(second)
    wrong = _density(first + second)
    assert float((wrong - correct).abs().max()) > _ATOL


@pytest.mark.parametrize(
    "name,wrong_record", [("E1", False), ("E1", True), ("E2", False)]
)
def test_omitted_wrong_record_or_reset_lost_record_correction_is_observable(
    name: str, wrong_record: bool, records: dict[str, tuple[_RecordReference, ...]]
) -> None:
    # Q=HTH commutes with X: undoing correction is an X on the final target.
    first, second = (outcome.coherent() for outcome in records[name])
    wrong_second = second[[1, 0, 3, 2]]
    wrong_first = first[[1, 0, 3, 2]] if wrong_record else first
    wrong = _density(wrong_first) + _density(wrong_second)
    correct = _density(first) + _density(second)
    assert float((wrong - correct).abs().max()) > _ATOL


def test_reset_aggregate_alone_cannot_qualify_record_weights(
    records: dict[str, tuple[_RecordReference, ...]],
) -> None:
    first, second = (outcome.coherent() for outcome in records["E2"])
    unit = _conditional(
        first, _weight(records["E2"][0].analytic), _weight(records["E2"][0].direct)
    )
    torch.testing.assert_close(
        _density(unit), _density(first) + _density(second), atol=_ATOL, rtol=0
    )
    assert abs(_weight(first) - 1) > _ATOL


@pytest.mark.parametrize("defect", ["shape", "nonfinite", "dtype", "weight", "zero"])
def test_invalid_conditional_reference_is_refused_before_division(
    defect: str, records: dict[str, tuple[_RecordReference, ...]]
) -> None:
    outcome = records["E0"][0]
    state = outcome.coherent().clone()
    expected, direct = 1.0, 1.0
    if defect == "shape":
        state = state.reshape(2, 2)
    elif defect == "nonfinite":
        state[0] = torch.tensor(complex(float("nan"), 0), dtype=state.dtype)
    elif defect == "dtype":
        state = state.to(torch.complex64)
    elif defect == "weight":
        state *= 2
    else:
        state = records["E0"][1].coherent()
        expected, direct = 0.0, 0.0
    with pytest.raises(ValueError):
        _conditional(state, expected, direct)


@pytest.mark.parametrize("name,record", [("E3", 0), ("E1", 2), ("E2", True)])
def test_unapproved_descriptions_are_refused(name: str, record: int) -> None:
    with pytest.raises(ValueError, match="approved E-H"):
        _projected_terms(name, record)
