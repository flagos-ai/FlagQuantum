"""Replaceable valid-input consumers for approved proposal 083.

The sampler fixture now consumes the admitted Runtime implementation. The
independent consumers and their physical/statistical assertions are retained
unchanged from the mathematical baseline.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

import pytest
import torch

from examples.terminal_instrument_reference import fock_reference
from flagquantum.core.ir import _decode_value
from flagquantum.operators import SuperOperator
from flagquantum.runtime.measurements import _DensityMatrixTarget
from flagquantum.runtime.result import MeasurementResult

pytestmark = pytest.mark.unit
Instrument = dict[str, tuple[torch.Tensor, ...]]


class TerminalSampler(Protocol):
    def __call__(
        self,
        density_matrix: torch.Tensor,
        instrument: Mapping[str, Sequence[torch.Tensor]],
        *,
        shots: int,
        seed: int | None = None,
    ) -> MeasurementResult: ...


@pytest.fixture
def terminal_sampler() -> TerminalSampler:
    from flagquantum.runtime import sample_terminal

    return sample_terminal


@pytest.fixture
def presence_instrument() -> Callable[[float, torch.dtype], Instrument]:
    def build(efficiency: float, dtype: torch.dtype) -> Instrument:
        return {
            "no_flag": (
                torch.diag(
                    torch.tensor([1, 1, math.sqrt(1 - efficiency)], dtype=dtype)
                ),
            ),
            "flag": (
                torch.diag(torch.tensor([0, 0, math.sqrt(efficiency)], dtype=dtype)),
            ),
        }

    return build


@pytest.fixture
def fock_instrument() -> Callable[[torch.dtype], Instrument]:
    def build(dtype: torch.dtype) -> Instrument:
        return {
            label: tuple(k.to(dtype) for k in operators)
            for label, operators in fock_reference()[1].items()
        }

    return build


def density(
    weights: Sequence[float], dtype: torch.dtype, coherent: bool
) -> torch.Tensor:
    diagonal = torch.tensor(weights, dtype=torch.float64).to(dtype)
    state = diagonal.sqrt()
    return torch.outer(state, state.conj()) if coherent else torch.diag(diagonal)


def assert_draws(
    result: MeasurementResult, expected: torch.Tensor, *, shots: int = 20_000
) -> None:
    assert result.kind == "sample" and result.qubits == ()
    assert result.statistics == {} and result.schema_version == "1.0"
    assert isinstance(result.value, torch.Tensor)
    assert result.value.dtype == torch.int64 and result.value.device.type == "cpu"
    assert result.shots == shots
    assert result.value.shape == (*expected.shape[:-1], shots)
    assert set(result.metadata) == {
        "outcome_labels",
        "probabilities",
        "probability_correction",
        "terminal_instrument",
    }
    budget = 1e-10 if expected.dtype == torch.float64 else 2e-5
    assert 0 <= result.metadata["probability_correction"] <= budget
    torch.testing.assert_close(
        torch.tensor(result.metadata["probabilities"], dtype=expected.dtype),
        expected,
        atol=budget,
        rtol=0,
    )
    assert result.shots is not None
    rows = result.value.reshape(-1, result.shots)
    for row, probabilities in zip(
        rows, expected.reshape(-1, expected.shape[-1]), strict=True
    ):
        frequencies = (
            torch.bincount(row, minlength=expected.shape[-1]).to(expected.dtype)
            / result.shots
        )
        bound = 6 * torch.sqrt(probabilities * (1 - probabilities) / result.shots)
        assert torch.all(
            (frequencies - probabilities).abs() <= bound + 2 / result.shots
        )


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("efficiency", [0.0, 0.8, 1.0])
@pytest.mark.parametrize(
    "weights",
    [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.6, 0.0, 0.4),
        (0.2, 0.3, 0.5),
    ],
)
@pytest.mark.parametrize("coherent", [False, True])
def test_presence_probabilities_conditionals_and_unread_backaction(
    terminal_sampler: TerminalSampler,
    presence_instrument: Callable[[float, torch.dtype], Instrument],
    dtype: torch.dtype,
    efficiency: float,
    weights: tuple[float, float, float],
    coherent: bool,
) -> None:
    rho = density(weights, dtype, coherent)
    branches = presence_instrument(efficiency, dtype)
    real_dtype = rho.real.dtype
    probability = efficiency * weights[2]
    expected = torch.tensor([1 - probability, probability], dtype=real_dtype)
    result = terminal_sampler(rho, branches, shots=20_000, seed=7)
    assert result.metadata["outcome_labels"] == ["no_flag", "flag"]
    assert_draws(result, expected)
    quiet, flag = (SuperOperator.from_kraus(k).apply(rho) for k in branches.values())
    budget = 1e-10 if dtype == torch.complex128 else 2e-5
    torch.testing.assert_close(
        sum(k.mH @ k for operators in branches.values() for k in operators),
        torch.eye(3, dtype=dtype),
        atol=budget,
        rtol=0,
    )
    torch.testing.assert_close(
        quiet.diagonal().real,
        torch.tensor(
            [weights[0], weights[1], weights[2] * (1 - efficiency)], dtype=real_dtype
        ),
        atol=budget,
        rtol=0,
    )
    assert (quiet + flag)[0, 2].real.item() == pytest.approx(
        math.sqrt(weights[0] * weights[2] * (1 - efficiency)) if coherent else 0,
        abs=budget,
    )
    if probability < 1:
        assert (quiet[0, 0].real / (1 - probability)).item() == pytest.approx(
            weights[0] / (1 - probability), abs=budget
        )
    else:
        torch.testing.assert_close(quiet, torch.zeros_like(quiet), atol=budget, rtol=0)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("batch_size", [None, 1, 2])
def test_fock_number_statistics_batch_axes_and_dephasing(
    terminal_sampler: TerminalSampler,
    fock_instrument: Callable[[torch.dtype], Instrument],
    dtype: torch.dtype,
    batch_size: int | None,
) -> None:
    rows = [(0.6, 0.3, 0.1), (0.1, 0.2, 0.7)]
    rho = density(rows[0], dtype, True)
    expected = torch.tensor(rows[0], dtype=rho.real.dtype)
    if batch_size is not None:
        rho = torch.stack([density(row, dtype, True) for row in rows[:batch_size]])
        expected = torch.tensor(rows[:batch_size], dtype=rho.real.dtype)
    branches = fock_instrument(dtype)
    torch.testing.assert_close(
        sum(k.mH @ k for operators in branches.values() for k in operators),
        torch.eye(3, dtype=dtype),
        atol=1e-10 if dtype == torch.complex128 else 2e-5,
        rtol=0,
    )
    result = terminal_sampler(rho, branches, shots=20_000, seed=11)
    assert result.metadata["outcome_labels"] == ["0", "1", "2"]
    assert_draws(result, expected)
    unread = sum(SuperOperator.from_kraus(k).apply(rho) for k in branches.values())
    torch.testing.assert_close(
        unread,
        torch.diag_embed(expected).to(dtype),
        atol=1e-10 if dtype == torch.complex128 else 2e-5,
        rtol=0,
    )


def test_pair_flag_order_is_not_hidden_loss_or_actor_reindexing(
    terminal_sampler: TerminalSampler,
    presence_instrument: Callable[[float, torch.dtype], Instrument],
) -> None:
    local = presence_instrument(0.8, torch.complex128)
    pair = {
        f"{left}/{right}": (torch.kron(a[0], b[0]),)
        for left, a in local.items()
        for right, b in local.items()
    }
    rho = torch.kron(
        density((0.7, 0, 0.3), torch.complex128, False),
        density((0.4, 0, 0.6), torch.complex128, False),
    )
    result = terminal_sampler(rho, pair, shots=30_000, seed=5)
    assert result.metadata["outcome_labels"] == [
        "no_flag/no_flag",
        "no_flag/flag",
        "flag/no_flag",
        "flag/flag",
    ]
    assert_draws(
        result,
        torch.tensor([0.3952, 0.3648, 0.1248, 0.1152], dtype=torch.float64),
        shots=30_000,
    )


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_fock_probabilities_are_covariant_under_a_complex_basis_change(
    terminal_sampler: TerminalSampler,
    fock_instrument: Callable[[torch.dtype], Instrument],
    dtype: torch.dtype,
) -> None:
    rho = density((0.6, 0.3, 0.1), dtype, True)
    coordinates = torch.arange(3, dtype=rho.real.dtype)
    phase = 2 * math.pi * torch.outer(coordinates, coordinates) / 3
    unitary = torch.polar(torch.full_like(phase, 1 / math.sqrt(3)), phase).to(dtype)
    branches = {
        label: tuple(unitary @ k @ unitary.mH for k in operators)
        for label, operators in fock_instrument(dtype).items()
    }
    rotated = unitary @ rho @ unitary.mH
    result = terminal_sampler(rotated, branches, shots=20_000, seed=7)
    expected = torch.tensor([0.6, 0.3, 0.1], dtype=rho.real.dtype)
    assert (rotated.diagonal().real - expected).abs().max() > 0.1
    assert_draws(result, expected)
    unread = sum(SuperOperator.from_kraus(k).apply(rotated) for k in branches.values())
    torch.testing.assert_close(
        unread,
        unitary @ torch.diag(expected).to(dtype) @ unitary.mH,
        atol=1e-10 if dtype == torch.complex128 else 2e-5,
        rtol=0,
    )


def test_equal_effects_do_not_choose_the_instrument_backaction(
    terminal_sampler: TerminalSampler,
    fock_instrument: Callable[[torch.dtype], Instrument],
) -> None:
    projectors = [k[0] for k in fock_instrument(torch.complex128).values()]
    resolved = {"even": (projectors[0], projectors[2]), "odd": (projectors[1],)}
    coherent = {"even": (projectors[0] + projectors[2],), "odd": (projectors[1],)}
    rho = density((0.6, 0.3, 0.1), torch.complex128, True)
    for branches in (resolved, coherent):
        result = terminal_sampler(rho, branches, shots=20_000, seed=7)
        assert_draws(result, torch.tensor([0.7, 0.3], dtype=torch.float64))
        captured = result.metadata["terminal_instrument"]["branches"]
        for record, (label, operators) in zip(captured, branches.items(), strict=True):
            assert record["label"] == label and len(record["kraus"]) == len(operators)
            for encoded, operator in zip(record["kraus"], operators, strict=True):
                torch.testing.assert_close(
                    _decode_value(encoded), operator, atol=0, rtol=0
                )
    a = sum(SuperOperator.from_kraus(k).apply(rho) for k in resolved.values())
    b = sum(SuperOperator.from_kraus(k).apply(rho) for k in coherent.values())
    assert a[0, 2].abs().item() < 1e-10
    assert b[0, 2].real.item() == pytest.approx(math.sqrt(0.06), abs=1e-10)


@pytest.mark.parametrize("seed", [None, 0, 2**64 - 1])
def test_capture_seed_replay_order_and_global_rng_isolation(
    terminal_sampler: TerminalSampler,
    fock_instrument: Callable[[torch.dtype], Instrument],
    seed: int | None,
) -> None:
    matrices = list(fock_instrument(torch.complex128).values())
    branches = {"two": matrices[2], "zero": matrices[0], "one": matrices[1]}
    rho = density((0.6, 0.3, 0.1), torch.complex128, False)
    global_state = torch.random.get_rng_state().clone()
    result = terminal_sampler(rho, branches, shots=256, seed=seed)
    assert_draws(result, torch.tensor([0.1, 0.6, 0.3], dtype=torch.float64), shots=256)
    captured = result.metadata["terminal_instrument"]
    assert set(captured) == {
        "schema",
        "version",
        "dimension",
        "dtype",
        "branches",
        "requested_seed",
        "effective_seed",
        "torch_version",
    }
    assert all(set(branch) == {"label", "kraus"} for branch in captured["branches"])
    assert captured["requested_seed"] == seed
    assert 0 <= captured["effective_seed"] <= 2**64 - 1
    replay = terminal_sampler(rho, branches, shots=256, seed=captured["effective_seed"])
    assert torch.equal(result.value, replay.value)
    assert torch.equal(global_state, torch.random.get_rng_state())
    assert result.metadata["outcome_labels"] == ["two", "zero", "one"]
    assert captured["schema"] == "flagquantum.terminal_instrument"
    assert captured["version"] == "1.0" and captured["dimension"] == 3
    assert captured["dtype"] == "complex128"
    assert captured["torch_version"] == str(torch.__version__)
    before = json.dumps(dict(result.metadata), sort_keys=True)
    saved = matrices[0][0].clone()
    matrices[0][0].zero_()
    rho.zero_()
    assert json.dumps(dict(result.metadata), sort_keys=True) == before
    torch.testing.assert_close(
        _decode_value(captured["branches"][1]["kraus"][0]), saved
    )


@pytest.mark.parametrize("format", ["index", "bits"])
def test_existing_qubit_density_draw_preserves_seed_and_output_axes(
    format: str,
) -> None:
    probabilities = torch.tensor(
        [[0.6, 0.2, 0.1, 0.1], [0.1, 0.2, 0.3, 0.4]], dtype=torch.float64
    )
    target = _DensityMatrixTarget(
        torch.diag_embed(probabilities).to(torch.complex128), None
    )
    expected = torch.multinomial(
        probabilities, 64, replacement=True, generator=torch.Generator().manual_seed(7)
    )
    if format == "bits":
        expected = (expected.unsqueeze(-1) >> torch.tensor([1, 0])) & 1
    actual = target.sample(
        64, format=format, generator=torch.Generator().manual_seed(7)
    )
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)
