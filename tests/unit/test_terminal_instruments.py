"""Production refusals and boundary behavior for CPU terminal instruments."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, NoReturn

import pytest
import torch

from flagquantum.errors import CapabilityError, ValidationError
from flagquantum.runtime import _terminal_instruments as terminal
from flagquantum.runtime import sample_terminal
from flagquantum.runtime.measurements import _DensityMatrixTarget
from flagquantum.simulation.lindblad_request import EvolutionValidationError

pytestmark = pytest.mark.unit
Instrument = dict[str, tuple[torch.Tensor, ...]]


@pytest.fixture
def valid_request() -> tuple[torch.Tensor, Instrument]:
    state = torch.eye(2, dtype=torch.complex128) / 2
    operators = {
        str(i): (torch.diag(torch.eye(2, dtype=state.dtype)[i]),) for i in range(2)
    }
    return state, operators


def forbidden(*args: Any, **kwargs: Any) -> NoReturn:
    raise AssertionError("refusal must precede capture, generator creation and drawing")


@pytest.fixture
def before_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(terminal, "_encode_value", forbidden)
    monkeypatch.setattr(terminal, "_draw_probability_indices", forbidden)
    monkeypatch.setattr(torch, "Generator", forbidden)


@pytest.mark.parametrize("shots", [True, False, 0, -1, 1.5, None, torch.tensor(1)])
def test_invalid_shots_do_not_reach_sampling(
    valid_request: tuple[torch.Tensor, Instrument], before_sampling: None, shots: Any
) -> None:
    with pytest.raises(ValidationError, match="shots"):
        sample_terminal(*valid_request, shots=shots)


@pytest.mark.parametrize("seed", [True, False, -1, 2**64, 0.5, "0", torch.tensor(1)])
def test_invalid_seeds_do_not_reach_sampling(
    valid_request: tuple[torch.Tensor, Instrument], before_sampling: None, seed: Any
) -> None:
    with pytest.raises(ValidationError, match="seed"):
        sample_terminal(*valid_request, shots=4, seed=seed)


@pytest.mark.parametrize(
    "problem,error",
    [
        ("not_tensor", TypeError),
        ("ket", ValidationError),
        ("scalar", ValidationError),
        ("nonsquare", ValidationError),
        ("empty", ValidationError),
        ("empty_batch", ValidationError),
        ("rank_four", ValidationError),
        ("real_dtype", ValidationError),
        ("nan", ValidationError),
        ("infinity", ValidationError),
        ("nonhermitian", ValidationError),
        ("trace", ValidationError),
        ("negative", ValidationError),
        ("sparse", ValidationError),
        ("gradient", CapabilityError),
        ("device", CapabilityError),
    ],
)
def test_invalid_original_state_is_not_coerced_or_sampled(
    valid_request: tuple[torch.Tensor, Instrument],
    before_sampling: None,
    problem: str,
    error: type[Exception],
) -> None:
    state, operators = valid_request
    malformed: dict[str, Any] = {
        "not_tensor": [[0.5, 0], [0, 0.5]],
        "ket": torch.ones(2, dtype=state.dtype),
        "scalar": torch.tensor(1, dtype=state.dtype),
        "nonsquare": torch.ones(2, 3, dtype=state.dtype),
        "empty": torch.empty(0, 0, dtype=state.dtype),
        "empty_batch": torch.empty(0, 2, 2, dtype=state.dtype),
        "rank_four": state.reshape(1, 1, 2, 2),
        "real_dtype": state.real,
        "nan": state * torch.nan,
        "infinity": state * torch.inf,
        "nonhermitian": torch.tensor([[0.5, 0.1], [0, 0.5]], dtype=state.dtype),
        "trace": state * 0.5,
        "negative": torch.diag(torch.tensor([-0.1, 1.1], dtype=state.dtype)),
        "sparse": state.to_sparse(),
        "gradient": state.clone().requires_grad_(),
        "device": torch.empty(2, 2, device="meta", dtype=state.dtype),
    }
    with pytest.raises(error):
        sample_terminal(malformed[problem], operators, shots=4)


@pytest.mark.parametrize(
    "problem,error",
    [
        ("not_mapping", TypeError),
        ("empty_mapping", ValidationError),
        ("empty_label", ValidationError),
        ("blank_label", ValidationError),
        ("numeric_label", ValidationError),
        ("not_sequence", TypeError),
        ("generator", TypeError),
        ("single_tensor", TypeError),
        ("empty_branch", ValidationError),
        ("not_tensor", TypeError),
        ("shape", ValidationError),
        ("dtype", ValidationError),
        ("nan", ValidationError),
        ("infinity", ValidationError),
        ("gradient", CapabilityError),
        ("device", CapabilityError),
        ("sparse", ValidationError),
        ("missing_outcome", ValidationError),
        ("gain", ValidationError),
    ],
)
def test_invalid_instrument_never_becomes_implicit_postselection(
    valid_request: tuple[torch.Tensor, Instrument],
    before_sampling: None,
    problem: str,
    error: type[Exception],
) -> None:
    state, operators = valid_request
    identity = torch.eye(2, dtype=state.dtype)

    def unconsumed() -> Iterator[torch.Tensor]:
        yield forbidden()

    malformed: dict[str, Any] = {
        "not_mapping": [],
        "empty_mapping": {},
        "empty_label": {"": (identity,)},
        "blank_label": {" \t": (identity,)},
        "numeric_label": {1: (identity,)},
        "not_sequence": {"a": None},
        "generator": {"a": unconsumed()},
        "single_tensor": {"a": identity},
        "empty_branch": {"a": ()},
        "not_tensor": {"a": ([[1, 0], [0, 1]],)},
        "shape": {"a": (torch.ones(2, 3, dtype=state.dtype),)},
        "dtype": {"a": (identity.to(torch.complex64),)},
        "nan": {"a": (identity * torch.nan,)},
        "infinity": {"a": (identity * torch.inf,)},
        "gradient": {"a": (identity.clone().requires_grad_(),)},
        "device": {"a": (torch.empty(2, 2, device="meta", dtype=state.dtype),)},
        "sparse": {"a": (identity.to_sparse(),)},
        "missing_outcome": {"0": operators["0"]},
        "gain": {"a": (2 * identity,)},
    }
    with pytest.raises(error):
        sample_terminal(state, malformed[problem], shots=4)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_roundoff_negative_weight_is_captured_before_clipping(
    dtype: torch.dtype,
) -> None:
    tolerance = 1e-5 if dtype == torch.complex64 else 1e-10
    state = torch.diag(torch.tensor([-tolerance / 4, 1 + tolerance / 4], dtype=dtype))
    operators = {str(i): (torch.diag(torch.eye(2, dtype=dtype)[i]),) for i in range(2)}
    result = sample_terminal(state, operators, shots=32, seed=0)
    assert isinstance(result.value, torch.Tensor)
    assert torch.equal(result.value, torch.ones(32, dtype=torch.int64))
    assert result.metadata["probabilities"][0] == pytest.approx(-tolerance / 4)
    assert tolerance / 5 < result.metadata["probability_correction"] < tolerance / 2


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_density_tolerance_does_not_hide_invalid_combined_branch_weight(
    dtype: torch.dtype, before_sampling: None
) -> None:
    tolerance = 1e-5 if dtype == torch.complex64 else 1e-10
    state = torch.diag(
        torch.tensor(
            [-0.75 * tolerance, -0.75 * tolerance, 1 + 1.5 * tolerance], dtype=dtype
        )
    )
    instrument = {
        "negative": (torch.diag(torch.tensor([1, 1, 0], dtype=dtype)),),
        "other": (torch.diag(torch.tensor([0, 0, 1], dtype=dtype)),),
    }
    with pytest.raises(ValidationError, match="branch probabilities"):
        sample_terminal(state, instrument, shots=4)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("batch", [None, 1, 2])
def test_exact_numeric_ceiling_accounts_for_batch_dtype_and_output(
    dtype: torch.dtype, batch: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = torch.eye(2, dtype=dtype) / 2
    if batch is not None:
        state = state.repeat(batch, 1, 1)
    operators = {str(i): (torch.diag(torch.eye(2, dtype=dtype)[i]),) for i in range(2)}
    members = batch or 1
    expected_bytes = (members + 2) * 4 * state.element_size()
    expected_bytes += members * 2 * state.real.element_size() + members * 4 * 8
    monkeypatch.setattr(terminal, "DEFAULT_DENSE_MATRIX_BYTES", expected_bytes - 1)
    with monkeypatch.context() as refusing:
        refusing.setattr(terminal, "_encode_value", forbidden)
        refusing.setattr(torch, "Generator", forbidden)
        with pytest.raises(CapabilityError, match="numeric bytes"):
            sample_terminal(state, operators, shots=4)
    monkeypatch.setattr(terminal, "DEFAULT_DENSE_MATRIX_BYTES", expected_bytes)
    result = sample_terminal(state, operators, shots=4, seed=0)
    assert isinstance(result.value, torch.Tensor)
    assert result.value.shape == ((4,) if batch is None else (batch, 4))


def test_unrepresentable_sample_count_is_refused_before_allocation(
    valid_request: tuple[torch.Tensor, Instrument], before_sampling: None
) -> None:
    with pytest.raises(CapabilityError, match="ceiling"):
        sample_terminal(*valid_request, shots=2**80)


def test_density_refusal_retains_the_original_numeric_cause(
    valid_request: tuple[torch.Tensor, Instrument], before_sampling: None
) -> None:
    state, operators = valid_request
    with pytest.raises(ValidationError) as error:
        sample_terminal(state * 0.5, operators, shots=4)
    assert isinstance(error.value.__cause__, EvolutionValidationError)
    assert error.value.__cause__.code == "unnormalized_initial_state"


@pytest.mark.parametrize("batch", [None, 1, 2])
def test_one_coordinate_state_does_not_infer_any_qubits(batch: int | None) -> None:
    state = torch.ones(
        (1, 1) if batch is None else (batch, 1, 1), dtype=torch.complex128
    )
    result = sample_terminal(
        state, {"  present  ": (torch.ones(1, 1, dtype=state.dtype),)}, shots=3, seed=0
    )
    assert result.qubits == () and result.metadata["outcome_labels"] == ["  present  "]
    assert isinstance(result.value, torch.Tensor)
    assert result.value.shape == ((3,) if batch is None else (batch, 3))
    assert not bool(result.value.any())


def test_noncontiguous_matrix_inputs_preserve_their_coordinates() -> None:
    state = torch.tensor([[0.6, 0.1j], [-0.1j, 0.4]], dtype=torch.complex128).T
    operators = {
        str(i): (torch.diag(torch.eye(2, dtype=state.dtype)[i]).T,) for i in range(2)
    }
    assert not state.is_contiguous()
    result = sample_terminal(state, operators, shots=4, seed=0)
    assert result.metadata["probabilities"] == pytest.approx([0.6, 0.4], abs=1e-10)


def test_old_unseeded_density_draw_keeps_global_generator_semantics() -> None:
    density = torch.diag(torch.tensor([0.25, 0.75], dtype=torch.complex128)).unsqueeze(
        0
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        expected = torch.multinomial(
            torch.tensor([[0.25, 0.75]], dtype=torch.float64), 12, replacement=True
        )
        expected_state = torch.get_rng_state()
        torch.manual_seed(17)
        actual = _DensityMatrixTarget(density, None).sample(12, format="index")
        assert torch.equal(actual, expected)
        assert torch.equal(torch.get_rng_state(), expected_state)
