"""Independent loss-channel witnesses for the private presence continuation."""

from __future__ import annotations

import math

import pytest
import torch

from flagquantum.core.rydberg import RydbergProgram
from flagquantum.errors import ExecutionError
from flagquantum.simulation.rydberg_presence import (
    _continue_rydberg_presence,
    _presence_model,
    _presence_operators,
    _rydberg_presence_prefix,
)

pytestmark = pytest.mark.unit


def program(efficiency: float) -> RydbergProgram:
    p = RydbergProgram(
        atoms=(("a", "home"),),
        sites={"home": (0.0, 0.0)},
        length_unit="um",
        time_unit="us",
        c6=0.0,
        loss_rate=0.2,
        minimum_separation=1.0,
        drive_channels=("laser",),
    )
    p.pulse("a", start=0.0, duration=1.0, amplitude=0.0, channel="laser")
    return p.measure_presence("a", at=0.5, efficiency=efficiency)


def loss(rho: torch.Tensor, duration: float) -> torch.Tensor:
    # Independent amplitude-loss Kraus matrices in the declared g/r/l basis.
    survival = math.exp(-0.2 * duration)
    k = torch.diag(
        torch.tensor(
            [math.sqrt(survival), math.sqrt(survival), 1.0], dtype=torch.complex128
        )
    )
    result = k @ rho @ k.mH
    for occupied in (0, 1):
        jump = torch.zeros((3, 3), dtype=torch.complex128)
        jump[2, occupied] = math.sqrt(1 - survival)
        result = result + jump @ rho @ jump.mH
    return result


@pytest.mark.parametrize("eta", [0.0, 0.8, 1.0])
@pytest.mark.parametrize("selected", [0, 1])
def test_selected_loss_trajectory_uses_raw_trace_and_absolute_time(
    eta: float, selected: int
) -> None:
    record = program(eta).to_ir()
    times = torch.linspace(0.0, 1.0, 129, dtype=torch.float64)
    numeric, bindings, boundary = _presence_model(record, times)
    assert bindings is None and boundary == 64
    initial = torch.tensor(
        [[0.5, 0.5, 0], [0.5, 0.5, 0], [0, 0, 0]], dtype=torch.complex128
    )
    saved = initial.clone()
    rng = torch.random.get_rng_state().clone()
    prefix = _rydberg_presence_prefix(numeric, initial, boundary)
    before = loss(initial, 0.5)
    assert torch.allclose(prefix[-1], before, atol=1e-10, rtol=0)
    diagonal = [1, 1, math.sqrt(1 - eta)] if selected == 0 else [0, 0, math.sqrt(eta)]
    independent = torch.diag(torch.tensor(diagonal, dtype=torch.complex128))
    branch = independent @ before @ independent.mH
    weight = torch.trace(branch).real
    if weight == 0:
        with pytest.raises(ExecutionError, match="positive finite trace"):
            _continue_rydberg_presence(record, numeric, prefix, selected)
    else:
        actual = _continue_rydberg_presence(record, numeric, prefix, selected)
        assert actual.shape == (129, 3, 3)
        assert torch.equal(actual[:boundary], prefix[:boundary])
        assert torch.allclose(actual[boundary], branch / weight, atol=1e-10, rtol=0)
        assert torch.allclose(
            actual[-1], loss(branch / weight, 0.5), atol=1e-10, rtol=0
        )
    assert torch.equal(initial, saved)
    assert torch.equal(torch.random.get_rng_state(), rng)
    k0, k1 = _presence_operators(record)
    assert torch.allclose(
        k0.mH @ k0 + k1.mH @ k1,
        torch.eye(3, dtype=torch.complex128),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize("pair,reverse", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("kind", ["ground", "rydberg", "coherent", "lost_coherence"])
@pytest.mark.parametrize("eta", [0.0, 0.8, 1.0])
def test_complete_density_matches_frozen_independent_oracle(
    pair: bool, reverse: bool, kind: str, eta: float
) -> None:
    from tests.unit.test_rydberg_presence_contract import _initial, _oracle, _program

    source = _program(RydbergProgram, pair, reverse).measure_presence(
        "a", at=0.5, efficiency=eta
    )
    record = source.to_ir()
    numeric, _, boundary = _presence_model(
        record, torch.linspace(0.0, 1.0, 129, dtype=torch.float64)
    )
    initial = _initial(pair, kind)
    prefix = _rydberg_presence_prefix(numeric, initial, boundary)
    saved = prefix.clone()
    for selected in (0, 1):
        try:
            expected, _ = _oracle(pair, initial, eta, selected, reverse=reverse)
        except ValueError:
            with pytest.raises(ExecutionError, match="positive finite trace"):
                _continue_rydberg_presence(record, numeric, prefix, selected)
        else:
            actual = _continue_rydberg_presence(record, numeric, prefix, selected)
            assert torch.allclose(actual, expected, atol=1e-10, rtol=0)
    assert torch.equal(prefix, saved)


@pytest.mark.parametrize("selected", [-1, 2, True, 0.0])
def test_invalid_selected_index_refused(selected: object) -> None:
    record = program(0.8).to_ir()
    numeric, _, boundary = _presence_model(record, [0.0, 0.5, 1.0])
    initial = torch.eye(3, dtype=torch.complex128) / 3
    prefix = _rydberg_presence_prefix(numeric, initial, boundary)
    with pytest.raises(ExecutionError, match="index 0 or 1"):
        _continue_rydberg_presence(record, numeric, prefix, selected)


def test_continuation_boundary_must_match_source() -> None:
    record = program(0.8).to_ir()
    numeric, _, _ = _presence_model(record, [0.0, 0.25, 0.5, 1.0])
    prefix = _rydberg_presence_prefix(
        numeric, torch.eye(3, dtype=torch.complex128) / 3, 1
    )
    with pytest.raises(ExecutionError, match="differs from source time"):
        _continue_rydberg_presence(record, numeric, prefix, 0)


@pytest.mark.parametrize("times", [[0.0, 1.0], [0.0, 0.25, 1.0]])
def test_absent_observation_grid_point_is_refused(times: list[float]) -> None:
    with pytest.raises(ExecutionError, match="distinct internal grid"):
        _presence_model(program(0.8).to_ir(), times)


@pytest.mark.parametrize("kind", ["batch", "gradient", "complex64"])
def test_initial_profile_refusals(kind: str) -> None:
    numeric, _, boundary = _presence_model(program(0.8).to_ir(), [0.0, 0.5, 1.0])
    initial = torch.eye(3, dtype=torch.complex128) / 3
    if kind == "batch":
        initial = initial.unsqueeze(0)
    elif kind == "gradient":
        initial.requires_grad_(True)
    else:
        initial = initial.to(torch.complex64)
    with pytest.raises(ExecutionError, match="without gradients"):
        _rydberg_presence_prefix(numeric, initial, boundary)


def test_third_actor_refused_before_kron(monkeypatch: pytest.MonkeyPatch) -> None:
    from flagquantum.errors import CapabilityError

    source = RydbergProgram(
        atoms=(("a", "a"), ("b", "b"), ("c", "c")),
        sites={"a": (0.0, 0.0), "b": (2.0, 0.0), "c": (4.0, 0.0)},
        length_unit="um",
        time_unit="us",
        c6=0.0,
        loss_rate=0.1,
        minimum_separation=1.0,
        drive_channels=("laser",),
    )
    source.pulse("a", start=0.0, duration=1.0, amplitude=0.0, channel="laser")
    source.measure_presence("a", at=0.5, efficiency=0.8)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("third actor reached tensor allocation")

    monkeypatch.setattr(torch, "kron", forbidden)
    with pytest.raises(CapabilityError, match="one or two actors"):
        _presence_operators(source.to_ir())


def test_interval_bindings_are_preserved_across_boundary() -> None:
    from tests.unit.test_rydberg_presence_contract import _initial, _oracle, _program

    source = _program(RydbergProgram, True, parameterized=True).measure_presence(
        "a", at=0.5, efficiency=0.8
    )
    values = [{"omega": 1.2 if i < 64 else 0.4} for i in range(128)]
    numeric, bindings, boundary = _presence_model(
        source.to_ir(),
        torch.linspace(0.0, 1.0, 129, dtype=torch.float64),
        parameter_values=values,
    )
    assert bindings == values
    initial = _initial(True, "coherent")
    prefix = _rydberg_presence_prefix(numeric, initial, boundary)
    for selected in (0, 1):
        actual = _continue_rydberg_presence(source.to_ir(), numeric, prefix, selected)
        expected, _ = _oracle(True, initial, 0.8, selected)
        assert torch.allclose(actual, expected, atol=1e-10, rtol=0)
