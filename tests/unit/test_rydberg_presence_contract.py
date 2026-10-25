"""Frozen proposal 092 consumers; replace only the four implementation factories."""

from __future__ import annotations

import math
import os
import random
import subprocess
import sys
from collections.abc import Callable
from functools import lru_cache
from typing import Any

import pytest
import torch

import flagquantum as fq
from flagquantum.core.ir import IRValidationError
from flagquantum.errors import CapabilityError, ExecutionError
from flagquantum.runtime import measurements
from flagquantum.runtime.result import ExecutionResult
from flagquantum.simulation.lindblad import EvolutionValidationError

pytestmark = pytest.mark.unit
DTYPE = torch.complex128


@pytest.fixture
def presence_source() -> Callable[..., Any]:
    from tests.unit.rydberg_presence_reference import ReferenceSource

    return ReferenceSource


@pytest.fixture
def presence_plan() -> Callable[..., Any]:
    from tests.unit.rydberg_presence_reference import plan

    return plan


@pytest.fixture
def presence_run() -> Callable[..., ExecutionResult]:
    from tests.unit.rydberg_presence_reference import run

    return run


@pytest.fixture
def presence_read() -> Callable[[str], Any]:
    from tests.unit.rydberg_presence_reference import read

    return read


def _program(
    factory: Callable[..., Any],
    pair: bool,
    reverse: bool = False,
    *,
    parameterized: bool = False,
) -> Any:
    roster = (
        (("b", "near"), ("a", "home")) if reverse else (("a", "home"), ("b", "near"))
    )
    source = factory(
        atoms=roster if pair else (("a", "home"),),
        sites={"home": (0.0, 0.0), "near": (2.0, 0.0), "far": (4.0, 0.0)},
        length_unit="um",
        time_unit="us",
        c6=32.0 if pair else 0.0,
        loss_rate=0.23,
        minimum_separation=1.0,
        transport_channels=("aod",),
        drive_channels=("laser_a", "laser_b"),
    )
    if pair:
        source.move(
            "b",
            source="near",
            destination="far",
            start=0.0,
            duration=1.0,
            channel="aod",
        )
    # Deliberately append the suffix first: source order is not physical time.
    actor_names = ("a", "b") if pair else ("a",)
    for name in actor_names:
        source.pulse(
            name,
            start=0.5,
            duration=0.5,
            amplitude=fq.Parameter("omega") if parameterized and name == "a" else 0.4,
            detuning=-0.5,
            phase=0.8,
            channel=f"laser_{name}",
        )
        source.pulse(
            name,
            start=0.0,
            duration=0.5,
            amplitude=(
                fq.Parameter("omega")
                if parameterized and name == "a"
                else (1.2 if name == "a" else 0.7)
            ),
            detuning=0.2,
            phase=0.3,
            channel=f"laser_{name}",
        )
    return source


def _embed(local: torch.Tensor, slot: int, count: int) -> torch.Tensor:
    result = torch.ones((1, 1), dtype=DTYPE)
    for index in range(count):
        result = torch.kron(
            result, local if index == slot else torch.eye(3, dtype=DTYPE)
        )
    return result


@lru_cache(maxsize=8)
def _propagators(pair: bool, intervals: int, reverse: bool) -> tuple[torch.Tensor, ...]:
    """Independent row-vectorized generator; no source/model/kernel matrices."""
    count = 2 if pair else 1
    dimension = 3**count
    names = ("b", "a") if reverse else (("a", "b") if pair else ("a",))
    excited = torch.diag(torch.tensor([0, 1, 0], dtype=DTYPE))
    lowering = torch.zeros((3, 3), dtype=DTYPE)
    lowering[0, 1] = 1
    identity = torch.eye(dimension, dtype=DTYPE)
    jumps = []
    for slot in range(count):
        for level in (0, 1):
            jump = torch.zeros((3, 3), dtype=DTYPE)
            jump[2, level] = math.sqrt(0.23)
            jumps.append(_embed(jump, slot, count))
    maps = []
    for index in range(intervals):
        midpoint = (index + 0.5) / intervals
        h = torch.zeros((dimension, dimension), dtype=DTYPE)
        for slot, name in enumerate(names):
            omega, delta, phase = (
                ((1.2 if name == "a" else 0.7), 0.2, 0.3)
                if midpoint < 0.5
                else (0.4, -0.5, 0.8)
            )
            drive = _embed(lowering, slot, count) * complex(
                math.cos(phase), math.sin(phase)
            )
            h += omega / 2 * (drive + drive.mH) - delta * _embed(excited, slot, count)
        if pair:
            h += (
                32
                / (2 + 2 * midpoint) ** 6
                * (_embed(excited, 0, count) @ _embed(excited, 1, count))
            )
        generator = -1j * (
            torch.kron(h, identity) - torch.kron(identity, h.T.contiguous())
        )
        for jump in jumps:
            effect = jump.mH @ jump
            generator += torch.kron(jump, jump.conj()) - 0.5 * (
                torch.kron(effect, identity)
                + torch.kron(identity, effect.T.contiguous())
            )
        maps.append(torch.matrix_exp(generator / intervals))
    return tuple(maps)


def _initial(pair: bool, kind: str) -> torch.Tensor:
    count = 2 if pair else 1
    local = torch.zeros(3, dtype=DTYPE)
    if kind == "ground":
        local[0] = 1
    elif kind == "rydberg":
        local[1] = 1
    elif kind == "coherent":
        local = torch.tensor([1, 1j, 0], dtype=DTYPE) / math.sqrt(2)
    else:
        local[0], local[2] = math.sqrt(0.4), math.sqrt(0.6)
    vector = local
    if count == 2:
        vector = torch.kron(local, local)
    return torch.outer(vector, vector.conj())


def _oracle(
    pair: bool,
    initial: torch.Tensor,
    eta: float,
    index: int,
    *,
    intervals: int = 128,
    reverse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    maps = _propagators(pair, intervals, reverse)
    rho = initial.clone()
    states = [rho]
    weights = torch.empty(2, dtype=torch.float64)
    slot = 1 if reverse else 0
    projectors = [
        _embed(torch.diag(torch.tensor(diagonal, dtype=DTYPE)), slot, 2 if pair else 1)
        for diagonal in ([1, 1, math.sqrt(1 - eta)], [0, 0, math.sqrt(eta)])
    ]
    for step, evolution in enumerate(maps):
        rho = (evolution @ rho.reshape(-1)).reshape(initial.shape)
        if step + 1 == intervals // 2:
            branches = [k @ rho @ k.mH for k in projectors]
            weights = torch.stack([torch.trace(x).real for x in branches])
            if float(weights[index]) <= 0:
                raise ValueError("zero contribution has no conditional density")
            rho = branches[index] / weights[index]
        states.append(rho)
    return torch.stack(states), weights


@pytest.mark.parametrize("pair", [False, True])
@pytest.mark.parametrize("initial", ["ground", "rydberg", "coherent", "lost_coherence"])
@pytest.mark.parametrize("eta", [0.0, 0.8, 1.0])
def test_complete_conditional_density_and_time_outputs(
    presence_source: Callable[..., Any],
    presence_plan: Callable[..., Any],
    presence_run: Callable[..., ExecutionResult],
    presence_read: Callable[[str], Any],
    pair: bool,
    initial: str,
    eta: float,
) -> None:
    source = _program(presence_source, pair)
    # The timed IR is canonical: this boundary precedes the suffix pulses.
    operation_index = sum(
        op.physical_data.start < 0.5
        for op in source.to_ir().functions[0].blocks[0].operations
    )
    source.measure_presence("a", at=0.5, efficiency=eta)
    state = _initial(pair, initial)
    times = torch.linspace(0, 1, 129, dtype=torch.float64)
    outputs = (fq.expectation(source.survival_operator(), name="survival"),)
    plan = presence_plan(
        source,
        state,
        times,
        outputs=outputs,
        return_density_matrices=True,
        options=fq.ExecutionOptions(shots=1, seed=17),
    )
    result = presence_run(plan)
    replay = presence_run(presence_read(plan.to_json()))
    observed = result.measurement("mid_circuit_measurement")
    assert isinstance(observed.value, torch.Tensor)
    assert observed.value.dtype == torch.int64 and observed.value.shape == (1, 1)
    chosen = int(observed.value[0, 0])
    expected, weights = _oracle(pair, state, eta, chosen)
    trajectory = result.measurement("density_matrix").value
    assert isinstance(trajectory, torch.Tensor) and result.state is not None
    torch.testing.assert_close(trajectory, expected, atol=1e-10, rtol=0)
    torch.testing.assert_close(result.state[0], expected[-1], atol=1e-10, rtol=0)
    assert result.state.shape == (1, state.shape[0], state.shape[1])
    assert result.state.dtype == DTYPE and result.state.device.type == "cpu"
    expected_survival = torch.einsum(
        "tij,ji->t", expected, source.survival_operator()
    ).real
    torch.testing.assert_close(
        result.expectation("survival"), expected_survival, atol=1e-10, rtol=0
    )
    assert result.expectation("survival").shape == (129,)
    assert [m.kind for m in result.measurements] == [
        "expectation",
        "density_matrix",
        "mid_circuit_measurement",
    ]
    assert result.value is None and result.samples is None
    assert set(observed.metadata) == {
        "outcome_labels",
        "probabilities",
        "probability_correction",
        "atom_id",
        "time_index",
        "time",
        "time_unit",
        "stage",
        "source_operation_index",
    }
    assert observed.qubits == () and observed.shots == 1
    assert observed.metadata["outcome_labels"] == ["no_flag", "flag"]
    assert observed.metadata["atom_id"] == "a" and observed.metadata["time_index"] == 64
    assert observed.metadata["time"] == 0.5 and observed.metadata["time_unit"] == "us"
    assert (
        observed.metadata["stage"] == "intermediate"
        and observed.metadata["source_operation_index"] == operation_index
    )
    torch.testing.assert_close(
        torch.tensor(observed.metadata["probabilities"], dtype=torch.float64)[0],
        weights,
        atol=1e-10,
        rtol=0,
    )
    initial_present = 0.4 if initial == "lost_coherence" else 1.0
    analytic_flag = eta * (1 - initial_present * math.exp(-0.23 * 0.5))
    assert abs(float(observed.metadata["probabilities"][0][1]) - analytic_flag) < 1e-10
    assert result.runtime["time_grid"] == times.tolist()
    assert result.runtime["state_axes"] == ["sample", "basis_row", "basis_column"]
    assert result.runtime["expectation_axes"] == ["time"]
    assert result.measurement("density_matrix").metadata["axes"] == [
        "time",
        "basis_row",
        "basis_column",
    ]
    assert replay.state is not None
    assert torch.equal(result.state, replay.state)
    assert torch.equal(result.expectation("survival"), replay.expectation("survival"))
    replay_index = replay.measurement("mid_circuit_measurement").value
    assert isinstance(replay_index, torch.Tensor)
    assert torch.equal(observed.value, replay_index)
    assert dict(observed.metadata) == dict(
        replay.measurement("mid_circuit_measurement").metadata
    )
    assert presence_read(plan.to_json()).to_json() == plan.to_json()


@pytest.mark.parametrize("index", [0, 1])
def test_both_branches_and_roster_permutation(
    presence_source: Callable[..., Any],
    presence_plan: Callable[..., Any],
    presence_run: Callable[..., ExecutionResult],
    monkeypatch: pytest.MonkeyPatch,
    index: int,
) -> None:
    def draw(
        probabilities: torch.Tensor, shots: int, *, generator: torch.Generator | None
    ) -> torch.Tensor:
        return torch.tensor([[index]], dtype=torch.int64)

    monkeypatch.setattr(measurements, "_draw_probability_indices", draw)
    results = []
    for reverse in [False, True]:
        source = _program(presence_source, True, reverse)
        source.measure_presence("a", at=0.5, efficiency=0.8)
        state = _initial(True, "coherent")
        plan = presence_plan(
            source,
            state,
            torch.linspace(0, 1, 129, dtype=torch.float64),
            options=fq.ExecutionOptions(shots=1, seed=17),
            return_density_matrices=True,
        )
        result = presence_run(plan)
        expected, _ = _oracle(True, state, 0.8, index, reverse=reverse)
        assert result.state is not None
        torch.testing.assert_close(result.state[0], expected[-1], atol=1e-10, rtol=0)
        results.append(result.state[0])
    swapped = results[1].reshape(3, 3, 3, 3).permute(1, 0, 3, 2).reshape(9, 9)
    torch.testing.assert_close(results[0], swapped, atol=1e-12, rtol=0)


def test_private_rng_snapshots_and_output_selection(
    presence_source: Callable[..., Any],
    presence_plan: Callable[..., Any],
    presence_run: Callable[..., ExecutionResult],
) -> None:
    source = _program(presence_source, False).measure_presence(
        "a", at=0.5, efficiency=0.8
    )
    state = _initial(False, "coherent")
    times = torch.linspace(0, 1, 65, dtype=torch.float64)
    operator = source.survival_operator()
    before_torch, before_python = torch.random.get_rng_state(), random.getstate()
    plan = presence_plan(
        source,
        state,
        times,
        outputs=fq.expectation(operator, name="survival"),
        options=fq.ExecutionOptions(shots=1, seed=2**64 - 1),
    )
    saved = plan.to_json()
    source.pulse("a", start=1.0, duration=0.5, amplitude=0.9, channel="laser_a")
    state.zero_()
    times.fill_(5)
    operator.zero_()
    result = presence_run(plan)
    assert plan.to_json() == saved
    assert result.expectation("survival").shape == (65,)
    assert all(m.kind != "density_matrix" for m in result.measurements)
    assert torch.equal(torch.random.get_rng_state(), before_torch)
    assert random.getstate() == before_python


@pytest.mark.parametrize(
    "options",
    [
        None,
        fq.ExecutionOptions(seed=1),
        fq.ExecutionOptions(shots=1),
        fq.ExecutionOptions(shots=2, seed=1),
        fq.ExecutionOptions(shots=1, seed=1, precision="complex64"),
        fq.ExecutionOptions(shots=1, seed=1, memory_limit_bytes=1000),
        fq.ExecutionOptions(shots=1, seed=1, require_gradients=True),
    ],
)
def test_unsupported_requests_fail_before_integration(
    presence_source: Callable[..., Any],
    presence_plan: Callable[..., Any],
    options: fq.ExecutionOptions | None,
) -> None:
    source = _program(presence_source, False).measure_presence(
        "a", at=0.5, efficiency=0.8
    )
    with pytest.raises(CapabilityError):
        presence_plan(
            source,
            _initial(False, "ground"),
            torch.linspace(0, 1, 65, dtype=torch.float64),
            options=options,
        )


@pytest.mark.parametrize("at", [0.0, 1.0, 0.51])
def test_internal_time_boundary_is_required(
    presence_source: Callable[..., Any],
    presence_plan: Callable[..., Any],
    at: float,
) -> None:
    source = _program(presence_source, False)
    with pytest.raises((IRValidationError, EvolutionValidationError)):
        source.measure_presence("a", at=at, efficiency=0.8)
        presence_plan(
            source,
            _initial(False, "ground"),
            torch.linspace(0, 1, 65, dtype=torch.float64),
            options=fq.ExecutionOptions(shots=1, seed=1),
        )


def test_selected_zero_weight_is_never_normalized(
    presence_source: Callable[..., Any],
    presence_plan: Callable[..., Any],
    presence_run: Callable[..., ExecutionResult],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _program(presence_source, False).measure_presence(
        "a", at=0.5, efficiency=0.0
    )
    plan = presence_plan(
        source,
        _initial(False, "ground"),
        torch.linspace(0, 1, 65, dtype=torch.float64),
        options=fq.ExecutionOptions(shots=1, seed=17),
    )

    def impossible(*args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.ones((1, 1), dtype=torch.int64)

    monkeypatch.setattr(measurements, "_draw_probability_indices", impossible)
    with pytest.raises(ExecutionError, match="zero-probability"):
        presence_run(plan)


@pytest.mark.parametrize("pair", [False, True])
@pytest.mark.parametrize("eta", [0.0, 0.8, 1.0])
def test_weighted_conditional_paths_recover_unread_continuation(
    presence_source: Callable[..., Any],
    presence_plan: Callable[..., Any],
    presence_run: Callable[..., ExecutionResult],
    monkeypatch: pytest.MonkeyPatch,
    pair: bool,
    eta: float,
) -> None:
    source = _program(presence_source, pair).measure_presence(
        "a", at=0.5, efficiency=eta
    )
    initial = _initial(pair, "lost_coherence")
    plan = presence_plan(
        source,
        initial,
        torch.linspace(0, 1, 129, dtype=torch.float64),
        options=fq.ExecutionOptions(shots=1, seed=17),
    )
    maps = _propagators(pair, 128, False)
    rho = initial.clone()
    for evolution in maps[:64]:
        rho = (evolution @ rho.reshape(-1)).reshape(initial.shape)
    matrices = [
        _embed(torch.diag(torch.tensor(d, dtype=DTYPE)), 0, 2 if pair else 1)
        for d in ([1, 1, math.sqrt(1 - eta)], [0, 0, math.sqrt(eta)])
    ]
    unread = sum((m @ rho @ m.mH for m in matrices), torch.zeros_like(rho))
    if eta:
        assert float(torch.linalg.norm(unread - rho)) > 0.01
    weights = [float(torch.trace(m @ rho @ m.mH).real) for m in matrices]
    for evolution in maps[64:]:
        unread = (evolution @ unread.reshape(-1)).reshape(initial.shape)
    combined = torch.zeros_like(initial)
    for index, weight in enumerate(weights):
        if weight == 0:
            continue

        def selected(*args: Any, _index: int = index, **kwargs: Any) -> torch.Tensor:
            return torch.tensor([[_index]], dtype=torch.int64)

        monkeypatch.setattr(measurements, "_draw_probability_indices", selected)
        result = presence_run(plan)
        assert result.state is not None
        combined += weight * result.state[0]
        assert abs(float(torch.trace(result.state[0]).real) - 1) < 1e-12
    torch.testing.assert_close(combined, unread, atol=1e-10, rtol=0)


@pytest.mark.parametrize("index", [0, 1])
def test_fixed_channel_algebra_uses_original_trace(
    presence_source: Callable[..., Any],
    presence_plan: Callable[..., Any],
    presence_run: Callable[..., ExecutionResult],
    monkeypatch: pytest.MonkeyPatch,
    index: int,
) -> None:
    source = presence_source(
        atoms=(("a", "home"),),
        sites={"home": (0.0, 0.0)},
        length_unit="um",
        time_unit="us",
        c6=0.0,
        loss_rate=0.0,
        minimum_separation=1.0,
        drive_channels=("laser",),
    )
    source.pulse("a", start=0.0, duration=1.0, amplitude=0.0, channel="laser")
    source.measure_presence("a", at=0.5, efficiency=0.8)
    scale = 1 + 5e-11  # Within the existing density validation budget.
    initial = _initial(False, "lost_coherence") * scale
    plan = presence_plan(
        source,
        initial,
        torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64),
        options=fq.ExecutionOptions(shots=1, seed=17),
        return_density_matrices=True,
    )

    def selected(*args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.tensor([[index]], dtype=torch.int64)

    monkeypatch.setattr(measurements, "_draw_probability_indices", selected)
    result = presence_run(plan)
    matrix = torch.diag(
        torch.tensor(
            [1.0, 1.0, math.sqrt(0.2)] if index == 0 else [0.0, 0.0, math.sqrt(0.8)],
            dtype=DTYPE,
        )
    )
    raw = matrix @ initial @ matrix.mH
    expected = raw / torch.trace(raw).real
    assert result.state is not None
    torch.testing.assert_close(result.state[0], expected, atol=1e-12, rtol=0)
    trajectory = result.measurement("density_matrix").value
    assert isinstance(trajectory, torch.Tensor)
    torch.testing.assert_close(trajectory[0], initial, atol=1e-12, rtol=0)
    torch.testing.assert_close(trajectory[1], expected, atol=1e-12, rtol=0)
    assert (
        abs(
            float(result.measurement(-1).metadata["probabilities"][0][1]) - 0.48 * scale
        )
        < 1e-12
    )
    assert result.measurement(-1).metadata["probability_correction"] > 1e-12


def test_independent_time_integration_convergence(
    presence_source: Callable[..., Any],
    presence_plan: Callable[..., Any],
    presence_run: Callable[..., ExecutionResult],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def selected(*args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.zeros((1, 1), dtype=torch.int64)

    monkeypatch.setattr(measurements, "_draw_probability_indices", selected)
    errors = []
    for intervals in (16, 32):
        source = _program(presence_source, False).measure_presence(
            "a", at=0.5, efficiency=0.8
        )
        # Full rank keeps the coarse RK grid inside the unchanged PSD budget.
        initial = 0.9 * _initial(False, "coherent") + torch.eye(3, dtype=DTYPE) / 30
        result = presence_run(
            presence_plan(
                source,
                initial,
                torch.linspace(0, 1, intervals + 1, dtype=torch.float64),
                options=fq.ExecutionOptions(shots=1, seed=17),
            )
        )
        expected, _ = _oracle(False, initial, 0.8, 0, intervals=intervals)
        assert result.state is not None
        errors.append(float(torch.linalg.norm(result.state[0] - expected[-1])))
    assert errors[0] > 0 and errors[1] < errors[0] / 8


def test_shared_private_single_draw_witness() -> None:
    before = torch.random.get_rng_state()
    generator = torch.Generator(device="cpu").manual_seed(17)
    probabilities = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    count = sum(
        int(
            measurements._draw_probability_indices(
                probabilities, 1, generator=generator
            )[0, 0]
        )
        for _ in range(4096)
    )
    assert 1856 <= count <= 2240
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize(
    "seed,exception",
    [(True, TypeError), (1.0, TypeError), (-1, ValueError), (2**64, ValueError)],
)
def test_seed_boundaries(
    presence_source: Callable[..., Any],
    presence_plan: Callable[..., Any],
    seed: Any,
    exception: type[Exception],
) -> None:
    source = _program(presence_source, False).measure_presence(
        "a", at=0.5, efficiency=0.8
    )
    with pytest.raises(exception):
        presence_plan(
            source,
            _initial(False, "ground"),
            torch.linspace(0, 1, 65, dtype=torch.float64),
            options=fq.ExecutionOptions(shots=1, seed=seed),
        )


@pytest.mark.parametrize("value", [torch.tensor(0.5), True, "0.5"])
def test_source_leaf_types_are_not_coerced(
    presence_source: Callable[..., Any], value: Any
) -> None:
    source = _program(presence_source, False)
    with pytest.raises(TypeError):
        source.measure_presence("a", at=value, efficiency=0.8)


@pytest.mark.parametrize("name", ["mid_circuit_measurement", "density_matrix"])
def test_output_name_collisions_are_refused(
    presence_source: Callable[..., Any], presence_plan: Callable[..., Any], name: str
) -> None:
    source = _program(presence_source, False).measure_presence(
        "a", at=0.5, efficiency=0.8
    )
    with pytest.raises(EvolutionValidationError):
        presence_plan(
            source,
            _initial(False, "ground"),
            torch.linspace(0, 1, 65, dtype=torch.float64),
            options=fq.ExecutionOptions(shots=1, seed=17),
            outputs=fq.expectation(source.survival_operator(), name=name),
            return_density_matrices=True,
        )


def test_snapshot_can_be_repeated_with_zero_seed(
    presence_source: Callable[..., Any],
    presence_plan: Callable[..., Any],
    presence_run: Callable[..., ExecutionResult],
    presence_read: Callable[[str], Any],
) -> None:
    source = _program(presence_source, False).measure_presence(
        "a", at=0.5, efficiency=0.8
    )
    p = presence_plan(
        source,
        _initial(False, "ground"),
        torch.linspace(0, 1, 65, dtype=torch.float64),
        options=fq.ExecutionOptions(shots=1, seed=0),
    )
    first, second = presence_run(p), presence_run(presence_read(p.to_json()))
    assert first.state is not None and second.state is not None
    assert torch.equal(first.state, second.state)
    assert first.runtime["seed"] == 0
    # A recipe can have its own identity; this asserts no production schema name.
    # Native-reader checks live in the compatibility test, outside frozen consumers.
    assert isinstance(p.identity, str) and p.identity


def test_gradients_are_refused_before_native_capture(
    presence_source: Callable[..., Any],
    presence_plan: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import flagquantum.lindblad as fql

    source = _program(presence_source, False).measure_presence(
        "a", at=0.5, efficiency=0.8
    )
    operator = source.survival_operator().requires_grad_()

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("native capture must not run")

    monkeypatch.setattr(fql, "plan", forbidden)
    with pytest.raises(CapabilityError):
        presence_plan(
            source,
            _initial(False, "ground"),
            torch.linspace(0, 1, 65, dtype=torch.float64),
            options=fq.ExecutionOptions(shots=1, seed=17),
            outputs=fq.expectation(operator, name="survival"),
        )


def test_ordinary_cpu_circuit_keeps_its_lazy_fast_path() -> None:
    code = """import sys
import torch
import flagquantum as fq
assert 'flagquantum.lindblad' not in sys.modules
result = fq.run(fq.Circuit(1).h(0))
assert result.state is not None
assert 'flagquantum.lindblad' not in sys.modules
assert not torch.distributed.is_initialized()
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=dict(os.environ),
        timeout=45,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_named_expectation_uses_density_name_when_trajectory_is_absent(
    presence_source: Callable[..., Any],
    presence_plan: Callable[..., Any],
    presence_run: Callable[..., ExecutionResult],
) -> None:
    source = _program(presence_source, False).measure_presence(
        "a", at=0.5, efficiency=0.8
    )
    plan = presence_plan(
        source,
        _initial(False, "ground"),
        torch.linspace(0, 1, 65, dtype=torch.float64),
        options=fq.ExecutionOptions(shots=1, seed=17),
        outputs=fq.expectation(source.survival_operator(), name="density_matrix"),
    )
    result = presence_run(plan)
    assert result.measurement("density_matrix").kind == "expectation"
    assert result.expectation("density_matrix").shape == (65,)


@pytest.mark.parametrize("eta", [1e-28, 1.0])
def test_tiny_positive_branch_is_distinct_from_zero(
    presence_source: Callable[..., Any],
    presence_plan: Callable[..., Any],
    presence_run: Callable[..., ExecutionResult],
    monkeypatch: pytest.MonkeyPatch,
    eta: float,
) -> None:
    source = _program(presence_source, False).measure_presence(
        "a", at=0.5, efficiency=eta
    )
    initial = torch.diag(torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE))
    plan = presence_plan(
        source,
        initial,
        torch.linspace(0, 1, 65, dtype=torch.float64),
        options=fq.ExecutionOptions(shots=1, seed=17),
    )

    def selected(*args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.ones((1, 1), dtype=torch.int64)

    monkeypatch.setattr(measurements, "_draw_probability_indices", selected)
    result = presence_run(plan)
    assert result.state is not None
    torch.testing.assert_close(result.state[0], initial, atol=1e-12, rtol=0)
    assert (
        abs(float(result.measurement(-1).metadata["probabilities"][0][1]) / eta - 1)
        < 1e-12
    )


@pytest.mark.parametrize("field,value", [("device", "cuda"), ("batch_size", 2)])
def test_device_and_batch_fail_before_capture(
    presence_source: Callable[..., Any],
    presence_plan: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
) -> None:
    import flagquantum.lindblad as fql

    source = _program(presence_source, False).measure_presence(
        "a", at=0.5, efficiency=0.8
    )

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("native capture must not run")

    monkeypatch.setattr(fql, "plan", forbidden)
    with pytest.raises(CapabilityError):
        presence_plan(
            source,
            _initial(False, "ground"),
            torch.linspace(0, 1, 65, dtype=torch.float64),
            options=fq.ExecutionOptions(shots=1, seed=17, **{field: value}),
        )


def test_duplicate_observation_is_not_a_second_event(
    presence_source: Callable[..., Any],
) -> None:
    source = _program(presence_source, False).measure_presence(
        "a", at=0.5, efficiency=0.8
    )
    with pytest.raises(IRValidationError):
        source.measure_presence("a", at=0.75, efficiency=0.8)


def test_full_interval_bindings_survive_the_observation_and_caller_edits(
    presence_source: Callable[..., Any],
    presence_plan: Callable[..., Any],
    presence_run: Callable[..., ExecutionResult],
    presence_read: Callable[[str], Any],
) -> None:
    source = _program(presence_source, True, parameterized=True).measure_presence(
        "a", at=0.5, efficiency=0.8
    )
    initial = _initial(True, "coherent")
    bindings = [
        {"omega": torch.tensor(1.2 if index < 64 else 0.4, dtype=torch.float64)}
        for index in range(128)
    ]
    plan = presence_plan(
        source,
        initial,
        torch.linspace(0, 1, 129, dtype=torch.float64),
        parameter_values=bindings,
        options=fq.ExecutionOptions(shots=1, seed=17),
    )
    saved = plan.to_json()
    for row in bindings:
        row["omega"].fill_(9)
    result = presence_run(plan)
    index_value = result.measurement(-1).value
    assert isinstance(index_value, torch.Tensor)
    expected, _ = _oracle(True, initial, 0.8, int(index_value[0, 0]))
    assert result.state is not None
    torch.testing.assert_close(result.state[0], expected[-1], atol=1e-10, rtol=0)
    replay = presence_run(presence_read(saved))
    assert replay.state is not None and torch.equal(result.state, replay.state)
    assert plan.to_json() == saved


def test_extending_the_output_grid_does_not_extend_the_control_horizon(
    presence_source: Callable[..., Any],
    presence_plan: Callable[..., Any],
) -> None:
    source = _program(presence_source, False)
    with pytest.raises(IRValidationError):
        source.measure_presence("a", at=1.5, efficiency=0.8)
        presence_plan(
            source,
            _initial(False, "ground"),
            torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0], dtype=torch.float64),
            options=fq.ExecutionOptions(shots=1, seed=17),
        )


@pytest.mark.parametrize("ambiguous", [False, True])
def test_observation_uses_existing_ulp_time_precision(
    presence_source: Callable[..., Any],
    presence_plan: Callable[..., Any],
    ambiguous: bool,
) -> None:
    at = 0.5 if ambiguous else 0.5 + 32 * math.ulp(0.5)
    grid = [0.0, 0.5, math.nextafter(0.5, 1.0), 1.0] if ambiguous else [0.0, 0.5, 1.0]
    source = _program(presence_source, False).measure_presence(
        "a", at=at, efficiency=0.8
    )
    with pytest.raises(EvolutionValidationError) as caught:
        presence_plan(
            source,
            _initial(False, "ground"),
            torch.tensor(grid, dtype=torch.float64),
            options=fq.ExecutionOptions(shots=1, seed=17),
        )
    assert (
        caught.value.code == "invalid_native_request" and caught.value.field == "times"
    )


def test_more_than_two_actors_refuses_before_operator_allocation(
    presence_source: Callable[..., Any],
    presence_plan: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = presence_source(
        atoms=(("a", "s0"), ("b", "s1"), ("c", "s2")),
        sites={"s0": (0.0, 0.0), "s1": (2.0, 0.0), "s2": (4.0, 0.0)},
        length_unit="um",
        time_unit="us",
        c6=0.0,
        loss_rate=0.0,
        minimum_separation=1.0,
        drive_channels=("laser",),
    )
    source.pulse("a", start=0.0, duration=1.0, amplitude=0.0, channel="laser")
    source.measure_presence("a", at=0.5, efficiency=0.8)
    initial = torch.zeros(27, dtype=DTYPE)
    initial[0] = 1

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("unsupported profile allocated exponential operators")

    monkeypatch.setattr(torch, "kron", forbidden)
    with pytest.raises(CapabilityError):
        presence_plan(
            source,
            initial,
            torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64),
            options=fq.ExecutionOptions(shots=1, seed=17),
        )
