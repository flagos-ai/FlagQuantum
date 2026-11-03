"""Finite example composition; no threshold or independent-engine claim."""

from __future__ import annotations

import dataclasses
import json
from statistics import NormalDist
from typing import Any, NamedTuple

import pytest

from examples.qec import surface_memory_sweep as ex
from flagquantum.qec import (
    DetectorErrorModel,
    Pauli,
    PhenomenologicalNoise,
    RotatedSurfaceCode,
    build_memory_circuit,
    get_decoder,
)
from flagquantum.qec.adapters.stim import (
    compile_stim_sampler,
    memory_to_stim_text,
    stim_circuit_to_dem_text,
)
from flagquantum.qec.circuits.memory import MemoryCircuit
from flagquantum.qec.execution.experiments import (
    MemoryExperimentResult,
    run_memory_experiment,
)

pytestmark = pytest.mark.unit


def _interval(k: int) -> tuple[float, float]:
    z2 = NormalDist().inv_cdf(0.975) ** 2

    def score(p: float) -> float:
        return (k - 35 * p) ** 2 - z2 * 35 * p * (1 - p)

    a, b = 0.0, k / 35
    for _ in range(64):
        m = (a + b) / 2
        if score(m) > 0:
            a = m
        else:
            b = m
    low = 0.0 if k == 0 else (a + b) / 2
    a, b = k / 35, 1.0
    for _ in range(64):
        m = (a + b) / 2
        if score(m) > 0:
            b = m
        else:
            a = m
    return low, 1.0 if k == 35 else (a + b) / 2


def _result(seed: int, k: int = 0) -> MemoryExperimentResult:
    return MemoryExperimentResult(
        35, k, k / 35, (k,), _interval(k), seed, "memory_circuit"
    )


def _rows(capsys: pytest.CaptureFixture[str]) -> list[dict[str, Any]]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


class _Call(NamedTuple):
    memory: MemoryCircuit
    noise: PhenomenologicalNoise
    decoder: str
    shots: int
    batch_size: int
    seed: int


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> list[_Call]:
    calls: list[_Call] = []
    monkeypatch.setattr(
        ex,
        "_versions",
        lambda: {
            "python": "test",
            "torch": "test",
            "stim": "test",
            "pymatching": "test",
        },
    )

    def run(
        memory: MemoryCircuit,
        *,
        noise: PhenomenologicalNoise,
        decoder: str,
        shots: int,
        batch_size: int,
        seed: int,
    ) -> MemoryExperimentResult:
        calls.append(_Call(memory, noise, decoder, shots, batch_size, seed))
        return _result(seed, seed - 53)

    monkeypatch.setattr(ex, "run_memory_experiment", run)
    return calls


def test_fixed_grid_has_eight_unique_keys() -> None:
    points = ex._points()
    assert len(points) == 8
    assert {(b, d, p) for b, d, p, _ in points} == {
        (b, d, p) for b in ("x", "z") for d in (3, 5) for p in ("low", "high")
    }


def test_explicit_seed_map() -> None:
    assert ex._points() == tuple(
        (b, d, p, 53 + i)
        for i, (b, d, p) in enumerate(
            (b, d, p) for b in ("x", "z") for d in (3, 5) for p in ("low", "high")
        )
    )


def test_frames_rounds_and_exact_noise(fake: list[_Call]) -> None:
    ex.main()
    for call, (basis, distance, profile, _) in zip(fake, ex._points(), strict=True):
        assert call.memory.readout_basis == basis
        assert len(call.memory.detectors) == 3 * (distance**2 - 1)
        assert call.noise == PhenomenologicalNoise(
            data_flip=0.005 if profile == "low" else 0.01,
            phase_flip=0.007 if profile == "low" else 0.014,
            measurement_flip=0.009 if profile == "low" else 0.018,
        )


def test_eight_calls_keep_shots_batch_and_decoder(fake: list[_Call]) -> None:
    ex.main()
    assert len(fake) == 8
    assert [(c.decoder, c.shots, c.batch_size, c.seed) for c in fake] == [
        ("pymatching", 35, 16, seed) for *_, seed in ex._points()
    ]


def test_per_point_counts_not_pooled(
    fake: list[_Call], capsys: pytest.CaptureFixture[str]
) -> None:
    ex.main()
    rows = _rows(capsys)[1:-1]
    assert [r["logical_failures"] for r in rows] == list(range(8))
    for k, row in enumerate(rows):
        assert row["completed_shots"] == 35 and row["logical_error_rate"] == k / 35
        assert row["observable_failures"] == [k]
        assert row["confidence_interval"] == pytest.approx(
            _interval(k), abs=1e-12, rel=0
        )


def test_zero_failure_upper_endpoint_positive() -> None:
    row = ex._result_fields(_result(53), 53)
    assert row["confidence_interval"] == _interval(0)
    assert _interval(0)[0] == 0 and _interval(0)[1] > 0


def test_reordering_preserves_keyed_seeds(
    fake: list[_Call], monkeypatch: pytest.MonkeyPatch
) -> None:
    points = ex._points()[::-1]
    monkeypatch.setattr(ex, "_points", lambda: points)
    ex.main()
    assert [c.seed for c in fake] == [p[3] for p in points]


def test_invalid_grid_before_first_call(
    fake: list[_Call], monkeypatch: pytest.MonkeyPatch
) -> None:
    points = ex._points()
    monkeypatch.setattr(ex, "_points", lambda: (points[0],) * 8)
    with pytest.raises(ValueError):
        ex.main()
    assert fake == []


def _fail_at(
    index: int, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    primary = RuntimeError("point failed")
    calls = 0

    def run(*args: Any, **kwargs: Any) -> MemoryExperimentResult:
        nonlocal calls
        calls += 1
        if calls == index:
            raise primary
        return _result(kwargs["seed"])

    monkeypatch.setattr(ex, "run_memory_experiment", run)
    with pytest.raises(RuntimeError) as caught:
        ex.main()
    assert caught.value is primary and calls == index
    output = capsys.readouterr()
    rows = [json.loads(line) for line in output.out.splitlines()]
    assert len(rows) == index and all(r["record_type"] != "complete" for r in rows)
    error = json.loads(output.err)
    assert error["record_type"] == "error" and error["point"]["seed"] == 52 + index


def test_first_failure_stops(
    fake: list[_Call],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _fail_at(1, monkeypatch, capsys)


def test_fourth_failure_retains_partial_diagnostics(
    fake: list[_Call],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _fail_at(4, monkeypatch, capsys)


def test_malformed_result_is_not_repaired(
    fake: list[_Call],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    good = _result(53)
    for bad in (
        dataclasses.replace(good, shots=34),
        dataclasses.replace(good, logical_failures=36),
        dataclasses.replace(good, observable_failures=(1,)),
        dataclasses.replace(good, seed=54),
        dataclasses.replace(good, confidence_interval=(0.0, 0.0)),
    ):
        with pytest.raises(ValueError):
            ex._result_fields(bad, 53)
    bad_source = dataclasses.replace(good)
    object.__setattr__(bad_source, "sampling_source", "other")
    with pytest.raises(ValueError):
        ex._result_fields(bad_source, 53)
    monkeypatch.setattr(ex, "run_memory_experiment", lambda *args, **kwargs: bad_source)
    with pytest.raises(ValueError):
        ex.main()
    output = capsys.readouterr()
    assert len(output.out.splitlines()) == 1
    assert json.loads(output.err)["record_type"] == "error"


def test_output_complete_only_after_eight_points(
    fake: list[_Call], capsys: pytest.CaptureFixture[str]
) -> None:
    ex.main()
    rows = _rows(capsys)
    assert len(rows) == 10
    assert rows[0]["record_type"] == "configuration" and set(rows[0]["versions"]) == {
        "python",
        "torch",
        "stim",
        "pymatching",
    }
    assert [r["record_type"] for r in rows[1:-1]] == ["point"] * 8
    assert rows[-1] == {
        "record_type": "complete",
        "points": 8,
        "total_requested_shots": 280,
    }


@pytest.mark.parametrize("point", ex._points())
def test_real_point_matches_original_source_reference(
    point: tuple[str, int, str, int],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("stim")
    pytest.importorskip("pymatching")
    basis, distance, profile, seed = point
    code = RotatedSurfaceCode(distance=distance)
    memory = build_memory_circuit(
        code,
        rounds=3,
        product=Pauli(x_qubits=code.data_qubits) if basis == "x" else None,
    )
    data, phase, measurement = ex._NOISE[profile]
    noise = PhenomenologicalNoise(
        data_flip=data, phase_flip=phase, measurement_flip=measurement
    )
    text = memory_to_stim_text(memory, noise=noise)
    model = DetectorErrorModel.from_stim_text(
        stim_circuit_to_dem_text(text, decompose_errors=False)
    )
    decoder = get_decoder("pymatching", model)
    failed = 0
    with compile_stim_sampler(text, seed=seed) as sampler:
        for size in (16, 16, 3):
            batch = sampler.sample(shots=size)
            for detection, observed in zip(
                batch.detectors, batch.observables, strict=True
            ):
                answer = decoder.decode(detection.nonzero().flatten().tolist())
                failed += int(bool(observed[0]) != (0 in answer.observables))
    # Each invocation validates the full grid and executes only its real point;
    # the other seven rows use controlled results, without stochastic draws.
    original = run_memory_experiment

    def run(*args: Any, **kwargs: Any) -> MemoryExperimentResult:
        return (
            original(*args, **kwargs)
            if kwargs["seed"] == seed
            else _result(kwargs["seed"])
        )

    monkeypatch.setattr(ex, "run_memory_experiment", run)
    ex.main()
    row = next(
        r
        for r in _rows(capsys)
        if r.get("record_type") == "point" and r["seed"] == seed
    )
    assert row["logical_failures"] == failed and row["observable_failures"] == [failed]
    assert row["logical_error_rate"] == failed / 35
    assert row["confidence_interval"] == pytest.approx(
        _interval(failed), abs=1e-12, rel=0
    )
