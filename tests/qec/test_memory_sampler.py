"""Reusable circuit sampling: ownership, immutable plans and stream semantics."""

from __future__ import annotations

import copy
import inspect
import json
import multiprocessing
import pickle
import weakref
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.connection import Connection
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import flagquantum as fq
import flagquantum.qec as qec
from flagquantum.noise import NoiseModel, bit_flip_channel
from flagquantum.qec.execution import sampling
from flagquantum.simulation.stabilizer import StabilizerDependencyError, engine

pytestmark = pytest.mark.unit


class _RecordSource:
    def __init__(self) -> None:
        self.draws: list[int] = []
        self.closed = False
        self.error: BaseException | None = None

    def sample(self, *, shots: int) -> torch.Tensor:
        self.draws.append(shots)
        if self.error is not None:
            raise self.error
        # Three rounds' two check measurements, then three terminal bits.
        return torch.tensor([[0, 1, 1, 0, 1, 1, 0, 1, 0]]).expand(shots, -1)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def record_source(monkeypatch: pytest.MonkeyPatch) -> _RecordSource:
    source = _RecordSource()
    monkeypatch.setattr(sampling, "_prepare_noisy_measurements", lambda *a, **k: source)
    return source


def _memory() -> qec.MemoryCircuit:
    return qec.build_memory_circuit(qec.RepetitionCode(distance=3), rounds=3)


def _handle() -> qec.MemorySampler:
    return qec.compile_memory_sampler(
        _memory(), noise=qec.PhenomenologicalNoise(), seed=53
    )


def test_exact_approved_surface_and_namespace() -> None:
    contract = json.loads(
        (Path(__file__).parents[2] / "contracts/qec-memory-sampler-v1.json").read_text()
    )
    assert contract["appended_exports"] == ["MemorySampler", "compile_memory_sampler"]
    assert (
        str(inspect.signature(qec.compile_memory_sampler))
        == contract["factory_signature"]
    )
    for method, expected in contract["methods"].items():
        assert str(inspect.signature(getattr(qec.MemorySampler, method))) == expected
    for name in contract["appended_exports"]:
        assert getattr(qec, name) is getattr(sampling, name)
        assert getattr(qec, name).__module__ == contract["owner"]
        assert not hasattr(fq, name)
    with pytest.raises(TypeError, match="compile_memory_sampler"):
        qec.MemorySampler()


def test_replacement_seam_parities_match_public_measurement_maps(
    record_source: _RecordSource,
) -> None:
    memory = _memory()
    noise = qec.PhenomenologicalNoise()
    inputs = qec.decoder_context_from_memory_circuit(
        memory, noise=noise
    ).full_component()
    raw = record_source.sample(shots=2)
    with qec.compile_memory_sampler(memory, noise=noise) as handle:
        sample = handle.sample(shots=2)
    assert torch.equal(
        sample.detectors, (raw @ inputs.measurement_to_detectors.dense().long().T) % 2
    )
    assert torch.equal(
        sample.observables,
        (raw @ inputs.measurement_to_observables.dense().long().T) % 2,
    )
    assert sample.detectors.dtype == sample.observables.dtype == torch.int8
    assert sample.detectors.device.type == sample.observables.device.type == "cpu"
    assert record_source.closed


@pytest.mark.parametrize("shots", [True, False, 1.5, "4", None, 0, -1])
def test_invalid_shots_leave_handle_and_stream_usable(
    shots: object, record_source: _RecordSource
) -> None:
    with _handle() as handle:
        with pytest.raises((TypeError, ValueError)):
            handle.sample(shots=shots)
        assert not record_source.draws
        assert not record_source.closed
        assert handle.sample(shots=2).shots == 2
    assert record_source.draws == [2]


def test_close_is_idempotent_and_context_does_not_suppress(
    record_source: _RecordSource,
) -> None:
    handle = _handle()
    sentinel = ValueError("caller error")
    with pytest.raises(ValueError) as caught, handle as entered:
        assert entered is handle
        raise sentinel
    assert caught.value is sentinel
    assert record_source.closed
    handle.close()
    for operation in (lambda: handle.sample(shots=1), handle.__enter__):
        with pytest.raises(RuntimeError, match="closed"):
            operation()


@pytest.mark.parametrize("error", [RuntimeError("draw failed"), KeyboardInterrupt()])
def test_execution_failure_poisoning_preserves_original_error(
    error: BaseException, record_source: _RecordSource
) -> None:
    handle = _handle()
    record_source.error = error
    with pytest.raises(type(error)) as caught:
        handle.sample(shots=1)
    assert caught.value is error
    assert record_source.closed
    with pytest.raises(RuntimeError, match="closed"):
        handle.sample(shots=1)


def test_parity_failure_also_closes_the_stream(
    monkeypatch: pytest.MonkeyPatch, record_source: _RecordSource
) -> None:
    handle = _handle()

    def broken(*args: object) -> torch.Tensor:
        raise ValueError("parity extraction failed")

    monkeypatch.setattr(sampling, "_parity_bits", broken)
    with pytest.raises(ValueError, match="parity extraction"):
        handle.sample(shots=1)
    assert record_source.closed
    with pytest.raises(RuntimeError, match="closed"):
        handle.sample(shots=1)


def _owner_violations(handle: qec.MemorySampler) -> list[str]:
    errors = []
    for operation in (lambda: handle.sample(shots=1), handle.close, handle.__enter__):
        try:
            operation()
        except RuntimeError as exc:
            errors.append(str(exc))
    return errors


def test_thread_misuse_cannot_draw_or_close_owner_state(
    record_source: _RecordSource,
) -> None:
    with _handle() as handle:
        with ThreadPoolExecutor(max_workers=1) as executor:
            errors = executor.submit(_owner_violations, handle).result(timeout=10)
        assert len(errors) == 3
        assert all("creator process and thread" in message for message in errors)
        assert not record_source.draws and not record_source.closed
        assert handle.sample(shots=1).shots == 1


def _child_misuse(handle: qec.MemorySampler, connection: Connection) -> None:
    connection.send(_owner_violations(handle))
    connection.close()


def test_fork_cannot_reuse_or_close_parent_stream(record_source: _RecordSource) -> None:
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("fork ownership applies only on platforms supporting fork")
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe(duplex=False)
    with _handle() as handle:
        worker = context.Process(target=_child_misuse, args=(handle, child))
        worker.start()
        child.close()
        try:
            assert parent.poll(10), "child failed to report ownership rejection"
            assert len(parent.recv()) == 3
            worker.join(timeout=10)
            assert worker.exitcode == 0
        finally:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
            parent.close()
        assert not record_source.draws and not record_source.closed
        assert handle.sample(shots=1).shots == 1


def test_copy_and_pickle_do_not_duplicate_live_stream(
    record_source: _RecordSource,
) -> None:
    with _handle() as handle:
        for operation in (copy.copy, copy.deepcopy, pickle.dumps):
            with pytest.raises(TypeError, match="cannot be copied or serialized"):
                operation(handle)
        assert not record_source.draws


def test_snapshot_owns_parity_columns_and_does_not_retain_results(
    record_source: _RecordSource,
) -> None:
    memory = _memory()
    with qec.compile_memory_sampler(
        memory, noise=qec.PhenomenologicalNoise()
    ) as handle:
        first = handle.sample(shots=2)
        expected = first.detectors.clone()
        object.__setattr__(memory.detectors.detectors[0], "parity", ())
        first.detectors.fill_(0)
        assert torch.equal(handle.sample(shots=2).detectors, expected)
        detectors = weakref.ref(first.detectors)
        observables = weakref.ref(first.observables)
        del first
        assert detectors() is None and observables() is None


def test_missing_engine_fails_at_compile_time(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable() -> None:
        raise StabilizerDependencyError("install the stim extra")

    monkeypatch.setattr(engine, "_stim", unavailable)
    with pytest.raises(StabilizerDependencyError, match="stim extra"):
        _handle()


@pytest.mark.parametrize("argument", ["circuit", "noise", "seed"])
def test_invalid_configuration_is_rejected_before_preparation(
    argument: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError("invalid configuration reached numerical preparation")

    monkeypatch.setattr(sampling, "_prepare_noisy_measurements", unexpected)
    kwargs = {"circuit": _memory(), "noise": qec.PhenomenologicalNoise(), "seed": 53}
    kwargs[argument] = object()
    with pytest.raises(TypeError, match=argument):
        qec.compile_memory_sampler(**kwargs)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("code", "basis"),
    [
        (qec.RotatedSurfaceCode(distance=3), "z"),
        (qec.SteaneCode(), "x"),
        (qec.SteaneCode(), "z"),
    ],
)
def test_readout_plans_match_existing_one_shot_route(
    code: qec.StabilizerCode, basis: str
) -> None:
    pytest.importorskip("stim")
    memory = qec.build_memory_circuit(code, rounds=3, readout_basis=basis)
    noise = qec.PhenomenologicalNoise(
        data_flip=0.02, phase_flip=0.03, measurement_flip=0.01
    )
    expected = qec.sample_memory_circuit(memory, noise=noise, shots=128, seed=29)
    with qec.compile_memory_sampler(memory, noise=noise, seed=29) as handle:
        assert handle.sample(shots=128) == expected
        assert handle.sample(shots=7).detectors.shape == (7, len(memory.detectors))


@pytest.mark.integration
def test_real_stream_compiles_once_and_preserves_one_shot_first_draw() -> None:
    pytest.importorskip("stim")
    memory = _memory()
    noise = qec.PhenomenologicalNoise(data_flip=0.1, measurement_flip=0.05)
    expected = qec.sample_memory_circuit(memory, noise=noise, shots=37, seed=53)
    with (
        patch.object(
            sampling, "_measurement_plan", wraps=sampling._measurement_plan
        ) as lower,
        patch.object(
            engine, "_noisy_engine_circuit", wraps=engine._noisy_engine_circuit
        ) as build,
    ):
        with qec.compile_memory_sampler(memory, noise=noise, seed=53) as first:
            assert first.sample(shots=37) == expected
            second = first.sample(shots=19)
        assert lower.call_count == build.call_count == 1
    with qec.compile_memory_sampler(memory, noise=noise, seed=53) as repeat:
        assert repeat.sample(shots=37) == expected
        assert repeat.sample(shots=19) == second


@pytest.mark.integration
def test_mutating_noise_after_compile_does_not_retune_handle() -> None:
    pytest.importorskip("stim")
    memory = _memory()
    noise = NoiseModel()
    with qec.compile_memory_sampler(memory, noise=noise, seed=53) as handle:
        noise.add("cx", bit_flip_channel(1.0), qubits=0)
        sample = handle.sample(shots=3)
        assert not sample.detectors.any() and not sample.observables.any()
    changed = qec.sample_memory_circuit(memory, noise=noise, shots=3, seed=53)
    assert changed.detectors.any()


@pytest.mark.integration
def test_invalid_shots_do_not_advance_real_rng() -> None:
    pytest.importorskip("stim")
    memory = _memory()
    noise = qec.PhenomenologicalNoise(data_flip=0.2)
    with (
        qec.compile_memory_sampler(memory, noise=noise, seed=19) as first,
        qec.compile_memory_sampler(memory, noise=noise, seed=19) as second,
    ):
        with pytest.raises(ValueError):
            first.sample(shots=0)
        assert first.sample(shots=128) == second.sample(shots=128)
