"""Replacement contracts for noiseless embedded sampling without an SDK."""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import pytest
import torch

import flagquantum as fq
from flagquantum.errors import CapabilityError, ValidationError
from flagquantum.simulation.stabilizer import _embedded_sampler, engine
from flagquantum.simulation.stabilizer.engine import _EngineOperation

pytestmark = pytest.mark.unit


class _Sampler:
    def __init__(
        self,
        operations: tuple[_EngineOperation, ...],
        seed: int | None,
        columns: int,
    ) -> None:
        self.operations = operations
        self.seed = seed
        self.columns = columns
        self.buffer = torch.tensor([[True, False]], dtype=torch.bool)
        self.draws: list[tuple[int, bool]] = []

    def sample(self, shots: int, *, bit_packed: bool) -> torch.Tensor:
        self.draws.append((shots, bit_packed))
        self.buffer = self.buffer.expand(shots, self.columns).clone()
        return self.buffer


@pytest.fixture
def embedded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[_Sampler]:
    (tmp_path / "_embedded_stim_profile.txt").write_text("stim-1.16.0\n")
    monkeypatch.setattr(engine, "Path", lambda filename: tmp_path / Path(filename).name)
    prepared: list[_Sampler] = []

    def prepare(
        operations: tuple[_EngineOperation, ...], seed: int | None, columns: int
    ) -> _Sampler:
        sampler = _Sampler(operations, seed, columns)
        prepared.append(sampler)
        return sampler

    def refuse_sdk() -> NoReturn:
        raise AssertionError("the embedded profile must not load the Stim SDK")

    monkeypatch.setattr(_embedded_sampler, "_EmbeddedSampler", prepare)
    monkeypatch.setattr(engine, "_stim", refuse_sdk)
    return prepared


def test_replacement_preserves_order_seed_and_owned_integer_results(
    embedded: list[_Sampler],
) -> None:
    result = engine.sample_stabilizer(
        fq.Circuit(3).x(1), shots=4, qubits=[1, 0], seed=7
    )
    assert len(embedded) == 1
    sampler = embedded[0]
    assert sampler.operations == (("X", (1,), None), ("M", (1, 0), None))
    assert sampler.seed == 7
    assert sampler.columns == 2
    assert sampler.draws == [(4, False)]
    assert result.dtype == torch.int64
    assert result.device.type == "cpu"
    assert result.tolist() == [[1, 0]] * 4
    sampler.buffer.zero_()
    assert result.tolist() == [[1, 0]] * 4


@pytest.mark.parametrize("shots", [0, -1, True, 1.5])
def test_invalid_shots_refuse_before_preparation(
    shots: object, embedded: list[_Sampler]
) -> None:
    sample = getattr(engine, "sample_stabilizer")
    expected = TypeError if isinstance(shots, (bool, float)) else ValidationError
    with pytest.raises(expected):
        sample(fq.Circuit(2), shots=shots)
    assert embedded == []


@pytest.mark.parametrize("seed", [-1, 2**64, True, 0.5])
def test_invalid_seeds_refuse_before_preparation(
    seed: object, embedded: list[_Sampler]
) -> None:
    sample = getattr(engine, "sample_stabilizer")
    with pytest.raises(ValidationError):
        sample(fq.Circuit(2), shots=1, seed=seed)
    assert embedded == []


def test_non_clifford_refuses_before_preparation(embedded: list[_Sampler]) -> None:
    with pytest.raises(CapabilityError, match="Clifford"):
        engine.sample_stabilizer(fq.Circuit(2).t(0), shots=1)
    assert embedded == []


@pytest.mark.parametrize(
    "failure", [ImportError("missing extension"), OSError("linker")]
)
def test_loading_failure_retains_cause_and_never_loads_sdk(
    failure: Exception, embedded: list[_Sampler], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(
        operations: tuple[_EngineOperation, ...], seed: int | None, columns: int
    ) -> NoReturn:
        raise failure

    monkeypatch.setattr(_embedded_sampler, "_EmbeddedSampler", fail)
    with pytest.raises(engine.StabilizerDependencyError) as info:
        engine.sample_stabilizer(fq.Circuit(2), shots=1)
    assert info.value.__cause__ is failure
    assert embedded == []


def test_marker_free_installation_preserves_sdk_route(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(engine, "Path", lambda filename: tmp_path / Path(filename).name)
    expected = torch.tensor([[True, False]])

    class Stream:
        def sample(self, *, shots: int) -> torch.Tensor:
            assert shots == 3
            return expected.expand(shots, 2)

    class Circuit:
        def compile_sampler(self, *, seed: int | None) -> Stream:
            assert seed == 11
            return Stream()

    def circuit(
        operations: tuple[tuple[str, tuple[int, ...]], ...], qubits: tuple[int, ...]
    ) -> Circuit:
        assert operations == (("X", (0,)),)
        assert qubits == (0, 1)
        return Circuit()

    monkeypatch.setattr(engine, "_engine_circuit", circuit)
    result = engine.sample_stabilizer(fq.Circuit(2).x(0), shots=3, seed=11)
    assert result.dtype == torch.int64
    assert result.tolist() == [[1, 0]] * 3
