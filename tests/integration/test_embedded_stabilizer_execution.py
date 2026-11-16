"""Public execution with a replaced numerical sampler, without SDK imports.

These are orchestration replacement contracts. Actual native execution must be
qualified separately against the built embedded installation.
"""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import pytest
import torch

import flagquantum as fq
from flagquantum.errors import CapabilityError
from flagquantum.simulation.stabilizer import _embedded_sampler, engine
from flagquantum.simulation.stabilizer.engine import _EngineOperation

pytestmark = pytest.mark.integration


@pytest.fixture
def prepared(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> list[tuple[_EngineOperation, ...]]:
    (tmp_path / "_embedded_stim_profile.txt").write_text("stim-1.16.0\n")
    monkeypatch.setattr(engine, "__file__", str(tmp_path / "engine.py"))
    calls: list[tuple[_EngineOperation, ...]] = []

    class Stream:
        def sample(self, shots: int, *, bit_packed: bool) -> torch.Tensor:
            assert not bit_packed
            return torch.tensor([[False, True, False]]).expand(shots, 3).clone()

    def prepare(
        operations: tuple[_EngineOperation, ...], seed: int | None, columns: int
    ) -> Stream:
        assert seed == 7
        assert columns == 3
        assert operations == (("X", (1,), None), ("M", (0, 1, 2), None))
        calls.append(operations)
        return Stream()

    def refuse_sdk() -> NoReturn:
        raise AssertionError("public embedded execution must not load the SDK")

    monkeypatch.setattr(_embedded_sampler, "_EmbeddedSampler", prepare)
    monkeypatch.setattr(engine, "_stim", refuse_sdk)
    return calls


def test_public_samples_use_the_replacement(
    prepared: list[tuple[_EngineOperation, ...]],
) -> None:
    result = fq.run(
        fq.Circuit(3).x(1),
        options=fq.ExecutionOptions(mode="stabilizer", shots=8, seed=7),
        outputs=fq.samples(),
    )
    assert result.samples is not None
    assert result.samples.tolist() == [[[0, 1, 0]] * 8]
    assert result.samples.dtype == torch.int64
    assert result.runtime["execution_path"] == "local_stabilizer"
    assert result.runtime["simulation_engine"] == "pauli_stabilizer_tableau"
    assert len(prepared) == 1


def test_public_counts_use_the_same_replacement(
    prepared: list[tuple[_EngineOperation, ...]],
) -> None:
    result = fq.run(
        fq.Circuit(3).x(1),
        options=fq.ExecutionOptions(mode="stabilizer", shots=8, seed=7),
        outputs=fq.counts(),
    )
    assert result.measurements[0].value[0] == {"010": 8}
    assert len(prepared) == 1


def test_t_refuses_before_numerical_preparation(
    prepared: list[tuple[_EngineOperation, ...]],
) -> None:
    with pytest.raises(CapabilityError, match="Clifford"):
        fq.run(
            fq.Circuit(3).t(0),
            options=fq.ExecutionOptions(mode="stabilizer", shots=8, seed=7),
            outputs=fq.samples(),
        )
    assert prepared == []


def test_controlled_h_refuses_before_numerical_preparation(
    prepared: list[tuple[_EngineOperation, ...]],
) -> None:
    with pytest.raises(CapabilityError, match="Clifford"):
        fq.run(
            fq.Circuit(3).ch(0, 1),
            options=fq.ExecutionOptions(mode="stabilizer", shots=8, seed=7),
            outputs=fq.samples(),
        )
    assert prepared == []
