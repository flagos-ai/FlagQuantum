"""Prepared numerical seam, checked against independently authored Stim records."""

from __future__ import annotations

import gc
import math
import weakref

import pytest
import torch

from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.errors import ValidationError
from flagquantum.noise import bit_flip_channel
from flagquantum.simulation.stabilizer import engine

pytestmark = pytest.mark.unit


def _program() -> CircuitIR:
    return CircuitIR(
        n_wires=2,
        instructions=(
            Instruction(name="h", wires=(0,)),
            Instruction(name="cx", wires=(0, 1)),
            Instruction(name="measure", wires=(0,), metadata={"is_dynamic": True}),
            Instruction(name="reset", wires=(0,), metadata={"is_dynamic": True}),
            Instruction(
                name="bit_flip",
                wires=(1,),
                matrix=bit_flip_channel(0.25).kraus,
                metadata={"is_channel": True},
            ),
        ),
    )


def test_repeated_native_draws_match_independent_reference() -> None:
    stim = pytest.importorskip("stim")
    reference = stim.Circuit("H 0\nCX 0 1\nM 0\nR 0\nX_ERROR(0.25) 1\nM 1")
    expected = reference.compile_sampler(seed=71)
    prepared = engine._prepare_noisy_measurements(
        _program(), terminal_qubits=(1,), seed=71
    )
    try:
        actual_counts = torch.zeros(4, dtype=torch.int64)
        reference_counts = torch.zeros(4, dtype=torch.int64)
        sizes = (10_000, 7777, 3131)
        for shots in sizes:
            actual = prepared.sample(shots=shots)
            assert actual.dtype == torch.int64
            external = torch.as_tensor(expected.sample(shots=shots)).long()
            actual_counts += torch.bincount(
                actual[:, 0] + 2 * actual[:, 1], minlength=4
            )
            reference_counts += torch.bincount(
                external[:, 0] + 2 * external[:, 1], minlength=4
            )
        # Bell preparation makes the first bit fair; an independent 1/4 flip
        # changes the second. Different channel encodings need distributional,
        # not same-seed bitwise, comparisons.
        total = sum(sizes)
        for index, probability in enumerate((0.375, 0.125, 0.125, 0.375)):
            actual_rate = float(actual_counts[index]) / total
            reference_rate = float(reference_counts[index]) / total
            sigma = math.sqrt(probability * (1 - probability) / total)
            assert abs(actual_rate - probability) < 6 * sigma
            assert abs(reference_rate - probability) < 6 * sigma
            assert abs(actual_rate - reference_rate) < 6 * math.sqrt(2) * sigma
    finally:
        prepared.close()
    prepared.close()
    with pytest.raises(RuntimeError, match="closed"):
        prepared.sample(shots=1)


def test_close_releases_compiled_state() -> None:
    class Stream:
        pass

    stream = Stream()
    reference = weakref.ref(stream)
    prepared = engine._PreparedMeasurements(stream, 1)
    del stream
    assert reference() is not None
    prepared.close()
    gc.collect()
    assert reference() is None


def test_one_compilation_no_eager_draws_and_repeated_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compilations = []
    draws = []

    class Stream:
        def sample(self, *, shots: int) -> torch.Tensor:
            draws.append(shots)
            return torch.zeros((shots, 2), dtype=torch.bool)

    class Circuit:
        def compile_sampler(self, *, seed: int) -> Stream:
            compilations.append(seed)
            return Stream()

    monkeypatch.setattr(engine, "_noisy_engine_circuit", lambda *args: Circuit())
    prepared = engine._prepare_noisy_measurements(
        _program(), terminal_qubits=(1,), seed=71
    )
    assert compilations == [71] and not draws
    try:
        for shots in (4, 3, 2):
            assert prepared.sample(shots=shots).shape == (shots, 2)
    finally:
        prepared.close()
    assert compilations == [71] and draws == [4, 3, 2]


@pytest.mark.parametrize("buffer", [torch.zeros((2, 2)), torch.full((2, 1), 0.5)])
def test_invalid_engine_records_are_refused_before_integer_coercion(
    buffer: torch.Tensor,
) -> None:
    class Stream:
        def sample(self, *, shots: int) -> torch.Tensor:
            return buffer

    prepared = engine._PreparedMeasurements(Stream(), 1)
    try:
        with pytest.raises(ValidationError):
            prepared.sample(shots=2)
    finally:
        prepared.close()
