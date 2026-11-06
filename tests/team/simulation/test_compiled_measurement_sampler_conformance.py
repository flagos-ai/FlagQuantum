"""Compare fixed prepared samplers to Stim record order and b8."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from flagquantum.dynamic import DynamicCircuit

pytestmark = pytest.mark.integration


def test_feedback_uses_latest_record_for_the_overwritten_slot(tmp_path: Path) -> None:
    stim = pytest.importorskip("stim")
    program = (
        DynamicCircuit(2)
        .x(0)
        .measure(0, classical_bit=0)
        .reset(0)
        .measure(0, classical_bit=0)
        .conditional("x", 1, classical_bit=0)
        .measure(1, classical_bit=1)
    )
    expected = stim.Circuit("X 0\nM 0\nR 0\nM 0\nCX rec[-1] 1\nM 1")
    stale = stim.Circuit("X 0\nM 0\nR 0\nM 0\nCX rec[-2] 1\nM 1")
    sampler = program.compile_sampler(seed=7)
    records = sampler.sample(8)
    assert records.dtype == torch.bool
    assert records.tolist() == expected.compile_sampler(seed=7).sample(8).tolist()
    assert records.tolist() == [[True, False, False]] * 8
    assert (
        stale.compile_sampler(seed=7).sample(8, bit_packed=True).tolist() == [[5]] * 8
    )
    path = tmp_path / "feedback.b8"
    sampler.sample_write(8, filepath=path, format="b8")
    assert path.read_bytes() == b"\x01" * 8
    actual = stim.read_shot_data_file(path=str(path), format="b8", num_measurements=3)
    assert actual.tolist() == records.tolist()


def test_inverted_feedback_and_cross_byte_output(tmp_path: Path) -> None:
    stim = pytest.importorskip("stim")
    complement = (
        DynamicCircuit(1)
        .h(0)
        .measure(0)
        .reset(0)
        .conditional("x", 0, classical_bit=0, equals=0)
        .measure(0)
    )
    dense = complement.compile_sampler(seed=7).sample(32)
    assert (dense[:, 0] ^ dense[:, 1]).all()
    assert set(map(tuple, dense.tolist())) == {(True, False), (False, True)}
    program = DynamicCircuit(1).x(0)
    for _ in range(9):
        program.measure(0)
    sampler = program.compile_sampler(seed=7)
    packed = sampler.sample(3, bit_packed=True)
    assert packed.dtype == torch.uint8 and packed.tolist() == [[255, 1]] * 3
    path = tmp_path / "cross-byte.b8"
    sampler.sample_write(3, filepath=path, format="b8")
    assert path.read_bytes() == b"\xff\x01" * 3
    rows = stim.read_shot_data_file(
        path=str(path), format="b8", num_measurements=9, bit_packed=True
    )
    assert rows.tolist() == packed.tolist()
