"""Check paired outcomes on shared Surface memory detector rows."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from examples.qec import paired_surface_decoder_outcomes as example
from flagquantum.errors import ValidationError
from flagquantum.qec.adapters.stim import (
    compile_stim_sampler,
    memory_to_stim_text,
    stim_circuit_to_dem_text,
)
from flagquantum.qec.circuits.memory import build_memory_circuit
from flagquantum.qec.codes.surface import RotatedSurfaceCode
from flagquantum.qec.decoders.registry import get_decoder
from flagquantum.qec.dem import DetectorErrorModel
from flagquantum.qec.noise import PhenomenologicalNoise


@pytest.fixture
def controlled(monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, Mock]:
    memory = build_memory_circuit(RotatedSurfaceCode(distance=3), rounds=3)
    detectors = len(memory.detectors)
    model = SimpleNamespace(num_detectors=detectors, num_observables=1)
    monkeypatch.setattr(example, "memory_to_stim_text", Mock(return_value="source"))
    monkeypatch.setattr(example, "stim_circuit_to_dem_text", Mock(return_value="dem"))
    monkeypatch.setattr(DetectorErrorModel, "from_stim_text", Mock(return_value=model))
    monkeypatch.setattr(example, "get_decoder", Mock(side_effect=lambda name, _: name))

    def sample(*, shots: int) -> SimpleNamespace:
        rows = torch.zeros((shots, detectors), dtype=torch.bool)
        for i in range(shots):
            rows[i, :2] = torch.tensor([bool(i & 2), bool(i & 1)])
        return SimpleNamespace(detectors=rows, observables=torch.zeros((shots, 1)))

    def decode(
        name: str, rows: tuple[tuple[int, ...], ...]
    ) -> tuple[SimpleNamespace, ...]:
        column = 0 if name == "minimum_weight_matching" else 1
        return tuple(
            SimpleNamespace(observables=(0,) if column in row else ()) for row in rows
        )

    sampler = Mock()
    sampler.sample.side_effect = sample
    monkeypatch.setattr(example, "compile_stim_sampler", Mock(return_value=sampler))
    batch_decode = Mock(side_effect=decode)
    monkeypatch.setattr(example, "decode_batch", batch_decode)
    return sampler, batch_decode


def test_four_bins_and_shared_partial_batch(
    controlled: tuple[Mock, Mock], capsys: pytest.CaptureFixture[str]
) -> None:
    sampler, decode = controlled
    example.main()
    report = json.loads(capsys.readouterr().out)
    assert report["shots"] == 35
    assert report["counts"] == dict(
        both_correct=9, native_only_wrong=9, pymatching_only_wrong=9, both_wrong=8
    )
    assert [call.kwargs["shots"] for call in sampler.sample.call_args_list] == [
        16,
        16,
        3,
    ]
    for left, right in zip(
        decode.call_args_list[2::2], decode.call_args_list[3::2], strict=True
    ):
        assert left.args[1] is right.args[1]
    sampler.close.assert_called_once_with()


@pytest.mark.parametrize(
    "failure", ["short_sample", "short_decode", "invalid_index", "cleanup"]
)
def test_failure_closes_without_reporting(
    controlled: tuple[Mock, Mock], capsys: pytest.CaptureFixture[str], failure: str
) -> None:
    sampler, decode = controlled
    expected: type[Exception] = ValidationError
    if failure == "short_sample":
        sampler.sample.side_effect = lambda **_: SimpleNamespace(
            detectors=torch.zeros((0, 0)), observables=torch.zeros((0, 1))
        )
    elif failure == "short_decode":
        decode.side_effect = None
        decode.return_value = ()
    elif failure == "invalid_index":
        decode.side_effect = lambda _, rows: tuple(
            SimpleNamespace(observables=(1,)) for _ in rows
        )
    else:
        sampler.sample.side_effect = ValueError("primary")
        sampler.close.side_effect = RuntimeError("cleanup")
        expected = ValueError
    with pytest.raises(expected) as caught:
        example.main()
    if failure == "cleanup":
        assert str(caught.value) == "primary"
        assert caught.value.__cause__ is sampler.close.side_effect
    sampler.close.assert_called_once_with()
    assert capsys.readouterr().out == ""


def test_actual_pair_matches_scalar_reference() -> None:
    pytest.importorskip("stim")
    pytest.importorskip("pymatching")
    report = example._report()
    memory = build_memory_circuit(RotatedSurfaceCode(distance=3), rounds=3)
    text = memory_to_stim_text(
        memory,
        noise=PhenomenologicalNoise(
            data_flip=0.005, phase_flip=0.007, measurement_flip=0.009
        ),
    )
    model = DetectorErrorModel.from_stim_text(
        stim_circuit_to_dem_text(text, decompose_errors=False)
    )
    native = get_decoder("minimum_weight_matching", model)
    matching = get_decoder("pymatching", model)
    expected = dict(
        both_correct=0, native_only_wrong=0, pymatching_only_wrong=0, both_wrong=0
    )
    sampler = compile_stim_sampler(text, seed=53)
    try:
        for size in (16, 16, 3):
            batch = sampler.sample(shots=size)
            for row, observed in zip(
                batch.detectors.tolist(), batch.observables.tolist(), strict=True
            ):
                syndrome = tuple(i for i, bit in enumerate(row) if bit)
                native_ok = (0 in native.decode(syndrome).observables) == bool(
                    observed[0]
                )
                matching_ok = (0 in matching.decode(syndrome).observables) == bool(
                    observed[0]
                )
                if native_ok and matching_ok:
                    expected["both_correct"] += 1
                elif native_ok:
                    expected["pymatching_only_wrong"] += 1
                elif matching_ok:
                    expected["native_only_wrong"] += 1
                else:
                    expected["both_wrong"] += 1
    finally:
        sampler.close()
    assert report["counts"] == expected
