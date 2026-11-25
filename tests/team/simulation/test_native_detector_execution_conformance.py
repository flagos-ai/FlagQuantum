"""Consumer conformance for detector bits, independent of numerical engines."""

from pathlib import Path
from typing import Literal, Protocol

import pytest
import torch

pytestmark = [pytest.mark.unit]


class DetectorStream(Protocol):
    def sample(
        self,
        shots: int,
        *,
        separate_observables: bool = False,
        append_observables: bool = False,
        bit_packed: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...


def consume(stream: DetectorStream) -> int:
    total = 0
    for shots in (16, 16, 3):
        pair = stream.sample(shots, separate_observables=True)
        assert isinstance(pair, tuple)
        detectors, observables = pair
        assert detectors.shape == (shots, 2) and observables.shape == (shots, 1)
        assert detectors.dtype == observables.dtype == torch.bool
        total += len(detectors)
    return total


class FakeStream:
    def sample(
        self,
        shots: int,
        *,
        separate_observables: bool = False,
        append_observables: bool = False,
        bit_packed: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert separate_observables and not append_observables and not bit_packed
        return torch.zeros((shots, 2), dtype=torch.bool), torch.zeros(
            (shots, 1), dtype=torch.bool
        )


def test_independent_consumer_accepts_contract_fake() -> None:
    assert consume(FakeStream()) == 35


@pytest.mark.integration
def test_same_consumer_accepts_numerical_replacement() -> None:
    pytest.importorskip("stim")
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.qec.execution.context import MeasurementMap

    circuit = DynamicCircuit(2).measure(0, classical_bit=0).measure(1, classical_bit=1)
    sampler = circuit.compile_detector_sampler(
        detector_map=MeasurementMap(2, ((0,), (1,))),
        observable_map=MeasurementMap(2, ((0,),)),
        seed=7,
    )
    assert consume(sampler) == 35


@pytest.mark.integration
@pytest.mark.parametrize("encoding", ["01", "b8", "r8", "hits", "dets", "ptb64"])
def test_file_outputs_match_pinned_sdk_and_independent_faults(
    tmp_path: Path, encoding: str
) -> None:
    from typing import Any

    stim = pytest.importorskip("stim")
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.qec.execution.context import MeasurementMap

    for mask in range(4):
        a, b = mask & 1, mask >> 1
        circuit = (
            DynamicCircuit(2)
            .x(0)
            .bit_flip(0, float(a))
            .bit_flip(1, float(b))
            .measure(0, classical_bit=0)
            .measure(1, classical_bit=1)
        )
        handle = circuit.compile_detector_sampler(
            detector_map=MeasurementMap(2, ((0,), (0, 1))),
            observable_map=MeasurementMap(2, ((1,),)),
            seed=7,
        )
        sdk = stim.Circuit(
            f"X 0\nX_ERROR({a}) 0\nX_ERROR({b}) 1\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"
        ).compile_detector_sampler(seed=7)
        actual, reference = tmp_path / f"a{mask}", tmp_path / f"r{mask}"
        options: dict[str, Any] = {"format": encoding, "append_observables": True}
        handle.sample_write(64, filepath=actual, **options)
        sdk.sample_write(64, filepath=reference, **options)
        assert actual.read_bytes() == reference.read_bytes()
        # Each independent fault contributes (a, a^b, b), after reference normalization.
        if encoding == "b8":
            assert actual.read_bytes() == bytes([a | ((a ^ b) << 1) | (b << 2)]) * 64


@pytest.mark.integration
def test_separate_files_preserve_random_same_shot_correlation(tmp_path: Path) -> None:
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.qec.execution.context import MeasurementMap

    pytest.importorskip("stim")
    circuit = DynamicCircuit(1).bit_flip(0, 0.5).measure(0)
    layout = MeasurementMap(1, ((0,),))
    sampler = circuit.compile_detector_sampler(
        detector_map=layout, observable_map=layout, seed=7
    )
    first, second = tmp_path / "d", tmp_path / "o"
    sampler.sample_write(
        1025, filepath=first, format="b8", obs_out_filepath=second, obs_out_format="01"
    )
    bits = first.read_bytes()
    assert set(bits) == {0, 1}
    assert second.read_bytes() == b"".join(str(bit).encode() + b"\n" for bit in bits)


class FileConsumer(Protocol):
    def sample_write(
        self,
        shots: int,
        *,
        filepath: str | Path,
        format: Literal["01", "b8", "r8", "hits", "dets", "ptb64"] = "01",
    ) -> None: ...


def consume_file(writer: FileConsumer, path: Path) -> bytes:
    writer.sample_write(64, filepath=path, format="b8")
    return path.read_bytes()


def test_file_consumer_replaces_engine_without_consumer_changes(tmp_path: Path) -> None:
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.qec.execution.context import MeasurementMap

    class FakeWriter:
        def sample_write(
            self,
            shots: int,
            *,
            filepath: str | Path,
            format: Literal["01", "b8", "r8", "hits", "dets", "ptb64"] = "01",
        ) -> None:
            assert format == "b8"
            Path(filepath).write_bytes(b"\x01" * shots)

    assert consume_file(FakeWriter(), tmp_path / "fake") == b"\x01" * 64
    pytest.importorskip("stim")
    circuit = DynamicCircuit(1).bit_flip(0, 1.0).measure(0)
    writer = circuit.compile_detector_sampler(
        detector_map=MeasurementMap(1, ((0,),)),
        observable_map=MeasurementMap(1, ()),
        seed=7,
    )
    assert consume_file(writer, tmp_path / "real") == b"\x01" * 64
