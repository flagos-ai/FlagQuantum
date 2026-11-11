"""Verify the installed embedded measurement route without the Stim SDK."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import sys
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import Literal
from unittest.mock import patch

import torch

from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.execution.context import MeasurementMap


def _verify_shot_file_reading(native: object) -> None:
    from contextlib import closing

    from flagquantum.qec.execution.shot_data import (
        iter_shot_data_file,
        read_shot_data_file,
    )

    formats: dict[Literal["01", "b8", "r8", "ptb64", "hits", "dets"], bytes] = {
        "01": b"101\n" * 128,
        "b8": b"\x05" * 128,
        "r8": b"\x00\x01\x00" * 128,
        "hits": b"0,2\n" * 128,
        "dets": b"shot D0 L0\n" * 128,
        "ptb64": (b"\xff" * 8 + b"\x00" * 8 + b"\xff" * 8) * 2,
    }
    with TemporaryDirectory() as directory:
        path = Path(directory) / "records"
        for encoding, data in formats.items():
            path.write_bytes(data)
            with patch.object(
                native, "read_shot_reader", wraps=getattr(native, "read_shot_reader")
            ) as observed:
                result = read_shot_data_file(
                    path=path,
                    format=encoding,
                    num_detectors=2,
                    num_observables=1,
                    separate_observables=True,
                    bit_packed=True,
                )
                observed.assert_called_once()
            assert isinstance(result, tuple)
            assert result[0].tolist() == result[1].tolist() == [[1]] * 128
            for size in (1, 17, 63, 64, 65, 1024, 65536):
                total = 0
                with closing(
                    iter_shot_data_file(
                        path=path,
                        format=encoding,
                        num_detectors=2,
                        num_observables=1,
                        separate_observables=True,
                        bit_packed=True,
                        batch_size=size,
                    )
                ) as batches:
                    for batch in batches:
                        assert isinstance(batch, tuple)
                        assert (
                            batch[0].tolist()
                            == batch[1].tolist()
                            == [[1]] * len(batch[0])
                        )
                        total += len(batch[0])
                assert total == 128
        path.write_bytes(b"\x01\x80")
        with closing(
            iter_shot_data_file(path=path, format="b8", num_detectors=1)
        ) as batches:
            try:
                next(batches)
            except ValueError:
                pass
            else:
                raise AssertionError("bad tail was delivered")
    print(
        "embedded shot files: six codecs, seven batch sizes, split padding, actual calls, bad-tail refusal"
    )


def _verify_shot_data_writing(native: ModuleType) -> None:
    from flagquantum.qec.execution.shot_data import write_shot_data_file

    expected: dict[Literal["01", "b8", "r8", "ptb64", "hits", "dets"], bytes] = {
        "01": b"111\n" * 64,
        "b8": b"\x07" * 64,
        "r8": b"\x00\x00\x00\x00" * 64,
        "ptb64": b"\xff" * 24,
        "hits": b"0,1,2\n" * 64,
        "dets": b"shot D0 D1 D2\n" * 64,
    }
    with TemporaryDirectory() as directory:
        for encoding, witness in expected.items():
            with patch.object(
                native, "write_shot_records", wraps=native.write_shot_records
            ) as write:
                path = Path(directory) / encoding
                write_shot_data_file(
                    data=torch.ones((64, 3), dtype=torch.bool),
                    path=path,
                    format=encoding,
                    num_detectors=3,
                )
                write.assert_called_once()
                assert path.read_bytes() == witness
    print("embedded shot-data writing: six formats, exact bytes, actual native calls")


def main() -> None:
    assert importlib.util.find_spec("stim") is None
    try:
        importlib.metadata.distribution("stim")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise AssertionError("the Stim SDK distribution is installed")

    native = import_module("flagquantum.simulation.stabilizer._embedded_stim")
    assert native.__file__ is not None
    native_path = Path(native.__file__).resolve()
    assert native_path.is_relative_to(Path(sys.prefix).resolve())
    assert native.upstream_commit == "e2fc1eca7fd21684d433aa5f10f4504ea4860d07"
    _verify_shot_file_reading(native)
    _verify_shot_data_writing(native)
    circuit = DynamicCircuit(1).x(0).measure(0)
    with patch.object(native, "prepare", wraps=native.prepare) as prepare:
        sampler = circuit.compile_sampler(seed=7)
        prepare.assert_called_once()
    with patch.object(native, "sample", wraps=native.sample) as sample:
        assert sampler.sample(2, bit_packed=True).tolist() == [[1], [1]]
        sample.assert_called_once()
    repeated_slot = (
        DynamicCircuit(1)
        .x(0)
        .measure(0, classical_bit=0)
        .reset(0)
        .measure(0, classical_bit=0)
    )
    expected: dict[Literal["01", "b8", "r8", "hits", "dets", "ptb64"], bytes] = {
        "01": b"10\n" * 64,
        "b8": b"\x01" * 64,
        "r8": b"\x00\x01" * 64,
        "hits": b"0\n" * 64,
        "dets": b"shot M0\n" * 64,
        "ptb64": b"\xff" * 8 + b"\x00" * 8,
    }
    with TemporaryDirectory() as directory:
        for format, reference in expected.items():
            writer = "write_b8" if format == "b8" else "write_records"
            with patch.object(native, writer, wraps=getattr(native, writer)) as write:
                output = Path(directory) / format
                repeated_slot.compile_sampler(seed=7).sample_write(
                    64, filepath=output, format=format
                )
                write.assert_called_once()
                assert output.read_bytes() == reference
    print(
        "embedded measurement files: six formats, exact bytes, repeated slot, actual native calls"
    )
    with patch.object(
        native, "prepare_converter", wraps=native.prepare_converter
    ) as prepare_converter:
        converter = circuit.compile_m2d_converter(
            detector_map=MeasurementMap(1, ((0,),)),
            observable_map=MeasurementMap(1, ()),
        )
        prepare_converter.assert_called_once()
    with patch.object(
        native, "convert_records", wraps=native.convert_records
    ) as convert:
        dets, obs = converter.convert(
            measurements=torch.tensor([[True], [False]]), separate_observables=True
        )
        convert.assert_called_once()
        assert dets.tolist() == [[False], [True]] and obs.shape == (2, 0)
    repeated = (
        DynamicCircuit(1)
        .measure(0, classical_bit=0)
        .measure(0, classical_bit=0)
        .conditional("x", 0, classical_bit=0)
        .measure(0, classical_bit=0)
    )
    file_converter = repeated.compile_m2d_converter(
        detector_map=MeasurementMap(3, ((0, 1), (0, 2))),
        observable_map=MeasurementMap(3, ((1,),)),
    )
    file_inputs: dict[Literal["01", "b8", "r8", "ptb64", "hits", "dets"], bytes] = {
        "01": b"010\n" * 64,
        "b8": b"\x02" * 64,
        "r8": b"\x01\x01" * 64,
        "hits": b"1\n" * 64,
        "dets": b"shot M1\n" * 64,
        "ptb64": b"\0" * 8 + b"\xff" * 8 + b"\0" * 8,
    }
    file_outputs: dict[Literal["01", "b8", "r8", "hits", "dets"], bytes] = {
        "01": b"101\n" * 64,
        "b8": b"\x05" * 64,
        "r8": b"\x00\x01\x00" * 64,
        "hits": b"0,2\n" * 64,
        "dets": b"shot D0 L0\n" * 64,
    }
    with (
        TemporaryDirectory() as directory,
        patch.object(native, "convert_file", wraps=native.convert_file) as file_convert,
    ):
        root = Path(directory)
        for input_format, raw in file_inputs.items():
            source_path = root / ("input." + input_format)
            source_path.write_bytes(raw)
            for output_format, expected_file_bytes in file_outputs.items():
                target = root / (input_format + "." + output_format)
                logical = root / (input_format + "." + output_format + ".obs")
                file_converter.convert_file(
                    measurements_filepath=source_path,
                    measurements_format=input_format,
                    detection_events_filepath=target,
                    detection_events_format=output_format,
                    append_observables=True,
                    obs_out_filepath=logical,
                    obs_out_format="b8",
                )
                assert (
                    target.read_bytes() == expected_file_bytes
                    and logical.read_bytes() == b"\x01" * 64
                )
        assert file_convert.call_count == 30
    print(
        "embedded file conversion: 30 fixed-byte cases; actual native calls; no Stim SDK"
    )
    noisy = DynamicCircuit(1).x(0).bit_flip(0, 1.0).measure(0)
    maps = MeasurementMap(1, ((0,),))
    with patch.object(
        native, "prepare_detector", wraps=native.prepare_detector
    ) as prepare_detector:
        detector_sampler = noisy.compile_detector_sampler(
            detector_map=maps, observable_map=maps, seed=7
        )
        prepare_detector.assert_called_once()
    with patch.object(
        native, "sample_detectors", wraps=native.sample_detectors
    ) as draw:
        packed = detector_sampler.sample(2, append_observables=True, bit_packed=True)
        assert isinstance(packed, torch.Tensor)
        assert packed.tolist() == [[3], [3]]
        draw.assert_called_once()
    detector_witnesses: dict[
        Literal["01", "b8", "r8", "hits", "dets", "ptb64"], bytes
    ] = {
        "01": b"11\n" * 64,
        "b8": b"\x03" * 64,
        "r8": b"\x00\x00\x00" * 64,
        "hits": b"0,1\n" * 64,
        "dets": b"shot D0 L0\n" * 64,
        "ptb64": b"\xff" * 16,
    }
    with TemporaryDirectory() as directory:
        for encoding, reference in detector_witnesses.items():
            with patch.object(
                native, "write_detectors", wraps=native.write_detectors
            ) as write:
                path = Path(directory) / ("detectors." + encoding)
                detector_sampler.sample_write(
                    64, filepath=path, format=encoding, append_observables=True
                )
                write.assert_called_once()
                assert path.read_bytes() == reference
        with patch.object(
            native, "write_detectors", wraps=native.write_detectors
        ) as write:
            detectors = Path(directory) / "separate.dets"
            observables = Path(directory) / "separate.b8"
            detector_sampler.sample_write(
                64,
                filepath=detectors,
                format="dets",
                obs_out_filepath=observables,
                obs_out_format="b8",
            )
            write.assert_called_once()
            assert detectors.read_bytes() == b"shot D0\n" * 64
            assert observables.read_bytes() == b"\x01" * 64
    print(
        "embedded detector files: six formats, same-call observable output, actual native calls"
    )
    with patch.object(
        native, "detector_error_model", wraps=native.detector_error_model
    ) as analyze:
        model = noisy.detector_error_model(detector_map=maps, observable_map=maps)
        assert model.num_detectors == model.num_observables == 1
        assert model.errors[0].probability == 1.0
        analyze.assert_called_once()
    detector_map = MeasurementMap(3, ((0, 1), (1, 2)))
    observable_map = MeasurementMap(3, ((0,),))
    for mask in range(8):
        forced = DynamicCircuit(3).x(0).x(1).x(2)
        for qubit in range(3):
            forced.bit_flip(qubit, float((mask >> qubit) & 1))
        forced.measure(0, classical_bit=0).measure(1, classical_bit=1).measure(
            2, classical_bit=2
        )
        handle = forced.compile_detector_sampler(
            detector_map=detector_map, observable_map=observable_map, seed=7
        )
        result = handle.sample(1, separate_observables=True)
        assert isinstance(result, tuple)
        a, b, c = ((mask >> qubit) & 1 for qubit in range(3))
        assert result[0].tolist() == [[bool(a ^ b), bool(b ^ c)]]
        assert result[1].tolist() == [[bool(a)]]
    crossing = noisy.compile_detector_sampler(
        detector_map=MeasurementMap(1, ((0,),) * 7),
        observable_map=MeasurementMap(1, ((0,),) * 2),
        seed=7,
    )
    packed = crossing.sample(1, append_observables=True, bit_packed=True)
    assert isinstance(packed, torch.Tensor) and packed.tolist() == [[255, 1]]
    repeated = (
        DynamicCircuit(3)
        .h(0)
        .measure(0, classical_bit=0)
        .conditional("x", 1, classical_bit=0)
        .reset(0)
        .conditional("x", 0, classical_bit=0, equals=0)
        .measure(0, classical_bit=0)
        .conditional("x", 2, classical_bit=0)
        .measure(1, classical_bit=1)
        .measure(2, classical_bit=2)
    )
    normalized = repeated.compile_detector_sampler(
        detector_map=MeasurementMap(4, ((0, 1), (0, 2), (1, 3))),
        observable_map=MeasurementMap(4, ((2, 3),)),
        seed=7,
    ).sample(35, append_observables=True)
    assert isinstance(normalized, torch.Tensor) and not normalized.any()
    print(
        "embedded detectors: eight independent forced signatures, repeated-slot X feedback, 7+2 packed boundary"
    )
    from flagquantum.errors import CapabilityError

    for refused in (
        DynamicCircuit(1).t(0).measure(0),
        DynamicCircuit(1).measure(0).conditional("h", 0, classical_bit=0).measure(0),
    ):
        with patch.object(native, "prepare", wraps=native.prepare) as untouched_sampler:
            try:
                refused.compile_sampler(seed=7)
            except CapabilityError:
                pass
            else:
                raise AssertionError("unsupported measurement program was admitted")
            untouched_sampler.assert_not_called()
        width = len([g for g in refused.to_ir().instructions if g.name == "measure"])
        with patch.object(
            native, "prepare_detector", wraps=native.prepare_detector
        ) as untouched:
            try:
                refused.compile_detector_sampler(
                    detector_map=MeasurementMap(width, ((0,),)),
                    observable_map=MeasurementMap(width, ()),
                )
            except CapabilityError:
                pass
            else:
                raise AssertionError("unsupported program was admitted")
            untouched.assert_not_called()
    print(
        "embedded detectors: one prepare/draw/analyze; T and conditional H refuse before preparation"
    )
    print("embedded conversion: one prepare, one convert; no Stim SDK")
    print("embedded module", native_path)
    print("embedded sha256", hashlib.sha256(native_path.read_bytes()).hexdigest())
    print("upstream commit", native.upstream_commit)


if __name__ == "__main__":
    main()
