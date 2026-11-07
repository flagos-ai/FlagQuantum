"""Verify the installed embedded measurement route without the Stim SDK."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import sys
from importlib import import_module
from pathlib import Path
from unittest.mock import patch

import torch

from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.execution.context import MeasurementMap


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
    circuit = DynamicCircuit(1).x(0).measure(0)
    with patch.object(native, "prepare", wraps=native.prepare) as prepare:
        sampler = circuit.compile_sampler(seed=7)
        prepare.assert_called_once()
    with patch.object(native, "sample", wraps=native.sample) as sample:
        assert sampler.sample(2, bit_packed=True).tolist() == [[1], [1]]
        sample.assert_called_once()
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
    print("embedded conversion: one prepare, one convert; no Stim SDK")
    print("embedded module", native_path)
    print("embedded sha256", hashlib.sha256(native_path.read_bytes()).hexdigest())
    print("upstream commit", native.upstream_commit)


if __name__ == "__main__":
    main()
