"""Compare the native-record contract with an independent upstream circuit."""

from __future__ import annotations

import pytest
import torch

from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.execution.context import MeasurementMap

pytestmark = pytest.mark.integration


def _program() -> DynamicCircuit:
    return (
        DynamicCircuit(3).h(0).measure(0, classical_bit=0)
        .conditional("x", 1, classical_bit=0).reset(0)
        .conditional("x", 0, classical_bit=0, equals=0).measure(0, classical_bit=0)
        .conditional("x", 2, classical_bit=0).measure(1).measure(2)
    )


def test_explicit_records_match_handwritten_stim_reference() -> None:
    stim = pytest.importorskip("stim")
    circuit = stim.Circuit(
        "H 0\nM 0\nCX rec[-1] 1\nR 0\nX 0\nCX rec[-1] 0\n"
        "M 0\nCX rec[-1] 2\nM 1\nM 2\nDETECTOR rec[-4] rec[-3]\n"
        "DETECTOR rec[-4] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-3] rec[-1]"
    )
    records = torch.tensor(
        [[True, False, True, False], [False, True, False, True],
         [True, False, False, False], [True, False, True, True]]
    )
    converter = _program().compile_m2d_converter(
        detector_map=MeasurementMap(4, ((0, 1), (0, 2))),
        observable_map=MeasurementMap(4, ((1, 3),)),
    )
    detectors, observables = converter.convert(measurements=records, separate_observables=True)
    expected = circuit.compile_m2d_converter().convert(
        measurements=records.numpy(), separate_observables=True
    )
    assert detectors.tolist() == expected[0].tolist() == [
        [False, False], [False, False], [False, True], [False, False]
    ]
    assert observables.tolist() == expected[1].tolist() == [[False], [False], [False], [True]]
