"""Public explicit-measurement conversion with a core-only implementation fake."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Callable

import pytest
import torch

import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit
from flagquantum.errors import CapabilityError
from flagquantum.qec.execution.context import MeasurementMap
from flagquantum.qec.execution import measurement_conversion as conversion

pytestmark = pytest.mark.unit


def _program() -> DynamicCircuit:
    return DynamicCircuit(1).x(0).measure(0).reset(0).measure(0)


def test_explicit_converter_signature_and_column_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert tuple(inspect.signature(fq.Circuit.compile_m2d_converter).parameters) == (
        "self", "detector_map", "observable_map"
    )
    calls = []

    def prepare(
        program: object,
        *,
        detector_columns: tuple[tuple[int, ...], ...],
        observable_columns: tuple[tuple[int, ...], ...],
    ) -> tuple[tuple[int, ...], Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]]:
        calls.append((detector_columns, observable_columns))

        def convert(records: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return (~records[:, :1]).clone(), torch.empty((len(records), 0), dtype=torch.bool)

        return (0, 0), convert

    monkeypatch.setattr(conversion, "_prepare_explicit_record_conversion", prepare)
    handle = _program().compile_m2d_converter(
        detector_map=MeasurementMap(2, ((0,),)),
        observable_map=MeasurementMap(2, ()),
    )
    assert calls == [(((0,),), ())]
    assert handle.measurement_qubits == (0, 0)
    detectors, observables = handle.convert(
        measurements=torch.tensor([[True, False]]), separate_observables=True
    )
    assert detectors.tolist() == [[False]] and observables.shape == (1, 0)
    again, _ = handle.convert(
        measurements=torch.tensor([[False, False]]), separate_observables=True
    )
    detectors.fill_(True)
    assert again.tolist() == [[True]] and len(calls) == 1


@pytest.mark.parametrize(
    "program",
    [
        DynamicCircuit(1).h(0).t(0).measure(0),
        DynamicCircuit(2).measure(0).conditional("h", 1, classical_bit=0),
    ],
)
def test_unsupported_program_refuses_before_backend(
    program: DynamicCircuit, monkeypatch: pytest.MonkeyPatch
) -> None:
    from flagquantum.simulation.stabilizer import _record_conversion as implementation

    def unexpected(*args: object) -> None:
        pytest.fail("unsupported program loaded a conversion backend")

    monkeypatch.setattr(implementation, "_prepare_explicit_backend", unexpected)
    with pytest.raises(CapabilityError):
        program.compile_m2d_converter(
            detector_map=MeasurementMap(1, ((0,),)),
            observable_map=MeasurementMap(1, ()),
        )


@pytest.mark.parametrize(
    ("detectors", "observables", "message"),
    [
        (MeasurementMap(1, ((0,),)), MeasurementMap(2, ()), "width"),
        (MeasurementMap(2, ()), MeasurementMap(2, ()), "at least one detector"),
    ],
)
def test_invalid_maps_refuse_before_backend(
    detectors: MeasurementMap, observables: MeasurementMap, message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("invalid maps entered conversion preparation")

    monkeypatch.setattr(conversion, "_prepare_explicit_record_conversion", unexpected)
    with pytest.raises(ValueError, match=message):
        _program().compile_m2d_converter(detector_map=detectors, observable_map=observables)
