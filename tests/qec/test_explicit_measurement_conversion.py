"""Public explicit-measurement conversion with a core-only implementation fake."""

from __future__ import annotations

import inspect
from collections.abc import Callable

import pytest
import torch

import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit
from flagquantum.errors import CapabilityError
from flagquantum.qec.execution import measurement_conversion as conversion
from flagquantum.qec.execution.context import MeasurementMap

pytestmark = pytest.mark.unit


def _program() -> DynamicCircuit:
    return DynamicCircuit(1).x(0).measure(0).reset(0).measure(0)


def test_explicit_converter_signature_and_column_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert tuple(inspect.signature(fq.Circuit.compile_m2d_converter).parameters) == (
        "self",
        "detector_map",
        "observable_map",
    )
    calls = []

    def prepare(
        program: object,
        *,
        detector_columns: tuple[tuple[int, ...], ...],
        observable_columns: tuple[tuple[int, ...], ...],
    ) -> tuple[
        tuple[int, ...], Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
    ]:
        calls.append((detector_columns, observable_columns))

        def convert(records: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return (~records[:, :1]).clone(), torch.empty(
                (len(records), 0), dtype=torch.bool
            )

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
    detectors: MeasurementMap,
    observables: MeasurementMap,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("invalid maps entered conversion preparation")

    monkeypatch.setattr(conversion, "_prepare_explicit_record_conversion", unexpected)
    with pytest.raises(ValueError, match=message):
        _program().compile_m2d_converter(
            detector_map=detectors, observable_map=observables
        )


def test_record_envelope_and_budget_refuse_before_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flagquantum.simulation.stabilizer import _record_conversion as implementation

    calls: list[torch.Tensor] = []

    def backend(
        text: str, detectors: int, observables: int
    ) -> Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        def convert(records: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            calls.append(records)
            return records[:, :1].clone(), torch.empty(
                (len(records), 0), dtype=torch.bool
            )

        return convert

    monkeypatch.setattr(implementation, "_prepare_explicit_backend", backend)
    handle = _program().compile_m2d_converter(
        detector_map=MeasurementMap(2, ((0,),)), observable_map=MeasurementMap(2, ())
    )
    for records, error in (
        (torch.zeros((1, 2), dtype=torch.int64), TypeError),
        (torch.zeros((1, 1), dtype=torch.bool), ValueError),
        (torch.zeros((2,), dtype=torch.bool), ValueError),
    ):
        with pytest.raises(error):
            handle.convert(measurements=records, separate_observables=True)
    empty, obs = handle.convert(
        measurements=torch.empty((0, 2), dtype=torch.bool), separate_observables=True
    )
    assert empty.shape == (0, 1) and obs.shape == (0, 0) and not calls
    monkeypatch.setattr(implementation, "_MAX_EXPLICIT_CONVERSION_BYTES", 1)
    with pytest.raises(CapabilityError, match="byte limit"):
        handle.convert(
            measurements=torch.zeros((1, 2), dtype=torch.bool),
            separate_observables=True,
        )
    assert not calls


@pytest.mark.parametrize(
    "rows, width, message", [(65537, 1, "row limit"), (50001, 2, "reference limit")]
)
def test_map_budget_precedes_backend(
    rows: int, width: int, message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("map excess reached backend")

    monkeypatch.setattr(conversion, "_prepare_explicit_record_conversion", unexpected)
    program = DynamicCircuit(1).measure(0)
    if width == 2:
        program.measure(0)
    with pytest.raises(CapabilityError, match=message):
        program.compile_m2d_converter(
            detector_map=MeasurementMap(width, (tuple(range(width)),) * rows),
            observable_map=MeasurementMap(width, ()),
        )
