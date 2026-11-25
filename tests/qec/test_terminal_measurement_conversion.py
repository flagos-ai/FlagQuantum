"""Contract and engine replacement checks without optional SDKs."""

import json
import pickle
from collections.abc import Callable
from dataclasses import replace
from inspect import signature
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum.core.ir import CircuitIR, MeasurementNode
from flagquantum.errors import CapabilityError
from flagquantum.qec.execution import measurement_conversion as bridge
from flagquantum.qec.execution.context import MeasurementMap

pytestmark = pytest.mark.unit


def test_public_contract_and_prepared_engine_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = 0
    calls = 0

    def prepare(
        program: CircuitIR,
        *,
        terminal_qubits: tuple[int, ...],
        detector_columns: tuple[tuple[int, ...], ...],
        observable_columns: tuple[tuple[int, ...], ...],
    ) -> Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        nonlocal prepared
        prepared += 1
        assert tuple(gate.name for gate in program.instructions) == ("x",)
        assert terminal_qubits == (1, 0)
        assert detector_columns == ((0, 1),)
        assert observable_columns == ((1,),)

        def convert(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal calls
            calls += 1
            return rows[:, :1] ^ ~rows[:, 1:], ~rows[:, 1:]

        return convert

    monkeypatch.setattr(bridge, "_prepare_terminal_record_conversion", prepare)
    converter = bridge.compile_m2d_converter(
        fq.Circuit(2).x(0),
        measurement_qubits=(1, 0),
        detector_map=MeasurementMap(2, ((0, 1),)),
        observable_map=MeasurementMap(2, ((1,),)),
    )
    assert converter.measurement_qubits == (1, 0)
    assert isinstance(converter, bridge.CompiledMeasurementsToDetectionEventsConverter)
    first = converter.convert(
        measurements=torch.tensor([[False, True]]), separate_observables=True
    )
    second = converter.convert(
        measurements=torch.tensor([[True, True], [False, False]]),
        separate_observables=True,
    )
    assert [value.tolist() for value in first] == [[[False]], [[False]]]
    assert [value.tolist() for value in second] == [[[True], [True]], [[False], [True]]]
    with pytest.raises(ValueError, match="width"):
        converter.convert(
            measurements=torch.zeros((1, 1), dtype=torch.bool),
            separate_observables=True,
        )
    with pytest.raises(TypeError, match="bool"):
        converter.convert(
            measurements=torch.zeros((1, 2), dtype=torch.uint8),
            separate_observables=True,
        )
    assert (prepared, calls) == (1, 2)
    with pytest.raises(TypeError, match="serialize"):
        pickle.dumps(converter)
    with pytest.raises(TypeError, match="compile_m2d_converter"):
        bridge.CompiledMeasurementsToDetectionEventsConverter()

    contract_path = (
        Path(__file__).parents[2]
        / "contracts/qec-terminal-measurement-converter-v1.json"
    )
    contract = json.loads(contract_path.read_text())
    assert bridge.__all__ == ("convert_measurement_records", *contract["exports_added"])
    assert all(not hasattr(fq, name) for name in contract["exports_added"])
    assert tuple(signature(bridge.compile_m2d_converter).parameters) == (
        "program",
        "measurement_qubits",
        "detector_map",
        "observable_map",
    )
    assert tuple(signature(converter.convert).parameters) == (
        "measurements",
        "separate_observables",
    )


@pytest.mark.parametrize("qubits", [(0, 0), (0, 2), ()])
def test_invalid_readout_is_refused(qubits: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="measurement_qubits"):
        bridge.compile_m2d_converter(
            fq.Circuit(2),
            measurement_qubits=qubits,
            detector_map=MeasurementMap(2, ((0,),)),
            observable_map=MeasurementMap(2, ()),
        )


@pytest.mark.parametrize(
    ("detectors", "observables", "message"),
    [
        (MeasurementMap(1, ((0,),)), MeasurementMap(2, ()), "width"),
        (MeasurementMap(2, ((0,),)), MeasurementMap(1, ()), "width"),
        (MeasurementMap(2, ()), MeasurementMap(2, ()), "at least one detector"),
    ],
)
def test_invalid_maps_are_refused(
    detectors: MeasurementMap, observables: MeasurementMap, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        bridge.compile_m2d_converter(
            fq.Circuit(2),
            measurement_qubits=(0, 1),
            detector_map=detectors,
            observable_map=observables,
        )


def test_rejects_unsupported_program_metadata() -> None:
    program = fq.Circuit(2).x(0).to_ir()
    for invalid in (
        replace(program, metadata={"routing": {}}),
        replace(
            program,
            metadata={
                "layout": {
                    "n_wires": 2,
                    "logical_to_physical": (1, 0),
                    "physical_to_logical": (0, 1),
                }
            },
        ),
        replace(program, measurements=(MeasurementNode("sample", (0,)),)),
    ):
        with pytest.raises(CapabilityError):
            bridge.compile_m2d_converter(
                invalid,
                measurement_qubits=(0,),
                detector_map=MeasurementMap(1, ((0,),)),
                observable_map=MeasurementMap(1, ()),
            )
