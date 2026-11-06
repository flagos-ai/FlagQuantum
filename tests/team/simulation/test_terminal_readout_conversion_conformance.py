"""Verify terminal readout against independent parity and physical column order."""

import math
from dataclasses import replace
from itertools import product

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler import Layout, apply_layout
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.errors import CapabilityError
from flagquantum.qec.execution.context import MeasurementMap
from flagquantum.qec.execution.measurement_conversion import compile_m2d_converter
from flagquantum.simulation.stabilizer import _record_conversion
from flagquantum.simulation.stabilizer.engine import _annotated_stim_source

pytestmark = pytest.mark.integration


def test_native_compiled_and_placed_match_independent_parities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stim = pytest.importorskip("stim")
    theta = fq.Parameter("theta")
    circuit = (
        fq.Circuit(3, dtype=torch.complex128).h(0).cx(0, 1).rx(2, theta=theta).x(1).x(1)
    )
    bound = circuit.bind_parameters({theta: math.pi})
    compiled = fq.compile(bound)
    placed = apply_layout(compiled, Layout((2, 0, 1)))
    rows = torch.tensor(list(product((False, True), repeat=3)))
    order = (2, 0, 1)
    maps = dict(
        detector_map=MeasurementMap(3, ((1, 2),)),
        observable_map=MeasurementMap(3, ((0,),)),
    )
    emitted: list[tuple[int, ...]] = []

    def annotate(
        program: CircuitIR,
        *,
        terminal_qubits: tuple[int, ...],
        detector_columns: tuple[tuple[int, ...], ...],
        observable_columns: tuple[tuple[int, ...], ...],
    ) -> str:
        text = _annotated_stim_source(
            program,
            terminal_qubits=terminal_qubits,
            detector_columns=detector_columns,
            observable_columns=observable_columns,
        )
        (measurement,) = [op for op in stim.Circuit(text) if op.name == "M"]
        emitted.append(tuple(target.value for target in measurement.targets_copy()))
        return text

    monkeypatch.setattr(_record_conversion, "_annotated_stim_source", annotate)
    for program in (bound, compiled, placed):
        converter = compile_m2d_converter(program, measurement_qubits=order, **maps)
        assert converter.measurement_qubits == order
        for batch in (rows[:3], rows[3:]):
            detectors, observables = converter.convert(
                measurements=batch, separate_observables=True
            )
            torch.testing.assert_close(detectors[:, 0], batch[:, 1] ^ batch[:, 2])
            torch.testing.assert_close(observables[:, 0], ~batch[:, 0])
    assert emitted == [(2, 0, 1), (2, 0, 1), (1, 2, 0)]
    with pytest.raises(CapabilityError, match="nondeterministic"):
        compile_m2d_converter(
            replace(placed, metadata={}), measurement_qubits=order, **maps
        )


def test_zero_shots_no_observables_and_owned_results() -> None:
    pytest.importorskip("stim")
    converter = compile_m2d_converter(
        fq.Circuit(1).x(0),
        measurement_qubits=(0,),
        detector_map=MeasurementMap(1, ((0,),)),
        observable_map=MeasurementMap(1, ()),
    )
    detectors, observables = converter.convert(
        measurements=torch.empty((0, 1), dtype=torch.bool), separate_observables=True
    )
    assert detectors.shape == (0, 1) and observables.shape == (0, 0)
    raw = torch.tensor([[True], [False]])
    detectors, observables = converter.convert(
        measurements=raw, separate_observables=True
    )
    raw.fill_(True)
    assert detectors.tolist() == [[False], [True]]
    assert observables.shape == (2, 0)


def test_unsupported_and_nondeterministic_circuits_refuse() -> None:
    pytest.importorskip("stim")
    theta = fq.Parameter("theta")
    for circuit, message in (
        (fq.Circuit(1).rx(0, theta=theta), "Clifford"),
        (fq.Circuit(1).rx(0, theta=math.nextafter(math.pi, 0)), "Clifford"),
        (DynamicCircuit(1).measure(0, classical_bit=0), "static"),
        (fq.Circuit(1).h(0), "nondeterministic"),
    ):
        with pytest.raises(CapabilityError, match=message):
            compile_m2d_converter(
                circuit,
                measurement_qubits=(0,),
                detector_map=MeasurementMap(1, ((0,),)),
                observable_map=MeasurementMap(1, ()),
            )
