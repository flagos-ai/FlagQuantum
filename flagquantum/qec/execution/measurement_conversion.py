"""Convert externally recorded measurement data into QEC events."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, NoReturn, SupportsIndex

import torch

from ...circuit import Circuit
from ...compiler.layout import Layout
from ...core.ir import CircuitIR, ensure_circuit_ir
from ...errors import CapabilityError
from ...simulation.stabilizer._record_conversion import (
    _convert_stim_measurements,
    _prepare_explicit_record_conversion,
    _prepare_terminal_record_conversion,
)
from ..dem.model import DemSample
from .context import MeasurementMap

__all__ = (
    "convert_measurement_records",
    "CompiledMeasurementsToDetectionEventsConverter",
    "compile_m2d_converter",
)


def _check_dense_cpu_records(record: torch.Tensor, name: str) -> None:
    if not isinstance(record, torch.Tensor) or record.layout != torch.strided:
        raise TypeError(f"{name} must be a dense tensor")
    if record.device.type != "cpu":
        raise CapabilityError(f"{name} must be on CPU")
    if record.ndim != 2:
        raise ValueError(f"{name} must have shot and record dimensions")


def _check_record(record: torch.Tensor, packed: bool, name: str) -> None:
    _check_dense_cpu_records(record, name)
    if packed:
        if record.dtype != torch.uint8:
            raise TypeError(f"packed {name} must be uint8")
        return
    if record.dtype == torch.bool:
        return
    if record.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
    ):
        raise TypeError(f"{name} must have bool or integer dtype")
    if not bool(((record == 0) | (record == 1)).all()):
        raise ValueError(f"{name} must be binary")


def convert_measurement_records(
    circuit_text: str,
    measurements: torch.Tensor,
    *,
    circuit_format: Literal["stim"],
    sweep_bits: torch.Tensor | None = None,
    bit_packed_measurements: bool = False,
    bit_packed_sweep_bits: bool = False,
) -> DemSample:
    """Convert external CPU records to reference-normalized detector and logical flips.

    Each input uses shot-major rows. Packed inputs use LSB-first uint8 bytes
    with zero padding. Omitted sweep controls are zero. The circuit format
    identifies the source text dialect. The current implementation requires
    the Stim extra and refuses nondeterministic declared parities.

    Examples:
        >>> from importlib.util import find_spec
        >>> text = "R 0\\nX 0\\nM 0\\nDETECTOR rec[-1]"
        >>> if find_spec("stim") is not None:
        ...     batch = convert_measurement_records(
        ...         text, torch.tensor([[1], [0]]), circuit_format="stim"
        ...     )
        ...     assert batch.detectors.tolist() == [[False], [True]]
    """
    if circuit_format != "stim":
        raise CapabilityError("unsupported measurement circuit format")
    if (
        type(bit_packed_measurements) is not bool
        or type(bit_packed_sweep_bits) is not bool
    ):
        raise TypeError("packed flags must be bool")
    _check_record(measurements, bit_packed_measurements, "measurements")
    if sweep_bits is None:
        if bit_packed_sweep_bits:
            raise ValueError("packed sweep controls require sweep_bits")
    else:
        _check_record(sweep_bits, bit_packed_sweep_bits, "sweep_bits")
        if sweep_bits.shape[0] != measurements.shape[0]:
            raise ValueError(
                "sweep_bits and measurements must have the same shot count"
            )
    detectors, observables = _convert_stim_measurements(
        circuit_text,
        measurements,
        sweep_bits=sweep_bits,
        bit_packed_measurements=bit_packed_measurements,
        bit_packed_sweep_bits=bit_packed_sweep_bits,
    )
    return DemSample(detectors, observables)


class CompiledMeasurementsToDetectionEventsConverter:
    """Reusable reference-normalized converter for declared measurement records."""

    __slots__ = ("_measurement_qubits", "_convert_records")
    _measurement_qubits: tuple[int, ...]
    _convert_records: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]

    def __init__(self) -> None:
        raise TypeError("create a converter with compile_m2d_converter()")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        raise TypeError("compiled terminal converters cannot be serialized")

    @property
    def measurement_qubits(self) -> tuple[int, ...]:
        """Logical qubit of each input column in its declared record order."""
        return self._measurement_qubits

    def convert(
        self,
        *,
        measurements: torch.Tensor,
        separate_observables: Literal[True],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return separate owned detector and logical flips for each input row."""
        if separate_observables is not True:
            raise ValueError("terminal conversion requires separate_observables=True")
        _check_dense_cpu_records(measurements, "measurements")
        if measurements.dtype != torch.bool:
            raise TypeError("terminal measurements must have bool dtype")
        if measurements.shape[1] != len(self._measurement_qubits):
            raise ValueError("measurements have the wrong declared width")
        return self._convert_records(measurements)


def _physical_terminal_qubits(
    ir: CircuitIR, logical: tuple[int, ...]
) -> tuple[int, ...]:
    if "routing" in ir.metadata or any(
        "routing_phase" in gate.metadata for gate in ir.instructions
    ):
        raise CapabilityError("terminal conversion does not accept routed circuits")
    if "layout" not in ir.metadata:
        return logical

    report = ir.metadata["layout"]
    if (
        not isinstance(report, dict)
        or set(report) != {"n_wires", "logical_to_physical", "physical_to_logical"}
        or type(report["n_wires"]) is not int
        or report["n_wires"] != ir.n_wires
    ):
        raise CapabilityError("terminal conversion requires an apply_layout assignment")
    placement = report["logical_to_physical"]
    inverse = report["physical_to_logical"]
    if (
        not isinstance(placement, (list, tuple))
        or len(placement) != ir.n_wires
        or not isinstance(inverse, (list, tuple))
        or any(type(item) is not int for item in inverse)
    ):
        raise CapabilityError("invalid terminal circuit layout assignment")
    try:
        layout = Layout(placement)
    except ValueError as error:
        raise CapabilityError("invalid terminal circuit layout assignment") from error
    if tuple(inverse) != layout.physical_to_logical:
        raise CapabilityError(
            "terminal circuit layout inverse disagrees with its assignment"
        )
    return layout.map_qubits(logical)


def compile_m2d_converter(
    program: Circuit | CircuitIR,
    *,
    measurement_qubits: tuple[int, ...],
    detector_map: MeasurementMap,
    observable_map: MeasurementMap,
) -> CompiledMeasurementsToDetectionEventsConverter:
    """Prepare conversion of terminal Z records from a bound Clifford circuit.

    Map rows index the supplied logical readout columns. The prepared reference
    is reused across calls to the returned handle. The Stim extra is required.

    Examples:
        >>> from importlib.util import find_spec
        >>> import flagquantum as fq
        >>> if find_spec("stim") is not None:
        ...     converter = compile_m2d_converter(
        ...         fq.Circuit(1).x(0), measurement_qubits=(0,),
        ...         detector_map=MeasurementMap(1, ((0,),)),
        ...         observable_map=MeasurementMap(1, ()),
        ...     )
        ...     dets, obs = converter.convert(
        ...         measurements=torch.tensor([[True], [False]]),
        ...         separate_observables=True,
        ...     )
        ...     assert dets.tolist() == [[False], [True]] and obs.shape == (2, 0)
    """
    if not isinstance(program, (Circuit, CircuitIR)):
        raise TypeError("program must be a Circuit or CircuitIR")
    if isinstance(program, Circuit) and program.circuit_param.get("inputs") is not None:
        raise CapabilityError("terminal conversion requires the default initial state")
    ir = ensure_circuit_ir(program)
    if ir.measurements:
        raise CapabilityError(
            "terminal conversion does not accept existing measurement requests"
        )
    if (
        not isinstance(measurement_qubits, tuple)
        or not measurement_qubits
        or any(
            type(wire) is not int or not 0 <= wire < ir.n_wires
            for wire in measurement_qubits
        )
        or len(set(measurement_qubits)) != len(measurement_qubits)
    ):
        raise ValueError("measurement_qubits must be distinct in-range logical qubits")
    if not isinstance(detector_map, MeasurementMap) or not isinstance(
        observable_map, MeasurementMap
    ):
        raise TypeError(
            "detector_map and observable_map must be MeasurementMap instances"
        )
    if detector_map.num_measurements != len(measurement_qubits) or (
        observable_map.num_measurements != len(measurement_qubits)
    ):
        raise ValueError("both measurement maps must match the logical record width")
    if not detector_map.num_rows:
        raise ValueError("terminal conversion requires at least one detector")
    physical = _physical_terminal_qubits(ir, measurement_qubits)
    convert = _prepare_terminal_record_conversion(
        ir,
        terminal_qubits=physical,
        detector_columns=detector_map.rows,
        observable_columns=observable_map.rows,
    )
    handle = object.__new__(CompiledMeasurementsToDetectionEventsConverter)
    handle._measurement_qubits = measurement_qubits
    handle._convert_records = convert
    return handle


def _compile_explicit_m2d_converter(
    program: Circuit,
    *,
    detector_map: MeasurementMap,
    observable_map: MeasurementMap,
) -> CompiledMeasurementsToDetectionEventsConverter:
    if (
        torch.device(program.device).type != "cpu"
        or program.runtime_config.backend != "pytorch"
        or program.bsz != 1
        or program.circuit_param.get("inputs") is not None
    ):
        raise CapabilityError(
            "explicit conversion requires one CPU PyTorch circuit with its default initial state"
        )
    ir = program.to_ir()
    width = sum(gate.name == "measure" for gate in ir.instructions)
    if not isinstance(detector_map, MeasurementMap) or not isinstance(
        observable_map, MeasurementMap
    ):
        raise TypeError(
            "detector_map and observable_map must be MeasurementMap instances"
        )
    if (
        detector_map.num_measurements != width
        or observable_map.num_measurements != width
    ):
        raise ValueError("both measurement maps must match the explicit record width")
    if not detector_map.num_rows:
        raise ValueError("explicit conversion requires at least one detector")
    if detector_map.num_rows + observable_map.num_rows > 65536:
        raise CapabilityError("explicit conversion exceeds the 65536 parity row limit")
    if detector_map.num_entries + observable_map.num_entries > 100000:
        raise CapabilityError(
            "explicit conversion exceeds the 100000 parity reference limit"
        )
    qubits, convert = _prepare_explicit_record_conversion(
        ir, detector_columns=detector_map.rows, observable_columns=observable_map.rows
    )
    handle = object.__new__(CompiledMeasurementsToDetectionEventsConverter)
    handle._measurement_qubits = qubits
    handle._convert_records = convert
    return handle
