"""Internal normalization for statically executable terminal measurements."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from ...core.classical_control import (
    is_classical_control_instruction,
    validate_classical_control,
)
from ...core.ir import CircuitIR, ensure_circuit_ir
from ...errors import ExecutionError
from ...observables import OutputRequest


@dataclass(frozen=True, slots=True)
class TerminalMeasurementNormalization:
    """Immutable proof that one dynamic IR is equivalent to static counts."""

    source_ir: CircuitIR
    unitary_ir: CircuitIR
    measurement_mapping: tuple[tuple[int, int], ...]
    classical_width: int
    output_name: str | None
    normalization_id: str = field(init=False)

    def __post_init__(self) -> None:
        payload = {
            "source_ir": self.source_ir.content_hash,
            "unitary_ir": self.unitary_ir.content_hash,
            "measurement_mapping": [list(item) for item in self.measurement_mapping],
            "classical_width": self.classical_width,
            "output": {
                "kind": "counts",
                "qubits": list(range(self.classical_width)),
                "name": self.output_name,
            },
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        object.__setattr__(
            self,
            "normalization_id",
            hashlib.sha256(encoded).hexdigest(),
        )

    def summary(self) -> dict[str, Any]:
        """Return a JSON-safe inspectable description of the normalization."""

        return {
            "normalization_id": self.normalization_id,
            "source_ir_sha256": self.source_ir.content_hash,
            "unitary_ir_sha256": self.unitary_ir.content_hash,
            "measurement_mapping": [
                {"qubit": qubit, "classical_bit": classical_bit}
                for qubit, classical_bit in self.measurement_mapping
            ],
            "classical_width": self.classical_width,
            "output_kind": "counts",
            "output_name": self.output_name,
        }

    def project_counts(self, counts: Mapping[str, int]) -> dict[str, int]:
        """Project qubit-ordered counts into declared classical-bit order."""

        projected: dict[str, int] = {}
        for raw_outcome, raw_count in counts.items():
            if (
                not isinstance(raw_outcome, str)
                or len(raw_outcome) != self.classical_width
                or set(raw_outcome) - {"0", "1"}
            ):
                raise ExecutionError(
                    "terminal measurement result has an invalid provider bitstring"
                )
            if (
                isinstance(raw_count, bool)
                or not isinstance(raw_count, int)
                or raw_count < 0
            ):
                raise ExecutionError(
                    "terminal measurement result has an invalid provider count"
                )
            classical = [""] * self.classical_width
            for qubit, classical_bit in self.measurement_mapping:
                classical[classical_bit] = raw_outcome[qubit]
            outcome = "".join(classical)
            projected[outcome] = projected.get(outcome, 0) + raw_count
        return projected


def _terminal_mapping(ir: CircuitIR) -> tuple[tuple[int, int], ...] | None:
    measure_indices = tuple(
        index
        for index, instruction in enumerate(ir.instructions)
        if instruction.name == "measure"
    )
    if not measure_indices:
        return None
    first_measurement = measure_indices[0]
    prefix = ir.instructions[:first_measurement]
    suffix = ir.instructions[first_measurement:]
    if any(is_classical_control_instruction(item) for item in prefix):
        return None
    if any(item.name != "measure" for item in suffix):
        return None
    if len(suffix) != ir.n_wires:
        return None

    validate_classical_control(ir.instructions, n_qubits=ir.n_wires)
    pairs: list[tuple[int, int]] = []
    for instruction in suffix:
        if set(instruction.metadata) != {"is_dynamic", "classical_bit"}:
            return None
        if instruction.metadata["is_dynamic"] is not True:
            return None
        classical_bit = instruction.metadata["classical_bit"]
        if type(classical_bit) is not int:
            return None
        pairs.append((instruction.wires[0], classical_bit))

    mapping = tuple(sorted(pairs))
    expected = set(range(ir.n_wires))
    if {qubit for qubit, _ in mapping} != expected:
        return None
    if {classical_bit for _, classical_bit in mapping} != expected:
        return None
    return mapping


def _counts_output(
    outputs: OutputRequest | Sequence[OutputRequest] | None,
    *,
    n_qubits: int,
) -> OutputRequest | None:
    if outputs is None:
        return None
    requested: tuple[OutputRequest, ...]
    if isinstance(outputs, OutputRequest):
        requested = (outputs,)
    elif isinstance(outputs, Sequence) and not isinstance(outputs, str | bytes):
        requested = tuple(outputs)
    else:
        raise TypeError("outputs must contain OutputRequest values")
    if len(requested) != 1 or not isinstance(requested[0], OutputRequest):
        raise ValueError(
            "terminal measurement normalization requires one full-register "
            "fq.counts() output"
        )
    output = requested[0]
    if (
        output.kind != "counts"
        or output.observable is not None
        or output.qubits not in {(), tuple(range(n_qubits))}
    ):
        raise ValueError(
            "terminal measurement normalization requires one full-register "
            "fq.counts() output"
        )
    return output


def normalize_terminal_measurements(
    program: Any,
    *,
    outputs: OutputRequest | Sequence[OutputRequest] | None,
) -> TerminalMeasurementNormalization | None:
    """Return the V1 static normalization, or ``None`` for other semantics."""

    source_ir = ensure_circuit_ir(program)
    mapping = _terminal_mapping(source_ir)
    if mapping is None:
        return None
    output = _counts_output(outputs, n_qubits=source_ir.n_wires)
    first_measurement = next(
        index
        for index, instruction in enumerate(source_ir.instructions)
        if instruction.name == "measure"
    )
    unitary_ir = replace(
        source_ir,
        instructions=source_ir.instructions[:first_measurement],
    )
    return TerminalMeasurementNormalization(
        source_ir=source_ir,
        unitary_ir=unitary_ir,
        measurement_mapping=mapping,
        classical_width=source_ir.n_wires,
        output_name=None if output is None else output.name,
    )


__all__: tuple[str, ...] = ()
