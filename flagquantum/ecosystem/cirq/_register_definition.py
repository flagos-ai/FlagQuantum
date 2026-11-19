"""Lower explicit native definitions for a shared SDK measurement history."""

from collections.abc import Hashable, Mapping
from typing import Any

from ...compiler.registers._local_register_binding import _bind_local_register_program
from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.signatures import Signature
from ._register_gate import RegisterCirqGate
from .conversion import _cirq


def _expand_native_definitions(
    operations: tuple[Any, ...], declared: Mapping[Hashable, FunctionRecord]
) -> tuple[Any, ...]:
    """Preserve native definition precedence and isolate local measurement keys."""
    cirq = _cirq()
    definitions: dict[str, FunctionRecord] = {}
    selected: list[FunctionRecord | None] = []
    for operation in operations:
        gate = operation.without_classical_controls().gate
        definition = declared.get(gate) if isinstance(gate, Hashable) else None
        if definition is None:
            definition = getattr(gate, "flagquantum_register_definition", None)
        if definition is not None and (
            not isinstance(definition, FunctionRecord) or definition.registers is None
        ):
            raise TypeError(
                "Cirq gate definition requires a typed native FunctionRecord"
            )
        selected.append(definition)
    for definition in (*declared.values(), *(f for f in selected if f is not None)):
        previous = definitions.setdefault(definition.name, definition)
        if previous != definition:
            raise ValueError(f"Conflicting Cirq gate definition {definition.name!r}")
    keys = {
        key for operation in operations for key in cirq.measurement_key_names(operation)
    }
    keys.update(
        str(key) for operation in operations for key in cirq.control_keys(operation)
    )
    result = []
    for index, (operation, definition) in enumerate(
        zip(operations, selected, strict=True)
    ):
        if definition is None:
            result.append(operation)
            continue
        prefix = f"fq_native_{index}"
        while any(key.startswith(prefix + "_") for key in keys):
            prefix += "_"
        bound = _bind_local_register_program(
            ProgramRecord(tuple(definitions.values())), definition.name
        )
        assert definition.registers is not None
        gate = RegisterCirqGate(
            bound, Signature(definition.registers), feedback_key_prefix=prefix
        )
        result.append(
            gate.on(*operation.qubits).with_classical_controls(
                *operation.classical_controls
            )
        )
    return tuple(result)
