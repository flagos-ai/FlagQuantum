"""Standard OpenQASM 3 dynamic dialect implementation."""

from typing import Any

from .._conditions import classical_width as _classical_width
from .._conditions import instruction_conditions as _instruction_conditions
from ..circuit import DynamicCircuit


def export_dynamic_qasm3(circuit: DynamicCircuit) -> str:
    """Export the supported dynamic subset to OpenQASM 3."""

    if not isinstance(circuit, DynamicCircuit):
        raise TypeError("dynamic QASM export requires DynamicCircuit")
    classical_width = _classical_width(circuit)
    lines = [
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
        f"qubit[{circuit.n_wires}] q;",
        *([f"bit[{classical_width}] c;"] if classical_width else []),
    ]
    for instruction in circuit._instructions:
        wire_text = ", ".join(f"q[{wire}]" for wire in instruction.wires)
        if instruction.name == "measure":
            bit = int(instruction.metadata["classical_bit"])
            statement = f"c[{bit}] = measure {wire_text};"
        elif instruction.name == "reset":
            statement = f"reset {wire_text};"
        else:
            params = ""
            if instruction.params:
                params = "(" + ", ".join(
                    str(value) for value in instruction.params.values()
                ) + ")"
            statement = f"{instruction.name}{params} {wire_text};"
        conditions = _instruction_conditions(instruction)
        if conditions:
            expression = " && ".join(
                f"c[{bit}] == {'true' if value else 'false'}"
                for bit, value in conditions
            )
            statement = f"if ({expression}) {{ {statement} }}"
        lines.append(statement)
    return "\n".join(lines) + "\n"


def export_dynamic_qasm3_for_backend(
    circuit: DynamicCircuit, backend: Any
) -> str:
    """Export using the backend-declared dynamic OpenQASM dialect."""

    dialect = getattr(backend, "dynamic_dialect", None) or "openqasm3"
    if dialect == "openqasm3":
        return export_dynamic_qasm3(circuit)
    if dialect == "braket_iqm":
        from .braket_iqm import (
            export_braket_iqm_dynamic_qasm3,
            iqm_qubit_groups,
        )

        return export_braket_iqm_dynamic_qasm3(
            circuit,
            qubit_groups=iqm_qubit_groups(backend),
        )
    raise ValueError(f"unsupported_dynamic_dialect:{dialect}")

__all__ = ("export_dynamic_qasm3", "export_dynamic_qasm3_for_backend")
