"""Private physical-program contract for Quafu SDK submission."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

_QREG_DECLARATION = re.compile(r"\bqreg\s+q\s*\[\s*(\d+)\s*\]\s*;")
_CREG_DECLARATION = re.compile(r"\bcreg\s+c\s*\[\s*(\d+)\s*\]\s*;")
_QUBIT_REFERENCE = re.compile(r"\bq\s*\[\s*(\d+)\s*\]")
_MEASUREMENT = re.compile(
    r"^\s*measure\s+q\s*\[\s*(\d+)\s*\]\s*->\s*c\s*\[\s*(\d+)\s*\]\s*;\s*$"
)
_GATE_STATEMENT = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*(?:\([^;]*\))?\s+(.+)\s*;\s*$"
)
_CX_STATEMENT = re.compile(
    r"^\s*cx\s+q\s*\[\s*(\d+)\s*\]\s*,\s*q\s*\[\s*(\d+)\s*\]\s*;\s*$",
    re.IGNORECASE,
)


def backend_snapshot_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def prepare_physical_qasm(
    logical_qasm: str,
    *,
    target_qubits: Sequence[int],
    backend_metadata: Mapping[str, Any],
) -> tuple[str, tuple[dict[str, int], ...]]:
    """Encode a QSteed mapping in QASM and validate one SDK snapshot."""

    capacity = backend_metadata.get("n_qubits")
    if type(capacity) is not int or capacity <= 0:
        raise RuntimeError("Quafu SDK backend metadata has no positive n_qubits")
    if any(qubit >= capacity for qubit in target_qubits):
        raise ValueError("QSteed physical qubit exceeds Quafu backend capacity")

    gates = backend_metadata.get("basis_gates")
    if not isinstance(gates, Sequence) or isinstance(gates, (str, bytes)):
        raise RuntimeError("Quafu SDK backend metadata has invalid basis_gates")
    basis_gates = {str(gate).lower() for gate in gates}
    if not basis_gates:
        raise RuntimeError("Quafu SDK backend metadata has no basis_gates")

    raw_coupling = backend_metadata.get("coupling_map")
    if not isinstance(raw_coupling, Sequence) or isinstance(raw_coupling, (str, bytes)):
        raise RuntimeError("Quafu SDK backend metadata has invalid coupling_map")
    directed_edges: set[tuple[int, int]] = set()
    for edge in raw_coupling:
        if (
            not isinstance(edge, Sequence)
            or isinstance(edge, (str, bytes))
            or len(edge) != 2
            or any(type(qubit) is not int for qubit in edge)
        ):
            raise RuntimeError("Quafu SDK backend metadata has invalid coupling_map")
        left, right = int(edge[0]), int(edge[1])
        if left < 0 or right < 0 or left >= capacity or right >= capacity:
            raise RuntimeError("Quafu SDK coupling_map exceeds backend capacity")
        directed_edges.add((left, right))

    declaration = _QREG_DECLARATION.search(logical_qasm)
    if declaration is None or int(declaration.group(1)) != len(target_qubits):
        raise ValueError("QSteed QASM must declare one logical qubit per mapping entry")
    classical = _CREG_DECLARATION.search(logical_qasm)
    if classical is None or int(classical.group(1)) != len(target_qubits):
        raise ValueError("QSteed QASM must measure every logical qubit exactly once")

    lines = []
    for line in logical_qasm.splitlines():
        if _QREG_DECLARATION.search(line) is not None:
            lines.append(_QREG_DECLARATION.sub(f"qreg q[{capacity}];", line, count=1))
        else:
            mapped = _QUBIT_REFERENCE.sub(
                lambda match: f"q[{target_qubits[int(match.group(1))]}]",
                line,
            )
            cx = _CX_STATEMENT.fullmatch(mapped)
            if cx is not None and "cx" not in basis_gates:
                if not {"h", "cz"}.issubset(basis_gates):
                    raise ValueError(
                        "QSteed operation 'cx' cannot be lowered to the Quafu basis"
                    )
                control, target = map(int, cx.groups())
                lines.extend(
                    (
                        f"h q[{target}];",
                        f"cz q[{control}], q[{target}];",
                        f"h q[{target}];",
                    )
                )
            else:
                lines.append(mapped)
    program = "\n".join(lines)

    measurements: list[dict[str, int]] = []
    for line in program.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("OPENQASM", "include", "qreg", "creg")):
            continue
        measurement = _MEASUREMENT.fullmatch(line)
        if measurement is not None:
            physical, cbit = map(int, measurement.groups())
            measurements.append(
                {
                    "cbit": cbit,
                    "logical_qubit": cbit,
                    "physical_qubit": physical,
                }
            )
            continue
        gate = _GATE_STATEMENT.fullmatch(line)
        if gate is None:
            raise ValueError("QSteed QASM contains an unsupported statement")
        operation = gate.group(1).lower()
        if operation not in basis_gates:
            raise ValueError(
                f"QSteed operation {operation!r} is not native to the Quafu backend"
            )
        operands = tuple(
            int(index) for index in _QUBIT_REFERENCE.findall(gate.group(2))
        )
        if len(operands) not in {1, 2}:
            raise ValueError("QSteed QASM gates must act on one or two physical qubits")
        if len(operands) == 2 and tuple(operands) not in directed_edges:
            raise ValueError(
                f"QSteed physical edge {operands[0]}->{operands[1]} is unavailable"
            )

    expected = tuple(
        {
            "cbit": logical,
            "logical_qubit": logical,
            "physical_qubit": physical,
        }
        for logical, physical in enumerate(target_qubits)
    )
    if tuple(sorted(measurements, key=lambda item: item["cbit"])) != expected:
        raise ValueError("QSteed QASM measurement mapping is incomplete or ambiguous")
    return program, expected


def validate_physical_result(
    *,
    expected_measurements: Any,
    expected_calibration_id: Any,
    response: Mapping[str, Any],
) -> None:
    actual = response.get("measurements")
    if not isinstance(expected_measurements, Sequence) or isinstance(
        expected_measurements, (str, bytes)
    ):
        raise RuntimeError("Quafu physical receipt has no measurement mapping")
    if not isinstance(actual, Sequence) or isinstance(actual, (str, bytes)):
        raise RuntimeError("Quafu SDK physical result has no measurement mapping")

    def normalized(rows: Sequence[Any]) -> tuple[tuple[int, int], ...]:
        pairs = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise RuntimeError(
                    "Quafu SDK physical result has invalid measurement mapping"
                )
            cbit = row.get("cbit")
            physical = row.get("physical_qubit")
            if type(cbit) is not int or type(physical) is not int:
                raise RuntimeError(
                    "Quafu SDK physical result has invalid measurement mapping"
                )
            pairs.append((cbit, physical))
        return tuple(sorted(pairs))

    if normalized(actual) != normalized(expected_measurements):
        raise RuntimeError(
            "Quafu SDK physical result does not preserve the logical measurement mapping"
        )
    if response.get("n_clbits") != len(expected_measurements):
        raise RuntimeError("Quafu SDK physical result has an unexpected bit width")
    if response.get("calibration_id") != expected_calibration_id:
        raise RuntimeError(
            "Quafu SDK physical result calibration identity does not match submission"
        )
    compiler = response.get("compiler")
    if not isinstance(compiler, Mapping) or compiler.get("name") != "none":
        raise RuntimeError("Quafu SDK physical result does not confirm compiler='none'")


__all__: tuple[str, ...] = ()
