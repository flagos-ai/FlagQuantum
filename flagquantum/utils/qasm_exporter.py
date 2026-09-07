"""
OpenQASM Exporter for FlagQuantum
Supports exporting to QASM 2.0 and 3.0 formats
"""

from typing import Any, List, Optional, Union

import torch

from ..core.ir import CircuitIR
from ..core.parameters import is_parameterized_value, parameter_names_in_value


class QASMExporter:
    """
    Export FlagQuantum's op_history to OpenQASM 2.0 or 3.0 format.

    Example:
        >>> exporter = QASMExporter()
        >>> # Export to QASM 3.0
        >>> exporter.export(qdev, "circuit_qasm3.qasm", version=3.0)
        >>> # Export to QASM 2.0 (Qiskit compatible)
        >>> exporter.export(qdev, "circuit_qasm2.qasm", version=2.0)
    """

    def __init__(self):
        # Common gate mapping (only standard OpenQASM gates)
        self.gate_map = {
            # Single-qubit gates
            "i": "id",
            "x": "x",
            "y": "y",
            "z": "z",
            "h": "h",
            "s": "s",
            "sdg": "sdg",  # Not standard in QASM 2.0
            "t": "t",
            "tdg": "tdg",  # Not standard in QASM 2.0
            "sx": "sx",
            "sxdg": "sxdg",  # Not standard
            # Two-qubit gates
            "cx": "cx",
            "cnot": "cx",
            "cy": "cy",
            "cz": "cz",
            "swap": "swap",
            "cphase": "cp",
            "ccx": "ccx",
            "cswap": "cswap",
            # Rotation gates (QASM standard)
            "rx": "rx",
            "ry": "ry",
            "rz": "rz",
            "u1": "u1",
            "u2": "u2",
            "u3": "u3",
            # Phase gates
            "p": "p",
            "phase": "p",
            # Measurement
            "measure_allZ": "measure",
        }

        # Gates that need decomposition (not standard in OpenQASM)
        self.decompose_gates = {
            "rxx": self._decompose_rxx,
            "ryy": self._decompose_ryy,
            "rzz": self._decompose_rzz,
            "crx": self._decompose_crx,
            "cry": self._decompose_cry,
            "crz": self._decompose_crz,
        }

        # Gates that need special handling for multi-qubit formatting
        self.multi_qubit_gates = {
            "cx": 2,
            "cnot": 2,
            "cy": 2,
            "cz": 2,
            "swap": 2,
            "cphase": 2,
            "cp": 2,
            "ccx": 3,
            "cswap": 3,
        }

        # Gates that require parameters
        self.parameterized_gates = {"rx", "ry", "rz", "u1", "u2", "u3", "p", "phase"}

    # ==================== Decomposition Methods ====================

    def _decompose_rxx(self, wires, params, version: float = 3.0) -> List[str]:
        """Decompose RXX gate into CX + RY + CX"""
        # RXX(θ) = (CX ⊗ I) * (RY(θ) ⊗ RY(θ)) * (CX ⊗ I)
        # Not the full decomposition, simplified
        theta = self._get_param_value(params) or 0.0
        if version == 3.0:
            return [
                f"  h q[{wires[0]}];",
                f"  cx q[{wires[0]}], q[{wires[1]}];",
                f"  ry({theta / 2}) q[{wires[1]}];",
                f"  cx q[{wires[0]}], q[{wires[1]}];",
                f"  ry({theta / 2}) q[{wires[0]}];",
                f"  h q[{wires[0]}];",
            ]
        else:
            return [
                f"h q[{wires[0]}];",
                f"cx q[{wires[0]}], q[{wires[1]}];",
                f"ry({theta / 2}) q[{wires[1]}];",
                f"cx q[{wires[0]}], q[{wires[1]}];",
                f"ry({theta / 2}) q[{wires[0]}];",
                f"h q[{wires[0]}];",
            ]

    def _decompose_ryy(self, wires, params, version: float = 3.0) -> List[str]:
        """Decompose RYY gate"""
        # Simplified: RYY(θ) uses RY rotations
        theta = self._get_param_value(params) or 0.0
        if version == 3.0:
            return [
                f"  ry({theta}) q[{wires[0]}];",
                f"  ry({theta}) q[{wires[1]}];",
            ]
        else:
            return [
                f"ry({theta}) q[{wires[0]}];",
                f"ry({theta}) q[{wires[1]}];",
            ]

    def _decompose_rzz(self, wires, params, version: float = 3.0) -> List[str]:
        """Decompose RZZ gate into CX + RZ + CX"""
        theta = self._get_param_value(params) or 0.0
        if version == 3.0:
            return [
                f"  cx q[{wires[0]}], q[{wires[1]}];",
                f"  rz({theta}) q[{wires[1]}];",
                f"  cx q[{wires[0]}], q[{wires[1]}];",
            ]
        else:
            return [
                f"cx q[{wires[0]}], q[{wires[1]}];",
                f"rz({theta}) q[{wires[1]}];",
                f"cx q[{wires[0]}], q[{wires[1]}];",
            ]

    def _decompose_crx(self, wires, params, version: float = 3.0) -> List[str]:
        """Decompose CRX gate into H + CZ + RY"""
        theta = self._get_param_value(params) or 0.0
        if version == 3.0:
            return [
                f"  h q[{wires[1]}];",
                f"  cx q[{wires[0]}], q[{wires[1]}];",
                f"  ry({theta}) q[{wires[1]}];",
                f"  cx q[{wires[0]}], q[{wires[1]}];",
                f"  h q[{wires[1]}];",
            ]
        else:
            return [
                f"h q[{wires[1]}];",
                f"cx q[{wires[0]}], q[{wires[1]}];",
                f"ry({theta}) q[{wires[1]}];",
                f"cx q[{wires[0]}], q[{wires[1]}];",
                f"h q[{wires[1]}];",
            ]

    def _decompose_cry(self, wires, params, version: float = 3.0) -> List[str]:
        """Decompose CRY gate"""
        theta = self._get_param_value(params) or 0.0
        if version == 3.0:
            return [
                f"  ry({theta / 2}) q[{wires[1]}];",
                f"  cx q[{wires[0]}], q[{wires[1]}];",
                f"  ry({-theta / 2}) q[{wires[1]}];",
                f"  cx q[{wires[0]}], q[{wires[1]}];",
            ]
        else:
            return [
                f"ry({theta / 2}) q[{wires[1]}];",
                f"cx q[{wires[0]}], q[{wires[1]}];",
                f"ry({-theta / 2}) q[{wires[1]}];",
                f"cx q[{wires[0]}], q[{wires[1]}];",
            ]

    def _decompose_crz(self, wires, params, version: float = 3.0) -> List[str]:
        """Decompose CRZ gate"""
        theta = self._get_param_value(params) or 0.0
        if version == 3.0:
            return [
                f"  rz({theta / 2}) q[{wires[1]}];",
                f"  cx q[{wires[0]}], q[{wires[1]}];",
                f"  rz({-theta / 2}) q[{wires[1]}];",
                f"  cx q[{wires[0]}], q[{wires[1]}];",
            ]
        else:
            return [
                f"rz({theta / 2}) q[{wires[1]}];",
                f"cx q[{wires[0]}], q[{wires[1]}];",
                f"rz({-theta / 2}) q[{wires[1]}];",
                f"cx q[{wires[0]}], q[{wires[1]}];",
            ]

    # ==================== Common Methods ====================

    def _get_param_values(self, params, n_params: int = 1) -> Optional[List[float]]:
        """Extract n parameter values."""
        if params is None:
            return None

        if isinstance(params, (int, float)):
            if n_params == 1:
                return [params]
            return None

        if isinstance(params, list):
            values = params[:n_params]
            if len(values) < n_params:
                values += [0.0] * (n_params - len(values))
            return values

        if isinstance(params, torch.Tensor):
            values = params.detach().cpu().numpy().flatten().tolist()[:n_params]
            if len(values) < n_params:
                values += [0.0] * (n_params - len(values))
            return values

        return None

    def _get_param_value(self, params) -> Optional[float]:
        """Extract a single parameter value."""
        if params is None:
            return None
        if isinstance(params, (int, float)):
            return params
        if isinstance(params, list) and len(params) > 0:
            return params[0]
        if isinstance(params, torch.Tensor) and params.numel() > 0:
            return params.flatten()[0].item()
        return None

    def _format_params(self, params) -> str:
        """Format parameters for QASM output."""
        if params is None:
            return ""
        if is_parameterized_value(params):
            names = ", ".join(parameter_names_in_value(params))
            raise ValueError(
                f"Unbound circuit parameter(s): {names}. Call bind_parameters first."
            )

        if isinstance(params, (int, float)):
            values = [params]
        elif isinstance(params, list):
            values = params
        elif isinstance(params, torch.Tensor):
            values = params.detach().cpu().numpy().flatten().tolist()
        else:
            return ""

        if len(values) == 1:
            return f"({values[0]})"
        return f"({', '.join(str(v) for v in values)})"

    def _normalize_program(self, program: Any) -> tuple[int, list[dict[str, Any]]]:
        """Accept native circuits, native IR, or recorded operation data."""

        if isinstance(program, CircuitIR):
            return program.n_wires, [
                {
                    "name_or_mat": instruction.name,
                    "wires": list(instruction.wires),
                    "params": instruction.params,
                }
                for instruction in program.instructions
            ]

        if hasattr(program, "to_ir"):
            return self._normalize_program(program.to_ir())

        if hasattr(program, "op_history"):
            return program.n_wires, list(program.op_history)

        raise TypeError(
            "QASM export expects a FlagQuantum Circuit, CircuitIR, or device with op_history."
        )

    def _params_for_gate(self, gate_name: str, params: Any) -> Any:
        """Convert native named parameter dictionaries into positional QASM params."""

        if not isinstance(params, dict):
            return params

        order = {
            "rx": ("theta",),
            "ry": ("theta",),
            "rz": ("theta",),
            "p": ("theta",),
            "phase": ("theta",),
            "u1": ("theta",),
            "u2": ("phi", "lbd"),
            "u3": ("theta", "phi", "lbd"),
            "crx": ("theta",),
            "cry": ("theta",),
            "crz": ("theta",),
            "cphase": ("theta",),
            "rxx": ("theta",),
            "ryy": ("theta",),
            "rzz": ("theta",),
        }.get(gate_name)
        if order is None:
            return None

        values = [params.get(name) for name in order]
        if any(value is None for value in values):
            return None
        return values[0] if len(values) == 1 else values

    def _format_wires(self, wires: Union[int, List[int]], version: float = 3.0) -> str:
        """Format wires for QASM output."""
        if isinstance(wires, int):
            wires = [wires]

        if version == 3.0:
            return ", ".join(f"q[{w}]" for w in wires)
        else:
            if len(wires) == 1:
                return f"q[{wires[0]}]"
            return ", ".join(f"q[{w}]" for w in wires)

    # ==================== QASM 3.0 Format ====================

    def _format_gate_call_qasm3(self, gate_name: str, wires, params=None) -> List[str]:
        """Format a gate call for QASM 3.0. Returns list of lines."""
        # Check if gate needs decomposition
        if gate_name in self.decompose_gates:
            return self.decompose_gates[gate_name](wires, params, version=3.0)

        if gate_name == "sxdg":
            # 使用 power 修饰符构建三倍的 sx 门
            wires_str = self._format_wires(wires, version=3.0)
            return [f"  pow(-1) @ sx {wires_str};"]

        qasm_name = self.gate_map.get(gate_name, gate_name)
        wires_str = self._format_wires(wires, version=3.0)
        params_str = self._format_params(params) if params else ""

        if params_str:
            return [f"  {qasm_name}{params_str} {wires_str};"]
        return [f"  {qasm_name} {wires_str};"]

    def _handle_measure_qasm3(self, qdev, has_measure: bool) -> List[str]:
        """Handle measurement for QASM 3.0."""
        lines = []
        if has_measure:
            lines.append(f"  bit[{qdev.n_wires}] c;")
            lines.append("  c = measure q;")
        else:
            lines.append("")
            lines.append("// Final measurements")
            for i in range(qdev.n_wires):
                lines.append(f"  bit b{i};")
                lines.append(f"  b{i} = measure q[{i}];")
        return lines

    def export_qasm3(self, qdev, filename: Optional[str] = None) -> Optional[str]:
        """Export to OpenQASM 3.0 format."""
        n_wires, op_history = self._normalize_program(qdev)
        lines = [
            "OPENQASM 3.0;",
            'include "stdgates.inc";',
            "// Generated by FlagQuantum",
            f"// Number of qubits: {n_wires}",
            f"// Total operations: {len(op_history)}",
            "",
            f"qubit[{n_wires}] q;",
            "",
            "// Initialize to |0...0>",
            "",
        ]

        has_measure = False

        for op in op_history:
            gate_name = op.get("name_or_mat", "unknown")
            wires = op.get("wires", [])
            params = self._params_for_gate(gate_name, op.get("params", None))

            if isinstance(wires, int):
                wires = [wires]

            if gate_name == "unknown":
                continue

            if gate_name == "measure_allZ":
                has_measure = True
                continue

            if gate_name == "cnot":
                gate_name = "cx"

            gate_lines = self._format_gate_call_qasm3(gate_name, wires, params)
            lines.extend(gate_lines)

        if has_measure:
            lines.append(f"  bit[{n_wires}] c;")
            lines.append("  c = measure q;")
        else:
            lines.append("")
            lines.append("// Final measurements")
            for i in range(n_wires):
                lines.append(f"  bit b{i};")
                lines.append(f"  b{i} = measure q[{i}];")

        qasm_str = "\n".join(lines)

        if filename:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(qasm_str)
            print(f"✅ QASM 3.0 exported to {filename}")

        return qasm_str

    # ==================== QASM 2.0 Format ====================

    def _format_gate_call_qasm2(self, gate_name: str, wires, params=None) -> List[str]:
        """Format a gate call for QASM 2.0. Returns list of lines."""
        # Check if gate needs decomposition
        if gate_name in self.decompose_gates:
            return self.decompose_gates[gate_name](wires, params, version=2.0)

        # Ensure wires list has correct length for multi-qubit gates
        expected_wires = self.multi_qubit_gates.get(gate_name, 1)
        if len(wires) < expected_wires:
            wires = wires + [wires[-1]] * (expected_wires - len(wires))

        wires_str = self._format_wires(wires, version=2.0)

        # CPHASE / CP gate - requires parameter
        if gate_name in ["cphase", "cp"]:
            p = self._get_param_value(params)
            p_str = f"({p})" if p is not None else "(0)"  # 默认参数 0
            return [f"cp{p_str} {wires_str};"]

        # U3 gate - 3 parameters
        if gate_name == "u3":
            p = self._get_param_values(params, 3)
            if p:
                return [f"u3({p[0]}, {p[1]}, {p[2]}) {wires_str};"]
            return [f"u3(0, 0, 0) {wires_str};"]

        # U2 gate - 2 parameters
        if gate_name == "u2":
            p = self._get_param_values(params, 2)
            if p:
                return [f"u2({p[0]}, {p[1]}) {wires_str};"]
            return [f"u2(0, 0) {wires_str};"]

        # U1 gate - 1 parameter
        if gate_name == "u1":
            p = self._get_param_value(params)
            p_str = f"({p})" if p is not None else ""
            return [f"u1{p_str} {wires_str};"]

        # RX, RY, RZ, P, PHASE - 1 parameter
        if gate_name in ["rx", "ry", "rz", "p", "phase"]:
            gate = "p" if gate_name == "phase" else gate_name
            p = self._get_param_value(params)
            p_str = f"({p})" if p is not None else ""
            return [f"{gate}{p_str} {wires_str};"]

        # Standard gates
        qasm_name = self.gate_map.get(gate_name, gate_name)
        return [f"{qasm_name} {wires_str};"]

    def export_qasm2(self, qdev, filename: Optional[str] = None) -> Optional[str]:
        """Export to QASM 2.0 format (Qiskit compatible)."""
        n_wires, op_history = self._normalize_program(qdev)
        lines = [
            "OPENQASM 2.0;",
            'include "qelib1.inc";',
            "// Generated by FlagQuantum",
            f"qreg q[{n_wires}];",
            f"creg c[{n_wires}];",
            "",
        ]

        for op in op_history:
            gate_name = op.get("name_or_mat", "unknown")
            wires = op.get("wires", [])
            params = self._params_for_gate(gate_name, op.get("params", None))

            if isinstance(wires, int):
                wires = [wires]

            if gate_name == "unknown":
                continue

            if gate_name == "measure_allZ":
                continue

            gate_lines = self._format_gate_call_qasm2(gate_name, wires, params)
            lines.extend(gate_lines)

        # Add measurements
        for wire in range(n_wires):
            lines.append(f"measure q[{wire}] -> c[{wire}];")

        qasm_str = "\n".join(lines)

        if filename:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(qasm_str)
            print(f"✅ QASM 2.0 exported to {filename}")

        return qasm_str

    # ==================== Unified Export Interface ====================

    def export(
        self, qdev, filename: Optional[str] = None, version: float = 3.0
    ) -> Optional[str]:
        """
        Unified export interface.

        Args:
            qdev: FlagQuantum circuit, IR, or recorded operation data
            filename: Output filename. If None, returns string instead
            version: QASM version, either 2.0 or 3.0

        Returns:
            QASM string if filename is None, otherwise None
        """
        if version == 2.0:
            return self.export_qasm2(qdev, filename)
        elif version == 3.0:
            return self.export_qasm3(qdev, filename)
        else:
            raise ValueError(f"Unsupported QASM version: {version}. Use 2.0 or 3.0")


# ==================== Convenience Functions ====================


def export_to_qasm(qdev, filename: str, version: float = 3.0) -> None:
    """Export circuit to a QASM file."""
    exporter = QASMExporter()
    exporter.export(qdev, filename, version)


def export_to_qasm_str(qdev, version: float = 3.0) -> str:
    """Export circuit to a QASM string."""
    exporter = QASMExporter()
    return exporter.export(qdev, version=version)


# ==================== Module Exports ====================

__all__ = [
    "QASMExporter",
    "export_to_qasm",
    "export_to_qasm_str",
]
