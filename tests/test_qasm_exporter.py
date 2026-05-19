"""
Pytest unit tests for OpenQASM Exporter
"""

import os
import re
import tempfile

import pytest

from flagquantum.utils.qasm_exporter import (
    QASMExporter,
    export_to_qasm,
    export_to_qasm_str,
)

# ============================================================================
# Mock Classes
# ============================================================================


class MockQDevice:
    """Mock quantum device for testing"""

    def __init__(self, n_wires=3, op_history=None):
        self.n_wires = n_wires
        self.op_history = op_history or []
        self.bsz = 1
        self.device = "cpu"


# ============================================================================
# Fixtures (必须放在类定义之前)
# ============================================================================


@pytest.fixture
def exporter():
    """Create QASMExporter instance"""
    return QASMExporter()


@pytest.fixture
def simple_circuit():
    """Simple circuit with basic gates"""
    return MockQDevice(
        n_wires=3,
        op_history=[
            {"name_or_mat": "h", "wires": [0], "params": None},
            {"name_or_mat": "cx", "wires": [0, 1], "params": None},
            {"name_or_mat": "measure_allZ", "wires": [], "params": None},
        ],
    )


@pytest.fixture
def parameterized_circuit():
    """Circuit with parameterized gates (using Python floats for precision)"""
    return MockQDevice(
        n_wires=3,
        op_history=[
            {"name_or_mat": "rx", "wires": [0], "params": 0.5},
            {"name_or_mat": "ry", "wires": [1], "params": 0.3},
            {"name_or_mat": "rz", "wires": [2], "params": 0.7},
            {"name_or_mat": "u3", "wires": [0], "params": [0.1, 0.2, 0.3]},
            {"name_or_mat": "measure_allZ", "wires": [], "params": None},
        ],
    )


@pytest.fixture
def ising_circuit():
    """Circuit with Ising gates"""
    return MockQDevice(
        n_wires=3,
        op_history=[
            {"name_or_mat": "rxx", "wires": [0, 1], "params": 0.5},
            {"name_or_mat": "ryy", "wires": [1, 2], "params": 0.3},
            {"name_or_mat": "rzz", "wires": [0, 2], "params": 0.7},
            {"name_or_mat": "measure_allZ", "wires": [], "params": None},
        ],
    )


@pytest.fixture
def controlled_circuit():
    """Circuit with controlled gates"""
    return MockQDevice(
        n_wires=3,
        op_history=[
            {"name_or_mat": "crx", "wires": [0, 1], "params": 0.5},
            {"name_or_mat": "cry", "wires": [1, 2], "params": 0.3},
            {"name_or_mat": "crz", "wires": [0, 2], "params": 0.7},
            {"name_or_mat": "cphase", "wires": [0, 1], "params": 0.5},
            {"name_or_mat": "measure_allZ", "wires": [], "params": None},
        ],
    )


@pytest.fixture
def multi_qubit_circuit():
    """Circuit with multi-qubit gates"""
    return MockQDevice(
        n_wires=3,
        op_history=[
            {"name_or_mat": "ccx", "wires": [0, 1, 2], "params": None},
            {"name_or_mat": "cswap", "wires": [0, 1, 2], "params": None},
            {"name_or_mat": "swap", "wires": [0, 1], "params": None},
            {"name_or_mat": "measure_allZ", "wires": [], "params": None},
        ],
    )


@pytest.fixture
def single_qubit_circuit():
    """Circuit with all single-qubit gates"""
    gates = ["i", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "sxdg"]
    return MockQDevice(
        n_wires=1,
        op_history=[
            {"name_or_mat": gate, "wires": [0], "params": None} for gate in gates
        ]
        + [{"name_or_mat": "measure_allZ", "wires": [], "params": None}],
    )


# ============================================================================
# Helper Functions
# ============================================================================


def parse_qasm_gates(qasm_str: str) -> list:
    """
    Parse QASM string and extract gate operations.
    Returns list of (gate_name, wires, params)
    """
    gates = []
    lines = qasm_str.strip().split("\n")

    for line in lines:
        line = line.strip()
        # Skip comments and empty lines
        if line.startswith("//") or not line:
            continue
        # Skip header and declarations
        if line.startswith("OPENQASM") or line.startswith("include"):
            continue
        if line.startswith("qubit[") or line.startswith("qreg"):
            continue
        if line.startswith("bit[") or line.startswith("creg"):
            continue
        if line == "c = measure q;":
            continue

        # Parse gate: gate_name(params) wires;
        match = re.match(r"(\w+)(?:\(([^)]+)\))?\s+(.+?);", line)
        if match:
            gate_name = match.group(1)
            params_str = match.group(2)
            wires_str = match.group(3)

            # Parse parameters
            params = []
            if params_str:
                for p in params_str.split(","):
                    try:
                        params.append(float(p.strip()))
                    except ValueError:
                        params.append(p.strip())

            # Parse wires
            wires = []
            for w in wires_str.split(","):
                w_clean = w.strip().replace("q[", "").replace("]", "")
                try:
                    wires.append(int(w_clean))
                except ValueError:
                    wires.append(w_clean)

            gates.append((gate_name, wires, params))

    return gates


def parse_qasm2_measurements(qasm_str: str) -> list:
    """Parse QASM 2.0 measurements"""
    measurements = []
    for line in qasm_str.split("\n"):
        if "measure" in line and "->" in line:
            match = re.search(r"measure q\[(\d+)\] -> c\[(\d+)\]", line)
            if match:
                measurements.append((int(match.group(1)), int(match.group(2))))
    return measurements


# ============================================================================
# Export Format Tests
# ============================================================================


class TestExportFormat:
    """Test basic export format"""

    def test_export_qasm3_format(self, exporter, simple_circuit):
        qasm_str = exporter.export_qasm3(simple_circuit)

        # Verify header
        assert "OPENQASM 3.0;" in qasm_str
        assert 'include "stdgates.inc";' in qasm_str
        assert "qubit[3] q;" in qasm_str

        # Verify gates
        assert "h q[0];" in qasm_str
        assert "cx q[0], q[1];" in qasm_str

        # Verify measurement
        assert "c = measure q;" in qasm_str

    def test_export_qasm2_format(self, exporter, simple_circuit):
        qasm_str = exporter.export_qasm2(simple_circuit)

        # Verify header
        assert "OPENQASM 2.0;" in qasm_str
        assert 'include "qelib1.inc";' in qasm_str
        assert "qreg q[3];" in qasm_str
        assert "creg c[3];" in qasm_str

        # Verify gates
        assert "h q[0];" in qasm_str
        assert "cx q[0], q[1];" in qasm_str

        # Verify measurements
        measurements = parse_qasm2_measurements(qasm_str)
        assert len(measurements) == 3

    def test_unified_export_qasm3(self, exporter, simple_circuit):
        qasm_str = exporter.export(simple_circuit, version=3.0)
        assert "OPENQASM 3.0;" in qasm_str

    def test_unified_export_qasm2(self, exporter, simple_circuit):
        qasm_str = exporter.export(simple_circuit, version=2.0)
        assert "OPENQASM 2.0;" in qasm_str

    def test_unified_export_invalid_version(self, exporter, simple_circuit):
        with pytest.raises(ValueError, match="Unsupported QASM version"):
            exporter.export(simple_circuit, version=1.0)


# ============================================================================
# Parameterized Gates Tests
# ============================================================================


class TestParameterizedGates:
    """Test parameterized gate export"""

    def test_parameterized_gates_qasm3(self, exporter, parameterized_circuit):
        qasm_str = exporter.export_qasm3(parameterized_circuit)
        gates = parse_qasm_gates(qasm_str)

        gate_names = [g[0] for g in gates]
        assert "rx" in gate_names
        assert "ry" in gate_names
        assert "rz" in gate_names
        assert "u3" in gate_names

    def test_parameterized_gates_qasm2(self, exporter, parameterized_circuit):
        qasm_str = exporter.export_qasm2(parameterized_circuit)

        assert "rx" in qasm_str
        assert "ry" in qasm_str
        assert "rz" in qasm_str
        assert "u3" in qasm_str


# ============================================================================
# Ising Gates Decomposition Tests
# ============================================================================


class TestIsingGates:
    """Test Ising gate decomposition"""

    def test_rxx_decomposition_qasm3(self, exporter, ising_circuit):
        qasm_str = exporter.export_qasm3(ising_circuit)

        assert "rxx" not in qasm_str
        assert "h q[0];" in qasm_str
        assert "cx q[0], q[1];" in qasm_str

    def test_ryy_decomposition_qasm3(self, exporter, ising_circuit):
        qasm_str = exporter.export_qasm3(ising_circuit)

        assert "ryy" not in qasm_str
        assert "ry" in qasm_str

    def test_rzz_decomposition_qasm3(self, exporter, ising_circuit):
        qasm_str = exporter.export_qasm3(ising_circuit)

        assert "rzz" not in qasm_str
        assert "cx q[0], q[2];" in qasm_str
        assert "rz" in qasm_str
        assert "cx q[0], q[2];" in qasm_str


# ============================================================================
# Controlled Gates Tests
# ============================================================================


class TestControlledGates:
    """Test controlled gate decomposition"""

    def test_controlled_rotation_decomposition_qasm3(
        self, exporter, controlled_circuit
    ):
        qasm_str = exporter.export_qasm3(controlled_circuit)

        assert "crx" not in qasm_str
        assert "cx q[0], q[1];" in qasm_str
        assert "ry" in qasm_str


# ============================================================================
# Multi-Qubit Gates Tests
# ============================================================================


class TestMultiQubitGates:
    """Test multi-qubit gate export"""

    def test_multi_qubit_gates_qasm3(self, exporter, multi_qubit_circuit):
        qasm_str = exporter.export_qasm3(multi_qubit_circuit)

        assert "ccx q[0], q[1], q[2];" in qasm_str
        assert "cswap q[0], q[1], q[2];" in qasm_str
        assert "swap q[0], q[1];" in qasm_str

    def test_multi_qubit_gates_qasm2(self, exporter, multi_qubit_circuit):
        qasm_str = exporter.export_qasm2(multi_qubit_circuit)

        assert "ccx q[0], q[1], q[2];" in qasm_str
        assert "cswap q[0], q[1], q[2];" in qasm_str
        assert "swap q[0], q[1];" in qasm_str


# ============================================================================
# Single-Qubit Gates Tests
# ============================================================================


class TestSingleQubitGates:
    """Test single-qubit gate export"""

    def test_single_qubit_gates_qasm3(self, exporter, single_qubit_circuit):
        qasm_str = exporter.export_qasm3(single_qubit_circuit)

        # Standard gates
        assert "id q[0];" in qasm_str
        assert "x q[0];" in qasm_str
        assert "y q[0];" in qasm_str
        assert "z q[0];" in qasm_str
        assert "h q[0];" in qasm_str
        assert "s q[0];" in qasm_str
        assert "sdg q[0];" in qasm_str
        assert "t q[0];" in qasm_str
        assert "tdg q[0];" in qasm_str
        assert "sx q[0];" in qasm_str

        # SXDG should use power modifier
        assert "pow(-1) @ sx q[0];" in qasm_str
        assert "sxdg" not in qasm_str


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases"""

    def test_empty_circuit(self, exporter):
        mock_qdev = MockQDevice(n_wires=2, op_history=[])

        qasm_str = exporter.export_qasm3(mock_qdev)

        assert "OPENQASM 3.0;" in qasm_str
        assert "qubit[2] q;" in qasm_str

    def test_unknown_gate_handling(self, exporter):
        op_history = [
            {"name_or_mat": "unknown_gate", "wires": [0], "params": None},
            {"name_or_mat": "measure_allZ", "wires": [], "params": None},
        ]
        mock_qdev = MockQDevice(n_wires=1, op_history=op_history)

        qasm_str = exporter.export_qasm3(mock_qdev)

        # The gate name appears
        assert "unknown_gate" in qasm_str

    def test_cnot_alias_handling(self, exporter):
        op_history = [
            {"name_or_mat": "cnot", "wires": [0, 1], "params": None},
            {"name_or_mat": "measure_allZ", "wires": [], "params": None},
        ]
        mock_qdev = MockQDevice(n_wires=2, op_history=op_history)

        qasm_str = exporter.export_qasm3(mock_qdev)

        assert "cx q[0], q[1];" in qasm_str
        assert "cnot" not in qasm_str


# ============================================================================
# Convenience Functions Tests
# ============================================================================


class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_export_to_qasm_file(self, simple_circuit):
        with tempfile.NamedTemporaryFile(suffix=".qasm", delete=False) as tmp:
            filename = tmp.name

        try:
            export_to_qasm(simple_circuit, filename, version=3.0)

            assert os.path.exists(filename)
            with open(filename, "r", encoding="utf-8") as f:
                content = f.read()
                assert "OPENQASM 3.0;" in content
                assert "qubit[3] q;" in content
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    def test_export_to_qasm_str(self, simple_circuit):
        qasm_str = export_to_qasm_str(simple_circuit, version=3.0)

        assert isinstance(qasm_str, str)
        assert "OPENQASM 3.0;" in qasm_str
        assert "qubit[3] q;" in qasm_str


# ============================================================================
# Wire Formatting Tests
# ============================================================================


class TestWireFormatting:
    """Test wire formatting"""

    def test_single_wire_formatting(self, exporter):
        assert exporter._format_wires(0, version=3.0) == "q[0]"
        assert exporter._format_wires(0, version=2.0) == "q[0]"

    def test_multiple_wires_formatting_qasm3(self, exporter):
        assert exporter._format_wires([0, 1], version=3.0) == "q[0], q[1]"

    def test_multiple_wires_formatting_qasm2(self, exporter):
        assert exporter._format_wires([0, 1], version=2.0) == "q[0], q[1]"
