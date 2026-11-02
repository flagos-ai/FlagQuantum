"""RZ reuses the constant-angle register contract and preserves its convention."""

import pytest

from flagquantum.compiler._construction.register_lowering import (
    schedule_register_program,
)
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from tests.register_program_cases import Registers

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("angle", [0.0, 1.9, -0.25, 1e-76])
def test_existing_rz_opcode_receives_named_theta(angle: float) -> None:
    r = Registers()
    q = r.declare("q", 1)
    r.emit("qreg.allocate", q)
    r.emit("qreg.gate", q, gate="rz", index=0, angle=angle)
    row = schedule_register_program(r.finish(), n_qubits=1)
    assert len(row.circuit.instructions) == 1
    gate = row.circuit.instructions[0]
    assert (gate.name, gate.wires, gate.params) == ("rz", (0,), {"theta": angle})
    assert row.peak_workspace == 0


@pytest.mark.parametrize("angle", [True, 1, None, "1.9", float("nan"), float("inf")])
def test_rz_angle_stays_a_finite_python_float(angle: object) -> None:
    r = Registers()
    q = r.declare("q", 1)
    r.emit("qreg.allocate", q)
    r.emit("qreg.gate", q, gate="rz", index=0, angle=angle)
    with pytest.raises(HybridVerificationError, match="finite float"):
        schedule_register_program(r.finish(), n_qubits=1)


@pytest.mark.parametrize("gate", [{"rz": 1}, ["rz"], None])
def test_malformed_gate_does_not_escape_as_a_hashing_type_error(gate: object) -> None:
    r = Registers()
    q = r.declare("q", 1)
    r.emit("qreg.allocate", q)
    r.emit("qreg.gate", q, gate=gate, index=0)
    with pytest.raises(HybridVerificationError):
        schedule_register_program(r.finish(), n_qubits=1)
