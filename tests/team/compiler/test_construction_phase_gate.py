"""Constant phase parameters belong to the existing register gate boundary."""

from dataclasses import replace

import pytest

from flagquantum.compiler._construction.register_lowering import (
    schedule_register_program,
)
from flagquantum.compiler._construction.register_profile import verify_register_program
from flagquantum.compiler._hybrid.model import Block, HybridProgram, Region
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from tests.register_program_cases import Registers

pytestmark = pytest.mark.unit


def phase_program(
    angle: object, *, initialized: bool = True, index: object = 0
) -> HybridProgram:
    r = Registers()
    q = r.declare("q", 1)
    if initialized:
        r.emit("qreg.allocate", q)
    r.emit("qreg.gate", q, gate="phase", index=index, angle=angle)
    return r.finish()


@pytest.mark.parametrize("angle", [0.0, 1.0, -1.0, 3.141592653589793, float(2**53)])
def test_compiler_preserves_phase_parameter_without_rz_substitution(
    angle: float,
) -> None:
    p = phase_program(angle)
    row = schedule_register_program(p, n_qubits=1)
    assert row.peak_workspace == 0
    assert len(row.circuit.instructions) == 1
    gate = row.circuit.instructions[0]
    assert (gate.name, gate.wires, gate.params) == ("phase", (0,), {"theta": angle})


@pytest.mark.parametrize(
    "angle", [True, 1, None, "1", (), float("nan"), float("inf"), -float("inf")]
)
def test_noncanonical_or_nonfinite_angle_is_refused(angle: object) -> None:
    with pytest.raises(HybridVerificationError, match="finite float"):
        verify_register_program(phase_program(angle))


@pytest.mark.parametrize("index", [True, -1, 1, 0.0])
def test_gate_index_is_still_validated(index: object) -> None:
    with pytest.raises(HybridVerificationError, match="register.gate"):
        verify_register_program(phase_program(1.0, index=index))


def test_initialization_and_exact_attribute_boundary() -> None:
    with pytest.raises(HybridVerificationError, match="uninitialized"):
        verify_register_program(phase_program(1.0, initialized=False))
    p = phase_program(1.0)
    block = p.body.blocks[0]
    gate = block.operations[-2]
    for attributes in [
        {"gate": "phase", "index": 0},
        {"gate": "h", "index": 0, "angle": 1.0},
        {**gate.attributes, "params": (1.0,)},
    ]:
        changed = replace(gate, attributes=attributes)
        broken = replace(
            p,
            body=Region(
                (
                    Block(
                        block.arguments,
                        (*block.operations[:-2], changed, block.operations[-1]),
                    ),
                )
            ),
        )
        with pytest.raises(HybridVerificationError, match="register.operation"):
            verify_register_program(broken)


def test_phase_emission_keeps_complete_operation_budget() -> None:
    r = Registers()
    q = r.declare("q", 1)
    r.emit("qreg.allocate", q)
    r.emit("qreg.gate", q, gate="phase", index=0, angle=1.0)
    r.emit("qreg.gate", q, gate="phase", index=0, angle=2.0)
    with pytest.raises(HybridVerificationError, match="budget|operations"):
        schedule_register_program(r.finish(), n_qubits=1, max_operations=1)
