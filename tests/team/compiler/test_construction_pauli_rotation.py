"""Owned compact Pauli effects are independently checked and exactly mapped."""

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler._construction.pauli_rotation import Support
from flagquantum.compiler._construction.register_lowering import (
    schedule_register_program,
)
from flagquantum.compiler._construction.register_profile import verify_register_program
from flagquantum.compiler._hybrid.model import Block, HybridProgram, Region
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from tests.register_program_cases import Registers

pytestmark = pytest.mark.unit
DTYPE = torch.complex128


@pytest.mark.parametrize(
    "support", [(), ((0, "X"),), ((1, "Y"), (0, "Z")), ((2, "Z"), (0, "Y"), (1, "X"))]
)
@pytest.mark.parametrize("angle", [0.0, 0.37, -0.81])
@pytest.mark.parametrize("control_index", [None, 0, 1])
def test_owned_pauli_matches_dense_exponential_on_coherent_input(
    support: Support, angle: float, control_index: int | None
) -> None:
    p = Registers()
    target = p.declare("data", 3)
    control = p.declare("control", 2)
    reference = fq.Circuit(5, dtype=DTYPE)
    for value, width, offset in ((target, 3, 0), (control, 2, 3)):
        p.emit("qreg.allocate", value)
        for index in range(width):
            p.emit("qreg.gate", value, gate="h", index=index)
            p.emit("qreg.gate", value, gate="rz", index=index, angle=0.17 * (index + 1))
            reference.h(offset + width - 1 - index).rz(
                offset + width - 1 - index, theta=0.17 * (index + 1)
            )
    kwargs = {} if control_index is None else {"control_index": control_index}
    p.emit(
        "qreg.pauli_rotation",
        target,
        *(() if control_index is None else (control,)),
        support=support,
        angle=angle,
        **kwargs,
    )
    model = p.finish()
    candidate = schedule_register_program(model, (), n_qubits=5)
    matrices = {
        "I": torch.eye(2, dtype=DTYPE),
        "X": torch.tensor([[0, 1], [1, 0]], dtype=DTYPE),
        "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=DTYPE),
        "Z": torch.diag(torch.tensor([1, -1], dtype=DTYPE)),
    }
    axes = dict(support)
    pauli = torch.ones((1, 1), dtype=DTYPE)
    for index in reversed(range(3)):
        pauli = torch.kron(pauli, matrices[axes.get(index, "I")])
    unitary = torch.matrix_exp(-1j * angle * pauli)
    if control_index is None:
        full = torch.kron(unitary, torch.eye(4, dtype=DTYPE))
    else:
        projector = torch.diag(
            torch.tensor([(n >> control_index) & 1 for n in range(4)], dtype=DTYPE)
        )
        full = torch.kron(unitary, projector) + torch.kron(
            torch.eye(8, dtype=DTYPE), torch.eye(4, dtype=DTYPE) - projector
        )
    reference.any(*range(5), matrix=full)
    actual = fq.Circuit.from_ir(candidate.circuit, dtype=DTYPE).state().reshape(-1)
    torch.testing.assert_close(
        actual, fq.run(reference).statevector().reshape(-1), atol=1e-12, rtol=0
    )
    assert candidate.peak_workspace == 0


@pytest.mark.parametrize(
    "attributes",
    [
        {"support": "X", "angle": 0.1},
        {"support": ((True, "X"),), "angle": 0.1},
        {"support": ((2, "Z"),), "angle": 0.1},
        {"support": ((-1, "Z"),), "angle": 0.1},
        {"support": ((0, "I"),), "angle": 0.1},
        {"support": ((0, "X"), (0, "Y")), "angle": 0.1},
        {"support": ((0, "X"),), "angle": True},
        {"support": ((0, "X"),), "angle": float("nan")},
        {"support": ((0, "X"),), "angle": 1e308},
        {"support": (), "angle": 0.1, "control_index": 0},
    ],
)
def test_owned_verifier_rejects_malformed_pauli_records(
    attributes: dict[str, object],
) -> None:
    p = Registers()
    q = p.declare("q", 2)
    p.emit("qreg.allocate", q)
    p.emit("qreg.pauli_rotation", q, **attributes)
    with pytest.raises(HybridVerificationError, match="pauli"):
        verify_register_program(p.finish())


@pytest.mark.parametrize("index", [True, -1, 2])
def test_pauli_control_index_is_checked(index: object) -> None:
    p = Registers()
    q = p.declare("q", 2)
    c = p.declare("c", 2)
    p.emit("qreg.allocate", q)
    p.emit("qreg.allocate", c)
    p.emit("qreg.pauli_rotation", q, c, support=(), angle=0.37, control_index=index)
    with pytest.raises(HybridVerificationError, match="control"):
        verify_register_program(p.finish())


def test_pauli_control_alias_and_uninitialized_target_are_rejected() -> None:
    p = Registers()
    q = p.declare("q", 2)
    p.emit("qreg.pauli_rotation", q, support=(), angle=0.37)
    with pytest.raises(HybridVerificationError, match="uninitialized"):
        verify_register_program(p.finish())
    p = Registers()
    q = p.declare("q", 2)
    p.emit("qreg.allocate", q)
    p.emit(
        "qreg.pauli_rotation", q, q, support=((0, "X"),), angle=0.37, control_index=1
    )
    with pytest.raises(HybridVerificationError, match="disjoint"):
        verify_register_program(p.finish())


def test_const_pauli_target_is_rejected_at_the_compiler_boundary() -> None:
    p = Registers()
    q = p.declare("q", 2)
    p.emit("qreg.pauli_rotation", q, support=((0, "X"),), angle=0.37)
    block = p.finish().body.blocks[0]
    helper = HybridProgram(
        "borrowed_pauli", Region((Block((p.entry, q), block.operations[1:]),))
    )
    outer = Registers()
    data = outer.declare("data", 2)
    outer.emit("qreg.allocate", data)
    with pytest.raises(HybridVerificationError, match="Const|writable"):
        verify_register_program(outer.finish(), (helper,))


def test_pauli_physical_expansion_still_obeys_final_instruction_budget() -> None:
    p = Registers()
    q = p.declare("q", 3)
    p.emit("qreg.allocate", q)
    p.emit("qreg.pauli_rotation", q, support=((0, "X"), (1, "Y"), (2, "Z")), angle=0.37)
    with pytest.raises(HybridVerificationError, match="operation budget"):
        schedule_register_program(p.finish(), (), n_qubits=3, max_operations=2)


def test_compact_rotation_does_not_bypass_the_owned_node_cap() -> None:
    p = Registers()
    q = p.declare("q", 1)
    p.emit("qreg.allocate", q)
    for _ in range(4096):
        p.emit("qreg.pauli_rotation", q, support=(), angle=0.0)
    with pytest.raises(HybridVerificationError, match="node budget"):
        verify_register_program(p.finish())
