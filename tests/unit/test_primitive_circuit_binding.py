"""Relocated fragments preserve every branch and occupied caller slots."""

from itertools import product

import pytest
import torch

from flagquantum import Circuit
from flagquantum.compiler.registers.primitives.circuit_binding import (
    bind_primitive_circuit,
)
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.core.ir import CircuitIR, Instruction, MeasurementNode, ObservableNode
from tests.unit.measurement_checks import forced_branch

pytestmark = pytest.mark.unit


def fragment() -> CircuitIR:
    return CircuitIR(
        3,
        (
            Instruction(
                "measure", (0,), metadata={"is_dynamic": True, "classical_bit": 0}
            ),
            Instruction(
                "measure", (1,), metadata={"is_dynamic": True, "classical_bit": 1}
            ),
            ClassicalXor((0, 1), 2, True).instruction(),
            Instruction(
                "ry", (2,), {"theta": 0.23}, metadata={"conditions": ((0, 1), (1, 0))}
            ),
            Instruction(
                "z", (2,), metadata={"condition_clauses": (((0, 0),), ((1, 1),))}
            ),
            Instruction(
                "reset",
                (0,),
                metadata={
                    "is_dynamic": True,
                    "guarded_effect": True,
                    "conditions": ((2, 1),),
                },
            ),
        ),
    ).validate()


@pytest.mark.parametrize("quantum", ((3, 1, 4), (5, 4, 2)))
@pytest.mark.parametrize("slots", ((7, 2, 9), (5, 1, 3)))
def test_quantum_classical_permutations_preserve_instrument_and_previous_measurement(
    quantum: tuple[int, ...], slots: tuple[int, ...]
) -> None:
    source = fragment()
    original = source.to_dict()
    rebound = bind_primitive_circuit(source, quantum, classical_bits=slots)
    assert source.to_dict() == original
    free = tuple(wire for wire in range(6) if wire not in quantum)
    order = (*quantum, *free)
    axes = tuple(order.index(wire) for wire in range(6))
    values = torch.arange(32, dtype=torch.float64)
    vector = torch.complex(values + 1, values % 5 - 2).reshape(8, 4)
    vector /= vector.norm()
    canonical = torch.zeros((8, 2, 4), dtype=torch.complex128)
    canonical[:, 1, :] = vector
    canonical = canonical.flatten()

    def physical(state: torch.Tensor) -> torch.Tensor:
        return state.reshape((2,) * 6).permute(axes).flatten()

    prefix = Instruction(
        "measure", (free[0],), metadata={"is_dynamic": True, "classical_bit": 0}
    )
    suffix = Instruction("x", (free[1],), metadata={"conditions": ((0, 1),)})
    flip = torch.arange(64) ^ 2  # Canonical spectator is wire four.
    for outcomes in product((0, 1), repeat=2):
        expected = forced_branch(canonical, source.instructions, outcomes)[flip]
        actual = forced_branch(
            physical(canonical), (prefix, *rebound, suffix), (1, *outcomes)
        )
        torch.testing.assert_close(actual, physical(expected), atol=2e-12, rtol=0)


@pytest.mark.parametrize(
    "quantum,slots",
    (
        ((0, 1), (0, 1, 2)),
        ((0, 0, 1), (0, 1, 2)),
        ((0, 1, -1), (0, 1, 2)),
        ((0, 1, True), (0, 1, 2)),
        ((0, 1, 2), (0, 1)),
        ((0, 1, 2), (0, 0, 1)),
        ((0, 1, 2), (0, -1, 2)),
        ((0, 1, 2), (0, True, 2)),
    ),
)
def test_incomplete_or_aliasing_storage_is_rejected(
    quantum: tuple[int, ...], slots: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError):
        bind_primitive_circuit(fragment(), quantum, classical_bits=slots)


def test_unresolved_external_classical_read_is_rejected() -> None:
    source = CircuitIR(1, (Instruction("x", (0,), metadata={"conditions": ((3, 1),)}),))
    with pytest.raises(ValueError):
        bind_primitive_circuit(source, (7,), classical_bits=(1, 2, 3, 4))


def test_pure_quantum_fragment_needs_no_classical_storage() -> None:
    source = CircuitIR(
        2, (Instruction("ry", (1,), {"theta": 0.17}), Instruction("cx", (1, 0)))
    )
    assert bind_primitive_circuit(source, (4, 2)) == (
        Instruction("ry", (2,), {"theta": 0.17}),
        Instruction("cx", (2, 4)),
    )


@pytest.mark.parametrize(
    "result", (ObservableNode("z", (0,)), MeasurementNode("probabilities", (0,)))
)
def test_program_result_declarations_are_not_silently_dropped(
    result: ObservableNode | MeasurementNode,
) -> None:
    source = CircuitIR(
        1,
        (),
        observables=(result,) if isinstance(result, ObservableNode) else (),
        measurements=(result,) if isinstance(result, MeasurementNode) else (),
    )
    with pytest.raises(ValueError, match="result declarations"):
        bind_primitive_circuit(source, (3,))


def test_custom_matrix_phase_and_spectator_survive_binding() -> None:
    matrix = torch.tensor([[1, 1j], [1j, 1]], dtype=torch.complex128) / 2**0.5
    source = Circuit(1).any(0, matrix=matrix).to_ir()
    bound = bind_primitive_circuit(source, (1,))
    state = torch.tensor([1, 2j, 1 - 1j, -3], dtype=torch.complex128)
    state /= state.norm()
    actual = (
        Circuit(2, inputs=state, dtype=state.dtype)
        .compose(CircuitIR(2, bound))
        .state()
        .flatten()
    )
    expected = torch.kron(torch.eye(2, dtype=state.dtype), matrix) @ state
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=0)


@pytest.mark.ftqc_reference
def test_actual_foreign_erasure_import_binds_to_existing_native_primitive() -> None:
    import cirq
    from qualtran.bloqs.mcmt import And

    from flagquantum.compiler.registers.primitives.and_compute import and_uncompute_ir
    from flagquantum.ecosystem.cirq._feedback_import import import_cirq_feedback

    external = And().adjoint().to_clifford_t_circuit().unfreeze()
    ports = tuple(cirq.NamedQubit(name) for name in ("ctrl_0", "ctrl_1", "target"))
    snapshot = import_cirq_feedback(external, qubit_order=ports).ir
    assert bind_primitive_circuit(
        snapshot, (5, 2, 7), classical_bits=(9,)
    ) == and_uncompute_ir(5, 2, 7, 9)
