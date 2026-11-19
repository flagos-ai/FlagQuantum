"""Build the two original H2 walks with bounded native synthesis."""

from typing import Annotated

import torch

import flagquantum as fq
from examples.synthesis.algorithms.quantum_phase_estimation.qpe_with_qubitization.qpe_for_molecule_with_qubitization.main import (
    main as optimized_qpe,
)
from examples.synthesis.algorithms.quantum_phase_estimation.qpe_with_qubitization.qpe_for_molecule_with_qubitization.operations import (
    walk,
)
from examples.synthesis.algorithms.quantum_phase_estimation.qpe_with_qubitization.qpe_for_molecule_with_qubitization.physics import (
    selection_members,
)
from flagquantum import synth
from flagquantum.synth import (
    H,
    Output,
    QArray,
    QBit,
    QNum,
    X,
    adjoint,
    allocate,
    control,
    qft,
    qfunc,
)

from .model import Model

BLOCK = tuple(range(4))
PHASE = tuple(range(4, 9))
DATA = tuple(range(9, 13))
LOGICAL_WIDTH = len(BLOCK) + len(PHASE) + len(DATA)
BUDGET = 100_000


@qfunc
def controlled_block(
    probabilities: list[float],
    x_members: list[int],
    y_members: list[int],
    z_members: list[int],
    block: Output[Annotated[QNum, 4]],
    data: Output[QArray[QBit]],
    flag: Output[QBit],
) -> None:
    allocate(block)
    allocate(4, data)
    allocate(flag)
    control(
        flag, lambda: walk(probabilities, x_members, y_members, z_members, block, data)
    )


@qfunc
def prepare_hartree_fock(data: Output[QArray[QBit]]) -> None:
    allocate(4, data)
    X(data[0])
    X(data[2])


@qfunc
def prepare_phase(bits: Output[QArray[QBit]]) -> None:
    allocate(5, bits)
    for bit in range(5):
        H(bits[bit])


@qfunc
def inverse_fourier(bits: Output[QArray[QBit]]) -> None:
    allocate(5, bits)
    with adjoint():
        qft(bits)


def bindings(model: Model) -> dict[str, list[float] | list[int]]:
    return {
        "probabilities": list(model.probabilities),
        "x_members": list(selection_members("X")),
        "y_members": list(selection_members("Y")),
        "z_members": list(selection_members("Z")),
    }


def optimized_program(model: Model) -> synth.Program:
    inputs = bindings(model)
    inputs["powers"] = [2**index for index in range(4)]
    return synth.synthesize(
        optimized_qpe,
        classical_inputs=inputs,
        max_qubits=24,
        max_operations=BUDGET,
        memory_limit_bytes=1024**3,
    )


def naive_circuit(model: Model) -> fq.Circuit:
    """Compile a reusable controlled walk, then compose the original 31 copies."""
    program = synth.synthesize(
        controlled_block,
        classical_inputs=bindings(model),
        max_qubits=24,
        max_operations=BUDGET,
        memory_limit_bytes=1024**3,
    )
    # This block's allocation contract is used with arbitrary data, selector
    # and flag states. Only the compiler workspace starts and ends in zero.
    ir = program.circuit
    count = 2 ** len(PHASE) - 1
    if len(ir.instructions) * count + 100 > BUDGET:
        raise ValueError("The composed walk would exceed the operation budget")
    occupied = tuple(
        wire for name in ("block", "data", "flag") for wire in program.registers[name]
    )
    if occupied != tuple(range(9)):
        raise ValueError(
            "The reusable walk must place its workspace after its three named inputs"
        )
    workspace = tuple(range(LOGICAL_WIDTH, LOGICAL_WIDTH + ir.n_wires - len(occupied)))
    circuit = fq.Circuit(LOGICAL_WIDTH + len(workspace), dtype=torch.complex128)
    circuit.compose(synth.synthesize(prepare_hartree_fock).circuit, qubits=DATA)
    circuit.compose(synth.synthesize(prepare_phase).circuit, qubits=PHASE)
    for bit, wire in enumerate(reversed(PHASE)):
        for _ in range(2**bit):
            circuit.compose(ir, qubits=(*BLOCK, *DATA, wire, *workspace))
    return circuit.compose(synth.synthesize(inverse_fourier).circuit, qubits=PHASE)
