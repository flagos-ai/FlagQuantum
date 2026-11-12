"""CX/rotation lowering with explicit phase compensation on clean workspace."""

import math
from dataclasses import replace

from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.quantum_walk.classiq_discrete_quantum_walk.overview.exact_basis import (
    exact_rules,
)
from flagquantum import synth
from flagquantum.compiler.basis_conversion import convert_basis
from flagquantum.compiler.resource_estimation import estimate_resources
from flagquantum.core.ir import CircuitIR, Instruction


def rotation_basis(
    program: synth.Program, *, max_added_operations: int = 20_000
) -> CircuitIR:
    """Preserve the walk's clean-workspace isometry in CX/RZ/RY/RX.

    This is not an operator identity on dirty workspace. The acceptance tests
    prove the input-zero and output-zero contract for every logical column.
    U3(theta,phi,lambda) = exp(i*(phi+lambda)/2) RZ(phi) RY(theta) RZ(lambda).
    A final RZ(-2*phase) on a clean zero work qubit restores that phase.
    """
    logical = {wire for wires in program.registers.values() for wire in wires}
    hidden = tuple(
        wire
        for wire in range(estimate_resources(program.circuit).n_qubits)
        if wire not in logical
    )
    if not hidden:
        raise ValueError(
            "Rotation phase compensation requires a verified clean work qubit."
        )
    source = convert_basis(
        program.circuit,
        gates=("cx", "u3"),
        max_added_operations=max_added_operations,
        rules=exact_rules(),
    ).program
    operations: list[Instruction] = []
    phase = 0.0
    for instruction in source.instructions:
        if instruction.name == "cx":
            operations.append(instruction)
            continue
        angles = tuple(instruction.params[name] for name in ("theta", "phi", "lbd"))
        if any(
            isinstance(angle, bool)
            or not isinstance(angle, (float, int))
            or not math.isfinite(angle)
            for angle in angles
        ):
            raise ValueError(
                "This fixed-reference workflow requires finite real U3 angles."
            )
        theta, phi, lbd = angles
        phase += (phi + lbd) / 2
        for name, angle in (("rz", lbd), ("ry", theta), ("rz", phi)):
            operations.append(
                Instruction(
                    name,
                    instruction.wires,
                    params={"theta": angle},
                    metadata=instruction.metadata,
                )
            )
    operations.append(
        Instruction(
            "rz",
            (hidden[0],),
            params={"theta": -2 * phase},
            metadata={"phase_compensation": "input_zero_output_zero_workspace"},
        )
    )
    if len(operations) - len(program.circuit.instructions) > max_added_operations:
        raise ValueError("Rotation-basis expansion exceeds max_added_operations.")
    return replace(source, instructions=tuple(operations))
