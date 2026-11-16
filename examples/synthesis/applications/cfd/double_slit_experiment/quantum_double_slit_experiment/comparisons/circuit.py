"""Independent public Circuit construction of the complete double-slit model.

Wire numbers are explicit here to contrast with the register-based native program.
Clean Toffoli ladders implement signed controls without compiler synthesis.
"""

import math
from dataclasses import replace

import torch

import flagquantum as fq
from flagquantum.compiler import optimize
from flagquantum.core.ir import CircuitIR

from ..physics import ABSORBING_PHASE, PHASES

X = (3, 2, 1, 0)
Y = (7, 6, 5, 4)
SELECT = (9, 8)
EDGE, SCATTER, AXIS, DIRECTION, AUX = range(10, 15)
WORK = tuple(range(15, 24))
Condition = tuple[tuple[int, int], ...]


def _word(wires: tuple[int, ...], value: int) -> Condition:
    return tuple((wire, (value >> index) & 1) for index, wire in enumerate(wires))


def _flip(
    circuit: fq.Circuit,
    condition: Condition,
    target: int,
    scratch: tuple[int, ...] = WORK,
) -> None:
    controls = tuple(wire for wire, _ in condition)
    negative = tuple(wire for wire, value in condition if value == 0)
    if len(set((*controls, target))) != len(controls) + 1:
        raise ValueError("Reference controls and target must be distinct")
    count = max(len(controls) - 2, 0)
    ladder = tuple(wire for wire in scratch if wire not in (*controls, target))[:count]
    if len(ladder) != count:
        raise ValueError("Reference Toffoli ladder needs clean scratch")
    for wire in negative:
        circuit.x(wire)
    if not controls:
        circuit.x(target)
    elif len(controls) == 1:
        circuit.cx(controls[0], target)
    elif len(controls) == 2:
        circuit.ccx(*controls, target)
    else:
        circuit.ccx(controls[0], controls[1], ladder[0])
        for index in range(1, count):
            circuit.ccx(ladder[index - 1], controls[index + 1], ladder[index])
        circuit.ccx(ladder[-1], controls[-1], target)
        for index in reversed(range(1, count)):
            circuit.ccx(ladder[index - 1], controls[index + 1], ladder[index])
        circuit.ccx(controls[0], controls[1], ladder[0])
    for wire in reversed(negative):
        circuit.x(wire)


def _h(circuit: fq.Circuit, condition: Condition, target: int) -> None:
    _flip(circuit, condition, WORK[0], WORK[1:])
    circuit.gate("ch", (WORK[0], target))
    _flip(circuit, condition, WORK[0], WORK[1:])


def _phase(circuit: fq.Circuit, condition: Condition, angle: float) -> None:
    _flip(circuit, condition, WORK[0], WORK[1:])
    circuit.gate("phase", WORK[0], theta=angle)
    _flip(circuit, condition, WORK[0], WORK[1:])


def _shift(
    circuit: fq.Circuit, wires: tuple[int, ...], condition: Condition, delta: int
) -> None:
    order = reversed(range(len(wires))) if delta == 1 else range(len(wires))
    for index in order:
        _flip(
            circuit, (*condition, *_word(wires[:index], (1 << index) - 1)), wires[index]
        )


def _scatter(circuit: fq.Circuit, condition: Condition, target: int) -> None:
    # The two excluded slit windows are disjoint, so XOR realizes the union.
    band = (*condition, *_word(Y[1:], 3))
    _flip(circuit, band, target, WORK[3:])
    for slit in (2, 5):
        _flip(circuit, (*band, *_word(X[1:], slit)), target, WORK[3:])


def _boundary(
    circuit: fq.Circuit, wires: tuple[int, ...], condition: Condition, target: int
) -> None:
    for value in (0, 15):
        _flip(circuit, (*condition, *_word(wires, value)), target, WORK[3:])


def _boundary_union(circuit: fq.Circuit) -> None:
    compute = fq.Circuit(24, dtype=torch.complex128)
    _boundary(compute, X, (), WORK[0])
    _boundary(compute, Y, (), WORK[1])
    _scatter(compute, (), WORK[2])
    circuit.compose(compute)
    _flip(circuit, _word(SELECT, 0), EDGE, WORK[3:])
    _flip(circuit, (*_word(SELECT, 0), *_word(WORK[:3], 0)), EDGE, WORK[3:])
    circuit.compose(compute.adjoint())


def _selector() -> fq.Circuit:
    # Analytic two-qubit state preparation, independent of the native loader.
    a, b, c, d = 0.58132999996477, 0.664357048150546, 0.469771373876329, 0.0
    circuit = fq.Circuit(24, dtype=torch.complex128)
    circuit.ry(SELECT[1], 2 * math.atan2(math.hypot(c, d), math.hypot(a, b)))
    for control_value, angle in ((0, 2 * math.atan2(b, a)), (1, 2 * math.atan2(d, c))):
        if control_value == 0:
            circuit.x(SELECT[1])
        circuit.ry(SELECT[0], angle / 2).cx(SELECT[1], SELECT[0])
        circuit.ry(SELECT[0], -angle / 2).cx(SELECT[1], SELECT[0])
        if control_value == 0:
            circuit.x(SELECT[1])
    return circuit


def _hamiltonian() -> fq.Circuit:
    circuit = fq.Circuit(24, dtype=torch.complex128)
    selector = _selector()
    circuit.compose(selector)
    laplace = _word(SELECT, 0)
    _scatter(circuit, laplace, SCATTER)
    _h(circuit, laplace, AXIS)
    _h(circuit, laplace, DIRECTION)
    for axis, wires in ((0, X), (1, Y)):
        for direction, delta in ((0, 1), (1, -1)):
            _shift(
                circuit, wires, (*laplace, (AXIS, axis), (DIRECTION, direction)), delta
            )
    _h(circuit, laplace, DIRECTION)
    _h(circuit, laplace, AXIS)
    _boundary_union(circuit)
    absorbing = _word(SELECT, 1)
    _h(circuit, absorbing, AXIS)
    for axis, wires in ((0, X), (1, Y)):
        selected = (*absorbing, (AXIS, axis))
        _flip(circuit, selected, wires[0])
        _boundary(circuit, wires, selected, DIRECTION)
        _flip(circuit, selected, DIRECTION)
    _h(circuit, absorbing, AXIS)
    _boundary(circuit, X, absorbing, SCATTER)
    _boundary(circuit, Y, absorbing, SCATTER)
    _flip(circuit, absorbing, SCATTER)
    _phase(circuit, absorbing, ABSORBING_PHASE)
    _phase(circuit, _word(SELECT, 2), math.pi)
    return circuit.compose(selector.adjoint())


def _project(circuit: fq.Circuit, angle: float) -> None:
    condition = _word((*SELECT, AXIS, DIRECTION, SCATTER, EDGE), 0)
    _flip(circuit, condition, AUX)
    circuit.rz(AUX, angle)
    _flip(circuit, condition, AUX)


def build_reference(*, cycles: int = 95) -> CircuitIR:
    """Construct all 192 phases and 95 cycles, or a bounded diagnostic prefix."""
    if type(cycles) is not int or not 1 <= cycles <= 95:
        raise ValueError("Cycle count must be an integer in 1..95")
    phases = list(PHASES[: 2 * cycles + 2])
    if cycles != 95:
        phases[-1] = -0.271
    circuit = fq.Circuit(24, dtype=torch.complex128)
    for wire in X[:3]:
        circuit.x(wire)
    circuit.h(X[3])
    for wire in X[:3]:
        circuit.cx(X[3], wire)
    circuit.x(Y[1]).h(AUX)
    hamiltonian = _hamiltonian()
    inverse = hamiltonian.adjoint()
    _project(circuit, phases[0])
    for index in range(cycles):
        circuit.compose(inverse)
        _project(circuit, phases[2 * index + 1])
        circuit.compose(hamiltonian)
        _project(circuit, phases[2 * index + 2])
    circuit.compose(inverse)
    _project(circuit, phases[-1])
    circuit.h(AUX)
    # This layout hint is independently checked for every emitted gate; it
    # cannot assert that scratch is clean or bypass numerical execution.
    return replace(
        optimize(circuit, optimization_level=2),
        metadata={"construction_basis_workspace_prefix": 15},
    )


def compare(*, cycles: int = 95) -> None:
    """Check the native and independent Circuit states, including every work bit."""
    from ..main import build_program

    reference = (
        fq.run(build_reference(cycles=cycles)).statevector().reshape(1 << 15, -1)
    )
    actual = fq.run(build_program(cycles=cycles)).statevector().reshape(1 << 15, -1)
    # Clean work makes this the full-state comparison under zero embedding,
    # even when the two independent constructions use different work widths.
    torch.testing.assert_close(actual[:, 0], reference[:, 0], atol=1e-10, rtol=0)
    assert torch.count_nonzero(actual[:, 1:]) == 0
    assert torch.count_nonzero(reference[:, 1:]) == 0
    print("Complete state, absolute phase and all clean work bits match.")


if __name__ == "__main__":
    compare()
