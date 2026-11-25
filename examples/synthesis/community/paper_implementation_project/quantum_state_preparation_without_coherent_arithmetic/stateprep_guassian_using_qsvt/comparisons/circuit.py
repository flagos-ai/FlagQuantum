"""Independent controlled-Y/projector matrices and clean CCX zero reflection."""

import cmath
import math

import torch

import flagquantum as fq

from ..data import EVEN_PHASES, ODD_PHASES, SINE_ANGLES, SUCCESS_AMPLITUDE

Gate = tuple[tuple[int, ...], torch.Tensor]


def direct_sum(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    n, m = left.shape[0], right.shape[0]
    result = torch.zeros((n + m, n + m), dtype=torch.complex128)
    result[:n, :n] = left
    result[n:, n:] = right
    return result


def rotation(angle: float) -> torch.Tensor:
    c, s = math.cos(angle / 2), math.sin(angle / 2)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.complex128)


def preparation_gates() -> list[Gate]:
    """Keep the original projector-X/RZ/X order, distinct from phase polynomials."""
    identity = torch.eye(2, dtype=torch.complex128)
    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    h = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / math.sqrt(2)
    gates: list[Gate] = []
    for wire in range(8):
        gates.append(((wire,), h))
    gates.extend([((10,), h), ((9,), h)])
    projector = torch.eye(4, dtype=torch.complex128)
    projector[:2, :2] = x

    def sine(inverse: bool = False, selected: bool = False) -> list[Gate]:
        block: list[Gate] = [((8,), x)] if inverse else []
        if selected:
            controlled_x = direct_sum(x, identity)
            block = [((10, 8), controlled_x)] if inverse else []
        for bit, angle in enumerate(SINE_ANGLES):
            ry = rotation(-angle if inverse else angle)
            controlled_y = direct_sum(identity, ry)
            matrix = (
                direct_sum(controlled_y, torch.eye(4, dtype=torch.complex128))
                if selected
                else controlled_y
            )
            wires = (10, 7 - bit, 8) if selected else (7 - bit, 8)
            block.append((wires, matrix))
        if not inverse:
            block.append(((8,), x))
        return block

    for index in range(29):
        if index:
            gates.extend(sine(inverse=index % 2 == 0, selected=index == 28))
        gates.append(((8, 9), projector))
        diagonal = [
            cmath.exp(
                0.5j
                * (
                    EVEN_PHASES[index]
                    if branch == 0
                    else ODD_PHASES[index] if index < 28 else 0
                )
                * sign
            )
            for branch in range(2)
            for sign in (-1, 1)
        ]
        gates.append(
            ((10, 9), torch.diag(torch.tensor(diagonal, dtype=torch.complex128)))
        )
        gates.append(((8, 9), projector))
    gates.extend([((9,), h), ((10,), h), ((10,), h)])
    return gates


def replay(circuit: fq.Circuit, gates: list[Gate], inverse: bool = False) -> None:
    for wires, matrix in reversed(gates) if inverse else gates:
        circuit.any(*wires, matrix=matrix.mH if inverse else matrix)


def zero_group(circuit: fq.Circuit, controls: tuple[int, ...], target: int) -> None:
    """Toggle one clean marker on six zero controls with four restored scratch bits."""
    for wire in controls:
        circuit.x(wire)
    circuit.ccx(controls[0], controls[1], 14)
    for index in range(2, 5):
        circuit.ccx(12 + index, controls[index], 13 + index)
    circuit.ccx(17, controls[5], target)
    for index in reversed(range(2, 5)):
        circuit.ccx(12 + index, controls[index], 13 + index)
    circuit.ccx(controls[0], controls[1], 14)
    for wire in controls:
        circuit.x(wire)


def reference_circuit() -> fq.Circuit:
    # Twelve algorithm bits, two group markers and four shared scratch bits.
    circuit = fq.Circuit(18, dtype=torch.complex128)
    gates = preparation_gates()
    replay(circuit, gates)
    beta = 2 * math.acos(math.sin(math.pi / 10) / SUCCESS_AMPLITUDE)
    circuit.ry(11, theta=beta)
    good = torch.eye(16, dtype=torch.complex128)
    good[0, 0] = -1
    for _ in range(2):
        circuit.any(8, 9, 10, 11, matrix=good)
        circuit.ry(11, theta=-beta)
        replay(circuit, gates, inverse=True)
        zero_group(circuit, tuple(range(6)), 12)
        zero_group(circuit, tuple(range(6, 12)), 13)
        circuit.cz(12, 13)
        zero_group(circuit, tuple(range(6, 12)), 13)
        zero_group(circuit, tuple(range(6)), 12)
        circuit.any(0, matrix=-torch.eye(2, dtype=torch.complex128))
        replay(circuit, gates)
        circuit.ry(11, theta=beta)
    return circuit


def run_example() -> None:
    from ..main import build_program

    program = build_program()
    actual = fq.run(program).statevector().reshape(2048, 4, -1)
    expected = fq.run(reference_circuit()).statevector().reshape(2048, 2, -1)
    torch.testing.assert_close(actual[:, 0, 0], expected[:, 0, 0], atol=1e-10, rtol=0)
    assert float(actual[:, 1:].abs().square().sum()) < 1e-22
    assert float(actual[:, :, 1:].abs().square().sum()) < 1e-22
    assert float(expected[:, 1:].abs().square().sum()) < 1e-22
    assert float(expected[:, :, 1:].abs().square().sum()) < 1e-22
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
