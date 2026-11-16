"""Independent local-matrix Circuit for the complete Vlasov-Ampere program."""

import math
from collections.abc import Callable

import torch

import flagquantum as fq

DTYPE = torch.complex128


def _block_diag(*blocks: torch.Tensor) -> torch.Tensor:
    size = sum(block.shape[0] for block in blocks)
    result = torch.zeros((size, size), dtype=DTYPE)
    offset = 0
    for block in blocks:
        width = block.shape[0]
        result[offset : offset + width, offset : offset + width] = block
        offset += width
    return result


def _phase_table(suffix: str) -> torch.Tensor:
    if suffix == "0":
        angles = [1.5707963267949 + 3.14159265358979 * (bit - 0.5) for bit in range(2)]
    else:
        angles = [
            1.17809724509617
            - 2.35619449019234 * (low - 0.5)
            - 0.785398163397448 * ((low ^ high) - 0.5)
            - 0.785398163397448 * (high - 0.5)
            for high in range(2)
            for low in range(2)
        ]
    return torch.diag(torch.exp(1j * torch.tensor(angles, dtype=torch.float64)))


def _ry(angle: float) -> torch.Tensor:
    c, s = (math.cos(angle / 2), math.sin(angle / 2))
    return torch.tensor([[c, -s], [s, c]], dtype=DTYPE)


def _permutation(width: int, map_index: Callable[[int], int]) -> torch.Tensor:
    matrix = torch.zeros((1 << width, 1 << width), dtype=DTYPE)
    for basis in range(1 << width):
        matrix[map_index(basis), basis] = 1
    return matrix


def _embed(matrix: torch.Tensor, wires: tuple[int, ...], width: int) -> torch.Tensor:
    masks = tuple(1 << width - wire - 1 for wire in wires)
    remove = sum(masks)
    result = torch.zeros((1 << width, 1 << width), dtype=DTYPE)
    for basis in range(1 << width):
        input_index = 0
        for mask in masks:
            input_index = input_index << 1 | bool(basis & mask)
        for subspace in range(1 << len(wires)):
            output = basis & ~remove
            for bit, mask in enumerate(reversed(masks)):
                if subspace & 1 << bit:
                    output |= mask
            result[output, basis] = matrix[subspace, input_index]
    return result


def _real_preparation(amplitudes: list[float]) -> torch.Tensor:
    """Prefix-conditioned real rotations, derived directly from the amplitudes."""
    width = len(amplitudes).bit_length() - 1
    result = torch.eye(len(amplitudes), dtype=DTYPE)
    for level in range(width):
        tail = 1 << width - level - 1
        matrices = []
        for offset in range(0, len(amplitudes), 2 * tail):
            low = amplitudes[offset : offset + tail]
            high = amplitudes[offset + tail : offset + 2 * tail]
            lower, upper = (
                (low[0], high[0])
                if tail == 1
                else (math.hypot(*low), math.hypot(*high))
            )
            rotation = _ry(2 * math.atan2(upper, lower))
            matrices.append(torch.kron(rotation, torch.eye(tail, dtype=DTYPE)))
        result = _block_diag(*matrices) @ result
    return result


def _derivative() -> torch.Tensor:
    a = [0.5, 0.5]
    prepare_lcu2 = _real_preparation([math.sqrt(value) for value in a])
    plus_minus = _block_diag(
        _permutation(4, lambda word: (word - 1) % 16),
        _permutation(4, lambda word: (word + 1) % 16),
    )
    prepare = torch.kron(prepare_lcu2, torch.eye(16, dtype=DTYPE))
    diagonal = torch.kron(_phase_table("0"), torch.eye(16, dtype=DTYPE))
    dirichlet = prepare.mH @ diagonal @ plus_minus @ prepare
    reflection = _permutation(4, lambda word: word ^ (6 if word & 8 else 0))
    z = torch.diag(torch.tensor([1] * 8 + [-1] * 8, dtype=DTYPE))
    a = [-0.688247201611685, 0.688247201611685, -0.229415733870562, 0.0]
    inverse = _embed(_real_preparation(a).mH, (1, 2), 4)
    mark = _permutation(4, lambda word: word ^ bool(word >> 1 & 3))
    boundary = reflection @ mark @ inverse @ z @ reflection
    d0 = _embed(dirichlet, (4, 3, 0, 1, 2), 5)
    d1 = _embed(boundary, (0, 1, 2, 3), 5)
    select = _embed(_block_diag(d0, d1), (4, 0, 1, 2, 3, 5), 6)
    a = [0.314519859138756, 0.685480140861244]
    prepare_lcu1 = _embed(_real_preparation([math.sqrt(value) for value in a]), (4,), 6)
    return prepare_lcu1.mH @ select @ prepare_lcu1


def reference_circuit() -> fq.Circuit:
    circuit = fq.Circuit(22, dtype=DTYPE)
    for wire in range(7):
        circuit.h(wire).cx(wire, 7 + wire)
    outer = (14, 15)
    electric, x, v = (0, (1, 2, 3), (4, 5, 6))
    lcu2, lcu1, indicator, block_flag, zeta_flag, flag = (16, 17, 18, 19, 20, 21)

    def controlled(
        selector: tuple[int, ...],
        pattern: int,
        wires: tuple[int, ...],
        matrix: torch.Tensor,
    ) -> None:
        identity = torch.eye(matrix.shape[0], dtype=DTYPE)
        diagonal = _block_diag(
            *(
                matrix if basis == pattern else identity
                for basis in range(1 << len(selector))
            )
        )
        circuit.any(*selector, *wires, matrix=diagonal)

    def rotate_then_flip(table: list[float]) -> torch.Tensor:
        x_gate = torch.tensor([[0, 1], [1, 0]], dtype=DTYPE)
        return _block_diag(*(x_gate @ _ry(angle) for angle in table))

    probabilities = [0.0960237307895, 0.817686639833667, 0.086289629376833, 0.0]
    preparation = _real_preparation([math.sqrt(value) for value in probabilities])
    circuit.any(*outer, matrix=preparation)
    controlled(outer, 0, (*x, flag, lcu1, lcu2), _derivative())
    table = [
        0.0,
        0.505360510284157,
        1.047197551196598,
        1.696124157962962,
        -3.141592653589793,
        -1.696124157962962,
        -1.047197551196598,
        -0.505360510284157,
    ]
    controlled(outer, 0, (*v, indicator), rotate_then_flip(table))

    def mark_zeta(word: int) -> int:
        x_value = word >> 4
        v_raw = word >> 1 & 7
        signed = v_raw if v_raw < 4 else v_raw - 8
        return word ^ (x_value == 0 and signed > 0 or (x_value == 7 and signed <= 0))

    controlled(outer, 0, (*x, *v, zeta_flag), _permutation(7, mark_zeta))
    controlled(
        outer, 0, (electric, block_flag), _permutation(2, lambda word: word ^ word >> 1)
    )
    controlled(outer, 1, (electric,), _permutation(1, lambda word: word ^ 1))
    mark_velocity = _permutation(4, lambda word: word ^ bool(word >> 1))
    a = [
        -0.0,
        -0.644914408684129,
        -0.287799710583439,
        -0.0354360582708,
        0.001426768318105,
        0.0354360582708,
        0.287799710583439,
        0.644914408684129,
    ]
    force = _embed(_real_preparation(a), (0, 1, 2), 4) @ mark_velocity
    a = [
        0.0,
        0.150755672288882,
        0.301511344577764,
        0.452267016866645,
        -0.603022689155527,
        -0.452267016866645,
        -0.301511344577764,
        -0.150755672288882,
    ]
    current = mark_velocity @ _embed(_real_preparation(a).mH, (0, 1, 2), 4)
    controlled((*outer, electric), 2, (*v, flag), force)
    controlled((*outer, electric), 3, (*v, flag), current)
    table = [0.099026155446035, 3.141592653589793]
    controlled(outer, 1, (electric, indicator), rotate_then_flip(table))
    circuit.any(*outer, matrix=_phase_table("1"))
    circuit.any(*outer, matrix=preparation.mH.contiguous())
    return circuit


def run_example() -> None:
    from ..main import build_program

    program = build_program()
    expected = fq.run(reference_circuit()).statevector().flatten()
    actual = fq.run(program).statevector().reshape(1 << 22, -1)
    dirty = 0.0
    for start in range(0, len(expected), 65536):
        torch.testing.assert_close(
            actual[start : start + 65536, 0],
            expected[start : start + 65536],
            atol=1e-11,
            rtol=0,
        )
        dirty += float(actual[start : start + 65536, 1:].abs().square().sum())
    assert dirty < 1e-22
    print(program.report["resources"])
    print("Aggregate dirty-work probability:", dirty)


if __name__ == "__main__":
    run_example()
