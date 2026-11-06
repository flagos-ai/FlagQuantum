"""Check bounded registered-gate synthesis against independent dense Circuits."""

import cmath
from dataclasses import replace

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler._construction.register_lowering import (
    schedule_register_program,
)
from flagquantum.compiler._construction.register_profile import verify_register_program
from flagquantum.compiler._hybrid.model import HybridProgram
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from tests.register_program_cases import Registers

pytestmark = pytest.mark.integration
DTYPE = torch.complex128


def matrix_fixture(width: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(104 + width)
    raw = torch.randn(1 << width, 1 << width, dtype=DTYPE, generator=generator)
    unitary, _ = torch.linalg.qr(raw)
    return unitary * cmath.exp(0.37j)


def model_for(
    matrix: torch.Tensor, *, power: int = 1, controlled: bool = False
) -> HybridProgram:
    p = Registers()
    data = p.declare("data", matrix.shape[0].bit_length() - 1)
    p.emit("qreg.allocate", data)
    ports = [data]
    attributes: dict[str, object] = {
        "matrix": tuple(
            tuple((float(z.real), float(z.imag)) for z in row)
            for row in matrix.tolist()
        ),
        "power": power,
    }
    if controlled:
        control = p.declare("control", 2)
        p.emit("qreg.allocate", control)
        ports.append(control)
        attributes["control_index"] = 0
    p.emit("qreg.unitary", *ports, **attributes)
    return p.finish()


@pytest.mark.parametrize(
    "width,power,controlled",
    [(1, 128, True), (2, 7, True), (3, 1, False), (3, 2, True), (3, 128, True)],
)
def test_complex_unitary_all_columns_and_controlled_phase(
    width: int, power: int, controlled: bool
) -> None:
    matrix = matrix_fixture(width)
    n = width + (2 if controlled else 0)
    row = schedule_register_program(
        model_for(matrix, power=power, controlled=controlled), n_qubits=n
    )
    assert row.circuit.n_wires == n
    assert row.peak_workspace == 0
    assert all(g.matrix is None for g in row.circuit.instructions)
    expected = fq.Circuit(n, dtype=DTYPE)
    unitary = torch.linalg.matrix_power(matrix, power)
    if controlled:
        expected.any(
            n - 1,
            *range(width),
            matrix=torch.block_diag(torch.eye(1 << width, dtype=DTYPE), unitary),
        )
    else:
        expected.any(*range(width), matrix=unitary)
    # Every column certifies asymmetric basis order and the relative control phase.
    for basis in range(1 << n):
        initial = torch.zeros(1 << n, dtype=DTYPE)
        initial[basis] = 1
        actual = fq.Circuit.from_ir(row.circuit, dtype=DTYPE, inputs=initial).state()
        reference = fq.Circuit.from_ir(
            expected.to_ir(), dtype=DTYPE, inputs=initial
        ).state()
        torch.testing.assert_close(actual, reference, atol=1e-12, rtol=0)
    initial = torch.randn(
        1 << n, dtype=DTYPE, generator=torch.Generator().manual_seed(403)
    )
    initial /= initial.norm()
    actual = fq.Circuit.from_ir(row.circuit, dtype=DTYPE, inputs=initial).state()
    reference = fq.Circuit.from_ir(
        expected.to_ir(), dtype=DTYPE, inputs=initial
    ).state()
    torch.testing.assert_close(actual, reference, atol=1e-12, rtol=0)


def test_four_qubit_unitary_without_clean_workspace() -> None:
    matrix = matrix_fixture(4)
    row = schedule_register_program(model_for(matrix), n_qubits=4)
    assert row.peak_workspace == 0
    inputs = torch.eye(16, dtype=DTYPE)
    actual = fq.Circuit.from_ir(row.circuit, dtype=DTYPE, inputs=inputs, bsz=16).state()
    torch.testing.assert_close(actual, matrix.T, atol=1e-12, rtol=0)


@pytest.mark.parametrize(
    "attributes",
    [
        {"power": 0},
        {"power": 129},
        {"power": True},
        {"control_index": -1},
        {"control_index": 2},
        {"control_index": False},
        {"other": 1},
    ],
)
def test_reject_malformed_power_and_control(attributes: dict[str, object]) -> None:
    model = model_for(matrix_fixture(1), controlled=True)
    block = model.body.blocks[0]
    operations = tuple(
        (
            replace(op, attributes={**op.attributes, **attributes})
            if op.name == "qreg.unitary"
            else op
        )
        for op in block.operations
    )
    model = replace(
        model, body=replace(model.body, blocks=(replace(block, operations=operations),))
    )
    with pytest.raises(HybridVerificationError):
        verify_register_program(model)


@pytest.mark.parametrize(
    "case", ["alias", "uninitialized", "nonunitary", "width", "duplicate", "index"]
)
def test_reject_unsafe_unitary_programs(case: str) -> None:
    p = Registers()
    width = 6 if case == "width" else 3
    data = p.declare("data", width)
    p.emit("qreg.allocate", data)
    control = data if case == "alias" else p.declare("control", 1)
    if case not in {"alias", "uninitialized"}:
        p.emit("qreg.allocate", control)
    matrix = torch.eye(1 << width, dtype=DTYPE)
    if case == "nonunitary":
        matrix[0, 0] = 1.01
    attributes: dict[str, object] = {
        "matrix": tuple(
            tuple((float(z.real), float(z.imag)) for z in row)
            for row in matrix.tolist()
        ),
        "power": 128,
        "control_index": 0,
    }
    targets = [data]
    if case in {"duplicate", "index"}:
        targets = [data, data]
        attributes["indices"] = (0, 0) if case == "duplicate" else (0, 3)
    p.emit("qreg.unitary", *targets, control, **attributes)
    with pytest.raises(HybridVerificationError):
        verify_register_program(p.finish())


def test_indexed_three_qubit_order_preserves_sentinel_and_control_phase() -> None:
    matrix = matrix_fixture(3)
    p = Registers()
    data = p.declare("data", 4)
    control = p.declare("control", 1)
    p.emit("qreg.allocate", data)
    p.emit("qreg.allocate", control)
    entries = tuple(
        tuple((float(z.real), float(z.imag)) for z in row) for row in matrix.tolist()
    )
    p.emit(
        "qreg.unitary",
        data,
        data,
        data,
        control,
        matrix=entries,
        indices=(0, 3, 1),
        control_index=0,
    )
    row = schedule_register_program(p.finish(), n_qubits=5)
    initial = torch.randn(32, dtype=DTYPE, generator=torch.Generator().manual_seed(407))
    initial /= initial.norm()
    reference = fq.Circuit(5, dtype=DTYPE).any(
        4, 3, 0, 2, matrix=torch.block_diag(torch.eye(8, dtype=DTYPE), matrix)
    )
    actual = fq.Circuit.from_ir(row.circuit, dtype=DTYPE, inputs=initial).state()
    expected = fq.Circuit.from_ir(
        reference.to_ir(), dtype=DTYPE, inputs=initial
    ).state()
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)


def test_scalar_global_phase_remains_observable_under_control() -> None:
    matrix = cmath.exp(0.73j) * torch.eye(8, dtype=DTYPE)
    row = schedule_register_program(
        model_for(matrix, power=7, controlled=True), n_qubits=5
    )
    initial = torch.full((32,), 1 / 32**0.5, dtype=DTYPE)
    expected = initial.clone()
    expected[1::2] *= cmath.exp(7 * 0.73j)
    actual = (
        fq.Circuit.from_ir(row.circuit, dtype=DTYPE, inputs=initial).state().flatten()
    )
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
