"""Source-ordered sparse Kraus products preserve phase and differentiability."""

import pytest
import torch

from flagquantum import Circuit
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.qec.resources.fifteen_to_one import FifteenToOneFactory
from flagquantum.simulation.density_matrix import (
    apply_kraus_density,
    density_matrix_from_ir,
    expand_operator,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
@pytest.mark.parametrize("batched_operator", (False, True))
def test_sparse_kraus_density_and_state_gradient(
    dtype: torch.dtype, batched_operator: bool
) -> None:
    generator = torch.Generator().manual_seed(832)
    state = torch.randn(3, 4, 4, dtype=dtype, generator=generator)
    density = (state @ state.mH).requires_grad_()
    matrices = torch.tensor(
        [[[0, 0.2 + 0.3j], [-0.7j, 0]], [[0.4j, 0], [0.1 - 0.2j, 0]]],
        dtype=dtype,
    )
    operators = tuple(matrices)
    if batched_operator:
        operators = tuple(op.expand(3, -1, -1) for op in operators)
    actual = apply_kraus_density(density, operators, (1,), 2)
    expected = torch.zeros_like(density)
    for operator in operators:
        full = expand_operator(operator, (1,), 2, dtype=dtype)
        expected = expected + full @ density @ full.mH
    torch.testing.assert_close(actual, expected)
    actual_gradient = torch.autograd.grad(actual.abs().square().sum(), density)[0]
    expected_gradient = torch.autograd.grad(expected.abs().square().sum(), density)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)


def test_trainable_zero_kraus_entries_retain_dense_derivatives() -> None:
    operator = torch.tensor(
        [[0.7 + 0.1j, 0], [0, -0.2 + 0.5j]],
        dtype=torch.complex128,
        requires_grad=True,
    )
    density = torch.tensor(
        [[[0.4, 0.1 + 0.2j], [0.1 - 0.2j, 0.6]]], dtype=torch.complex128
    )
    actual = apply_kraus_density(density, (operator,), (0,), 1)
    batched = operator.unsqueeze(0)
    expected = torch.bmm(torch.bmm(batched, density), batched.mH)
    actual_gradient = torch.autograd.grad(actual.real.sum(), operator)[0]
    expected_gradient = torch.autograd.grad(expected.real.sum(), operator)[0]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=0, rtol=0)
    assert bool(torch.all(actual_gradient[operator == 0] != 0))


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("distances", ((7, 3, 3), (11, 5, 5), (17, 7, 7)))
@pytest.mark.parametrize("random_input", (False, True))
def test_complete_factory_channels_match_each_source_density_bitwise(
    distances: tuple[int, int, int], random_input: bool
) -> None:
    cirq = pytest.importorskip("cirq")
    np = pytest.importorskip("numpy")
    pytest.importorskip("qualtran")
    from qualtran.surface_code import LogicalErrorModel, QECScheme
    from qualtran.surface_code.fifteen_to_one_factory import _build_factory

    model = LogicalErrorModel(
        physical_error=1e-4,
        qec_scheme=QECScheme(error_rate_scaler=0.1, error_rate_threshold=0.01),
    )
    dx, dz, dm = distances
    original = tuple(
        _build_factory(
            d_X=dx, d_Z=dz, d_m=dm, logical_error_model=model
        ).all_operations()
    )
    native = FifteenToOneFactory(*distances).to_ir(
        physical_error=1e-4, error_rate_scaler=0.1
    )
    # Initial factor preparation and final projection have independent owners.
    expected = (
        cirq.DensityMatrixSimulator(dtype=np.complex128)
        .simulate(cirq.Circuit(original[:5]))
        .final_density_matrix.copy()
    )
    if random_input:
        generator = np.random.default_rng(832)
        state = generator.normal(size=32) + 1j * generator.normal(size=32)
        state /= np.linalg.norm(state)
        expected = np.outer(state, state.conj())
    actual = torch.from_numpy(expected.copy()).unsqueeze(0)
    assert len(original) == len(native.instructions) == 100
    for source_op, instruction in zip(
        original[5:], native.instructions[5:], strict=True
    ):
        assert instruction.matrix is not None
        actual = apply_kraus_density(actual, instruction.matrix, instruction.wires, 5)
        arguments = cirq.ApplyChannelArgs(
            target_tensor=expected.reshape((2,) * 10).copy(),
            out_buffer=np.empty((2,) * 10, dtype=np.complex128),
            auxiliary_buffer0=np.empty((2,) * 10, dtype=np.complex128),
            auxiliary_buffer1=np.empty((2,) * 10, dtype=np.complex128),
            left_axes=tuple(qubit.x for qubit in source_op.qubits),
            right_axes=tuple(qubit.x + 5 for qubit in source_op.qubits),
        )
        expected = cirq.apply_channel(source_op, arguments).reshape(32, 32).copy()
        np.testing.assert_array_equal(actual[0].detach().numpy(), expected)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("distances", ((7, 3, 3), (11, 5, 5), (17, 7, 7)))
def test_normal_factory_entry_preserves_complete_source_density(
    distances: tuple[int, int, int],
) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("qualtran")
    from qualtran.surface_code import FifteenToOne, LogicalErrorModel, QECScheme

    model = LogicalErrorModel(
        physical_error=1e-4,
        qec_scheme=QECScheme(error_rate_scaler=0.1, error_rate_threshold=0.01),
    )
    original = FifteenToOne(*distances)
    ir = FifteenToOneFactory(*distances).to_ir(
        physical_error=1e-4, error_rate_scaler=0.1
    )
    actual = density_matrix_from_ir(ir, dtype=torch.complex128)
    np.testing.assert_array_equal(actual[0].numpy(), original._final_state(model))


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_product_prefix_retains_parameter_batches_and_derivatives(
    dtype: torch.dtype,
) -> None:
    angles = torch.tensor((0.2, -0.4, 0.7), dtype=torch.float64, requires_grad=True)
    cosine, sine = torch.cos(angles / 2), torch.sin(angles / 2)
    ir = CircuitIR(
        2,
        (
            Instruction("h", (1,)),
            Instruction("ry", (0,), params={"theta": angles}),
            Instruction("cx", (0, 1)),
        ),
    )
    actual = density_matrix_from_ir(ir, bsz=3, dtype=dtype)
    vectors = torch.stack((cosine, cosine, sine, sine), dim=-1).to(dtype) / 2**0.5
    expected = vectors.unsqueeze(-1) * vectors.conj().unsqueeze(-2)
    torch.testing.assert_close(actual, expected)
    actual_gradient = torch.autograd.grad(actual.real[:, 3, 3].sum(), angles)[0]
    expected_gradient = torch.autograd.grad(expected.real[:, 3, 3].sum(), angles)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)


def test_circuit_owned_entangled_input_retains_its_state_gradient() -> None:
    state = torch.tensor(((1, 0, 0, 1j),), dtype=torch.complex128, requires_grad=True)
    initial = state / 2**0.5
    circuit = Circuit(2, inputs=initial, dtype=torch.complex128).h(0)
    actual = density_matrix_from_ir(circuit)
    density = initial.unsqueeze(-1) * initial.conj().unsqueeze(-2)
    hadamard = torch.tensor(((1, 1), (1, -1)), dtype=state.dtype) / 2**0.5
    full = torch.kron(hadamard, torch.eye(2, dtype=state.dtype)).unsqueeze(0)
    expected = torch.bmm(torch.bmm(full, density), full.mH)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual_gradient = torch.autograd.grad(
        actual.real[:, 0, 3].sum(), state, retain_graph=True
    )[0]
    expected_gradient = torch.autograd.grad(expected.real[:, 0, 3].sum(), state)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=0, rtol=0)


def test_product_prefix_uses_an_explicit_matrix_override() -> None:
    matrix = torch.diag(torch.tensor((1, -1), dtype=torch.complex128))
    ir = CircuitIR(1, (Instruction("h", (0,), matrix=matrix),))
    actual = density_matrix_from_ir(ir, dtype=torch.complex128)
    expected = torch.tensor((((1, 0), (0, 0)),), dtype=torch.complex128)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
