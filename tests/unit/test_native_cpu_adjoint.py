"""Correctness and rollback contracts for the native CPU adjoint operator."""

from __future__ import annotations

import pytest
import torch

import flagquantum as fq
from flagquantum.runtime.executors.statevector import (
    forward_rzz_segment,
    forward_sweep,
    reverse_adjoint_cx,
    reverse_adjoint_rotation_segment,
    reverse_adjoint_sweep,
)
from flagquantum.runtime.executors.statevector.forward_executor import (
    execute_torch_distributed_statevector,
)
from flagquantum.runtime.executors.statevector.reverse import (
    execute_torch_distributed_statevector_reverse,
)
from flagquantum.simulation.native_cpu import (
    fused_cx_adjoint_gather,
    fused_rotation_adjoint_,
    fused_rotation_block_forward_,
    fused_rotation_segment_adjoint_,
    fused_rzz_segment_adjoint_,
    fused_rzz_segment_forward_,
    native_cpu_adjoint_available,
    native_cpu_cx_adjoint_gather_available,
    native_cpu_parallel_build_available,
    native_cpu_rotation_available,
    native_cpu_rzz_available,
)
from flagquantum.simulation.statevector.operations import _apply_matrix


def test_native_cpu_build_preserves_torch_parallel_backend() -> None:
    if not native_cpu_adjoint_available():
        pytest.skip("native CPU adjoint extension is unavailable")
    torch_backend = torch.__config__.parallel_info()
    if "ATen parallel backend: OpenMP" in torch_backend:
        assert native_cpu_parallel_build_available()


pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
@pytest.mark.parametrize("index_dtype", (torch.int32, torch.int64))
def test_native_cx_adjoint_gather_matches_two_index_selects(
    dtype: torch.dtype,
    index_dtype: torch.dtype,
) -> None:
    if not native_cpu_cx_adjoint_gather_available():
        pytest.skip("native CPU extension is not built in this source checkout")
    generator = torch.Generator().manual_seed(8123)
    ket = (
        torch.randn((2, 64), generator=generator)
        + 1j * torch.randn((2, 64), generator=generator)
    ).to(dtype)
    adjoint = (
        torch.randn((2, 64), generator=generator)
        + 1j * torch.randn((2, 64), generator=generator)
    ).to(dtype)
    index = torch.randperm(64, generator=generator, dtype=index_dtype)

    gathered = fused_cx_adjoint_gather(ket, adjoint, index)

    assert gathered is not None
    torch.testing.assert_close(gathered[0], torch.index_select(ket, 1, index))
    torch.testing.assert_close(gathered[1], torch.index_select(adjoint, 1, index))


def test_native_cx_adjoint_gather_has_explicit_environment_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_NATIVE_CPU_CX_ADJOINT_GATHER", "0")
    ket = torch.ones((1, 4), dtype=torch.complex128)
    index = torch.arange(4, dtype=torch.int32)

    assert fused_cx_adjoint_gather(ket, ket, index) is None


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
@pytest.mark.parametrize("wires", ((0, 2), (4, 1, 3), (3, 0, 4, 2), (5, 1, 4, 0, 3, 2)))
def test_native_rotation_block_matches_sequential_pytorch(
    dtype: torch.dtype, wires: tuple[int, ...]
) -> None:
    if not native_cpu_rotation_available():
        pytest.skip("native CPU extension is not built in this source checkout")
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    generator = torch.Generator().manual_seed(1307 + sum(wires))
    n_wires = max(5, len(wires))
    width = 1 << n_wires
    state = (
        torch.randn((2, width), generator=generator)
        + 1j * torch.randn((2, width), generator=generator)
    ).to(dtype)
    matrices = torch.stack(
        tuple(
            _rotation_matrix(
                ("rx", "ry", "rz")[index % 3],
                torch.tensor(0.17 * (index + 1), dtype=real_dtype),
                dtype,
            )
            for index in range(len(wires))
        )
    ).contiguous()
    expected = state.clone()
    for matrix, wire in zip(matrices, wires, strict=True):
        expected = _apply_matrix(expected, matrix, (wire,), n_wires)

    applied = fused_rotation_block_forward_(
        state,
        matrices,
        torch.tensor(wires, dtype=torch.int64),
        n_wires=n_wires,
    )

    assert applied
    tolerance = 3e-5 if dtype == torch.complex64 else 2e-12
    torch.testing.assert_close(state, expected, atol=tolerance, rtol=tolerance)


def test_native_rotation_block_has_explicit_environment_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_NATIVE_CPU_ROTATION_FUSION", "0")
    state = torch.ones((1, 4), dtype=torch.complex128)
    matrices = torch.eye(2, dtype=torch.complex128).expand(2, -1, -1).contiguous()

    applied = fused_rotation_block_forward_(
        state,
        matrices,
        torch.tensor((0, 1), dtype=torch.int64),
        n_wires=2,
    )

    assert not applied
    torch.testing.assert_close(state, torch.ones_like(state))


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
@pytest.mark.parametrize("pairs", (((0, 1), (1, 2), (2, 3), (3, 4)), ((0, 3), (2, 4))))
def test_native_rzz_segment_forward_matches_sequential_pytorch(
    dtype: torch.dtype, pairs: tuple[tuple[int, int], ...]
) -> None:
    if not native_cpu_rzz_available():
        pytest.skip("native CPU extension is not built in this source checkout")
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    generator = torch.Generator().manual_seed(1811 + len(pairs))
    state = (
        torch.randn((2, 32), generator=generator)
        + 1j * torch.randn((2, 32), generator=generator)
    ).to(dtype)
    angles = torch.tensor(
        tuple(
            0.19 if len(pairs) > 2 else 0.11 * (index + 1)
            for index in range(len(pairs))
        ),
        dtype=real_dtype,
    )
    expected = state.clone()
    for angle, wires in zip(angles, pairs, strict=True):
        expected = _apply_matrix(
            expected,
            _rotation_matrix("rzz", angle, dtype),
            wires,
            5,
        )

    applied = fused_rzz_segment_forward_(
        state,
        angles,
        torch.tensor(tuple(pair[0] for pair in pairs), dtype=torch.int64),
        torch.tensor(tuple(pair[1] for pair in pairs), dtype=torch.int64),
        n_wires=5,
    )

    assert applied
    tolerance = 3e-5 if dtype == torch.complex64 else 2e-12
    torch.testing.assert_close(state, expected, atol=tolerance, rtol=tolerance)


def test_native_rzz_segment_forward_has_explicit_environment_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_NATIVE_CPU_RZZ_FUSION", "0")
    state = torch.ones((1, 8), dtype=torch.complex128)
    applied = fused_rzz_segment_forward_(
        state,
        torch.tensor((0.1, 0.1), dtype=torch.float64),
        torch.tensor((0, 1), dtype=torch.int64),
        torch.tensor((1, 2), dtype=torch.int64),
        n_wires=3,
    )

    assert not applied
    torch.testing.assert_close(state, torch.ones_like(state))


def test_forward_sweep_fuses_contiguous_rzz_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not native_cpu_rzz_available():
        pytest.skip("native CPU extension is not built in this source checkout")
    circuit = fq.Circuit(6, dtype=torch.complex128)
    for wire in range(5):
        circuit.rzz(wire, wire + 1, 0.23)
    monkeypatch.setenv("FQ_NATIVE_CPU_RZZ_FUSION", "0")
    expected = execute_torch_distributed_statevector(
        circuit, dtype=torch.complex128
    ).shard_state.amplitudes
    calls = 0
    native = forward_rzz_segment.fused_rzz_segment_forward_

    def counted(*args: object, **kwargs: object) -> bool:
        nonlocal calls
        calls += 1
        return native(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(forward_rzz_segment, "fused_rzz_segment_forward_", counted)
    monkeypatch.setenv("FQ_NATIVE_CPU_RZZ_FUSION", "1")
    actual = execute_torch_distributed_statevector(
        circuit, dtype=torch.complex128
    ).shard_state.amplitudes

    assert calls == 1
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)


def test_forward_sweep_fuses_composed_rotation_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not native_cpu_rotation_available():
        pytest.skip("native CPU extension is not built in this source checkout")
    circuit = fq.Circuit(8, dtype=torch.complex128)
    for wire in range(8):
        circuit.rx(wire, 0.11 + wire * 0.01)
        circuit.ry(wire, -0.07 + wire * 0.02)
        circuit.rz(wire, 0.03 - wire * 0.01)
    monkeypatch.setenv("FQ_NATIVE_CPU_ROTATION_FUSION", "0")
    expected = execute_torch_distributed_statevector(
        circuit, dtype=torch.complex128
    ).shard_state.amplitudes
    calls = 0
    native = forward_sweep.fused_rotation_block_forward_

    def counted(
        state: torch.Tensor,
        matrices: torch.Tensor,
        wires: torch.Tensor,
        *,
        n_wires: int,
    ) -> bool:
        nonlocal calls
        calls += 1
        return native(state, matrices, wires, n_wires=n_wires)

    monkeypatch.setattr(forward_sweep, "fused_rotation_block_forward_", counted)
    monkeypatch.setenv("FQ_NATIVE_CPU_ROTATION_FUSION", "1")

    actual = execute_torch_distributed_statevector(
        circuit, dtype=torch.complex128
    ).shard_state.amplitudes

    assert calls == 2
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)


def test_forward_sweep_fuses_single_rotation_layer_with_explicit_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not native_cpu_rotation_available():
        pytest.skip("native CPU extension is not built in this source checkout")
    circuit = fq.Circuit(4, dtype=torch.complex128)
    for wire in range(4):
        circuit.rx(wire, 0.1 + wire * 0.02)
    monkeypatch.setenv("FQ_NATIVE_CPU_ONE_QUBIT_LAYER", "0")
    expected = execute_torch_distributed_statevector(
        circuit, dtype=torch.complex128
    ).shard_state.amplitudes
    calls = 0
    native = forward_sweep.fused_rotation_block_forward_

    def counted(
        state: torch.Tensor,
        matrices: torch.Tensor,
        wires: torch.Tensor,
        *,
        n_wires: int,
    ) -> bool:
        nonlocal calls
        calls += 1
        return native(state, matrices, wires, n_wires=n_wires)

    monkeypatch.setattr(forward_sweep, "fused_rotation_block_forward_", counted)
    monkeypatch.setenv("FQ_NATIVE_CPU_ONE_QUBIT_LAYER", "1")
    actual = execute_torch_distributed_statevector(
        circuit, dtype=torch.complex128
    ).shard_state.amplitudes

    assert calls == 1
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)


def _rotation_matrix(
    name: str, theta: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    cosine = torch.cos(theta / 2)
    sine = torch.sin(theta / 2)
    if name == "rx":
        matrix = torch.stack(
            (
                torch.stack((cosine + 0j, -1j * sine)),
                torch.stack((-1j * sine, cosine + 0j)),
            )
        )
    elif name == "ry":
        matrix = torch.stack(
            (torch.stack((cosine, -sine)), torch.stack((sine, cosine)))
        )
    elif name == "rz":
        matrix = torch.diag(
            torch.stack((torch.exp(-0.5j * theta), torch.exp(0.5j * theta)))
        )
    else:
        negative = torch.exp(-0.5j * theta)
        positive = torch.exp(0.5j * theta)
        matrix = torch.diag(torch.stack((negative, positive, positive, negative)))
    return matrix.to(dtype=dtype).contiguous()


def _reference(
    ket: torch.Tensor,
    adjoint: torch.Tensor,
    matrix: torch.Tensor,
    *,
    name: str,
    wire: int,
    n_wires: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    stride = 1 << (n_wires - wire - 1)
    ket_pairs = ket.reshape(ket.shape[0], -1, 2, stride)
    adjoint_pairs = adjoint.reshape(adjoint.shape[0], -1, 2, stride)
    ket_zero, ket_one = ket_pairs[:, :, 0], ket_pairs[:, :, 1]
    adjoint_zero, adjoint_one = adjoint_pairs[:, :, 0], adjoint_pairs[:, :, 1]

    def cross(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left.real * right.imag - left.imag * right.real

    if name == "rx":
        gradient = 0.5 * (
            cross(adjoint_zero, ket_one).sum() + cross(adjoint_one, ket_zero).sum()
        )
    elif name == "ry":
        gradient = 0.5 * (
            (adjoint_one.conj() * ket_zero).real.sum()
            - (adjoint_zero.conj() * ket_one).real.sum()
        )
    else:
        gradient = 0.5 * (
            cross(adjoint_zero, ket_zero).sum() - cross(adjoint_one, ket_one).sum()
        )

    inverse = matrix.mH
    previous_ket = torch.einsum("ij,abjs->abis", inverse, ket_pairs).reshape_as(ket)
    previous_adjoint = torch.einsum("ij,abjs->abis", inverse, adjoint_pairs).reshape_as(
        adjoint
    )
    return gradient, previous_ket, previous_adjoint


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
@pytest.mark.parametrize("name", ("rx", "ry", "rz"))
@pytest.mark.parametrize("wire", (0, 2, 4))
def test_native_rotation_adjoint_matches_pytorch_reference(
    dtype: torch.dtype, name: str, wire: int
) -> None:
    if not native_cpu_adjoint_available():
        pytest.skip("native CPU extension is not built in this source checkout")
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    generator = torch.Generator().manual_seed(1701 + wire)
    ket = (
        torch.randn((2, 32), generator=generator)
        + 1j * torch.randn((2, 32), generator=generator)
    ).to(dtype)
    adjoint = (
        torch.randn((2, 32), generator=generator)
        + 1j * torch.randn((2, 32), generator=generator)
    ).to(dtype)
    matrix = _rotation_matrix(name, torch.tensor(0.31, dtype=real_dtype), dtype)
    expected = _reference(
        ket.clone(),
        adjoint.clone(),
        matrix,
        name=name,
        wire=wire,
        n_wires=5,
    )

    gradient = fused_rotation_adjoint_(
        ket, adjoint, matrix, name=name, wire=wire, n_wires=5
    )

    assert gradient is not None
    tolerance = 2e-5 if dtype == torch.complex64 else 1e-12
    torch.testing.assert_close(gradient, expected[0], atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(ket, expected[1], atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(adjoint, expected[2], atol=tolerance, rtol=tolerance)


def test_native_rotation_adjoint_has_explicit_environment_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_NATIVE_CPU_ADJOINT", "0")
    ket = torch.ones((1, 2), dtype=torch.complex128)
    adjoint = torch.ones_like(ket)
    matrix = torch.eye(2, dtype=torch.complex128)

    result = fused_rotation_adjoint_(ket, adjoint, matrix, name="ry", wire=0, n_wires=1)

    assert result is None
    torch.testing.assert_close(ket, torch.ones_like(ket))
    torch.testing.assert_close(adjoint, torch.ones_like(adjoint))


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
@pytest.mark.parametrize("aggregate", (False, True))
def test_native_rotation_segment_adjoint_matches_sequential_reference(
    dtype: torch.dtype,
    aggregate: bool,
) -> None:
    if not native_cpu_adjoint_available():
        pytest.skip("native CPU extension is not built in this source checkout")
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    generator = torch.Generator().manual_seed(9871)
    ket = (
        torch.randn((2, 64), generator=generator)
        + 1j * torch.randn((2, 64), generator=generator)
    ).to(dtype)
    adjoint = (
        torch.randn((2, 64), generator=generator)
        + 1j * torch.randn((2, 64), generator=generator)
    ).to(dtype)
    names = ("rz", "ry", "rx", "rz", "ry", "rx", "rz", "ry", "rx")
    wires = (4, 4, 4, 2, 2, 2, 0, 0, 0)
    angles = torch.tensor(
        tuple(0.07 * (index + 1) for index in range(len(names))), dtype=real_dtype
    )
    matrices = torch.stack(
        tuple(
            _rotation_matrix(
                name,
                angles[index],
                dtype,
            )
            for index, name in enumerate(names)
        )
    ).contiguous()
    expected_ket = ket.clone()
    expected_adjoint = adjoint.clone()
    expected_gradients = []
    for name, wire, matrix in zip(names, wires, matrices, strict=True):
        gradient, expected_ket, expected_adjoint = _reference(
            expected_ket,
            expected_adjoint,
            matrix,
            name=name,
            wire=wire,
            n_wires=6,
        )
        expected_gradients.append(gradient)

    gradients = fused_rotation_segment_adjoint_(
        ket,
        adjoint,
        angles,
        torch.tensor(tuple({"rx": 0, "ry": 1, "rz": 2}[name] for name in names)),
        torch.tensor(wires, dtype=torch.int64),
        n_wires=6,
        aggregate_shared_parameter=aggregate,
    )

    assert gradients is not None
    tolerance = 4e-5 if dtype == torch.complex64 else 3e-12
    torch.testing.assert_close(
        gradients,
        (
            torch.stack(expected_gradients).sum().reshape(1)
            if aggregate
            else torch.stack(expected_gradients)
        ),
        atol=tolerance,
        rtol=tolerance,
    )
    torch.testing.assert_close(ket, expected_ket, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(
        adjoint, expected_adjoint, atol=tolerance, rtol=tolerance
    )


def test_native_rotation_segment_adjoint_has_explicit_environment_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_NATIVE_CPU_ROTATION_SEGMENT", "0")
    ket = torch.ones((1, 4), dtype=torch.complex128)
    adjoint = torch.ones_like(ket)
    angles = torch.zeros(2, dtype=torch.float64)

    gradients = fused_rotation_segment_adjoint_(
        ket,
        adjoint,
        angles,
        torch.tensor((0, 1), dtype=torch.int64),
        torch.tensor((0, 1), dtype=torch.int64),
        n_wires=2,
    )

    assert gradients is None
    torch.testing.assert_close(ket, torch.ones_like(ket))
    torch.testing.assert_close(adjoint, torch.ones_like(adjoint))


def test_reverse_sweep_fuses_shared_rotation_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not native_cpu_adjoint_available():
        pytest.skip("native CPU extension is not built in this source checkout")
    theta = torch.tensor(0.19, dtype=torch.float64, requires_grad=True)
    circuit = fq.Circuit(6, dtype=torch.complex128)
    for wire in range(6):
        circuit.h(wire)
    for wire in range(6):
        circuit.rx(wire, theta)

    monkeypatch.setenv("FQ_NATIVE_CPU_ROTATION_SEGMENT", "1")
    monkeypatch.setenv("FQ_NATIVE_CPU_SHARED_ROTATION_GRADIENT", "0")
    baseline = execute_torch_distributed_statevector_reverse(circuit)
    baseline.backward()
    expected_value = baseline.value.detach().clone()
    expected_gradient = theta.grad.detach().clone()
    theta.grad = None
    calls = 0
    aggregate_flags: list[bool] = []
    native = reverse_adjoint_rotation_segment.fused_rotation_segment_adjoint_

    def counted(*args: object, **kwargs: object) -> torch.Tensor | None:
        nonlocal calls
        calls += 1
        aggregate_flags.append(bool(kwargs.get("aggregate_shared_parameter")))
        return native(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        reverse_adjoint_rotation_segment,
        "fused_rotation_segment_adjoint_",
        counted,
    )
    monkeypatch.setenv("FQ_NATIVE_CPU_SHARED_ROTATION_GRADIENT", "1")
    actual = execute_torch_distributed_statevector_reverse(circuit)
    actual.backward()

    assert calls == 1
    assert aggregate_flags == [True]
    torch.testing.assert_close(actual.value, expected_value, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(theta.grad, expected_gradient, atol=2e-12, rtol=2e-12)


def test_reverse_sweep_uses_native_dual_cx_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not native_cpu_cx_adjoint_gather_available():
        pytest.skip("native CPU extension is not built in this source checkout")
    theta = torch.tensor(0.19, dtype=torch.float64, requires_grad=True)
    circuit = fq.Circuit(6, dtype=torch.complex128)
    circuit.ry(0, theta)
    for wire in range(5):
        circuit.cx(wire, wire + 1)

    monkeypatch.setenv("FQ_NATIVE_CPU_CX_ADJOINT_GATHER", "0")
    baseline = execute_torch_distributed_statevector_reverse(circuit)
    baseline.backward()
    expected_value = baseline.value.detach().clone()
    expected_gradient = theta.grad.detach().clone()
    theta.grad = None
    calls = 0
    native = reverse_adjoint_cx.fused_cx_adjoint_gather

    def counted(
        ket: torch.Tensor,
        adjoint: torch.Tensor,
        index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        nonlocal calls
        calls += 1
        return native(ket, adjoint, index)

    monkeypatch.setattr(reverse_adjoint_cx, "fused_cx_adjoint_gather", counted)
    monkeypatch.setenv("FQ_NATIVE_CPU_CX_ADJOINT_GATHER", "1")
    actual = execute_torch_distributed_statevector_reverse(circuit)
    actual.backward()

    assert calls == 1
    torch.testing.assert_close(actual.value, expected_value, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(theta.grad, expected_gradient, atol=2e-12, rtol=2e-12)


def test_reverse_sweep_fuses_fixed_hadamard_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not native_cpu_rotation_available():
        pytest.skip("native CPU extension is not built in this source checkout")
    theta = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    circuit = fq.Circuit(6, dtype=torch.complex128)
    for wire in range(6):
        circuit.h(wire)
    circuit.ry(0, theta)

    monkeypatch.setenv("FQ_NATIVE_CPU_ONE_QUBIT_LAYER", "0")
    baseline = execute_torch_distributed_statevector_reverse(circuit)
    baseline.backward()
    expected_value = baseline.value.detach().clone()
    expected_gradient = theta.grad.detach().clone()
    theta.grad = None
    calls = 0
    native = reverse_adjoint_sweep.fused_rotation_block_forward_

    def counted(
        state: torch.Tensor,
        matrices: torch.Tensor,
        wires: torch.Tensor,
        *,
        n_wires: int,
    ) -> bool:
        nonlocal calls
        calls += 1
        return native(state, matrices, wires, n_wires=n_wires)

    monkeypatch.setattr(reverse_adjoint_sweep, "fused_rotation_block_forward_", counted)
    monkeypatch.setenv("FQ_NATIVE_CPU_ONE_QUBIT_LAYER", "1")
    actual = execute_torch_distributed_statevector_reverse(circuit)
    actual.backward()

    assert calls == 4
    torch.testing.assert_close(actual.value, expected_value, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(theta.grad, expected_gradient, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
@pytest.mark.parametrize("wires", ((0, 1), (0, 4), (3, 1)))
def test_native_rzz_adjoint_matches_pytorch_reference(
    dtype: torch.dtype, wires: tuple[int, int]
) -> None:
    if not native_cpu_adjoint_available():
        pytest.skip("native CPU extension is not built in this source checkout")
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    generator = torch.Generator().manual_seed(2903 + sum(wires))
    ket = (
        torch.randn((2, 32), generator=generator)
        + 1j * torch.randn((2, 32), generator=generator)
    ).to(dtype)
    adjoint = (
        torch.randn((2, 32), generator=generator)
        + 1j * torch.randn((2, 32), generator=generator)
    ).to(dtype)
    matrix = _rotation_matrix("rzz", torch.tensor(0.31, dtype=real_dtype), dtype)
    cross = adjoint.real * ket.imag - adjoint.imag * ket.real
    indices = torch.arange(32)
    first = (indices >> (4 - wires[0])) & 1
    second = (indices >> (4 - wires[1])) & 1
    signs = (1 - 2 * (first ^ second)).to(cross.dtype)
    expected_gradient = 0.5 * (cross * signs).sum()
    inverse = torch.where(signs > 0, matrix[0, 0].conj(), matrix[1, 1].conj()).reshape(
        1, -1
    )
    expected_ket = ket * inverse
    expected_adjoint = adjoint * inverse

    gradient = fused_rotation_adjoint_(
        ket,
        adjoint,
        matrix,
        name="rzz",
        wire=wires[0],
        second_wire=wires[1],
        n_wires=5,
    )

    assert gradient is not None
    tolerance = 2e-5 if dtype == torch.complex64 else 1e-12
    torch.testing.assert_close(
        gradient, expected_gradient, atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(ket, expected_ket, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(
        adjoint, expected_adjoint, atol=tolerance, rtol=tolerance
    )


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_native_rzz_segment_matches_sequential_reference(dtype: torch.dtype) -> None:
    if not native_cpu_adjoint_available():
        pytest.skip("native CPU extension is not built in this source checkout")
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    generator = torch.Generator().manual_seed(4109)
    original_ket = (
        torch.randn((2, 32), generator=generator)
        + 1j * torch.randn((2, 32), generator=generator)
    ).to(dtype)
    original_adjoint = (
        torch.randn((2, 32), generator=generator)
        + 1j * torch.randn((2, 32), generator=generator)
    ).to(dtype)
    angles = torch.tensor((0.19, -0.37, 0.23), dtype=real_dtype)
    first_wires = torch.tensor((0, 3, 1), dtype=torch.int64)
    second_wires = torch.tensor((4, 1, 2), dtype=torch.int64)
    expected_ket = original_ket.clone()
    expected_adjoint = original_adjoint.clone()
    expected_gradients = []
    for angle, first, second in zip(angles, first_wires, second_wires, strict=True):
        matrix = _rotation_matrix("rzz", angle, dtype)
        gradient = fused_rotation_adjoint_(
            expected_ket,
            expected_adjoint,
            matrix,
            name="rzz",
            wire=int(first),
            second_wire=int(second),
            n_wires=5,
        )
        assert gradient is not None
        expected_gradients.append(gradient)
    ket = original_ket.clone()
    adjoint = original_adjoint.clone()

    gradients = fused_rzz_segment_adjoint_(
        ket,
        adjoint,
        angles,
        first_wires,
        second_wires,
        n_wires=5,
    )

    assert gradients is not None
    tolerance = 3e-5 if dtype == torch.complex64 else 2e-12
    torch.testing.assert_close(
        gradients,
        torch.stack(expected_gradients),
        atol=tolerance,
        rtol=tolerance,
    )
    torch.testing.assert_close(ket, expected_ket, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(
        adjoint, expected_adjoint, atol=tolerance, rtol=tolerance
    )


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_native_shared_rzz_path_aggregates_gradient(dtype: torch.dtype) -> None:
    if not native_cpu_adjoint_available():
        pytest.skip("native CPU extension is not built in this source checkout")
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    generator = torch.Generator().manual_seed(5113)
    original_ket = (
        torch.randn((2, 32), generator=generator)
        + 1j * torch.randn((2, 32), generator=generator)
    ).to(dtype)
    original_adjoint = (
        torch.randn((2, 32), generator=generator)
        + 1j * torch.randn((2, 32), generator=generator)
    ).to(dtype)
    angles = torch.full((4,), 0.29, dtype=real_dtype)
    first_wires = torch.tensor((3, 2, 1, 0), dtype=torch.int64)
    second_wires = torch.tensor((4, 3, 2, 1), dtype=torch.int64)
    expected_ket = original_ket.clone()
    expected_adjoint = original_adjoint.clone()
    expected_gradient = torch.zeros((), dtype=real_dtype)
    for angle, first, second in zip(angles, first_wires, second_wires, strict=True):
        gradient = fused_rotation_adjoint_(
            expected_ket,
            expected_adjoint,
            _rotation_matrix("rzz", angle, dtype),
            name="rzz",
            wire=int(first),
            second_wire=int(second),
            n_wires=5,
        )
        assert gradient is not None
        expected_gradient += gradient
    ket = original_ket.clone()
    adjoint = original_adjoint.clone()

    gradients = fused_rzz_segment_adjoint_(
        ket,
        adjoint,
        angles,
        first_wires,
        second_wires,
        n_wires=5,
        aggregate_shared_parameter=True,
    )

    assert gradients is not None
    tolerance = 3e-5 if dtype == torch.complex64 else 2e-12
    torch.testing.assert_close(
        gradients[0], expected_gradient, atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(
        gradients[1:], torch.zeros_like(gradients[1:]), atol=0, rtol=0
    )
    torch.testing.assert_close(ket, expected_ket, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(
        adjoint, expected_adjoint, atol=tolerance, rtol=tolerance
    )


@pytest.mark.parametrize("name", ("custom",))
def test_native_rotation_adjoint_rejects_unsupported_gates(name: str) -> None:
    ket = torch.ones((1, 4), dtype=torch.complex128)
    adjoint = torch.ones_like(ket)
    matrix = torch.eye(2, dtype=torch.complex128)

    assert (
        fused_rotation_adjoint_(ket, adjoint, matrix, name=name, wire=0, n_wires=2)
        is None
    )
