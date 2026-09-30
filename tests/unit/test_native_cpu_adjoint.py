"""Correctness and rollback contracts for the native CPU adjoint operator."""

from __future__ import annotations

import pytest
import torch

import flagquantum as fq
from flagquantum.runtime.executors.statevector import (
    forward_rzz_segment,
    forward_sweep,
    reverse_adjoint,
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
from flagquantum.runtime.executors.statevector.reverse_adjoint_cx import (
    apply_cpu_cx_adjoint_index,
    cpu_cx_permutation_index,
)
from flagquantum.simulation.native_cpu import (
    fused_cx_adjoint_gather,
    fused_cx_gather_out,
    fused_cx_rotation_segment_adjoint,
    fused_hadamard_block_adjoint_,
    fused_observable_adjoint_seed,
    fused_observable_expectation,
    fused_rotation_adjoint_,
    fused_rotation_block_forward_,
    fused_rotation_segment_adjoint_,
    fused_rzz_segment_adjoint_,
    fused_rzz_segment_forward_,
    native_cpu_adjoint_available,
    native_cpu_cx_adjoint_gather_available,
    native_cpu_cx_gather_available,
    native_cpu_parallel_build_available,
    native_cpu_rotation_available,
    native_cpu_rzz_available,
)
from flagquantum.simulation.native_cpu.rotation import (
    native_cpu_forward_rotation_tile_wires,
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
@pytest.mark.parametrize("batch", (1, 2))
def test_native_observable_boundary_matches_eager_reference(
    dtype: torch.dtype,
    batch: int,
) -> None:
    if not native_cpu_adjoint_available():
        pytest.skip("native CPU extension is unavailable")
    generator = torch.Generator().manual_seed(9181 + batch)
    ket = (
        torch.randn((batch, 64), generator=generator)
        + 1j * torch.randn((batch, 64), generator=generator)
    ).to(dtype)
    weights = torch.randn(
        64,
        generator=generator,
        dtype=torch.float32 if dtype == torch.complex64 else torch.float64,
    )

    value = fused_observable_expectation(ket, weights)
    adjoint = fused_observable_adjoint_seed(ket, weights)

    assert value is not None
    assert adjoint is not None
    tolerance = 2e-5 if dtype == torch.complex64 else 2e-12
    torch.testing.assert_close(
        value,
        (ket.abs().square() * weights.reshape(1, -1)).sum(),
        atol=tolerance,
        rtol=tolerance,
    )
    torch.testing.assert_close(
        adjoint,
        2 * ket * weights.reshape(1, -1),
        atol=tolerance,
        rtol=tolerance,
    )


def test_native_observable_boundary_has_explicit_environment_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ket = torch.ones((1, 4), dtype=torch.complex128)
    weights = torch.ones(4, dtype=torch.float64)
    monkeypatch.setenv("FQ_NATIVE_CPU_OBSERVABLE_BOUNDARY", "0")

    assert fused_observable_expectation(ket, weights) is None
    assert fused_observable_adjoint_seed(ket, weights) is None


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
@pytest.mark.parametrize("index_dtype", (torch.int32, torch.int64))
@pytest.mark.parametrize("batch", (1, 2))
def test_native_cx_gather_out_matches_index_select(
    dtype: torch.dtype,
    index_dtype: torch.dtype,
    batch: int,
) -> None:
    if not native_cpu_cx_gather_available():
        pytest.skip("native CPU extension is not built in this source checkout")
    generator = torch.Generator().manual_seed(9321 + batch)
    state = (
        torch.randn((batch, 64), generator=generator)
        + 1j * torch.randn((batch, 64), generator=generator)
    ).to(dtype)
    index = torch.randperm(64, generator=generator, dtype=index_dtype)
    output = torch.empty_like(state)

    applied = fused_cx_gather_out(state, index, output)

    assert applied
    torch.testing.assert_close(output, torch.index_select(state, 1, index))


def test_native_cx_gather_out_preserves_autograd_and_has_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = torch.arange(4, dtype=torch.int32)
    output = torch.empty((1, 4), dtype=torch.complex128)
    differentiable = torch.ones((1, 4), dtype=torch.complex128, requires_grad=True)

    assert not fused_cx_gather_out(differentiable, index, output)
    monkeypatch.setenv("FQ_NATIVE_CPU_CX_GATHER", "0")
    assert not fused_cx_gather_out(differentiable.detach(), index, output)


def test_forward_sweep_reuses_native_cx_gather_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not native_cpu_cx_gather_available():
        pytest.skip("native CPU extension is not built in this source checkout")
    circuit = fq.Circuit(6, dtype=torch.complex128)
    circuit.h(0).cx(0, 1).cx(1, 2).h(3).cx(2, 4).cx(4, 5)
    monkeypatch.setenv("FQ_NATIVE_CPU_CX_GATHER", "0")
    expected = execute_torch_distributed_statevector(
        circuit, dtype=torch.complex128
    ).shard_state.amplitudes
    calls: list[tuple[int, int]] = []
    native = forward_sweep.fused_cx_gather_out

    def counted(
        state: torch.Tensor,
        index: torch.Tensor,
        output: torch.Tensor,
    ) -> bool:
        calls.append((state.data_ptr(), output.data_ptr()))
        return native(state, index, output)

    monkeypatch.setattr(forward_sweep, "fused_cx_gather_out", counted)
    monkeypatch.setenv("FQ_NATIVE_CPU_CX_GATHER", "1")
    actual = execute_torch_distributed_statevector(
        circuit, dtype=torch.complex128
    ).shard_state.amplitudes

    assert len(calls) == 2
    assert calls[0] == (calls[1][1], calls[1][0])
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)


def test_adjoint_forward_caches_observable_weights_with_isolation_and_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reverse_adjoint._clear_observable_weight_cache()
    theta = torch.tensor(0.17, dtype=torch.float64, requires_grad=True)
    circuit = fq.Circuit(6, dtype=torch.complex128).ry(0, theta).cx(0, 1)
    terms = ((0.7, (0, 1)), (0.2, (2,)))
    calls = 0
    original = reverse_adjoint.z_hamiltonian_weights

    def counted(*args: object, **kwargs: object) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(reverse_adjoint, "z_hamiltonian_weights", counted)
    first = execute_torch_distributed_statevector_reverse(
        circuit, observable_terms=terms
    ).value
    second = execute_torch_distributed_statevector_reverse(
        circuit, observable_terms=terms
    ).value

    assert calls == 1
    torch.testing.assert_close(first, second)

    execute_torch_distributed_statevector_reverse(
        circuit, observable_terms=((0.8, (0, 1)), (0.2, (2,)))
    )
    assert calls == 2

    monkeypatch.setenv("FQ_STATEVECTOR_ADJOINT_OBSERVABLE_CACHE", "0")
    execute_torch_distributed_statevector_reverse(circuit, observable_terms=terms)
    assert calls == 3
    reverse_adjoint._clear_observable_weight_cache()


def test_adjoint_observable_weight_cache_evicts_least_recent_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reverse_adjoint._clear_observable_weight_cache()
    monkeypatch.setattr(reverse_adjoint, "_OBSERVABLE_WEIGHT_CACHE_MAX_ENTRIES", 1)
    theta = torch.tensor(0.11, dtype=torch.float64, requires_grad=True)
    circuit = fq.Circuit(4, dtype=torch.complex128).rx(0, theta)
    calls = 0
    original = reverse_adjoint.z_hamiltonian_weights

    def counted(*args: object, **kwargs: object) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(reverse_adjoint, "z_hamiltonian_weights", counted)
    first_terms = ((0.7, (0,)),)
    second_terms = ((0.2, (1,)),)
    execute_torch_distributed_statevector_reverse(circuit, observable_terms=first_terms)
    execute_torch_distributed_statevector_reverse(
        circuit, observable_terms=second_terms
    )
    execute_torch_distributed_statevector_reverse(circuit, observable_terms=first_terms)

    assert calls == 3
    reverse_adjoint._clear_observable_weight_cache()


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
@pytest.mark.parametrize(
    "wires",
    (
        (0, 2),
        (4, 1, 3),
        (3, 0, 4, 2),
        (5, 1, 4, 0, 3, 2),
        tuple(range(8)),
    ),
)
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


def test_forward_rotation_tile_width_has_explicit_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert native_cpu_forward_rotation_tile_wires(15) == 4
    assert native_cpu_forward_rotation_tile_wires(22) == 8

    monkeypatch.setenv("FQ_NATIVE_CPU_FORWARD_WIDE_ROTATION_TILES", "0")

    assert native_cpu_forward_rotation_tile_wires(22) == 6


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
def test_native_rotation_block_fuses_preceding_shared_rzz(
    dtype: torch.dtype,
) -> None:
    if not native_cpu_rotation_available():
        pytest.skip("native CPU extension is not built in this source checkout")
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    generator = torch.Generator().manual_seed(2031)
    n_wires = 4
    state = (
        torch.randn((2, 1 << n_wires), generator=generator)
        + 1j * torch.randn((2, 1 << n_wires), generator=generator)
    ).to(dtype)
    angle = torch.tensor(0.23, dtype=real_dtype)
    rzz_angles = angle.expand(3).contiguous()
    first_wires = torch.tensor((0, 1, 2), dtype=torch.int64)
    second_wires = torch.tensor((1, 2, 3), dtype=torch.int64)
    matrices = torch.stack(
        tuple(_rotation_matrix("rx", angle, dtype) for _ in range(n_wires))
    ).contiguous()
    phase = torch.exp(-0.5j * angle).to(dtype)
    inverse_phase = phase.conj()
    rzz_matrix = torch.diag(torch.stack((phase, inverse_phase, inverse_phase, phase)))
    expected = state.clone()
    for wires in zip(first_wires.tolist(), second_wires.tolist(), strict=True):
        expected = _apply_matrix(expected, rzz_matrix, wires, n_wires)
    for matrix, wire in zip(matrices, range(n_wires), strict=True):
        expected = _apply_matrix(expected, matrix, (wire,), n_wires)

    applied = fused_rotation_block_forward_(
        state,
        matrices,
        torch.arange(n_wires, dtype=torch.int64),
        n_wires=n_wires,
        rzz_angles=rzz_angles,
        rzz_first_wires=first_wires,
        rzz_second_wires=second_wires,
    )

    assert applied
    tolerance = 3e-5 if dtype == torch.complex64 else 2e-12
    torch.testing.assert_close(state, expected, atol=tolerance, rtol=tolerance)


def test_specialized_forward_rotation_arithmetic_has_explicit_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not native_cpu_rotation_available():
        pytest.skip("native CPU extension is not built in this source checkout")
    angle = torch.tensor(0.31, dtype=torch.float64)
    matrices = torch.stack(
        (_rotation_matrix("rx", angle, torch.complex128),) * 2
    ).contiguous()
    initial = torch.arange(4, dtype=torch.float64).to(torch.complex128).reshape(1, 4)
    specialized = initial.clone()
    rollback = initial.clone()

    monkeypatch.setenv("FQ_NATIVE_CPU_FORWARD_SPECIALIZED_ROTATIONS", "1")
    assert fused_rotation_block_forward_(
        specialized, matrices, torch.tensor((0, 1)), n_wires=2
    )
    monkeypatch.setenv("FQ_NATIVE_CPU_FORWARD_SPECIALIZED_ROTATIONS", "0")
    assert fused_rotation_block_forward_(
        rollback, matrices, torch.tensor((0, 1)), n_wires=2
    )

    torch.testing.assert_close(specialized, rollback, atol=2e-12, rtol=2e-12)


def test_native_rotation_block_rejects_nonshared_rzz_without_mutation() -> None:
    if not native_cpu_rotation_available():
        pytest.skip("native CPU extension is not built in this source checkout")
    state = torch.ones((1, 8), dtype=torch.complex128)
    initial = state.clone()
    matrices = torch.eye(2, dtype=torch.complex128).expand(2, -1, -1).contiguous()

    applied = fused_rotation_block_forward_(
        state,
        matrices,
        torch.tensor((0, 1), dtype=torch.int64),
        n_wires=3,
        rzz_angles=torch.tensor((0.1, 0.2), dtype=torch.float64),
        rzz_first_wires=torch.tensor((0, 1), dtype=torch.int64),
        rzz_second_wires=torch.tensor((1, 2), dtype=torch.int64),
    )

    assert not applied
    torch.testing.assert_close(state, initial)


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
@pytest.mark.parametrize("wire_count", (2, 6, 11))
def test_native_hadamard_block_adjoint_matches_two_sequential_states(
    dtype: torch.dtype, wire_count: int
) -> None:
    if not native_cpu_rotation_available():
        pytest.skip("native CPU extension is not built in this source checkout")
    generator = torch.Generator().manual_seed(1709 + wire_count)
    width = 1 << wire_count
    ket = (
        torch.randn((2, width), generator=generator)
        + 1j * torch.randn((2, width), generator=generator)
    ).to(dtype)
    adjoint = (
        torch.randn((2, width), generator=generator)
        + 1j * torch.randn((2, width), generator=generator)
    ).to(dtype)
    matrix = torch.tensor(((1.0, 1.0), (1.0, -1.0)), dtype=dtype) / 2**0.5
    expected_ket = ket.clone()
    expected_adjoint = adjoint.clone()
    for wire in range(wire_count):
        expected_ket = _apply_matrix(expected_ket, matrix, (wire,), wire_count)
        expected_adjoint = _apply_matrix(expected_adjoint, matrix, (wire,), wire_count)

    applied = fused_hadamard_block_adjoint_(
        ket,
        adjoint,
        torch.arange(wire_count, dtype=torch.int64),
        n_wires=wire_count,
    )

    assert applied
    tolerance = 3e-5 if dtype == torch.complex64 else 3e-12
    torch.testing.assert_close(ket, expected_ket, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(
        adjoint, expected_adjoint, atol=tolerance, rtol=tolerance
    )


def test_native_hadamard_block_adjoint_uses_wide_tile_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_NATIVE_CPU_ADJOINT_WIDE_TILES", "0")
    ket = torch.ones((1, 4), dtype=torch.complex128)
    adjoint = torch.ones_like(ket)

    applied = fused_hadamard_block_adjoint_(
        ket,
        adjoint,
        torch.tensor((0, 1), dtype=torch.int64),
        n_wires=2,
    )

    assert not applied
    torch.testing.assert_close(ket, torch.ones_like(ket))
    torch.testing.assert_close(adjoint, torch.ones_like(adjoint))


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


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_native_rotation_segment_fuses_preceding_hadamards(
    dtype: torch.dtype,
) -> None:
    if not native_cpu_adjoint_available():
        pytest.skip("native CPU extension is not built in this source checkout")
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    generator = torch.Generator().manual_seed(7129)
    n_wires = 6
    initial_ket = (
        torch.randn((2, 1 << n_wires), generator=generator)
        + 1j * torch.randn((2, 1 << n_wires), generator=generator)
    ).to(dtype)
    initial_adjoint = (
        torch.randn((2, 1 << n_wires), generator=generator)
        + 1j * torch.randn((2, 1 << n_wires), generator=generator)
    ).to(dtype)
    angle = torch.tensor(0.23, dtype=real_dtype)
    angles = angle.expand(n_wires).contiguous()
    wires = torch.arange(n_wires, dtype=torch.int64)
    rzz_angles = angle.expand(n_wires - 1).contiguous()
    first_wires = torch.arange(n_wires - 1, dtype=torch.int64)
    second_wires = first_wires + 1

    expected_ket = initial_ket.clone()
    expected_adjoint = initial_adjoint.clone()
    expected_gradients = fused_rotation_segment_adjoint_(
        expected_ket,
        expected_adjoint,
        angles,
        torch.zeros(n_wires, dtype=torch.int64),
        wires,
        n_wires=n_wires,
        aggregate_shared_parameter=True,
        rzz_angles=rzz_angles,
        rzz_first_wires=first_wires,
        rzz_second_wires=second_wires,
    )
    hadamard = torch.tensor(((1.0, 1.0), (1.0, -1.0)), dtype=dtype) / torch.sqrt(
        torch.tensor(2.0, dtype=real_dtype)
    )
    for wire in range(n_wires):
        expected_ket = _apply_matrix(expected_ket, hadamard, (wire,), n_wires)
        expected_adjoint = _apply_matrix(expected_adjoint, hadamard, (wire,), n_wires)

    ket = initial_ket.clone()
    adjoint = initial_adjoint.clone()
    gradients = fused_rotation_segment_adjoint_(
        ket,
        adjoint,
        angles,
        torch.zeros(n_wires, dtype=torch.int64),
        wires,
        n_wires=n_wires,
        aggregate_shared_parameter=True,
        rzz_angles=rzz_angles,
        rzz_first_wires=first_wires,
        rzz_second_wires=second_wires,
        fuse_preceding_hadamards=True,
    )

    assert expected_gradients is not None
    assert gradients is not None
    tolerance = 6e-5 if dtype == torch.complex64 else 5e-12
    torch.testing.assert_close(
        gradients, expected_gradients, atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(ket, expected_ket, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(
        adjoint, expected_adjoint, atol=tolerance, rtol=tolerance
    )


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
@pytest.mark.parametrize("n_wires", (6, 12))
def test_native_rotation_segment_fuses_observable_seed(
    dtype: torch.dtype, n_wires: int
) -> None:
    if not native_cpu_adjoint_available():
        pytest.skip("native CPU extension is not built in this source checkout")
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    generator = torch.Generator().manual_seed(8137)
    initial_ket = (
        torch.randn((2, 1 << n_wires), generator=generator)
        + 1j * torch.randn((2, 1 << n_wires), generator=generator)
    ).to(dtype)
    weights = torch.randn(1 << n_wires, generator=generator, dtype=real_dtype)
    angles = torch.linspace(0.07, 0.31, n_wires, dtype=real_dtype)
    kinds = torch.tensor(tuple(wire % 3 for wire in range(n_wires)), dtype=torch.int64)
    wires = torch.arange(n_wires, dtype=torch.int64)

    expected_ket = initial_ket.clone()
    expected_adjoint = 2 * initial_ket * weights.reshape(1, -1)
    expected_gradients = fused_rotation_segment_adjoint_(
        expected_ket,
        expected_adjoint,
        angles,
        kinds,
        wires,
        n_wires=n_wires,
    )
    ket = initial_ket.clone()
    adjoint = torch.empty_like(ket)
    gradients = fused_rotation_segment_adjoint_(
        ket,
        adjoint,
        angles,
        kinds,
        wires,
        n_wires=n_wires,
        observable_weights=weights,
    )

    assert expected_gradients is not None
    assert gradients is not None
    tolerance = 7e-5 if dtype == torch.complex64 else 5e-12
    torch.testing.assert_close(
        gradients, expected_gradients, atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(ket, expected_ket, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(
        adjoint, expected_adjoint, atol=tolerance, rtol=tolerance
    )


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_native_rotation_segment_fast_path_matches_exact_rollback(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
) -> None:
    if not native_cpu_adjoint_available():
        pytest.skip("native CPU extension is not built in this source checkout")
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    generator = torch.Generator().manual_seed(16127)
    initial_ket = (
        torch.randn((2, 128), generator=generator)
        + 1j * torch.randn((2, 128), generator=generator)
    ).to(dtype)
    initial_adjoint = (
        torch.randn((2, 128), generator=generator)
        + 1j * torch.randn((2, 128), generator=generator)
    ).to(dtype)
    names = ("rz", "ry", "rx") * 5 + ("rx", "rz")
    wires = tuple(wire for wire in (6, 5, 4, 3, 2) for _ in range(3)) + (1, 0)
    angles = torch.linspace(0.03, 0.39, len(names), dtype=real_dtype)
    gate_kinds = torch.tensor(
        tuple({"rx": 0, "ry": 1, "rz": 2}[name] for name in names),
        dtype=torch.int64,
    )
    wire_tensor = torch.tensor(wires, dtype=torch.int64)

    monkeypatch.setenv("FQ_NATIVE_CPU_ADJOINT_EULER_TRIPLES", "0")
    rollback_ket = initial_ket.clone()
    rollback_adjoint = initial_adjoint.clone()
    rollback_gradients = fused_rotation_segment_adjoint_(
        rollback_ket,
        rollback_adjoint,
        angles,
        gate_kinds,
        wire_tensor,
        n_wires=7,
    )

    monkeypatch.setenv("FQ_NATIVE_CPU_ADJOINT_EULER_TRIPLES", "1")
    monkeypatch.setenv("FQ_NATIVE_CPU_ADJOINT_FLAT_PAIR_SIMD", "0")
    nested_ket = initial_ket.clone()
    nested_adjoint = initial_adjoint.clone()
    nested_gradients = fused_rotation_segment_adjoint_(
        nested_ket,
        nested_adjoint,
        angles,
        gate_kinds,
        wire_tensor,
        n_wires=7,
    )

    monkeypatch.setenv("FQ_NATIVE_CPU_ADJOINT_FLAT_PAIR_SIMD", "1")
    fast_ket = initial_ket.clone()
    fast_adjoint = initial_adjoint.clone()
    fast_gradients = fused_rotation_segment_adjoint_(
        fast_ket,
        fast_adjoint,
        angles,
        gate_kinds,
        wire_tensor,
        n_wires=7,
    )

    assert rollback_gradients is not None
    assert nested_gradients is not None
    assert fast_gradients is not None
    tolerance = 5e-5 if dtype == torch.complex64 else 4e-12
    torch.testing.assert_close(
        fast_gradients, rollback_gradients, atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(fast_ket, rollback_ket, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(
        fast_adjoint, rollback_adjoint, atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(
        fast_gradients, nested_gradients, atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(fast_ket, nested_ket, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(
        fast_adjoint, nested_adjoint, atol=tolerance, rtol=tolerance
    )


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_native_cx_rotation_adjoint_matches_sequential_boundary(
    dtype: torch.dtype,
) -> None:
    if not native_cpu_adjoint_available():
        pytest.skip("native CPU extension is not built in this source checkout")
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    generator = torch.Generator().manual_seed(19723)
    initial_ket = (
        torch.randn((2, 128), generator=generator)
        + 1j * torch.randn((2, 128), generator=generator)
    ).to(dtype)
    initial_adjoint = torch.empty_like(initial_ket)
    weights = torch.randn(128, generator=generator, dtype=real_dtype)
    angles = torch.linspace(0.03, 0.29, 9, dtype=real_dtype)
    kinds = torch.tensor((2, 1, 0) * 3, dtype=torch.int64)
    wires = torch.tensor((6, 6, 6, 3, 3, 3, 0, 0, 0), dtype=torch.int64)
    index = cpu_cx_permutation_index(
        initial_ket,
        controls=(5, 4, 3, 2, 1, 0),
        targets=(6, 5, 4, 3, 2, 1),
        n_wires=7,
    )

    reference_ket, reference_adjoint = apply_cpu_cx_adjoint_index(
        initial_ket.clone(),
        2 * initial_ket * weights.reshape(1, -1),
        index,
    )
    reference_gradients = fused_rotation_segment_adjoint_(
        reference_ket,
        reference_adjoint,
        angles,
        kinds,
        wires,
        n_wires=7,
    )
    fused = fused_cx_rotation_segment_adjoint(
        initial_ket,
        initial_adjoint,
        index,
        angles,
        kinds,
        wires,
        n_wires=7,
        observable_weights=weights,
    )
    terminal_input = initial_ket.clone()
    terminal = fused_cx_rotation_segment_adjoint(
        terminal_input,
        initial_adjoint,
        index,
        angles,
        kinds,
        wires,
        n_wires=7,
        observable_weights=weights,
        restore_state=False,
    )

    assert reference_gradients is not None
    assert fused is not None
    assert terminal is not None
    gradients, ket, adjoint = fused
    terminal_gradients, _, _ = terminal
    tolerance = 7e-5 if dtype == torch.complex64 else 5e-12
    torch.testing.assert_close(
        gradients, reference_gradients, atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(ket, reference_ket, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(
        adjoint, reference_adjoint, atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(
        terminal_gradients, reference_gradients, atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(terminal_input, initial_ket, atol=0, rtol=0)


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


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_native_rotation_segment_wide_tiles_match_legacy_tiles(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
) -> None:
    if not native_cpu_adjoint_available():
        pytest.skip("native CPU extension is not built in this source checkout")
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    generator = torch.Generator().manual_seed(5317)
    initial_ket = (
        torch.randn((2, 128), generator=generator)
        + 1j * torch.randn((2, 128), generator=generator)
    ).to(dtype)
    initial_adjoint = (
        torch.randn((2, 128), generator=generator)
        + 1j * torch.randn((2, 128), generator=generator)
    ).to(dtype)
    names = ("rz", "ry", "rx") * 6
    wires = tuple(wire for wire in (6, 5, 4, 3, 2, 1) for _ in range(3))
    angles = torch.tensor(
        tuple(0.03 * (index + 1) for index in range(len(names))), dtype=real_dtype
    )
    gate_kinds = torch.tensor(
        tuple({"rx": 0, "ry": 1, "rz": 2}[name] for name in names),
        dtype=torch.int64,
    )
    wire_tensor = torch.tensor(wires, dtype=torch.int64)

    monkeypatch.setenv("FQ_NATIVE_CPU_ADJOINT_WIDE_TILES", "0")
    legacy_ket = initial_ket.clone()
    legacy_adjoint = initial_adjoint.clone()
    legacy_gradients = fused_rotation_segment_adjoint_(
        legacy_ket,
        legacy_adjoint,
        angles,
        gate_kinds,
        wire_tensor,
        n_wires=7,
    )

    monkeypatch.setenv("FQ_NATIVE_CPU_ADJOINT_WIDE_TILES", "1")
    wide_ket = initial_ket.clone()
    wide_adjoint = initial_adjoint.clone()
    wide_gradients = fused_rotation_segment_adjoint_(
        wide_ket,
        wide_adjoint,
        angles,
        gate_kinds,
        wire_tensor,
        n_wires=7,
    )

    assert legacy_gradients is not None
    assert wide_gradients is not None
    tolerance = 7e-5 if dtype == torch.complex64 else 8e-12
    torch.testing.assert_close(
        wide_gradients, legacy_gradients, atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(wide_ket, legacy_ket, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(
        wide_adjoint, legacy_adjoint, atol=tolerance, rtol=tolerance
    )


def test_native_rotation_segment_wide_tiles_cover_complete_large_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not native_cpu_adjoint_available():
        pytest.skip("native CPU extension is not built in this source checkout")
    generator = torch.Generator().manual_seed(9137)
    initial_ket = (
        torch.randn((1, 256), generator=generator)
        + 1j * torch.randn((1, 256), generator=generator)
    ).to(torch.complex128)
    initial_adjoint = (
        torch.randn((1, 256), generator=generator)
        + 1j * torch.randn((1, 256), generator=generator)
    ).to(torch.complex128)
    names = ("rz", "ry", "rx", "rz", "ry", "rx", "rz") * 8
    wires = tuple(wire for wire in range(7, -1, -1) for _ in range(7))
    angles = torch.tensor(
        tuple(0.01 * (index + 1) for index in range(len(names))),
        dtype=torch.float64,
    )
    gate_kinds = torch.tensor(
        tuple({"rx": 0, "ry": 1, "rz": 2}[name] for name in names),
        dtype=torch.int64,
    )
    wire_tensor = torch.tensor(wires, dtype=torch.int64)

    expected_ket = initial_ket.clone()
    expected_adjoint = initial_adjoint.clone()
    expected_gradients = []
    for name, wire, angle in zip(names, wires, angles, strict=True):
        gradient, expected_ket, expected_adjoint = _reference(
            expected_ket,
            expected_adjoint,
            _rotation_matrix(name, angle, torch.complex128),
            name=name,
            wire=wire,
            n_wires=8,
        )
        expected_gradients.append(gradient)

    monkeypatch.setenv("FQ_NATIVE_CPU_ADJOINT_WIDE_TILES", "0")
    assert (
        fused_rotation_segment_adjoint_(
            initial_ket.clone(),
            initial_adjoint.clone(),
            angles,
            gate_kinds,
            wire_tensor,
            n_wires=8,
        )
        is None
    )

    monkeypatch.setenv("FQ_NATIVE_CPU_ADJOINT_WIDE_TILES", "1")
    actual_ket = initial_ket.clone()
    actual_adjoint = initial_adjoint.clone()
    actual_gradients = fused_rotation_segment_adjoint_(
        actual_ket,
        actual_adjoint,
        angles,
        gate_kinds,
        wire_tensor,
        n_wires=8,
    )

    assert actual_gradients is not None
    torch.testing.assert_close(
        actual_gradients, torch.stack(expected_gradients), atol=1e-11, rtol=1e-11
    )
    torch.testing.assert_close(actual_ket, expected_ket, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(actual_adjoint, expected_adjoint, atol=1e-11, rtol=1e-11)


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


def test_reverse_sweep_fuses_qaoa_hadamard_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not native_cpu_adjoint_available():
        pytest.skip("native CPU extension is not built in this source checkout")
    theta = torch.tensor(0.19, dtype=torch.float64, requires_grad=True)
    circuit = fq.Circuit(6, dtype=torch.complex128)
    for wire in range(6):
        circuit.h(wire)
    for wire in range(5):
        circuit.rzz(wire, wire + 1, theta)
    for wire in range(6):
        circuit.rx(wire, theta)

    monkeypatch.setenv("FQ_NATIVE_CPU_ADJOINT_RZZ_H_FUSION", "0")
    baseline = execute_torch_distributed_statevector_reverse(circuit)
    baseline.backward()
    expected_value = baseline.value.detach().clone()
    expected_gradient = theta.grad.detach().clone()
    theta.grad = None
    fusion_flags: list[bool] = []
    native = reverse_adjoint_rotation_segment.fused_rotation_segment_adjoint_

    def counted(*args: object, **kwargs: object) -> torch.Tensor | None:
        fusion_flags.append(bool(kwargs.get("fuse_preceding_hadamards")))
        return native(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        reverse_adjoint_rotation_segment,
        "fused_rotation_segment_adjoint_",
        counted,
    )
    monkeypatch.setenv("FQ_NATIVE_CPU_ADJOINT_RZZ_H_FUSION", "1")
    actual = execute_torch_distributed_statevector_reverse(circuit)
    actual.backward()

    assert fusion_flags == [True]
    torch.testing.assert_close(actual.value, expected_value, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(theta.grad, expected_gradient, atol=2e-12, rtol=2e-12)


def test_reverse_sweep_fuses_observable_seed_with_first_rotation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not native_cpu_adjoint_available():
        pytest.skip("native CPU extension is not built in this source checkout")
    theta = torch.tensor(0.17, dtype=torch.float64, requires_grad=True)
    circuit = fq.Circuit(6, dtype=torch.complex128)
    for wire in range(6):
        circuit.h(wire)
    for wire in range(6):
        circuit.rx(wire, theta)

    monkeypatch.setenv("FQ_NATIVE_CPU_OBSERVABLE_ROTATION_BOUNDARY", "0")
    baseline = execute_torch_distributed_statevector_reverse(circuit)
    baseline.backward()
    expected_value = baseline.value.detach().clone()
    expected_gradient = theta.grad.detach().clone()
    theta.grad = None
    observed_weights: list[bool] = []
    native = reverse_adjoint_rotation_segment.fused_rotation_segment_adjoint_

    def counted(*args: object, **kwargs: object) -> torch.Tensor | None:
        observed_weights.append(kwargs.get("observable_weights") is not None)
        return native(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        reverse_adjoint_rotation_segment,
        "fused_rotation_segment_adjoint_",
        counted,
    )
    monkeypatch.setenv("FQ_NATIVE_CPU_OBSERVABLE_ROTATION_BOUNDARY", "1")
    actual = execute_torch_distributed_statevector_reverse(circuit)
    actual.backward()

    assert observed_weights == [True]
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


def test_reverse_sweep_fuses_cx_and_rotation_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not native_cpu_adjoint_available():
        pytest.skip("native CPU extension is not built in this source checkout")
    parameters = torch.linspace(
        0.07, 0.31, 18, dtype=torch.float64, requires_grad=True
    ).reshape(6, 3)
    parameters.retain_grad()
    circuit = fq.Circuit(6, dtype=torch.complex128)
    for qubit in range(6):
        circuit.rx(qubit, parameters[qubit, 0])
        circuit.ry(qubit, parameters[qubit, 1])
        circuit.rz(qubit, parameters[qubit, 2])
    for qubit in range(5):
        circuit.cx(qubit, qubit + 1)

    monkeypatch.setenv("FQ_NATIVE_CPU_ADJOINT_CX_ROTATION_FUSION", "0")
    monkeypatch.setenv("FQ_NATIVE_CPU_ADJOINT_TERMINAL_NO_RESTORE", "0")
    baseline = execute_torch_distributed_statevector_reverse(circuit)
    baseline.backward()
    expected_value = baseline.value.detach().clone()
    assert parameters.grad is not None
    expected_gradient = parameters.grad.detach().clone()
    parameters.grad = None
    calls = 0
    native = reverse_adjoint_rotation_segment.fused_cx_rotation_segment_adjoint

    def counted(*args: object, **kwargs: object) -> tuple[torch.Tensor, ...] | None:
        nonlocal calls
        calls += 1
        return native(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        reverse_adjoint_rotation_segment,
        "fused_cx_rotation_segment_adjoint",
        counted,
    )
    monkeypatch.setenv("FQ_NATIVE_CPU_ADJOINT_CX_ROTATION_FUSION", "1")
    monkeypatch.setenv("FQ_NATIVE_CPU_ADJOINT_TERMINAL_NO_RESTORE", "1")
    actual = execute_torch_distributed_statevector_reverse(circuit)
    actual.backward()

    assert calls == 1
    torch.testing.assert_close(actual.value, expected_value, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(
        parameters.grad, expected_gradient, atol=2e-12, rtol=2e-12
    )


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
    native = reverse_adjoint_sweep.fused_hadamard_block_adjoint_

    def counted(
        ket: torch.Tensor,
        adjoint: torch.Tensor,
        wires: torch.Tensor,
        *,
        n_wires: int,
    ) -> bool:
        nonlocal calls
        calls += 1
        return native(ket, adjoint, wires, n_wires=n_wires)

    monkeypatch.setattr(reverse_adjoint_sweep, "fused_hadamard_block_adjoint_", counted)
    monkeypatch.setenv("FQ_NATIVE_CPU_ONE_QUBIT_LAYER", "1")
    actual = execute_torch_distributed_statevector_reverse(circuit)
    actual.backward()

    assert calls == 1
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
