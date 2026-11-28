"""Replace Simulation preparation through unchanged Runtime MPS consumers."""

from types import SimpleNamespace

import pytest

from flagquantum.runtime.executors.jax.mps import lowering, shards
from flagquantum.simulation.mps.jax import batched, kernels
from flagquantum.simulation.numerics import jax as numerics

pytestmark = pytest.mark.unit


@pytest.fixture
def engines():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    np = pytest.importorskip("numpy")
    previous_x64 = jax.config.x64_enabled
    previous_dtype = numerics._set_active_jax_compute_dtype("complex128")
    jax.config.update("jax_enable_x64", True)
    try:
        yield jax, jnp, np
    finally:
        numerics._set_active_jax_compute_dtype(previous_dtype)
        jax.config.update("jax_enable_x64", previous_x64)


@pytest.mark.parametrize("scan", [False, True])
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("precision", [None, "high", "highest"])
def test_lowering_uses_replaced_composition_and_operator_batches(
    engines, monkeypatch, scan, dtype, precision
):
    import flagquantum as fq

    _, jnp, np = engines
    numerics._set_active_jax_compute_dtype(dtype)
    monkeypatch.setenv("FQ_DISABLE_JAX_MPS_CX_SCAN", "0" if scan else "1")
    circuit = fq.Circuit(3)
    for layer in range(2):
        for wire in range(3):
            circuit.ry(wire, theta=0.21 + 0.17 * wire + 0.11 * layer)
            circuit.rz(wire, theta=-0.37 + 0.13 * wire)
        circuit.cx(0, 1).cx(1, 2)
    original = lowering._jax_mps_from_circuit(
        circuit, 3, None, max_bond=4, matmul_precision=precision
    )
    compositions, batches = [], []

    def compose(matrix, previous, matmul_precision):
        compositions.append(matmul_precision)
        return jnp.einsum("ij,jk->ik", matrix, previous, precision=matmul_precision)

    def stack(matrices):
        batches.append(len(matrices))
        return jnp.concatenate([matrix[None, ...] for matrix in matrices], axis=0)

    monkeypatch.setattr(numerics, "_jax_compose_gate_matrices", compose)
    monkeypatch.setattr(numerics, "_jax_stack_gate_matrices", stack)
    replaced = lowering._jax_mps_from_circuit(
        circuit, 3, None, max_bond=4, matmul_precision=precision
    )
    assert compositions and set(compositions) == {precision}
    # Both paths parse the nearest-CX layers only when scan admission is enabled.
    assert batches == ([3, 3] if scan else [])
    for before, after in zip(original, replaced, strict=True):
        assert before.shape == after.shape
        assert str(after.dtype) == dtype
        np.testing.assert_allclose(after, before, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(
        kernels.jax_mps_to_statevector(replaced, precision),
        kernels.jax_mps_to_statevector(original, precision),
        atol=2e-6,
        rtol=2e-6,
    )


@pytest.mark.parametrize("batch_size", [0, 1, 3])
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_rank_initialization_uses_replaced_numerics_and_runtime_device_placement(
    engines, monkeypatch, batch_size, dtype
):
    jax, jnp, np = engines
    calls, placements = [], []
    device = jax.devices("cpu")[0]

    def initial(size, complex_dtype):
        calls.append((size, complex_dtype))
        site = jnp.asarray([1.0 + 0.0j, 0.0j], dtype=complex_dtype)
        return jnp.broadcast_to(site.reshape(1, 1, 2, 1), (size, 1, 2, 1))

    def place(tensor, selected_device):
        placements.append(selected_device)
        return jax.device_put(tensor, selected_device)

    monkeypatch.setattr(batched, "jax_mps_initial_batched_site", initial)
    monkeypatch.setattr(shards, "_jnp_device_put", place)
    plans = [
        SimpleNamespace(rank=2, qubits=(2, 0)),
        SimpleNamespace(rank=0, qubits=(1,)),
        SimpleNamespace(rank=1, qubits=()),
    ]
    result = shards._initialize_jax_mps_rank_tensors(
        n_qubits=3,
        bsz=batch_size,
        shard_plans=plans,
        dtype=getattr(jnp, dtype),
        device=device,
    )
    assert list(result) == [2, 0, 1]
    assert [list(local) for local in result.values()] == [[2, 0], [1], []]
    assert calls == [(batch_size, getattr(jnp, dtype))] * 3
    assert placements == [device] * 3
    expected = np.zeros((batch_size, 1, 2, 1), dtype=dtype)
    expected[:, 0, 0, 0] = 1
    for local in result.values():
        for site in local.values():
            np.testing.assert_array_equal(site, expected)
            assert str(site.dtype) == dtype


def test_gate_composition_preserves_jit_and_parameter_gradient(engines):
    jax, jnp, np = engines
    previous = jnp.asarray([[1, 2j], [3j, 4]], dtype=jnp.complex128)

    def loss(theta):
        matrix = jnp.asarray([[theta, 2.0], [3.0, theta * theta]])
        composed = numerics._jax_compose_gate_matrices(matrix, previous, "highest")
        return jnp.real(jnp.sum(composed))

    theta = 0.37
    # Real sum is theta + 3 + 8 + 4 * theta**2.
    np.testing.assert_allclose(jax.jit(loss)(theta), theta + 11 + 4 * theta**2)
    np.testing.assert_allclose(jax.jit(jax.grad(loss))(theta), 1 + 8 * theta)


@pytest.mark.parametrize("parser", [False, True])
def test_empty_lowering_preserves_optional_engine_admission(monkeypatch, parser):
    def unavailable(name):
        assert name == "jax.numpy"
        raise ModuleNotFoundError("JAX dependency unavailable")

    monkeypatch.setattr(lowering, "import_module", unavailable)
    circuit = SimpleNamespace(to_ir=lambda: [])
    with pytest.raises(ModuleNotFoundError, match="JAX dependency unavailable"):
        if parser:
            lowering._parse_nearest_cx_layers(circuit, 0, "highest")
        else:
            lowering._jax_mps_from_circuit(circuit, 0, None)


def test_empty_rank_assignment_preserves_optional_engine_admission(monkeypatch):
    def unavailable():
        raise RuntimeError("JAX sharded statevector execution requires jax.")

    monkeypatch.setattr(shards, "_require_jax", unavailable)
    with pytest.raises(RuntimeError, match="requires jax"):
        shards._initialize_jax_mps_rank_tensors(
            n_qubits=0, bsz=1, shard_plans=(), dtype=object(), device=None
        )
