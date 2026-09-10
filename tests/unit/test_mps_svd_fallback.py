import torch

from flagquantum.simulation.mps import factorization as mps_factorization


def test_batched_gesvd_failure_retries_isolated_matrices(monkeypatch):
    calls = []

    def fake_svd(matrix, *, full_matrices, driver):
        calls.append((tuple(matrix.shape), full_matrices, driver))
        if matrix.ndim > 2:
            raise torch.linalg.LinAlgError("batched failure")
        size = matrix.shape[-1]
        return (
            torch.eye(size).to(matrix).reshape(size, size),
            torch.ones(size, dtype=matrix.real.dtype),
            torch.eye(size).to(matrix).reshape(size, size),
        )

    monkeypatch.setattr(mps_factorization.torch.linalg, "svd", fake_svd)
    monkeypatch.setattr(mps_factorization, "_is_cuda_tensor", lambda _matrix: True)
    mps_factorization.reset_mps_svd_fallback_stats()

    u, singular, vh = mps_factorization._cuda_svd(
        torch.ones(2, 1, 3, 3),
        driver="gesvd",
    )

    assert u.shape == (2, 1, 3, 3)
    assert singular.shape == (2, 1, 3)
    assert vh.shape == (2, 1, 3, 3)
    assert calls == [
        ((2, 1, 3, 3), False, "gesvd"),
        ((3, 3), False, "gesvd"),
        ((3, 3), False, "gesvd"),
    ]
    assert mps_factorization.mps_svd_fallback_stats() == {
        "requested_driver_failures": 1,
        "gesvd_driver_fallbacks": 0,
        "isolated_gesvd_retries": 1,
        "isolated_gesvd_matrices": 2,
        "cpu_lapack_fallbacks": 0,
        "cpu_lapack_matrices": 0,
        "nonfinite_svd_outputs": 0,
    }


def test_isolated_cuda_failure_uses_strict_cpu_lapack(monkeypatch):
    def fake_svd(matrix, *, full_matrices, driver=None):
        if driver == "gesvd":
            raise torch.linalg.LinAlgError("cuda failure")
        size = matrix.shape[-1]
        return (
            torch.eye(size).to(matrix),
            torch.ones(size, dtype=matrix.real.dtype),
            torch.eye(size).to(matrix),
        )

    monkeypatch.setattr(mps_factorization.torch.linalg, "svd", fake_svd)
    monkeypatch.setattr(mps_factorization, "_is_cuda_tensor", lambda _matrix: True)
    mps_factorization.reset_mps_svd_fallback_stats()

    result = mps_factorization._cuda_svd(
        torch.ones(2, 1, 3, 3),
        driver="gesvd",
    )

    assert result[0].shape == (2, 1, 3, 3)
    expected_vectors = torch.eye(3).expand(2, 1, 3, 3)
    torch.testing.assert_close(result[0], expected_vectors)
    torch.testing.assert_close(result[1], torch.ones(2, 1, 3))
    torch.testing.assert_close(result[2], expected_vectors)
    assert mps_factorization.mps_svd_fallback_stats() == {
        "requested_driver_failures": 1,
        "gesvd_driver_fallbacks": 0,
        "isolated_gesvd_retries": 0,
        "isolated_gesvd_matrices": 0,
        "cpu_lapack_fallbacks": 1,
        "cpu_lapack_matrices": 2,
        "nonfinite_svd_outputs": 0,
    }
