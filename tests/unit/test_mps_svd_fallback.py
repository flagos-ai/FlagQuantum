import pytest
import torch

from flagquantum.simulation.mps import factorization as mps_factorization

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
@pytest.mark.parametrize("factor_index", [0, 1, 2])
def test_nonfinite_factors_preserve_torch_exception_and_counters(
    monkeypatch: pytest.MonkeyPatch, nonfinite: float, factor_index: int
) -> None:
    def nonfinite_svd(
        matrix: torch.Tensor, *, full_matrices: bool, driver: str | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        factors = (torch.eye(2), torch.ones(2), torch.eye(2))
        factors[factor_index].reshape(-1)[0] = nonfinite
        return factors

    monkeypatch.setattr(torch.linalg, "svd", nonfinite_svd)
    mps_factorization.reset_mps_svd_fallback_stats()
    with pytest.raises(
        RuntimeError, match="MPS SVD returned non-finite factors"
    ) as error:
        mps_factorization._cuda_svd(torch.eye(2), driver=None)
    assert type(error.value) is getattr(torch.linalg, "LinAlgError")
    stats = mps_factorization.mps_svd_fallback_stats()
    assert stats["nonfinite_svd_outputs"] == 1
    assert stats["requested_driver_failures"] == 1


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
        "nonfinite_svd_outputs": 0,
    }


def test_isolated_cuda_failure_fails_closed_without_cpu_retry(monkeypatch):
    calls = []

    def fake_svd(matrix, *, full_matrices, driver=None):
        calls.append((str(matrix.device), tuple(matrix.shape), driver))
        if driver == "gesvd":
            raise torch.linalg.LinAlgError("cuda failure")
        raise AssertionError("CPU LAPACK must not be attempted")

    monkeypatch.setattr(mps_factorization.torch.linalg, "svd", fake_svd)
    monkeypatch.setattr(mps_factorization, "_is_cuda_tensor", lambda _matrix: True)
    mps_factorization.reset_mps_svd_fallback_stats()

    with pytest.raises(
        RuntimeError,
        match="MPS strict SVD failed for batched and isolated CUDA gesvd",
    ) as error:
        mps_factorization._cuda_svd(
            torch.ones(2, 1, 3, 3),
            driver="gesvd",
        )

    assert isinstance(error.value.__cause__, torch.linalg.LinAlgError)
    assert calls == [
        ("cpu", (2, 1, 3, 3), "gesvd"),
        ("cpu", (3, 3), "gesvd"),
    ]
    assert mps_factorization.mps_svd_fallback_stats() == {
        "requested_driver_failures": 1,
        "gesvd_driver_fallbacks": 0,
        "isolated_gesvd_retries": 0,
        "isolated_gesvd_matrices": 0,
        "nonfinite_svd_outputs": 0,
    }
