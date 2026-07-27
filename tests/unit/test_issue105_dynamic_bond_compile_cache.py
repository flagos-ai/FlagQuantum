import pytest
import torch

from flagquantum.runtime.backends.mps.site_kernels import (
    SiteKernelBucket,
    SiteKernelCachePolicy,
    apply_rxx_contraction_bucket,
    apply_ry_bucket,
    clear_site_kernel_cache,
    configure_site_kernel_cache,
    prewarm_site_kernel_buckets,
    require_warm_step_regression,
    reset_site_kernel_stats,
    site_kernel_bucket_capacity,
    site_kernel_cache_events,
    site_kernel_stats,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def isolated_cache():
    configure_site_kernel_cache(
        SiteKernelCachePolicy(
            bond_buckets=(1, 2, 4),
            max_entries=2,
            max_accounted_input_bytes=1 << 20,
            backend="eager",
            mode=None,
            policy_id="issue105_test",
        )
    )
    reset_site_kernel_stats(clear_cache=True)
    yield
    clear_site_kernel_cache()
    configure_site_kernel_cache(SiteKernelCachePolicy())
    reset_site_kernel_stats(clear_cache=True)


def _ry(left: int, right: int, *, dtype: torch.dtype = torch.complex64):
    tensor = torch.randn(1, 1, left, 2, right, dtype=dtype)
    matrix = torch.eye(2, dtype=dtype).reshape(1, 1, 2, 2)
    return apply_ry_bucket(tensor, matrix, compiled=True)


def test_cold_start_then_one_hundred_recurring_steps_do_not_recompile():
    expected = _ry(1, 2)
    for _ in range(100):
        _ry(1, 2)
    stats = site_kernel_stats()
    assert expected.shape == (1, 1, 1, 2, 2)
    assert stats["cache_misses"] == 1
    assert stats["dynamo_graphs"] == 1
    assert stats["cache_hits"] == 100
    assert stats["cache_entries"] == 1
    assert [event["event"] for event in site_kernel_cache_events()].count(
        "compile"
    ) == 1


def test_exact_shape_cache_can_compile_more_than_dynamo_default_recompile_limit():
    configure_site_kernel_cache(
        SiteKernelCachePolicy(
            bond_buckets=(1,),
            max_entries=16,
            max_accounted_input_bytes=1 << 20,
            backend="eager",
            mode=None,
            policy_id="more_than_dynamo_default_limit",
        )
    )
    for bucket_size in range(1, 10):
        left = torch.randn(bucket_size, 1, 1, 2, 1, dtype=torch.complex64)
        right = torch.randn(bucket_size, 1, 1, 2, 1, dtype=torch.complex64)
        matrix = (
            torch.eye(4, dtype=torch.complex64)
            .reshape(1, 1, 4, 4)
            .expand(bucket_size, -1, -1, -1)
        )
        output = apply_rxx_contraction_bucket(left, right, matrix, compiled=True)
        assert output.shape == (bucket_size, 1, 2, 2)
    stats = site_kernel_stats()
    assert stats["cache_misses"] == 9
    assert stats["dynamo_graphs"] == 9
    assert stats["cache_entries"] == 9


def test_prewarm_separates_setup_and_recurring_warm_execution():
    events = prewarm_site_kernel_buckets(
        (SiteKernelBucket("ry", left_bond=2, right_bond=4),),
        device="cpu",
    )
    assert any(event["event"] == "compile" and event["prewarm"] for event in events)
    reset_site_kernel_stats()
    _ry(2, 4)
    stats = site_kernel_stats()
    assert stats["cache_hits"] == 1
    assert stats["cache_misses"] == 0
    assert stats["compile_seconds"] == 0
    assert stats["warm_execution_seconds"] >= 0


def test_lru_eviction_is_bounded_and_dtype_is_part_of_key():
    _ry(1, 1, dtype=torch.complex64)
    _ry(1, 1, dtype=torch.complex128)
    _ry(2, 2, dtype=torch.complex64)
    stats = site_kernel_stats()
    assert stats["cache_entries"] == stats["cache_max_entries"] == 2
    assert stats["cache_evictions"] == 1
    assert (
        stats["cache_accounted_input_bytes"] <= stats["cache_max_accounted_input_bytes"]
    )


def test_unsupported_shape_falls_back_without_invalidating_hot_graph():
    hot = _ry(1, 2)
    entries = site_kernel_stats()["cache_entries"]
    rare = _ry(3, 3)
    torch.testing.assert_close(hot, hot)
    assert rare.shape == (1, 1, 3, 2, 3)
    stats = site_kernel_stats()
    assert stats["eager_fallbacks"] == 1
    assert stats["cache_entries"] == entries
    assert any(
        event["event"] == "unsupported_shape_fallback"
        for event in site_kernel_cache_events()
    )


def test_input_larger_than_cache_budget_falls_back_without_exceeding_budget():
    configure_site_kernel_cache(
        SiteKernelCachePolicy(
            bond_buckets=(1, 2, 4),
            max_entries=2,
            max_accounted_input_bytes=8,
            backend="eager",
            mode=None,
            policy_id="tiny_budget",
        )
    )
    _ry(1, 1)
    stats = site_kernel_stats()
    assert stats["eager_fallbacks"] == 1
    assert stats["cache_entries"] == 0
    assert (
        stats["cache_accounted_input_bytes"] <= stats["cache_max_accounted_input_bytes"]
    )


def test_bucket_capacity_bounds_packed_inputs_before_stacking():
    configure_site_kernel_cache(
        SiteKernelCachePolicy(
            bond_buckets=(1, 2, 4),
            max_entries=2,
            max_accounted_input_bytes=384,
            backend="eager",
            mode=None,
            policy_id="bounded_packing",
        )
    )
    left = torch.empty(1, 1, 2, 2, dtype=torch.complex64)
    right = torch.empty(1, 2, 2, 1, dtype=torch.complex64)
    matrix = torch.empty(1, 4, 4, dtype=torch.complex64)
    assert site_kernel_bucket_capacity(left, right, matrix) == 2


def test_dynamo_specialization_limit_accounts_for_microbatch_variants():
    from flagquantum.runtime.backends.mps import site_kernels as kernels

    configure_site_kernel_cache(
        SiteKernelCachePolicy(
            bond_buckets=(1, 2, 4),
            max_entries=9,
            max_accounted_input_bytes=1024,
            backend="eager",
            mode=None,
            policy_id="microbatch_variants",
        )
    )
    with kernels._recompile_limit_context():
        from torch._dynamo import config

        assert config.recompile_limit >= 36
        assert config.accumulated_recompile_limit >= 36


def test_policy_is_in_cache_lifecycle_and_error_fallback_fails_closed():
    _ry(1, 1)
    configure_site_kernel_cache(
        SiteKernelCachePolicy(
            bond_buckets=(1,),
            max_entries=1,
            max_accounted_input_bytes=1024,
            backend="eager",
            mode=None,
            unsupported_shape="error",
            policy_id="strict",
        )
    )
    assert site_kernel_stats()["cache_entries"] == 0
    with pytest.raises(RuntimeError, match="unsupported compiled site-kernel shape"):
        _ry(2, 2)


def test_warm_step_regression_gate_passes_and_rejects():
    report = require_warm_step_regression((0.9, 1.0), (1.0, 1.0), maximum_ratio=1.05)
    assert report["passed"] and report["compiled_to_eager_ratio"] == pytest.approx(0.95)
    with pytest.raises(RuntimeError, match="warm-step regression"):
        require_warm_step_regression((1.2,), (1.0,), maximum_ratio=1.05)
