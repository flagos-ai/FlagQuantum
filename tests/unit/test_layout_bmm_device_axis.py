"""The catalog, not a device literal, decides the fused complex-BMM route.

``complex_einsum_pair`` used to answer "can the fused layout BMM run here?" with
``left.is_cuda`` and ``left.dtype != torch.complex64``. Two lines below, the
same function authorized ``FQKI-TRITON-NUM-002-A`` through the kernel catalog, so
the device decision and the kernel decision were made by different authorities
and disagreed by construction: on CPU the catalog was never consulted at all.

These tests pin the replacement. The route is now selected by asking the catalog
whether an evidenced implementation covers the operands' device and precision.
The proof is a replacement: declaring a CPU implementation for the semantic makes
the fused route run on CPU with no edit to ``real_imag_kernels.py``. The control
differs only in that declaration, so the declaration is what moved the decision.
"""

from __future__ import annotations

import ast
import inspect

import pytest
import torch

import flagquantum.simulation.complex_bmm_dispatch as complex_bmm_dispatch
import flagquantum.simulation.real_imag_kernels as kernels_runtime
from flagquantum.kernels.catalog import (
    EVIDENCE,
    IMPLEMENTATIONS,
    SEMANTICS,
    KernelEvidence,
    KernelImplementation,
    KernelMatchResult,
    KernelRequest,
    match_kernel_implementations,
)
from flagquantum.simulation.complex_bmm_dispatch import _layout_complex_bmm_declared
from flagquantum.simulation.real_imag_kernels import complex_einsum_pair

pytestmark = pytest.mark.unit

EQUATION = "zab,zbc->zac"
_CORRECTNESS_TEST = "tests/unit/test_layout_bmm_device_axis.py::test_placeholder"
_CPU_IMPLEMENTATIONS = tuple(
    KernelImplementation(
        implementation_id=f"FQKI-PYTORCH-NUM-002-{suffix}",
        semantic_id="numerics.matmul.complex_batched_layout",
        provider="pytorch",
        module="flagquantum.kernels.pytorch.complex_bmm",
        symbol="fused_cpu_layout_bmm",
        devices=("cpu",),
        dtypes=(dtype,),
        layouts=("explicit_strided_batch",),
        directions=("forward",),
        addressing=("local",),
        maturity="experimental",
    )
    for suffix, dtype in (("B", "complex64"), ("C", "complex128"))
)
_CPU_EVIDENCE = tuple(
    KernelEvidence(
        evidence_id=record.implementation_id.replace("FQKI-", "FQKE-", 1),
        implementation_id=record.implementation_id,
        correctness_tests=(_CORRECTNESS_TEST,),
    )
    for record in _CPU_IMPLEMENTATIONS
)


def _operands(dtype: torch.dtype = torch.complex64):
    generator = torch.Generator().manual_seed(20240930)
    left = torch.randn(2, 3, 4, dtype=dtype, generator=generator)
    right = torch.randn(2, 4, 5, dtype=dtype, generator=generator)
    return left, right


def _declare_cpu_implementations(monkeypatch) -> None:
    """Add CPU records to the catalog the dispatcher actually reads."""

    def with_cpu_records(
        request: KernelRequest, **kwargs: object
    ) -> KernelMatchResult:
        return match_kernel_implementations(
            request,
            semantics=SEMANTICS,
            implementations=(*IMPLEMENTATIONS, *_CPU_IMPLEMENTATIONS),
            evidence=(*EVIDENCE, *_CPU_EVIDENCE),
        )

    monkeypatch.setattr(
        complex_bmm_dispatch, "match_kernel_implementations", with_cpu_records
    )


def _count_fused_route(monkeypatch) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    calls: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    monkeypatch.setattr(kernels_runtime, "_FUSED_WORKING_SET_BYTES", 0)

    def counted(
        equation: str, left: torch.Tensor, right: torch.Tensor, layout: object
    ) -> torch.Tensor:
        calls.append((equation, left, right))
        return torch.einsum(equation, left, right)

    monkeypatch.setattr(kernels_runtime, "_fused_layout_bmm", counted)
    return calls


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_a_declared_device_and_precision_select_the_fused_route(
    monkeypatch, dtype: torch.dtype
) -> None:
    """Replacing the catalog content moves the decision without touching code.

    Both axes are covered so that neither ``is_cuda`` nor ``complex64`` can be
    reinstated as a literal without a behavioural failure: the declaration is
    the only difference between this and the control below.
    """

    left, right = _operands(dtype)
    calls = _count_fused_route(monkeypatch)
    _declare_cpu_implementations(monkeypatch)

    assert _layout_complex_bmm_declared(left, right)
    actual = complex_einsum_pair(EQUATION, left, right)

    assert [equation for equation, _, _ in calls] == [EQUATION]
    torch.testing.assert_close(actual, torch.einsum(EQUATION, left, right))


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_the_same_call_without_the_declaration_stays_on_the_eager_route(
    monkeypatch, dtype: torch.dtype
) -> None:
    """The control: only the declaration above differs from this test."""

    left, right = _operands(dtype)
    calls = _count_fused_route(monkeypatch)

    assert not _layout_complex_bmm_declared(left, right)
    actual = complex_einsum_pair(EQUATION, left, right)

    assert calls == []
    torch.testing.assert_close(actual, torch.einsum(EQUATION, left, right))


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_no_declared_record_covers_cpu_at_either_precision(
    dtype: torch.dtype,
) -> None:
    """The shipped catalog declares CUDA-only complex64 records."""

    left, right = _operands(dtype)

    assert not _layout_complex_bmm_declared(left, right)


def test_an_undeclared_device_can_never_reach_a_cataloged_implementation() -> None:
    left, right = _operands()
    on_meta = left.to(device="meta"), right.to(device="meta")

    assert not _layout_complex_bmm_declared(*on_meta)


def test_operands_on_different_devices_are_not_declared(monkeypatch) -> None:
    """A declared CPU record must not cover a pair that straddles two devices."""

    left, right = _operands()
    _declare_cpu_implementations(monkeypatch)

    assert _layout_complex_bmm_declared(left, _operands()[0])
    assert not _layout_complex_bmm_declared(left, right.to(device="meta"))


def test_operands_at_different_precisions_are_not_declared(monkeypatch) -> None:
    """A complex64 record must not stand in for a mixed-precision pair."""

    left, right = _operands()
    _declare_cpu_implementations(monkeypatch)

    assert _layout_complex_bmm_declared(left, right)
    assert not _layout_complex_bmm_declared(left, right.to(torch.complex128))


def test_non_complex_operands_bypass_the_catalog_entirely() -> None:
    """A real contraction is not this semantic, so no device question is asked.

    ``match_kernel_implementations`` would refuse ``float32`` as a dtype
    mismatch rather than as a malformed request, so returning before the query
    keeps the refusal honest.
    """

    left, right = _operands()

    assert not _layout_complex_bmm_declared(left.real, right.real)
    assert complex_einsum_pair(EQUATION, left.real, right.real).dtype == torch.float32


def test_the_einsum_route_does_not_read_a_cuda_literal() -> None:
    """The device decision must come from the catalog, not from a device test.

    This is the regression guard for the two early returns that used
    ``left.is_cuda`` and ``left.dtype``; a device literal reintroduced here would
    silently bypass the catalog again while every behavioural test still passed
    on a CPU-only host.
    """

    attributes = {
        node.attr
        for node in ast.walk(ast.parse(inspect.getsource(complex_einsum_pair)))
        if isinstance(node, ast.Attribute)
    }

    assert attributes.isdisjoint({"is_cuda", "is_cpu", "device", "dtype"})
