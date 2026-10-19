"""Which authority answers the fused complex-BMM route question, at each seam.

``complex_einsum_pair`` answers "can the fused layout BMM run here?" with the
gates its build declares: the equation shape table, a gradient requirement, the
device, and ``complex64``. The kernel catalog is not consulted there. That is a
decided position rather than an oversight: ``cudaq-parity-reconciliation.md``
section 7.5 item D3 was ratified as "take MAIN's authorization and fallback, keep
the catalog declaration surface", and the merged function is exactly those gates
with ``_pair_contraction`` on every fallback. The earlier alternative — answer the
device question from the catalog inside this function — was the branch's design
and was not the one ratified.

The catalog declaration surface is not withdrawn, it is scoped. It is the
authority for the tensor-network and density-matrix routes, which consult
``_layout_complex_bmm_declared``, ``_layout_complex_bmm_declared_for``,
``_operand_kernel_axis``, and ``_catalog_declares_kernel``; the replacement proof
lives in ``tests/unit/test_tensor_network_catalog_route.py`` and
``tests/unit/test_density_matrix_catalog_dispatch.py``.

These tests pin both halves: that the declaration answers for the operands it
covers, and that declaring a CPU implementation does not move the layout-BMM route
inside ``complex_einsum_pair``. Re-wiring the catalog into that function fails
here, so it has to arrive as a proposal rather than as a merge detail.
"""

from __future__ import annotations

import ast
import inspect

import pytest
import torch

import flagquantum.simulation.kernel_dispatch as kernel_dispatch
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
    """Add CPU records to the catalog the dispatcher actually reads.

    The declared-route question is asked through
    ``flagquantum.simulation.kernel_dispatch._catalog_declares_kernel``, which is
    the one seam every fused contraction route in Simulation shares: the
    tensor-network, layout-BMM, and density-matrix callers all reach the catalog
    through it. Patching the catalog lookup there is what makes this a
    replacement proof rather than a test of one module's private helper.
    """

    def with_cpu_records(request: KernelRequest, **kwargs: object) -> KernelMatchResult:
        return match_kernel_implementations(
            request,
            semantics=SEMANTICS,
            implementations=(*IMPLEMENTATIONS, *_CPU_IMPLEMENTATIONS),
            evidence=(*EVIDENCE, *_CPU_EVIDENCE),
        )

    monkeypatch.setattr(
        kernel_dispatch, "match_kernel_implementations", with_cpu_records
    )


def _count_fused_route(monkeypatch) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    """Record fused-kernel calls, opening the shape gate to isolate the device axis.

    NUM-002 is narrowed to the one forward complex64 signature it was measured to
    win, and that narrowing is a second authority with its own coverage in
    ``tests/unit/test_real_imag_kernels.py``. These tests measure which authority
    answers the device question, so the shape gate is opened here and the catalog
    declaration below stays the only difference from the control.
    """

    calls: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    monkeypatch.setattr(
        kernels_runtime, "_layout_bmm_signature_supported", lambda *_: True
    )
    monkeypatch.setattr(
        kernels_runtime, "_layout_bmm_dispatch_supported", lambda *_: True
    )

    def counted(
        equation: str, left: torch.Tensor, right: torch.Tensor, layout: object
    ) -> torch.Tensor:
        calls.append((equation, left, right))
        return torch.einsum(equation, left, right)

    monkeypatch.setattr(kernels_runtime, "_fused_layout_bmm", counted)
    return calls


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_a_catalog_declaration_does_not_move_the_layout_bmm_route(
    monkeypatch, dtype: torch.dtype
) -> None:
    """The ratified boundary: this route's device gate is not the catalog's.

    A declaration that covers CPU moves the tensor-network and density-matrix
    seams, and it must leave this one where it is. The shape gate is opened below
    so that the device gate is the only thing between the operands and the fused
    route, which is what makes the comparison sensitive on a CPU host.
    """

    left, right = _operands(dtype)

    before = _count_fused_route(monkeypatch)
    complex_einsum_pair(EQUATION, left, right)
    without_declaration = len(before)

    _declare_cpu_implementations(monkeypatch)
    after = _count_fused_route(monkeypatch)
    complex_einsum_pair(EQUATION, left, right)

    assert _layout_complex_bmm_declared(left, right)
    assert len(after) == without_declaration


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


def test_the_layout_bmm_route_names_no_catalog_predicate() -> None:
    """The same boundary read from the function rather than from a call.

    ``complex_einsum_pair`` decides the device question itself, so it must not
    name any part of the catalog declaration surface. The behavioural test above
    is a reading of this host; this one holds wherever it runs, which is what
    makes a re-wire visible on a host where the device gate would mask it.
    """

    tree = ast.parse(inspect.getsource(complex_einsum_pair))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }

    assert names.isdisjoint(
        {
            "_layout_complex_bmm_declared",
            "_layout_complex_bmm_declared_for",
            "_catalog_declares_kernel",
            "_operand_kernel_axis",
        }
    )
