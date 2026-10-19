"""Catalog dispatch contracts for the density-matrix batched matrix product."""

from __future__ import annotations

import pytest
import torch

import flagquantum.simulation.kernel_dispatch as kernel_dispatch
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
from flagquantum.simulation import density_matrix_dispatch as dispatch
from flagquantum.simulation.density_matrix import (
    apply_kraus_density,
    apply_unitary_density,
)
from flagquantum.simulation.density_matrix_dispatch import (
    _density_bmm,
    _density_matmul_declared,
    _density_matmul_kernel_match,
    _require_density_matmul_kernel,
    density_matmul_route_stats,
    reset_density_matmul_route_stats,
)
from flagquantum.simulation.kernel_dispatch import (
    _catalog_declares_kernel,
    _operand_kernel_axis,
)

pytestmark = pytest.mark.unit

_CUDA_AXIS = ("cuda", "complex64")


@pytest.fixture(autouse=True)
def _reset_route_counters() -> None:
    reset_density_matmul_route_stats()


def _operands() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1301)
    left = torch.randn(3, 4, 4, dtype=torch.complex64, generator=generator)
    right = torch.randn(3, 4, 4, dtype=torch.complex64, generator=generator)
    return left, right


def _patch_device_axis(monkeypatch: pytest.MonkeyPatch, axis: tuple[str, str]) -> None:
    monkeypatch.setattr(dispatch, "_operand_kernel_axis", lambda operands: axis)


def test_density_matmul_dispatch_binds_exact_catalog_implementation() -> None:
    implementation = _require_density_matmul_kernel(
        device_type="cuda",
        dtype="complex64",
    )

    assert implementation.semantic_id == "numerics.matmul.complex_batched"
    assert implementation.implementation_id == "FQKI-TRITON-NUM-001-A"
    assert implementation.symbol == "fused_complex_bmm"
    assert implementation.layouts == ("batched_matrix",)
    assert implementation.directions == ("forward", "backward")


@pytest.mark.parametrize(
    ("device_type", "dtype", "mismatch"),
    (("cpu", "complex64", "device"), ("cuda", "complex128", "dtype")),
)
def test_density_matmul_dispatch_reports_catalog_mismatch(
    device_type: str,
    dtype: str,
    mismatch: str,
) -> None:
    match = _density_matmul_kernel_match(device_type=device_type, dtype=dtype)

    rejection = next(
        item
        for item in match.rejections
        if item.implementation.implementation_id == "FQKI-TRITON-NUM-001-A"
    )
    assert tuple(item.code for item in rejection.mismatches) == (mismatch,)


def test_density_matmul_dispatch_fails_closed_on_unsupported_input() -> None:
    with pytest.raises(RuntimeError, match="not authorized.*device"):
        _require_density_matmul_kernel(device_type="cpu", dtype="complex64")


@pytest.mark.parametrize(
    ("device_type", "dtype", "declared"),
    (
        ("cpu", "complex64", False),
        ("cpu", "float32", False),
        ("cuda", "complex64", True),
        ("cuda", "complex128", False),
        ("cuda", "float32", False),
        ("flagos", "complex64", False),
        ("mps", "complex64", False),
        ("meta", "complex64", False),
    ),
)
def test_catalog_declares_kernel_answers_for_the_declared_axis(
    device_type: str,
    dtype: str,
    declared: bool,
) -> None:
    assert (
        _catalog_declares_kernel(
            "numerics.matmul.complex_batched",
            device_type=device_type,
            dtype=dtype,
            layout="batched_matrix",
        )
        is declared
    )


def test_catalog_declares_kernel_rejects_an_unknown_semantic() -> None:
    assert not _catalog_declares_kernel(
        "numerics.matmul.not_declared",
        device_type="cuda",
        dtype="complex64",
        layout="batched_matrix",
    )


def test_operand_kernel_axis_requires_one_declared_device_and_dtype() -> None:
    left, right = _operands()

    assert _operand_kernel_axis((left, right)) == ("cpu", "complex64")
    assert _operand_kernel_axis(()) is None
    assert _operand_kernel_axis((left, left.to(torch.complex128))) is None

    on_meta = torch.zeros(2, dtype=torch.complex64, device="meta")
    assert _operand_kernel_axis((on_meta, on_meta)) is None


def test_density_matmul_declared_follows_the_catalog_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left, right = _operands()

    assert not _density_matmul_declared(left, right)

    _patch_device_axis(monkeypatch, _CUDA_AXIS)
    assert _density_matmul_declared(left, right)


def test_density_matmul_declared_requires_the_batched_matrix_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left, _ = _operands()

    _patch_device_axis(monkeypatch, _CUDA_AXIS)
    assert not _density_matmul_declared(left[0], left[0])
    assert _density_matmul_declared(left, left)


def test_declaring_a_cpu_implementation_moves_the_route_without_a_code_edit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The replacement proof: catalog content decides, not a device literal.

    The shipped catalog declares CUDA-only records for
    ``numerics.matmul.complex_batched``, so a CPU product takes the reference
    route. Declaring an evidenced CPU implementation for the same semantic must
    move the route on this host without editing ``density_matrix_dispatch.py``.
    Both halves run on the same machine, so the declaration is the only
    difference between them.
    """

    left, right = _operands()
    applied: list[tuple[torch.Tensor, torch.Tensor]] = []

    def counted(
        operand_left: torch.Tensor, operand_right: torch.Tensor
    ) -> torch.Tensor:
        applied.append((operand_left, operand_right))
        return torch.bmm(operand_left, operand_right)

    monkeypatch.setattr(dispatch, "_apply_cataloged_density_matmul", counted)

    assert not _density_matmul_declared(left, right)
    assert torch.allclose(_density_bmm(left, right), torch.bmm(left, right))
    assert applied == []
    assert density_matmul_route_stats() == {
        "cataloged_routes": 0,
        "reference_routes": 1,
    }

    reset_density_matmul_route_stats()
    record = KernelImplementation(
        implementation_id="FQKI-PYTORCH-NUM-001-B",
        semantic_id="numerics.matmul.complex_batched",
        provider="pytorch",
        module="flagquantum.kernels.pytorch.complex_bmm",
        symbol="fused_cpu_complex_bmm",
        devices=("cpu",),
        dtypes=("complex64",),
        layouts=("batched_matrix",),
        directions=("forward",),
        addressing=("local",),
        maturity="experimental",
    )
    evidence = KernelEvidence(
        evidence_id="FQKE-PYTORCH-NUM-001-B",
        implementation_id=record.implementation_id,
        correctness_tests=("tests/unit/test_density_matrix_catalog_dispatch.py",),
    )

    def with_cpu_record(request: KernelRequest, **kwargs: object) -> KernelMatchResult:
        return match_kernel_implementations(
            request,
            semantics=SEMANTICS,
            implementations=(*IMPLEMENTATIONS, record),
            evidence=(*EVIDENCE, evidence),
        )

    monkeypatch.setattr(
        kernel_dispatch, "match_kernel_implementations", with_cpu_record
    )

    assert _density_matmul_declared(left, right)
    assert torch.allclose(_density_bmm(left, right), torch.bmm(left, right))
    assert applied == [(left, right)]
    assert density_matmul_route_stats() == {
        "cataloged_routes": 1,
        "reference_routes": 0,
    }


def test_density_bmm_takes_the_cataloged_route_when_one_is_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left, right = _operands()
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def record(operand_left: torch.Tensor, operand_right: torch.Tensor) -> torch.Tensor:
        calls.append((operand_left, operand_right))
        return torch.bmm(operand_left, operand_right)

    _patch_device_axis(monkeypatch, _CUDA_AXIS)
    monkeypatch.setattr(dispatch, "_apply_cataloged_density_matmul", record)

    assert torch.allclose(_density_bmm(left, right), torch.bmm(left, right))
    assert calls == [(left, right)]
    assert density_matmul_route_stats() == {
        "cataloged_routes": 1,
        "reference_routes": 0,
    }


def test_density_bmm_falls_back_to_a_counted_reference_route() -> None:
    left, right = _operands()

    assert torch.allclose(_density_bmm(left, right), torch.bmm(left, right))
    assert density_matmul_route_stats() == {
        "cataloged_routes": 0,
        "reference_routes": 1,
    }


def test_density_matrix_simulator_products_go_through_the_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = torch.Generator().manual_seed(1307)
    state = torch.randn(1, 4, 4, dtype=torch.complex64, generator=generator)
    rho = state @ state.mH
    # A non-identity, non-Hermitian operator on wire 0. An identity would make
    # ``U rho`` and ``rho U`` the same product, so it cannot show which side of
    # the sandwich the operator was placed on.
    gate = torch.tensor(
        [[0.6 + 0.4j, 0.2 - 0.8j], [0.1j, -0.7 + 0.05j]], dtype=torch.complex64
    )
    full = torch.kron(gate, torch.eye(2, dtype=torch.complex64)).unsqueeze(0)
    expected = full @ rho @ full.mH

    _patch_device_axis(monkeypatch, _CUDA_AXIS)
    monkeypatch.setattr(dispatch, "_apply_cataloged_density_matmul", torch.bmm)

    actual = apply_unitary_density(rho, gate, (0,), 2)

    assert torch.allclose(actual, expected)
    # The sandwich is two products, and both must be the simulator's own
    # cataloged route rather than a direct torch.bmm call.
    assert density_matmul_route_stats() == {
        "cataloged_routes": 2,
        "reference_routes": 0,
    }

    reset_density_matmul_route_stats()
    channels = apply_kraus_density(rho, (gate, gate), (0,), 2)

    assert torch.allclose(channels, 2 * expected)
    # One product pair per Kraus operator, so the channel simulator is a client
    # of the same route and not a second device decision.
    assert density_matmul_route_stats() == {
        "cataloged_routes": 4,
        "reference_routes": 0,
    }
