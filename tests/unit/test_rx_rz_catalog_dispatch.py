"""Catalog dispatch contracts for fused local RX/RZ sequences."""

from __future__ import annotations

import pytest
import torch

from flagquantum.simulation.statevector.rx_rz_dispatch import (
    _apply_cataloged_rx_rz_sequence,
    _require_rx_rz_sequence_kernel,
    _rx_rz_sequence_kernel_match,
)

pytestmark = pytest.mark.unit


def test_rx_rz_sequence_dispatch_binds_exact_catalog_implementation() -> None:
    implementation = _require_rx_rz_sequence_kernel(
        device_type="cuda",
        dtype="complex64",
    )

    assert implementation.semantic_id == "statevector.apply.rx_rz_sequence.local"
    assert implementation.implementation_id == "FQKI-TRITON-SV-005-A"
    assert implementation.symbol == "repeated_rx_rz"


@pytest.mark.parametrize(
    ("device_type", "dtype", "mismatch"),
    (("cpu", "complex64", "device"), ("cuda", "complex128", "dtype")),
)
def test_rx_rz_sequence_dispatch_reports_catalog_mismatch(
    device_type: str,
    dtype: str,
    mismatch: str,
) -> None:
    match = _rx_rz_sequence_kernel_match(
        device_type=device_type,
        dtype=dtype,
    )

    rejection = next(
        item
        for item in match.rejections
        if item.implementation.implementation_id == "FQKI-TRITON-SV-005-A"
    )
    assert tuple(item.code for item in rejection.mismatches) == (mismatch,)


def test_rx_rz_sequence_dispatch_fails_closed_on_unsupported_input() -> None:
    with pytest.raises(RuntimeError, match="not authorized.*device"):
        _require_rx_rz_sequence_kernel(
            device_type="cpu",
            dtype="complex64",
        )


@pytest.mark.parametrize(
    ("dtype", "codes"),
    ((torch.complex64, ("device",)), (torch.complex128, ("device", "dtype"))),
)
def test_the_execution_wrapper_decides_on_the_real_device(
    dtype: torch.dtype, codes: tuple[str, ...]
) -> None:
    # Premise: the fused step is emitted only when the Triton loop is enabled, so
    # a CUDA complex64 tensor is the intended caller. The wrapper must still
    # authorize from the tensor's own device and dtype, because a decision the
    # caller may skip is not authorization. Before this inverted-guard change the
    # CPU call skipped the catalog entirely and fell through to the provider,
    # which is absent here: an unguarded route ends in ModuleNotFoundError rather
    # than the RuntimeError asserted below.
    state = torch.zeros(1, 8, dtype=dtype)
    angles = torch.zeros(1, 4, dtype=torch.float32)

    with pytest.raises(RuntimeError) as failure:
        _apply_cataloged_rx_rz_sequence(state, angles, angles)

    message = str(failure.value)
    assert message.startswith(
        "fused RX/RZ sequence kernel is not authorized by the kernel catalog: "
    )
    assert tuple(message.rsplit(": ", maxsplit=1)[1].split(", ")) == codes


def test_the_execution_wrapper_authorizes_even_where_the_caller_did_not() -> None:
    # Premise: the fused step is only emitted when the triton loop is enabled,
    # so a CUDA tensor is the intended caller. The wrapper must still decide on
    # the real device, because a decision made elsewhere is not authorization.
    state = torch.zeros(1, 8, dtype=torch.complex64)
    angles = torch.zeros(1, 4, dtype=torch.float32)

    with pytest.raises(
        RuntimeError, match="not authorized by the kernel catalog: device"
    ):
        _apply_cataloged_rx_rz_sequence(state, angles, angles)


def test_the_execution_wrapper_authorizes_before_importing_the_provider() -> None:
    # A CPU tensor must be refused by the catalog rather than by the provider's
    # import or its own internal fallback. Triton is absent in this environment,
    # so a ModuleNotFoundError here would mean authorization was skipped.
    state = torch.zeros(1, 4, dtype=torch.complex64)
    angles = torch.zeros(1, 2, dtype=torch.float32)

    with pytest.raises(RuntimeError) as failure:
        _apply_cataloged_rx_rz_sequence(state, angles, angles)

    assert "kernel catalog" in str(failure.value)
    assert "No module named 'triton'" not in str(failure.value)
