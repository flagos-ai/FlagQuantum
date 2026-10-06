"""Catalog and rollout contracts for local controlled one-qubit rotations."""

from __future__ import annotations

import sys

import pytest
import torch

import flagquantum as fq
import flagquantum.simulation.statevector.controlled_matrix_dispatch as dispatch
from flagquantum.simulation.statevector.controlled_matrix_dispatch import (
    _controlled_matrix_dispatch_enabled,
    _controlled_matrix_kernel_match,
    _controlled_matrix_shape_supported,
    _require_controlled_matrix_kernel,
    _try_apply_cataloged_controlled_rotation,
)

pytestmark = pytest.mark.unit


def test_controlled_matrix_dispatch_binds_exact_catalog_implementation() -> None:
    implementation = _require_controlled_matrix_kernel(
        device_type="cuda",
        dtype="complex64",
    )

    assert implementation.semantic_id == "statevector.apply.controlled_matrix_1q.local"
    assert implementation.implementation_id == "FQKI-TRITON-SV-012-A"
    assert implementation.symbol == "apply_complex64_local_controlled_1q"
    assert implementation.directions == ("forward",)
    assert implementation.maturity == "provisional"


@pytest.mark.parametrize(
    ("device_type", "dtype", "mismatch"),
    (("cpu", "complex64", "device"), ("cuda", "complex128", "dtype")),
)
def test_controlled_matrix_dispatch_reports_catalog_mismatch(
    device_type: str,
    dtype: str,
    mismatch: str,
) -> None:
    match = _controlled_matrix_kernel_match(device_type=device_type, dtype=dtype)

    rejection = next(
        item
        for item in match.rejections
        if item.implementation.implementation_id == "FQKI-TRITON-SV-012-A"
    )
    assert tuple(item.code for item in rejection.mismatches) == (mismatch,)


def test_controlled_matrix_dispatch_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="not authorized.*dtype"):
        _require_controlled_matrix_kernel(device_type="cuda", dtype="complex128")


@pytest.mark.parametrize("disabled", ("0", "false", "off", "no", " FALSE "))
def test_controlled_matrix_rollout_defaults_on_and_supports_kill_switch(
    monkeypatch: pytest.MonkeyPatch,
    disabled: str,
) -> None:
    monkeypatch.delenv("FQ_TRITON_CONTROLLED_MATRIX_1Q", raising=False)
    assert _controlled_matrix_dispatch_enabled()

    monkeypatch.setenv("FQ_TRITON_CONTROLLED_MATRIX_1Q", disabled)
    assert not _controlled_matrix_dispatch_enabled()


@pytest.mark.parametrize(
    (
        "state_shape",
        "angle_shape",
        "control",
        "target",
        "n_qubits",
        "opcode",
        "supported",
    ),
    (
        ((1, 1 << 20), (1,), 0, 19, 20, "crx", True),
        ((1, 1 << 24), (1,), 3, 19, 24, "cry", True),
        ((4, 1 << 20), (4,), 17, 3, 20, "crz", True),
        ((1, 1 << 16), (1,), 0, 15, 16, "crx", False),
        ((2, 1 << 20), (2,), 0, 19, 20, "cry", False),
        ((1, 1 << 20), (1,), 0, 19, 20, "cy", False),
        ((1, 1 << 20), (1,), 1, 1, 20, "crz", False),
        ((1, 1 << 20), (1,), 0, 20, 20, "crx", False),
        ((1, 1 << 20), (2,), 0, 19, 20, "cry", False),
        ((1, 1 << 19), (1,), 0, 18, 20, "crz", False),
    ),
)
def test_controlled_matrix_shape_policy_matches_evidenced_window(
    state_shape: tuple[int, ...],
    angle_shape: tuple[int, ...],
    control: int,
    target: int,
    n_qubits: int,
    opcode: str,
    supported: bool,
) -> None:
    assert (
        _controlled_matrix_shape_supported(
            state_shape,
            angle_shape,
            control_qubit=control,
            target_qubit=target,
            n_qubits=n_qubits,
            opcode=opcode,
        )
        is supported
    )


def test_controlled_matrix_cpu_falls_back_without_importing_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_TRITON_CONTROLLED_MATRIX_1Q", "1")
    module = "flagquantum.kernels.triton.statevector_controlled_matrix"
    sys.modules.pop(module, None)
    state = torch.randn(1, 1 << 20, dtype=torch.complex64)
    angles = torch.tensor([0.23], dtype=torch.float32)

    actual = _try_apply_cataloged_controlled_rotation(
        state,
        angles,
        control_qubit=0,
        target_qubit=19,
        n_qubits=20,
        opcode="crx",
    )

    assert actual is None
    assert module not in sys.modules


@pytest.mark.parametrize("opcode", ("crx", "cry", "crz"))
def test_local_statevector_program_calls_public_controlled_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    opcode: str,
) -> None:
    calls: list[tuple[int, int, int, str, torch.Tensor]] = []

    def capture(
        state: torch.Tensor,
        angles: torch.Tensor,
        *,
        control_qubit: int,
        target_qubit: int,
        n_qubits: int,
        opcode: str,
    ) -> torch.Tensor:
        calls.append(
            (control_qubit, target_qubit, n_qubits, opcode, angles.detach().clone())
        )
        return state.clone()

    monkeypatch.setattr(dispatch, "_try_apply_cataloged_controlled_rotation", capture)
    circuit = getattr(fq.Circuit(4), opcode)(0, 3, theta=0.23)

    actual = circuit.state(refresh=True)

    assert actual.shape == (1, 1 << 4)
    assert len(calls) == 1
    control, target, n_qubits, captured_opcode, angles = calls[0]
    assert (control, target, n_qubits, captured_opcode) == (0, 3, 4, opcode)
    torch.testing.assert_close(angles, torch.tensor([0.23]))


@pytest.mark.parametrize("opcode", ("crx", "cry", "crz"))
@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_controlled_rotation_public_path_uses_catalog(
    monkeypatch: pytest.MonkeyPatch,
    opcode: str,
) -> None:
    monkeypatch.setenv("FQ_TRITON_CONTROLLED_MATRIX_1Q", "1")
    routes: list[str] = []
    require_cataloged_kernel = dispatch._require_controlled_matrix_kernel

    def capture_catalog_route(*, device_type: str, dtype: str):
        implementation = require_cataloged_kernel(
            device_type=device_type,
            dtype=dtype,
        )
        routes.append(implementation.implementation_id)
        return implementation

    monkeypatch.setattr(
        dispatch,
        "_require_controlled_matrix_kernel",
        capture_catalog_route,
    )
    enabled = fq.Circuit(20, device="cuda", dtype=torch.complex64)
    disabled = fq.Circuit(20, device="cuda", dtype=torch.complex64)
    getattr(enabled, opcode)(0, 19, theta=0.23)
    getattr(disabled, opcode)(0, 19, theta=0.23)

    actual = enabled.state(refresh=True)
    monkeypatch.setenv("FQ_TRITON_CONTROLLED_MATRIX_1Q", "0")
    expected = disabled.state(refresh=True)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert routes == ["FQKI-TRITON-SV-012-A"]


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_controlled_matrix_kill_switch_preserves_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_TRITON_CONTROLLED_MATRIX_1Q", "0")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("disabled dispatch must not execute SV-012")

    monkeypatch.setattr(
        dispatch, "_apply_cataloged_controlled_rotation", fail_if_called
    )
    circuit = fq.Circuit(20, device="cuda", dtype=torch.complex64).crz(
        0,
        19,
        theta=0.23,
    )

    actual = circuit.state(refresh=True)

    assert actual.shape == (1, 1 << 20)
