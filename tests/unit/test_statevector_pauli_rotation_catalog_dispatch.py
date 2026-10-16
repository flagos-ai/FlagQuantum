"""Catalog and rollout contracts for local two-qubit Pauli rotations."""

from __future__ import annotations

import sys

import pytest
import torch

import flagquantum as fq
import flagquantum.simulation.statevector.pauli_rotation_dispatch as dispatch
from flagquantum.simulation.statevector.pauli_rotation_dispatch import (
    _pauli_rotation_dispatch_enabled,
    _pauli_rotation_kernel_match,
    _pauli_rotation_shape_supported,
    _require_pauli_rotation_kernel,
    _try_apply_cataloged_pauli_rotation,
)

pytestmark = pytest.mark.unit


def test_pauli_rotation_dispatch_binds_exact_catalog_implementation() -> None:
    implementation = _require_pauli_rotation_kernel(
        device_type="cuda",
        dtype="complex64",
    )

    assert implementation.semantic_id == "statevector.apply.pauli_rotation_2q.local"
    assert implementation.implementation_id == "FQKI-TRITON-SV-011-A"
    assert implementation.symbol == "apply_complex64_local_pauli_rotation_2q"
    assert implementation.directions == ("forward",)
    assert implementation.maturity == "provisional"


@pytest.mark.parametrize(
    ("device_type", "dtype", "mismatch"),
    (("cpu", "complex64", "device"), ("cuda", "complex128", "dtype")),
)
def test_pauli_rotation_dispatch_reports_catalog_mismatch(
    device_type: str,
    dtype: str,
    mismatch: str,
) -> None:
    match = _pauli_rotation_kernel_match(device_type=device_type, dtype=dtype)

    rejection = next(
        item
        for item in match.rejections
        if item.implementation.implementation_id == "FQKI-TRITON-SV-011-A"
    )
    assert tuple(item.code for item in rejection.mismatches) == (mismatch,)


def test_pauli_rotation_dispatch_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="not authorized.*dtype"):
        _require_pauli_rotation_kernel(device_type="cuda", dtype="complex128")


@pytest.mark.parametrize("disabled", ("0", "false", "off", "no", " FALSE "))
def test_pauli_rotation_rollout_defaults_on_and_supports_kill_switch(
    monkeypatch: pytest.MonkeyPatch,
    disabled: str,
) -> None:
    monkeypatch.delenv("FQ_TRITON_PAULI_ROTATION_2Q", raising=False)
    assert _pauli_rotation_dispatch_enabled()

    monkeypatch.setenv("FQ_TRITON_PAULI_ROTATION_2Q", disabled)
    assert not _pauli_rotation_dispatch_enabled()


@pytest.mark.parametrize(
    ("state_shape", "angle_shape", "qubits", "n_qubits", "opcode", "supported"),
    (
        ((1, 1 << 20), (1,), (0, 19), 20, "rxx", True),
        ((1, 1 << 24), (1,), (3, 19), 24, "ryy", True),
        ((4, 1 << 20), (4,), (1, 18), 20, "rzz", True),
        ((1, 1 << 16), (1,), (0, 15), 16, "rxx", False),
        ((2, 1 << 20), (2,), (0, 19), 20, "ryy", False),
        ((1, 1 << 20), (1,), (0, 19), 20, "rx", False),
        ((1, 1 << 20), (1,), (1, 1), 20, "rzz", False),
        ((1, 1 << 20), (1,), (0, 20), 20, "rxx", False),
        ((1, 1 << 20), (2,), (0, 19), 20, "ryy", False),
        ((1, 1 << 19), (1,), (0, 18), 20, "rzz", False),
    ),
)
def test_pauli_rotation_shape_policy_matches_evidenced_window(
    state_shape: tuple[int, ...],
    angle_shape: tuple[int, ...],
    qubits: tuple[int, ...],
    n_qubits: int,
    opcode: str,
    supported: bool,
) -> None:
    assert (
        _pauli_rotation_shape_supported(
            state_shape,
            angle_shape,
            qubits=qubits,
            n_qubits=n_qubits,
            opcode=opcode,
        )
        is supported
    )


def test_pauli_rotation_cpu_falls_back_without_importing_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_TRITON_PAULI_ROTATION_2Q", "1")
    module = "flagquantum.kernels.triton.statevector_pauli_rotation"
    sys.modules.pop(module, None)
    state = torch.randn(1, 1 << 20, dtype=torch.complex64)
    angles = torch.tensor([0.23], dtype=torch.float32)

    actual = _try_apply_cataloged_pauli_rotation(
        state,
        angles,
        qubits=(0, 19),
        n_qubits=20,
        opcode="rxx",
    )

    assert actual is None
    assert module not in sys.modules


def test_local_statevector_program_calls_public_pauli_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[int, ...], int, str, torch.Tensor]] = []

    def capture(
        state: torch.Tensor,
        angles: torch.Tensor,
        *,
        qubits: tuple[int, ...],
        n_qubits: int,
        opcode: str,
    ) -> torch.Tensor:
        calls.append((tuple(qubits), n_qubits, opcode, angles.detach().clone()))
        return state.clone()

    monkeypatch.setattr(dispatch, "_try_apply_cataloged_pauli_rotation", capture)
    circuit = fq.Circuit(4).rxx(0, 3, theta=0.23)

    actual = circuit.state(refresh=True)

    assert actual.shape == (1, 1 << 4)
    assert len(calls) == 1
    qubits, n_qubits, opcode, angles = calls[0]
    assert qubits == (0, 3)
    assert n_qubits == 4
    assert opcode == "rxx"
    torch.testing.assert_close(angles, torch.tensor([0.23]))


@pytest.mark.parametrize("opcode", ("rxx", "ryy", "rzz"))
@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_pauli_rotation_public_path_uses_catalog(
    monkeypatch: pytest.MonkeyPatch,
    opcode: str,
) -> None:
    monkeypatch.setenv("FQ_TRITON_PAULI_ROTATION_2Q", "1")
    routes: list[str] = []
    require_cataloged_kernel = dispatch._require_pauli_rotation_kernel

    def capture_catalog_route(*, device_type: str, dtype: str):
        implementation = require_cataloged_kernel(
            device_type=device_type,
            dtype=dtype,
        )
        routes.append(implementation.implementation_id)
        return implementation

    monkeypatch.setattr(
        dispatch,
        "_require_pauli_rotation_kernel",
        capture_catalog_route,
    )
    enabled = fq.Circuit(20, device="cuda", dtype=torch.complex64)
    disabled = fq.Circuit(20, device="cuda", dtype=torch.complex64)
    getattr(enabled, opcode)(0, 19, theta=0.23)
    getattr(disabled, opcode)(0, 19, theta=0.23)

    actual = enabled.state(refresh=True)
    monkeypatch.setenv("FQ_TRITON_PAULI_ROTATION_2Q", "0")
    expected = disabled.state(refresh=True)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert routes == ["FQKI-TRITON-SV-011-A"]


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_pauli_rotation_kill_switch_preserves_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_TRITON_PAULI_ROTATION_2Q", "0")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("disabled dispatch must not execute SV-011")

    monkeypatch.setattr(dispatch, "_apply_cataloged_pauli_rotation", fail_if_called)
    circuit = fq.Circuit(20, device="cuda", dtype=torch.complex64).rzz(
        0,
        19,
        theta=0.23,
    )

    actual = circuit.state(refresh=True)

    assert actual.shape == (1, 1 << 20)
