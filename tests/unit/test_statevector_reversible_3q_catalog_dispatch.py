"""Catalog and rollout contracts for local CCX and CSWAP dispatch."""

from __future__ import annotations

import sys

import pytest
import torch

import flagquantum as fq
import flagquantum.simulation.statevector.reversible_3q_dispatch as dispatch
from flagquantum.simulation.statevector.reversible_3q_dispatch import (
    _require_reversible_3q_kernel,
    _reversible_3q_dispatch_enabled,
    _reversible_3q_kernel_match,
    _reversible_3q_shape_supported,
    _try_apply_cataloged_reversible_3q,
)

pytestmark = pytest.mark.unit


def test_reversible_3q_dispatch_binds_exact_catalog_implementation() -> None:
    implementation = _require_reversible_3q_kernel(
        device_type="cuda",
        dtype="complex64",
    )

    assert implementation.semantic_id == (
        "statevector.apply.reversible_permutation_3q.local"
    )
    assert implementation.implementation_id == "FQKI-TRITON-SV-013-A"
    assert implementation.symbol == "apply_complex64_local_reversible_3q"
    assert implementation.directions == ("forward",)
    assert implementation.maturity == "provisional"


@pytest.mark.parametrize(
    ("device_type", "dtype", "mismatch"),
    (("cpu", "complex64", "device"), ("cuda", "complex128", "dtype")),
)
def test_reversible_3q_dispatch_reports_catalog_mismatch(
    device_type: str,
    dtype: str,
    mismatch: str,
) -> None:
    match = _reversible_3q_kernel_match(device_type=device_type, dtype=dtype)

    rejection = next(
        item
        for item in match.rejections
        if item.implementation.implementation_id == "FQKI-TRITON-SV-013-A"
    )
    assert tuple(item.code for item in rejection.mismatches) == (mismatch,)


def test_reversible_3q_dispatch_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="not authorized.*dtype"):
        _require_reversible_3q_kernel(device_type="cuda", dtype="complex128")


@pytest.mark.parametrize("disabled", ("0", "false", "off", "no", " FALSE "))
def test_reversible_3q_rollout_defaults_on_and_supports_kill_switch(
    monkeypatch: pytest.MonkeyPatch,
    disabled: str,
) -> None:
    monkeypatch.delenv("FQ_TRITON_REVERSIBLE_3Q", raising=False)
    assert _reversible_3q_dispatch_enabled()

    monkeypatch.setenv("FQ_TRITON_REVERSIBLE_3Q", disabled)
    assert not _reversible_3q_dispatch_enabled()


@pytest.mark.parametrize(
    ("state_shape", "qubits", "n_qubits", "opcode", "supported"),
    (
        ((1, 1 << 20), (0, 1, 19), 20, "ccx", True),
        ((1, 1 << 24), (3, 19, 7), 24, "cswap", True),
        ((4, 1 << 20), (1, 18, 7), 20, "ccx", True),
        ((1, 1 << 16), (0, 1, 15), 16, "ccx", False),
        ((2, 1 << 20), (0, 1, 19), 20, "cswap", False),
        ((1, 1 << 20), (0, 1, 19), 20, "swap", False),
        ((1, 1 << 20), (1, 1, 19), 20, "ccx", False),
        ((1, 1 << 20), (0, 1, 20), 20, "cswap", False),
        ((1, 1 << 19), (0, 1, 18), 20, "ccx", False),
    ),
)
def test_reversible_3q_shape_policy_matches_evidenced_window(
    state_shape: tuple[int, ...],
    qubits: tuple[int, ...],
    n_qubits: int,
    opcode: str,
    supported: bool,
) -> None:
    assert (
        _reversible_3q_shape_supported(
            state_shape,
            qubits=qubits,
            n_qubits=n_qubits,
            opcode=opcode,
        )
        is supported
    )


def test_reversible_3q_cpu_falls_back_without_importing_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_TRITON_REVERSIBLE_3Q", "1")
    module = "flagquantum.kernels.triton.statevector_reversible_3q"
    sys.modules.pop(module, None)
    state = torch.randn(1, 1 << 20, dtype=torch.complex64)

    actual = _try_apply_cataloged_reversible_3q(
        state,
        qubits=(0, 1, 19),
        n_qubits=20,
        opcode="ccx",
    )

    assert actual is None
    assert module not in sys.modules


@pytest.mark.parametrize("opcode", ("ccx", "cswap"))
def test_local_statevector_program_calls_public_reversible_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    opcode: str,
) -> None:
    calls: list[tuple[tuple[int, int, int], int, str]] = []

    def capture(
        state: torch.Tensor,
        *,
        qubits: tuple[int, int, int],
        n_qubits: int,
        opcode: str,
    ) -> torch.Tensor:
        calls.append((qubits, n_qubits, opcode))
        return state.clone()

    monkeypatch.setattr(dispatch, "_try_apply_cataloged_reversible_3q", capture)
    circuit = getattr(fq.Circuit(4), opcode)(0, 1, 3)

    actual = circuit.state(refresh=True)

    assert actual.shape == (1, 1 << 4)
    assert calls == [((0, 1, 3), 4, opcode)]


@pytest.mark.parametrize("opcode", ("ccx", "cswap"))
@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_reversible_3q_public_path_uses_catalog(
    monkeypatch: pytest.MonkeyPatch,
    opcode: str,
) -> None:
    monkeypatch.setenv("FQ_TRITON_REVERSIBLE_3Q", "1")
    generator = torch.Generator(device="cuda").manual_seed(261_013)
    inputs = torch.randn(
        1,
        1 << 20,
        generator=generator,
        device="cuda",
        dtype=torch.complex64,
    )
    routes: list[str] = []
    require_cataloged_kernel = dispatch._require_reversible_3q_kernel

    def capture_catalog_route(*, device_type: str, dtype: str):
        implementation = require_cataloged_kernel(
            device_type=device_type,
            dtype=dtype,
        )
        routes.append(implementation.implementation_id)
        return implementation

    monkeypatch.setattr(
        dispatch,
        "_require_reversible_3q_kernel",
        capture_catalog_route,
    )
    enabled = fq.Circuit(20, inputs=inputs, device="cuda", dtype=torch.complex64)
    disabled = fq.Circuit(20, inputs=inputs, device="cuda", dtype=torch.complex64)
    getattr(enabled, opcode)(0, 1, 19)
    getattr(disabled, opcode)(0, 1, 19)

    actual = enabled.state(refresh=True)
    monkeypatch.setenv("FQ_TRITON_REVERSIBLE_3Q", "0")
    expected = disabled.state(refresh=True)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert routes == ["FQKI-TRITON-SV-013-A"]


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_reversible_3q_kill_switch_preserves_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_TRITON_REVERSIBLE_3Q", "0")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("disabled dispatch must not execute SV-013")

    monkeypatch.setattr(dispatch, "_apply_cataloged_reversible_3q", fail_if_called)
    circuit = fq.Circuit(20, device="cuda", dtype=torch.complex64).ccx(0, 1, 19)

    actual = circuit.state(refresh=True)

    assert actual.shape == (1, 1 << 20)
