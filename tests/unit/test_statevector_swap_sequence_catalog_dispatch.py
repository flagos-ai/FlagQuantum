"""Catalog and rollout contracts for bounded SWAP-sequence dispatch."""

from __future__ import annotations

import sys

import pytest
import torch

import flagquantum as fq
import flagquantum.simulation.statevector.swap_sequence_dispatch as dispatch
from flagquantum.core.ir import Instruction
from flagquantum.simulation.statevector.execution_metrics import initial_runtime_metrics
from flagquantum.simulation.statevector.local import _execute_statevector_program
from flagquantum.simulation.statevector.operations import _compile_statevector_program
from flagquantum.simulation.statevector.program import (
    _StatevectorGateStep,
    _StatevectorSwapSequenceStep,
)
from flagquantum.simulation.statevector.swap_sequence_dispatch import (
    _require_swap_sequence_kernel,
    _swap_sequence_dispatch_enabled,
    _swap_sequence_kernel_match,
    _swap_sequence_shape_supported,
    _try_apply_cataloged_swap_sequence,
)

pytestmark = pytest.mark.unit

_FOUR_SWAPS = ((0, 19), (1, 18), (2, 17), (3, 16))


def test_swap_sequence_dispatch_binds_exact_catalog_implementation() -> None:
    implementation = _require_swap_sequence_kernel(
        device_type="cuda",
        dtype="complex64",
    )

    assert implementation.semantic_id == "statevector.apply.swap_sequence.local"
    assert implementation.implementation_id == "FQKI-TRITON-SV-014-A"
    assert implementation.symbol == "apply_complex64_local_swap_sequence"
    assert implementation.directions == ("forward",)
    assert implementation.maturity == "provisional"


@pytest.mark.parametrize(
    ("device_type", "dtype", "mismatch"),
    (("cpu", "complex64", "device"), ("cuda", "complex128", "dtype")),
)
def test_swap_sequence_dispatch_reports_catalog_mismatch(
    device_type: str,
    dtype: str,
    mismatch: str,
) -> None:
    match = _swap_sequence_kernel_match(device_type=device_type, dtype=dtype)

    rejection = next(
        item
        for item in match.rejections
        if item.implementation.implementation_id == "FQKI-TRITON-SV-014-A"
    )
    assert tuple(item.code for item in rejection.mismatches) == (mismatch,)


def test_swap_sequence_dispatch_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="not authorized.*dtype"):
        _require_swap_sequence_kernel(device_type="cuda", dtype="complex128")


@pytest.mark.parametrize("disabled", ("0", "false", "off", "no", " FALSE "))
def test_swap_sequence_rollout_defaults_on_and_supports_kill_switch(
    monkeypatch: pytest.MonkeyPatch,
    disabled: str,
) -> None:
    monkeypatch.delenv("FQ_TRITON_SWAP_SEQUENCE", raising=False)
    assert _swap_sequence_dispatch_enabled()

    monkeypatch.setenv("FQ_TRITON_SWAP_SEQUENCE", disabled)
    assert not _swap_sequence_dispatch_enabled()


@pytest.mark.parametrize(
    ("state_shape", "swaps", "n_qubits", "supported"),
    (
        ((1, 1 << 16), ((0, 15), (1, 14), (2, 13), (3, 12)), 16, True),
        ((1, 1 << 20), _FOUR_SWAPS, 20, True),
        ((1, 1 << 24), _FOUR_SWAPS * 2, 24, True),
        ((4, 1 << 20), _FOUR_SWAPS + ((4, 15),), 20, True),
        ((1, 1 << 18), _FOUR_SWAPS, 18, False),
        ((2, 1 << 20), _FOUR_SWAPS, 20, False),
        ((1, 1 << 20), _FOUR_SWAPS[:3], 20, False),
        ((1, 1 << 20), _FOUR_SWAPS * 3, 20, False),
        ((1, 1 << 20), ((0, 0),) * 4, 20, False),
        ((1, 1 << 20), ((0, 20),) * 4, 20, False),
        ((1, 1 << 19), _FOUR_SWAPS, 20, False),
    ),
)
def test_swap_sequence_shape_policy_matches_evidenced_window(
    state_shape: tuple[int, ...],
    swaps: tuple[tuple[int, int], ...],
    n_qubits: int,
    supported: bool,
) -> None:
    assert (
        _swap_sequence_shape_supported(
            state_shape,
            swaps=swaps,
            n_qubits=n_qubits,
        )
        is supported
    )


def test_swap_sequence_cpu_falls_back_without_importing_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_TRITON_SWAP_SEQUENCE", "1")
    module = "flagquantum.kernels.triton.statevector_swap"
    sys.modules.pop(module, None)
    state = torch.randn(1, 1 << 20, dtype=torch.complex64)

    actual = _try_apply_cataloged_swap_sequence(
        state,
        swaps=_FOUR_SWAPS,
        n_qubits=20,
    )

    assert actual is None
    assert module not in sys.modules


@pytest.mark.parametrize("length", (4, 5, 8))
def test_compiler_forms_only_kernel_supported_swap_sequences(length: int) -> None:
    swaps = tuple((index % 4, 7 - index % 4) for index in range(length))
    instructions = tuple(Instruction("swap", pair) for pair in swaps)

    program = _compile_statevector_program(
        instructions,
        8,
        enable_triton_loop=False,
        enable_triton_swap_sequence=True,
    )

    assert program == (_StatevectorSwapSequenceStep(swaps),)


@pytest.mark.parametrize("length", (3, 9))
def test_compiler_preserves_out_of_window_swap_sequences(length: int) -> None:
    instructions = tuple(
        Instruction("swap", (index % 4, 7 - index % 4)) for index in range(length)
    )

    program = _compile_statevector_program(
        instructions,
        8,
        enable_triton_loop=False,
        enable_triton_swap_sequence=True,
    )

    assert len(program) == length
    assert all(isinstance(step, _StatevectorGateStep) for step in program)


def test_executor_calls_catalog_dispatch_for_compiled_swap_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[tuple[int, int], ...], int]] = []

    def capture(
        state: torch.Tensor,
        *,
        swaps: tuple[tuple[int, int], ...],
        n_qubits: int,
    ) -> torch.Tensor:
        calls.append((swaps, n_qubits))
        return state.clone()

    monkeypatch.setattr(dispatch, "_try_apply_cataloged_swap_sequence", capture)
    circuit = fq.Circuit(20)
    program = (_StatevectorSwapSequenceStep(_FOUR_SWAPS),)
    circuit._last_statevector_runtime = initial_runtime_metrics(
        program,
        enable_triton_loop=False,
    )
    state = torch.randn(1, 1 << 20, dtype=torch.complex64)

    actual = _execute_statevector_program(circuit, program, state, None)

    assert actual.shape == state.shape
    assert calls == [(_FOUR_SWAPS, 20)]


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_swap_sequence_public_path_uses_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_TRITON_SWAP_SEQUENCE", "1")
    generator = torch.Generator(device="cuda").manual_seed(261_014)
    inputs = torch.randn(
        1,
        1 << 20,
        generator=generator,
        device="cuda",
        dtype=torch.complex64,
    )
    routes: list[str] = []
    require_cataloged_kernel = dispatch._require_swap_sequence_kernel

    def capture_catalog_route(*, device_type: str, dtype: str):
        implementation = require_cataloged_kernel(
            device_type=device_type,
            dtype=dtype,
        )
        routes.append(implementation.implementation_id)
        return implementation

    monkeypatch.setattr(
        dispatch,
        "_require_swap_sequence_kernel",
        capture_catalog_route,
    )
    enabled = fq.Circuit(20, inputs=inputs, device="cuda", dtype=torch.complex64)
    disabled = fq.Circuit(20, inputs=inputs, device="cuda", dtype=torch.complex64)
    for first, second in _FOUR_SWAPS:
        enabled.swap(first, second)
        disabled.swap(first, second)

    actual = enabled.state(refresh=True)
    monkeypatch.setenv("FQ_TRITON_SWAP_SEQUENCE", "0")
    expected = disabled.state(refresh=True)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert routes == ["FQKI-TRITON-SV-014-A"]
