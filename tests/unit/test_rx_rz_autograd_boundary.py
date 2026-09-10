"""CPU checks of RX/RZ tangent ordering and external Jacobian returns."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

pytestmark = pytest.mark.unit


@pytest.fixture
def loop_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    triton = ModuleType("triton")
    triton.jit = lambda function: function
    language = ModuleType("triton.language")
    triton.language = language
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", language)
    path = (
        Path(__file__).resolve().parents[2]
        / "flagquantum/simulation/triton_kernels/single_qubit_loop.py"
    )
    spec = importlib.util.spec_from_file_location("_test_rx_rz_loop", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("depth", [1, 3])
def test_cpu_tangents_match_interleaved_finite_differences(
    loop_module: ModuleType, batch: int, depth: int
) -> None:
    generator = torch.Generator().manual_seed(211)
    state = torch.randn(batch, 3, 2, dtype=torch.complex128, generator=generator)
    rx = torch.randn(batch, depth, dtype=torch.float64, generator=generator)
    rz = torch.randn(batch, depth, dtype=torch.float64, generator=generator)
    actual = loop_module.repeated_rx_rz_tangents(state, rx, rz)
    expected = []
    epsilon = 1e-6
    for layer in range(depth):
        perturbation = torch.zeros_like(rx)
        perturbation[:, layer] = epsilon
        for family in ("rx", "rz"):
            plus_rx, minus_rx = (
                (rx + perturbation, rx - perturbation) if family == "rx" else (rx, rx)
            )
            plus_rz, minus_rz = (
                (rz + perturbation, rz - perturbation) if family == "rz" else (rz, rz)
            )
            plus = loop_module.repeated_rx_rz(state, plus_rx, plus_rz)
            minus = loop_module.repeated_rx_rz(state, minus_rx, minus_rz)
            expected.append((plus - minus) / (2 * epsilon))
    assert actual.shape == (2 * depth, batch, 3, 2)
    torch.testing.assert_close(actual, torch.stack(expected), atol=1e-8, rtol=1e-7)


@pytest.mark.parametrize("value", [None, (torch.tensor(0.0),)])
def test_cpu_tangents_reject_invalid_jacobian(
    loop_module: ModuleType, monkeypatch: pytest.MonkeyPatch, value: object
) -> None:
    monkeypatch.setattr(
        torch.autograd.functional, "jacobian", lambda *args, **kwargs: value
    )
    with pytest.raises(TypeError, match="RX/RZ Jacobian must return a tensor"):
        loop_module.repeated_rx_rz_tangents(
            torch.ones(1, 1, 2, dtype=torch.complex128),
            torch.zeros(1, 1, dtype=torch.float64),
            torch.zeros(1, 1, dtype=torch.float64),
        )
