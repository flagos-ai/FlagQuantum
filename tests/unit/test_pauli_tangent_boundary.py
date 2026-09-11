"""CPU checks of layer-major two-qubit Pauli rotation tangents."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

pytestmark = pytest.mark.unit


@pytest.fixture
def tangent_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    triton = ModuleType("triton")
    triton.jit = lambda function: function
    language = ModuleType("triton.language")
    triton.language = language
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", language)
    path = (
        Path(__file__).resolve().parents[2]
        / "flagquantum/simulation/triton_kernels/two_qubit_pauli_tangent.py"
    )
    spec = importlib.util.spec_from_file_location("_test_pauli_tangent", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("depth", [1, 3])
def test_pauli_tangents_match_finite_differences(
    tangent_module: ModuleType, batch: int, depth: int
) -> None:
    generator = torch.Generator().manual_seed(531)
    state = torch.randn(batch, 2, 4, dtype=torch.complex128, generator=generator)
    angles = torch.randn(batch, depth, 3, dtype=torch.float64, generator=generator)
    actual = tangent_module.repeated_rxx_ryy_rzz_tangents(state, angles)
    expected = []
    epsilon = 1e-6
    for layer in range(depth):
        for family in range(3):
            shift = torch.zeros_like(angles)
            shift[:, layer, family] = epsilon
            plus = tangent_module._reference(state, angles + shift)
            minus = tangent_module._reference(state, angles - shift)
            expected.append((plus - minus) / (2 * epsilon))
    torch.testing.assert_close(actual, torch.stack(expected), atol=1e-8, rtol=1e-7)


@pytest.mark.parametrize("value", [None, (torch.tensor(0.0),)])
def test_pauli_tangents_reject_invalid_jacobian(
    tangent_module: ModuleType, monkeypatch: pytest.MonkeyPatch, value: object
) -> None:
    monkeypatch.setattr(
        torch.autograd.functional, "jacobian", lambda *args, **kwargs: value
    )
    with pytest.raises(TypeError, match="Pauli rotation Jacobian must return a tensor"):
        tangent_module.repeated_rxx_ryy_rzz_tangents(
            torch.ones(1, 1, 4, dtype=torch.complex128),
            torch.zeros(1, 1, 3, dtype=torch.float64),
        )
