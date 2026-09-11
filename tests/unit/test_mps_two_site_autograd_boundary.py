"""CPU gradient checks for the two-site wrapper using a replacement launch."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

pytestmark = pytest.mark.unit


def _cpu_two_site(
    left: torch.Tensor, gate: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    shaped_gate = gate.reshape(*gate.shape[:-2], 2, 2, 2, 2)
    equation = "ijst,blsm,bmtr->blijr" if gate.ndim == 2 else "bijst,blsm,bmtr->blijr"
    return torch.einsum(equation, shaped_gate, left, right).reshape(
        left.shape[0], left.shape[1] * 2, 2 * right.shape[-1]
    )


@pytest.mark.parametrize("batched_gate", [False, True])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_two_site_autograd_matches_direct_contraction(
    monkeypatch: pytest.MonkeyPatch, batched_gate: bool, dtype: torch.dtype
) -> None:
    triton = ModuleType("triton")
    triton.jit = lambda function: function
    language = ModuleType("triton.language")
    triton.language = language
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", language)
    path = (
        Path(__file__).resolve().parents[2]
        / "flagquantum/simulation/triton_kernels/mps_two_site.py"
    )
    spec = importlib.util.spec_from_file_location("_test_mps_two_site", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._launch = _cpu_two_site
    generator = torch.Generator().manual_seed(901)
    left = torch.randn(2, 3, 2, 4, dtype=dtype, generator=generator, requires_grad=True)
    right = torch.randn(
        2, 4, 2, 5, dtype=dtype, generator=generator, requires_grad=True
    )
    gate_shape = (2, 4, 4) if batched_gate else (4, 4)
    gate = torch.randn(
        *gate_shape, dtype=dtype, generator=generator, requires_grad=True
    )

    actual = module._FusedMPSTwoSite.apply(left, gate, right)
    expected = _cpu_two_site(left, gate, right)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual.abs().square().sum(), (left, gate, right))
    expected_grad = torch.autograd.grad(
        expected.abs().square().sum(), (left, gate, right)
    )
    for actual_value, expected_value in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual_value, expected_value)
