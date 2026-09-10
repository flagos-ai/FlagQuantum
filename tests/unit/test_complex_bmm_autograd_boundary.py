"""CPU checks of CUDA BMM autograd wrappers with replacement launch functions."""

import importlib.util
from math import prod
from pathlib import Path
from types import ModuleType

import pytest
import torch

pytestmark = pytest.mark.unit


@pytest.fixture
def bmm_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    import sys

    triton = ModuleType("triton")
    triton.jit = lambda function: function
    language = ModuleType("triton.language")
    triton.language = language
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", language)
    path = (
        Path(__file__).resolve().parents[2]
        / "flagquantum/simulation/triton_kernels/complex_bmm.py"
    )
    spec = importlib.util.spec_from_file_location("_test_complex_bmm", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cpu_launch(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    conjugate_left: bool = False,
    conjugate_right: bool = False,
) -> torch.Tensor:
    return torch.bmm(
        left.conj() if conjugate_left else left,
        right.conj() if conjugate_right else right,
    )


def _cpu_layout_launch(
    left: torch.Tensor,
    right: torch.Tensor,
    left_permutation: tuple[int, ...],
    right_permutation: tuple[int, ...],
    batch_shape: tuple[int, ...],
    left_shape: tuple[int, ...],
    contracted_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    *,
    conjugate_left: bool = False,
    conjugate_right: bool = False,
) -> torch.Tensor:
    batch, rows, reduction, cols = map(
        prod, (batch_shape, left_shape, contracted_shape, right_shape)
    )
    return _cpu_launch(
        left.permute(left_permutation).reshape(batch, rows, reduction),
        right.permute(right_permutation).reshape(batch, reduction, cols),
        conjugate_left=conjugate_left,
        conjugate_right=conjugate_right,
    )


@pytest.mark.parametrize("layout", [False, True])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_bmm_autograd_wrappers_match_torch(
    bmm_module: ModuleType, layout: bool, dtype: torch.dtype
) -> None:
    bmm_module._launch = _cpu_launch
    bmm_module._launch_layout = _cpu_layout_launch
    generator = torch.Generator().manual_seed(417)
    left = torch.randn(2, 3, 4, dtype=dtype, generator=generator)
    right = torch.randn(2, 4, 5, dtype=dtype, generator=generator)
    if layout:
        left, right = left.transpose(1, 2), right.transpose(1, 2)
    left.requires_grad_()
    right.requires_grad_()
    if layout:
        actual = bmm_module._FusedComplexLayoutBMM.apply(
            left, right, (0, 2, 1), (0, 2, 1), ((2,), (3,), (4,), (5,))
        )
        expected = torch.bmm(left.transpose(1, 2), right.transpose(1, 2))
    else:
        actual = bmm_module._FusedComplexBMM.apply(left, right)
        expected = torch.bmm(left, right)

    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual.abs().square().sum(), (left, right))
    expected_grad = torch.autograd.grad(expected.abs().square().sum(), (left, right))
    for actual_value, expected_value in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual_value, expected_value)
