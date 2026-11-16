"""Exact permutation and derivative conformance for FP64 Torch-FL CX kernels."""

import os
from importlib import import_module

import pytest

if os.environ.get("FLAGQUANTUM_TEST_TORCH_FL_FP64") != "1":
    pytest.skip(
        "set FLAGQUANTUM_TEST_TORCH_FL_FP64=1 in a validated Torch-FL GPU environment",
        allow_module_level=True,
    )

else:
    # Registration must happen before importing torch in an opted-in process.
    import_module("torch_fl")

import torch

from flagquantum.simulation.statevector.cx_fp64_autograd import apply_cx_sequence

pytestmark = [pytest.mark.gpu, pytest.mark.triton, pytest.mark.integration]


def reference(state, controls, targets, n):
    # Independent CPU dense permutation, constructed in chronological source order.
    values = torch.arange(1 << n, dtype=torch.int64)
    dest = values.clone()
    for c, t in zip(controls, targets, strict=True):
        dest ^= ((dest >> (n - c - 1)) & 1) << (n - t - 1)
    inverse = torch.argsort(dest)
    return state.index_select(1, inverse.to(state.device))


@pytest.mark.parametrize(
    "controls,targets",
    [
        ((0,), (5,)),
        ((5,), (0,)),
        ((0, 1, 2), (1, 2, 3)),
        ((5, 1, 3), (1, 3, 5)),
        ((0, 3, 1, 5), (3, 1, 5, 0)),
    ],
)
@pytest.mark.parametrize("batch", [1, 2])
def test_cx_sequence(controls, targets, batch):
    torch.manual_seed(37)
    x = torch.randn(batch, 64, dtype=torch.complex128).to("flagos:0")
    actual = apply_cx_sequence(x, controls, targets, 6)
    torch.testing.assert_close(
        actual, reference(x, controls, targets, 6), atol=0, rtol=0
    )


def test_cx_derivatives():
    torch.manual_seed(31)
    a = torch.randn(2, 64, dtype=torch.complex128).to("flagos:0")
    probe = torch.randn_like(a)
    results = []
    for fused in [False, True]:
        x = a.detach().requires_grad_()
        y = (
            apply_cx_sequence(x, (0, 2, 5), (2, 5, 0), 6)
            if fused
            else reference(x, (0, 2, 5), (2, 5, 0), 6)
        )
        loss = (y.conj() * probe).real.sum() + 0.13 * y.abs().square().sum()
        (first,) = torch.autograd.grad(loss, x, create_graph=True)
        (second,) = torch.autograd.grad(first.abs().square().sum(), x)
        results.append((y, first, second))
    for x, y in zip(*results, strict=True):
        torch.testing.assert_close(x, y, atol=1e-12, rtol=1e-12)
    x = torch.randn(1, 8, dtype=torch.complex128).to("flagos:0").requires_grad_()

    def f(x: torch.Tensor) -> torch.Tensor:
        return apply_cx_sequence(x, (0, 1), (1, 2), 3)

    assert torch.autograd.gradcheck(f, (x,))
    assert torch.autograd.gradgradcheck(f, (x,))
