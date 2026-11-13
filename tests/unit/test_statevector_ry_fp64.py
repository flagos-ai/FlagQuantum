"""FP64 Torch-FL RY kernel and derivative conformance."""

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

from flagquantum.kernels.catalog import (
    KernelRequest,
    match_kernel_implementations,
    validate_catalog,
)
from flagquantum.simulation.statevector.rotation_autograd import _RYRotation

pytestmark = [pytest.mark.gpu, pytest.mark.triton, pytest.mark.integration]


def reference(x, p, w, n):
    v = x.reshape(x.shape[0], 1 << w, 2, 1 << (n - w - 1))
    c = (p[:, 0] / 2).cos().reshape(-1, 1, 1)
    s = (p[:, 0] / 2).sin().reshape(-1, 1, 1)
    return torch.stack(
        (c * v[:, :, 0] - s * v[:, :, 1], s * v[:, :, 0] + c * v[:, :, 1]), 2
    ).reshape_as(x)


def matrix(p):
    c = (p[:, 0] / 2).cos()
    s = (p[:, 0] / 2).sin()
    r = torch.stack((c, -s, s, c), -1).reshape(-1, 2, 2)
    return torch.complex(r, torch.zeros_like(r))


@pytest.mark.parametrize("w", [0, 3, 5])
@pytest.mark.parametrize("batch,shared", [(1, True), (2, True), (2, False)])
def test_fused_actions(w, batch, shared):
    from flagquantum.kernels.triton.statevector_ry_fp64 import backward, forward

    torch.manual_seed(131)
    x = torch.randn(batch, 64, dtype=torch.complex128, device="cpu").to("flagos:0")
    g = torch.randn_like(x)
    p = torch.randn(1 if shared else batch, 1, dtype=torch.float64).to(x.device)
    # Independent differentiable tensor reference, also tests both complex components.
    with torch.enable_grad():
        a = x.detach().requires_grad_()
        t = p.detach().requires_grad_()
        r = reference(a, t, w, 6)
        gx, gp = torch.autograd.grad(r, (a, t), g)
    out = forward(x, matrix(p), w, 6)
    sg, pg = backward(g, out, matrix(p), w, 6)
    if shared:
        pg = pg.sum(0, keepdim=True)
    torch.testing.assert_close(out, r, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(sg, gx, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(pg, gp, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("w", [0, 3, 5])
@pytest.mark.parametrize("batch,shared", [(1, True), (2, True), (2, False)])
def test_rotation_derivatives(w, batch, shared):
    torch.manual_seed(17)
    base = torch.randn(batch, 64, dtype=torch.complex128)
    params = torch.randn(1 if shared else batch, 1, dtype=torch.float64)
    probe = torch.randn_like(base)
    results = []
    for fused in [False, True]:
        x = base.to("flagos:0").detach().requires_grad_()
        p = params.to(x.device).detach().requires_grad_()
        out = (
            _RYRotation.apply(x, p, matrix(p).detach(), w, 6)
            if fused
            else reference(x, p, w, 6)
        )
        loss = (
            out.conj() * probe.to(x.device)
        ).real.sum() + 0.13 * out.abs().square().sum()
        first = torch.autograd.grad(loss, (x, p), create_graph=True)
        second = torch.autograd.grad(
            first[0].abs().square().sum() + first[1].square().sum(), (x, p)
        )
        # Repeat ordinary backward to exercise the fused first-order VJP.
        first_fast = torch.autograd.grad(
            (
                (
                    _RYRotation.apply(x, p, matrix(p).detach(), w, 6)
                    if fused
                    else reference(x, p, w, 6)
                ).conj()
                * probe.to(x.device)
            ).real.sum(),
            (x, p),
        )
        results.append((out, *first, *second, *first_fast))
    for a, b in zip(*results, strict=True):
        torch.testing.assert_close(a, b, atol=2e-10, rtol=2e-10)


def test_catalog_precision_boundary():
    validate_catalog()
    for dtype, expected in [("complex128", True), ("complex64", False)]:
        result = match_kernel_implementations(
            KernelRequest(
                semantic_id="gradient.vjp.ry.local",
                device="flagos",
                dtype=dtype,
                layout="flat_statevector",
                direction="vjp",
                addressing=("local",),
                providers=("triton",),
            )
        )
        assert result.matched is expected


@pytest.mark.parametrize("state_grad", [False, True])
def test_one_gradient_requested(state_grad):
    torch.manual_seed(97)
    x = (
        torch.randn(2, 64, dtype=torch.complex128)
        .to("flagos:0")
        .requires_grad_(state_grad)
    )
    p = (
        torch.randn(1, 1, dtype=torch.float64)
        .to(x.device)
        .requires_grad_(not state_grad)
    )
    probe = torch.randn_like(x)
    results = []
    for fused in [False, True]:
        y = (
            _RYRotation.apply(x, p, matrix(p).detach(), 5, 6)
            if fused
            else reference(x, p, 5, 6)
        )
        results.append(
            torch.autograd.grad((y.conj() * probe).real.sum(), x if state_grad else p)[
                0
            ]
        )
    torch.testing.assert_close(results[0], results[1], atol=1e-11, rtol=1e-11)
