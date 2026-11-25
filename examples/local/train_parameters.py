"""Initialize, freeze, resume, and unfreeze parameters using native PyTorch."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from pathlib import Path

import torch

import flagquantum as fq


def build(parameters: Mapping[str, torch.Tensor]) -> fq.Circuit:
    return fq.Circuit(1).ry(0, parameters["encoder"]).rx(0, parameters["readout"])


def make_model() -> fq.Module:
    return fq.Module(
        build,
        parameters={"encoder": (), "readout": ()},
        init={"encoder": 0.4, "readout": 0.7},
        dtype=torch.float64,
    )


def main() -> None:
    model = make_model()
    encoder = model.parameter_groups["encoder"]
    encoder.requires_grad_(False)
    encoder.grad = None
    # Retain all parameters so later unfreezing needs no optimizer rebuild.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.01)
    frozen = encoder.detach().clone()
    readout_before = model.parameter_groups["readout"].detach().clone()

    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = model().square().mean()
        torch.autograd.backward(loss)
        optimizer.step()

    torch.testing.assert_close(encoder, frozen, atol=0, rtol=0)
    assert not torch.equal(model.parameter_groups["readout"], readout_before)

    with tempfile.TemporaryDirectory(prefix="fq-parameter-training-") as directory:
        checkpoint = model.save_checkpoint(
            Path(directory) / "training.pt", optimizer=optimizer, seed=11, step=3
        )
        restored = make_model()
        restored_optimizer = torch.optim.AdamW(
            restored.parameters(), lr=0.03, weight_decay=0.01
        )
        restored.load_checkpoint(checkpoint, optimizer=restored_optimizer)
        assert not restored.parameter_groups["encoder"].requires_grad
        assert all(parameter.grad is None for parameter in restored.parameters())
        for name, parameter in model.parameter_groups.items():
            torch.testing.assert_close(parameter, restored.parameter_groups[name])

        encoder = restored.parameter_groups["encoder"]
        encoder.requires_grad_(True)
        encoder.grad = None
        before = encoder.detach().clone()
        restored_optimizer.zero_grad(set_to_none=True)
        loss = restored().square().mean()
        torch.autograd.backward(loss)
        assert encoder.grad is not None
        assert bool(encoder.grad.abs() > 0)
        restored_optimizer.step()
        assert not torch.equal(encoder, before)

    print(
        "Frozen encoder unchanged; readout trained; checkpoint resumed; encoder unfrozen."
    )


if __name__ == "__main__":
    main()
