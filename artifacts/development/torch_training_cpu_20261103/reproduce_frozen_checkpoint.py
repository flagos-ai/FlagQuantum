import json
import subprocess
import tempfile
from pathlib import Path

import torch

import flagquantum as fq


def build(p, x):
    return (
        fq.Circuit(2, bsz=x.shape[0], dtype=torch.complex128)
        .ry(0, p[0].square() + p[1] * x[:, 0].sin())
        .rx(1, p[0].detach() * x[:, 1])
        .cx(0, 1)
    )


def model():
    return fq.Module(
        build,
        2,
        dtype=torch.float64,
        init=torch.tensor([0.23, -0.31], dtype=torch.float64),
        policy=fq.RuntimePolicy(observable="z_sum", observable_qubits=(0, 1)),
    )


records = []
for trainable in [False, True]:
    with tempfile.TemporaryDirectory() as directory:
        source = model()
        opt = torch.optim.Adam(source.parameters(), lr=0.03)
        for step in range(3):
            x = (
                torch.linspace(-0.3, 0.5, 16, dtype=torch.float64).reshape(8, 2)
                + step * 0.01
            ).requires_grad_(trainable)
            opt.zero_grad(set_to_none=True)
            source.execute(x).require_value().square().mean().backward()
            opt.step()
        path = Path(directory) / "saved.pt"
        source.save_checkpoint(path, optimizer=opt, seed=11, step=3)
        target = model()
        targetopt = torch.optim.Adam(target.parameters(), lr=0.03)
        target.execute(x.detach().clone().requires_grad_(trainable))
        try:
            target.load_checkpoint(path, optimizer=targetopt)
            record = {"trainable_features": trainable, "status": "restored"}
        except Exception as error:
            record = {
                "trainable_features": trainable,
                "status": "refused",
                "type": type(error).__name__,
                "reason": str(error),
            }
        records.append(record)
print(
    json.dumps(
        {
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(fq.__file__).resolve().parents[1],
                text=True,
                timeout=15,
            ).strip(),
            "records": records,
        },
        indent=2,
    )
)
