"""Public builder portability and gradient-cut reproducer; timings are out of scope."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import torch

import flagquantum as fq


def make_builder(recipe, calls):
    def build(p, x):
        calls.append(1)
        if recipe == "square_method":
            angle = p[0].square() + p[1] * x[:, 0]
        elif recipe == "torch_sin":
            angle = torch.sin(p[0]) + p[1] * x[:, 0]
        elif recipe == "gradient_cut":
            angle = p[0].detach() + p[1] * x[:, 0]
        else:
            angle = p[0] ** 2 + p[1] * x[:, 0]
        return fq.Circuit(1, bsz=x.shape[0]).ry(0, angle)

    return build


def reproduce():
    records = []
    for recipe in ("power", "square_method", "torch_sin", "gradient_cut"):
        for backend in ("pytorch", "jax"):
            calls = []
            build = make_builder(recipe, calls)

            model = fq.Module(
                build,
                2,
                init=torch.tensor([0.2, -0.3], dtype=torch.float64),
                dtype=torch.float64,
                policy=fq.RuntimePolicy(
                    execution_options=fq.ExecutionOptions(backend=backend)
                ),
            )
            row = dict(
                recipe=recipe,
                backend=backend,
                value_max_error=0.0,
                parameter_vjp_max_error=0.0,
                input_vjp_max_error=0.0,
            )
            try:
                for step in range(3):
                    x = (
                        torch.tensor([[0.1], [0.3]], dtype=torch.float64) + step * 0.01
                    ).requires_grad_()
                    p = model.parameters_tensor
                    value = model(x)
                    if recipe == "torch_sin":
                        angle = p[0].sin() + p[1] * x[:, 0]
                    elif recipe == "gradient_cut":
                        angle = p[0].detach() + p[1] * x[:, 0]
                    else:
                        angle = p[0].square() + p[1] * x[:, 0]
                    expected = angle.cos()
                    row["value_max_error"] = max(
                        row["value_max_error"],
                        float((value.detach() - expected.detach()).abs().max()),
                    )
                    torch.testing.assert_close(value, expected, atol=1e-10, rtol=1e-10)
                    weights = torch.tensor([-0.7, 1.3], dtype=torch.float64)
                    actual = torch.autograd.grad((value * weights).sum(), (p, x))
                    reference = torch.autograd.grad((expected * weights).sum(), (p, x))
                    for key, a, b in zip(
                        ("parameter_vjp_max_error", "input_vjp_max_error"),
                        actual,
                        reference,
                        strict=True,
                    ):
                        row[key] = max(row[key], float((a - b).abs().max()))
                        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
                row["status"] = "completed"
                with torch.no_grad():
                    details = model.execute(x)
                row["builder_lowering"] = details.runtime.get(
                    "builder_lowering", "not_reported"
                )
                row["jax_kernel_build_count"] = details.runtime.get(
                    "jax_kernel_build_count"
                )
            except Exception as error:
                row.update(
                    status="failed",
                    error_type=type(error).__name__,
                    cause_type=type(error.__cause__).__name__,
                )
            row["builder_calls"] = len(calls)
            records.append(row)
    root = Path(fq.__file__).resolve().parents[1]

    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True, timeout=10
        ).strip()

    return dict(
        source_commit=git("rev-parse", "HEAD"),
        source_dirty=bool(git("status", "--porcelain", "--untracked-files=no")),
        imported_package=str(Path(fq.__file__).resolve()),
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        versions={
            name: importlib.metadata.version(name)
            for name in ("torch", "jax", "jaxlib")
        },
        scope="public Module values, VJPs and Python builder calls; no performance claim",
        results=records,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = reproduce()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        {
            "completed": sum(row["status"] == "completed" for row in result["results"]),
            "cases": len(result["results"]),
        }
    )
