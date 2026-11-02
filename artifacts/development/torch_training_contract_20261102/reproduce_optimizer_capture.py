"""Check eager/identity-compiler optimizer trajectories without numeric codegen."""

import hashlib
import json
import subprocess
from pathlib import Path

import torch

import flagquantum as fq
from tests.unit.test_training_compilation_contract_fake import (
    model_and_optimizer,
    numerical_step,
)


def trajectory(kind, capture):
    torch._dynamo.reset()
    model, optimizer = model_and_optimizer(kind)
    graphs = []

    def identity(graph, inputs):
        graphs.append(tuple(str(node.target) for node in graph.graph.nodes))
        return graph.forward

    eager = numerical_step(model, optimizer)
    step = torch.compile(eager, backend=identity, fullgraph=False) if capture else eager
    losses = []
    for index in range(6):
        x = (
            torch.tensor([[0.1], [0.3]], dtype=torch.float64) + index * 0.01
        ).requires_grad_()
        losses.append(float(step(x)))
    return model.parameters_tensor.detach(), losses, graphs


torch.set_num_threads(1)
results = []
for kind in ("SGD", "Adam"):
    eager, losses, _ = trajectory(kind, False)
    compiled, compiled_losses, graphs = trajectory(kind, True)
    results.append(
        dict(
            optimizer=kind,
            parameter_max_error=float((eager - compiled).abs().max()),
            loss_max_error=max(
                abs(a - b) for a, b in zip(losses, compiled_losses, strict=True)
            ),
            eager_parameters=eager.tolist(),
            captured_parameters=compiled.tolist(),
            eager_losses=losses,
            captured_losses=compiled_losses,
            graph_count=len(graphs),
            status=(
                "conformant"
                if torch.allclose(eager, compiled, atol=1e-12, rtol=1e-12)
                else "unsupported_precision"
            ),
        )
    )
root = Path(fq.__file__).resolve().parents[1]
output = dict(
    source_commit=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=10
    ).strip(),
    torch_version=torch.__version__,
    source_dirty=bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            text=True,
            timeout=10,
        ).strip()
    ),
    fake_sha256=hashlib.sha256(
        (root / "tests/unit/test_training_compilation_contract_fake.py").read_bytes()
    ).hexdigest(),
    dtype="float64",
    scope="identity compiler backend; no codegen or performance claim",
    producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    results=results,
)
Path(__file__).with_name("optimizer-capture.json").write_text(
    json.dumps(output, indent=2) + "\n"
)
print([(r["optimizer"], r["status"], r["parameter_max_error"]) for r in results])
