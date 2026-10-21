"""Train and replay two CPU models through the same existing dynamics workflow.

Run with ``python -m examples.shared_dynamics_workflow``. Frequencies use the
inverse of the time-grid unit. The spin-mode input stays in the one-excitation
sector, so the three-level oscillator cutoff does not approximate this example.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch

import flagquantum as fq
import flagquantum.lindblad as fql
from flagquantum.observables.boson import annihilate


def models() -> (
    Iterator[tuple[str, tuple[int, ...], torch.Tensor, torch.Tensor, torch.Tensor]]
):
    """Yield each model's dimensions, transition, initial batch and observable."""
    transition = torch.zeros((3, 3), dtype=torch.complex128)
    transition[1, 2] = transition[2, 1] = 1
    initial = torch.tensor([[0, 1, 0], [1, 0, 0]], dtype=torch.complex128)
    projector = torch.diag(torch.tensor([0, 0, 1], dtype=torch.complex128))
    yield "three-level", (3,), transition, initial, projector

    lower = torch.tensor([[0, 1], [0, 0]], dtype=torch.complex128)
    ladder = annihilate(0).to_matrix({0: 3})
    exchange = torch.kron(lower.T.contiguous(), ladder) + torch.kron(
        lower, ladder.T.contiguous()
    )
    initial = torch.zeros((2, 6), dtype=torch.complex128)
    initial[0, 3] = initial[1, 0] = 1
    number = torch.diag(torch.tensor([0, 1, 2], dtype=torch.complex128))
    occupation = torch.kron(torch.eye(2, dtype=torch.complex128), number)
    yield "spin-mode", (2, 3), exchange, initial, occupation


def main() -> None:
    times = torch.linspace(0, 0.5, 51, dtype=torch.float64)
    for name, dimensions, transition, initial, observable in models():
        coupling = torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float64))
        optimizer = torch.optim.SGD([coupling], lr=1.0)
        output = fq.expectation(observable, name="transfer")
        options = fq.ExecutionOptions(batch_size=2, require_gradients=True)
        losses = []
        for step in range(5):
            optimizer.zero_grad()
            # A fresh plan keeps the current parameter graph after every update.
            plan = fql.plan(
                coupling * transition,
                initial,
                times,
                dimensions=dimensions,
                outputs=output,
                options=options,
            )
            result = fql.run(plan)
            values = result.expectation("transfer")
            if step == 0:
                (gradient,) = torch.autograd.grad(
                    values[-1, 0], coupling, retain_graph=True
                )
                print(f"{name}: axes={tuple(values.shape)}")
                print(f"Initial transfer: {values[-1, 0].item():.8f}")
                print(f"Coupling derivative: {gradient.item():.8f}")
            loss = (values[-1, 0] - 0.25).square()
            torch.autograd.backward(loss)
            optimizer.step()
            losses.append(loss.item())

        # JSON is a numeric artifact, not a saved PyTorch training graph.
        frozen = fql.plan(
            coupling.detach() * transition,
            initial,
            times,
            dimensions=dimensions,
            outputs=output,
            options=fq.ExecutionOptions(batch_size=2),
        )
        restored = fql.LindbladPlan.from_json(frozen.to_json())
        replay = fql.run(restored).expectation("transfer")
        original = fql.run(frozen).expectation("transfer")
        print(f"Loss decreased: {losses[-1] < losses[0]}")
        print(f"Exact numeric replay: {torch.equal(replay, original)}")


if __name__ == "__main__":
    main()
