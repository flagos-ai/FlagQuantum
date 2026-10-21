"""Use the shared expectation request on a three-level CPU Lindblad model."""

import torch

import flagquantum as fq
import flagquantum.lindblad as fql
from flagquantum.errors import CapabilityError


def main() -> None:
    coupling = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    transition = torch.zeros((3, 3), dtype=torch.complex128)
    transition[1, 2] = transition[2, 1] = 1
    projector = torch.diag(torch.tensor([0.0, 0.0, 1.0], dtype=torch.complex128))
    output = fq.expectation(projector, name="leakage")
    plan = fql.plan(
        coupling * transition,
        [0.0, 1.0, 0.0],
        torch.linspace(0.0, 0.5, 201, dtype=torch.float64),
        dimensions=(3,),
        outputs=output,
        options=fq.ExecutionOptions(require_gradients=True),
    )
    value = fql.run(plan).expectation("leakage")[-1]
    (derivative,) = torch.autograd.grad(value, coupling)
    print(f"Leakage expectation: {value.item():.8f}")
    print(f"Coupling derivative: {derivative.item():.8f}")
    try:
        fq.plan(fq.Circuit(1), outputs=fq.expectation(torch.eye(2)))
    except CapabilityError:
        print("Circuit matrix output: refused; use flagquantum.lindblad")


if __name__ == "__main__":
    main()
