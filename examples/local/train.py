"""Train a one-parameter quantum model with a PyTorch optimizer."""

import torch

import flagquantum as fq


def build(parameters: torch.Tensor) -> fq.Circuit:
    return fq.Circuit(1).ry(0, theta=parameters[0])


def main() -> None:
    model = fq.Module(build, n_parameters=1, init=torch.tensor([0.25]))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    training = fq.train(
        model,
        optimizer=optimizer,
        objective=lambda value: value.mean(),
        steps=8,
    )

    assert training.final_loss < training.losses[0]
    print({"initial_loss": training.losses[0], "final_loss": training.final_loss})


if __name__ == "__main__":
    main()
