"""Train one quantum parameter through fq.Module with ordinary PyTorch."""

import argparse
import json
import torch
import flagquantum as fq


def build_circuit(parameters, inputs=None):
    return fq.Circuit(1).ry(0, parameters["theta"][0])


def main(steps=80):
    if steps < 1:
        raise ValueError("steps must be positive")
    model = fq.Module(
        build_circuit,
        parameters={"theta": (1,)},
        init="uniform",
        seed=42,
        policy=fq.RuntimePolicy(
            execution_options=fq.ExecutionOptions(device="cpu", mode="statevector"),
            observable="z_sum",
            observable_wires=(0,),
        ),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.08)
    target = 0.25
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = (model() - target).square().mean()
        losses.append(float(loss.detach()))
        loss.backward()
        if not all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
        ):
            raise RuntimeError("Missing or nonfinite gradient")
        optimizer.step()
    prediction = float(model().detach().reshape(-1)[0])
    return {
        "experiment": "train_z",
        "target": target,
        "prediction": prediction,
        "initial_loss": losses[0],
        "final_loss": (prediction - target) ** 2,
        "losses": losses,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=80)
    print(json.dumps(main(parser.parse_args().steps), indent=2))
