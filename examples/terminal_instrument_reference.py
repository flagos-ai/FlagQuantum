"""Current-API mathematical references for proposal 083; no detector API exists.

SuperOperator supplies the quantum maps. Torch supplies the illustrative draws;
this script does not implement a framework instrument parser, detector, event
record, public sampling entry, feedback or artifact replay API.
"""

from __future__ import annotations

import math

import torch

from flagquantum.observables.boson import number
from flagquantum.operators import SuperOperator


def presence_reference() -> tuple[torch.Tensor, dict[str, tuple[torch.Tensor, ...]]]:
    state = torch.tensor([math.sqrt(0.6), 0.0, math.sqrt(0.4)], dtype=torch.complex128)
    rho = torch.outer(state, state.conj())
    quiet = torch.diag(torch.tensor([1.0, 1.0, math.sqrt(0.2)], dtype=torch.complex128))
    flag = torch.diag(torch.tensor([0.0, 0.0, math.sqrt(0.8)], dtype=torch.complex128))
    return rho, {"no_flag": (quiet,), "flag": (flag,)}


def fock_reference() -> tuple[torch.Tensor, dict[str, tuple[torch.Tensor, ...]]]:
    state = torch.sqrt(torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64)).to(
        torch.complex128
    )
    values, vectors = torch.linalg.eigh(number(0).to_matrix({0: 3}))
    branches = {
        str(round(value.item())): (torch.outer(vector, vector.conj()),)
        for value, vector in zip(values, vectors.unbind(-1), strict=True)
    }
    return torch.outer(state, state.conj()), branches


def main() -> None:
    for name, (rho, branches) in (
        ("presence", presence_reference()),
        ("fock", fock_reference()),
    ):
        states = [
            SuperOperator.from_kraus(operators).apply(rho)
            for operators in branches.values()
        ]
        probabilities = torch.stack([torch.trace(state).real for state in states])
        print(f"{name} labels:", tuple(branches))
        print(f"{name} probabilities:", [round(x, 8) for x in probabilities.tolist()])
        seed = 7
        draws = torch.multinomial(
            probabilities,
            10000,
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        replay = torch.multinomial(
            probabilities,
            10000,
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        print(f"{name} Torch draw replay:", torch.equal(draws, replay))
        if name == "presence":
            print(
                f"No-flag conditional ground: {(states[0][0, 0].real / probabilities[0]).item():.8f}"
            )
            print(
                "Unread measurement changes coherence:",
                not torch.allclose(sum(states), rho, atol=1e-10, rtol=0),
            )
    print("Framework terminal instrument API: flagquantum.runtime.sample_terminal")


if __name__ == "__main__":
    main()
