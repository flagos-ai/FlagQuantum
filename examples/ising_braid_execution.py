"""Execute and replay native Ising braids, then observe terminal pair charge."""

from __future__ import annotations

import torch

import flagquantum as fq
from flagquantum.experimental.simulation import IsingBraidProgram
from flagquantum.runtime import sample_terminal


def main() -> None:
    program = IsingBraidProgram().exchange(1).exchange(0, clockwise=True)
    output = fq.expectation(program.projector("psi"), name="psi")
    selected = fq.plan(program, outputs=output)
    direct = fq.run(program, outputs=output)
    replay = fq.ExecutionPlan.from_json(selected.to_json())
    result = fq.run(replay)
    assert direct.state is not None and result.state is not None
    instrument = {label: (program.projector(label),) for label in ("1", "psi")}
    before_rng = torch.random.get_rng_state().clone()
    sampled = sample_terminal(result.state[0], instrument, shots=2000, seed=7)
    assert isinstance(sampled.value, torch.Tensor)

    print("Plan schema:", selected.schema_version)
    print("Qubit count:", selected.summary()["n_wires"])
    print("State shape:", tuple(result.state.shape))
    print("P(psi):", round(float(result.expectation("psi")[0]), 12))
    print("Exact replay:", torch.equal(direct.state, result.state))
    print("Terminal shots:", sampled.value.numel())
    print(
        "Caller RNG unchanged:", torch.equal(before_rng, torch.random.get_rng_state())
    )


if __name__ == "__main__":
    main()
