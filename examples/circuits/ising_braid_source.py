"""Build a native Ising source and shared outputs before execution admission."""

from __future__ import annotations

import flagquantum as fq
from flagquantum.experimental.simulation import IsingBraidProgram


def main() -> None:
    program = IsingBraidProgram(initial_channel="1")
    program.exchange(1).exchange(0, clockwise=True)
    snapshot = program.to_ir()
    program.exchange(2)
    effect = program.projector("psi")
    output = fq.expectation(effect, name="fusion_psi")

    print("Fusion paths:", program.basis_labels)
    print("Independent snapshots:", snapshot != program.to_ir())
    print("Charge observable:", output.name)
    print("Projector:", tuple(effect.shape), effect.dtype, effect.device.type)


if __name__ == "__main__":
    main()
