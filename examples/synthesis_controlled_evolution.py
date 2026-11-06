"""BPDE original, existing native phase composition and independent Circuit.

The native comparison uses the admitted qfunc phase helpers and explicit static
Trotter steps. It does not add a native Hamiltonian-array or evolution API.
"""

import math
from pathlib import Path

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.services.synthesis import _synthesize
from flagquantum.synth import (
    Const,
    Output,
    Program,
    QBit,
    allocate,
    hadamard_transform,
    phase,
    qfunc,
)

H0 = (-0.530427259238194, 0.536032051970536, 0.196711389481122)
H1 = (-0.531672967442228, 0.534084446291781, 0.196869810923244)
I0 = -0.5 * H0[0]
ID = -0.5 * (H1[0] - H0[0])
ZG0 = -0.25 * H0[1]
ZGD = -0.25 * (H1[1] - H0[1])
Z0 = 0.5 * H0[1]
ZD = 0.5 * (H1[1] - H0[1])
XG0 = -0.5 * H0[2]
XGD = -0.5 * (H1[2] - H0[2])
X0 = H0[2]
XD = H1[2] - H0[2]


@qfunc
def identity_phase(r: Const[QBit]) -> None:
    phase(1, I0)
    phase(r, ID)


@qfunc
def z_half(r: Const[QBit], s: Const[QBit]) -> None:
    phase(1, ZG0)
    phase(r, ZGD)
    phase(s, Z0)
    phase(r * s, ZD)


@qfunc
def x_full_in_z_basis(r: Const[QBit], s: Const[QBit]) -> None:
    phase(1, XG0)
    phase(r, XGD)
    phase(s, X0)
    phase(r * s, XD)


def native_program(delta: float) -> Program:
    phase_angle = 4 * delta

    @qfunc
    def main(readout: Output[QBit], state: Output[QBit]) -> None:
        allocate(readout)
        state |= 1
        hadamard_transform(readout)
        # Strang step 1/8, dt=0.5.
        identity_phase(readout)
        z_half(readout, state)
        hadamard_transform(state)
        x_full_in_z_basis(readout, state)
        hadamard_transform(state)
        z_half(readout, state)
        # Strang step 2/8, dt=0.5.
        identity_phase(readout)
        z_half(readout, state)
        hadamard_transform(state)
        x_full_in_z_basis(readout, state)
        hadamard_transform(state)
        z_half(readout, state)
        # Strang step 3/8, dt=0.5.
        identity_phase(readout)
        z_half(readout, state)
        hadamard_transform(state)
        x_full_in_z_basis(readout, state)
        hadamard_transform(state)
        z_half(readout, state)
        # Strang step 4/8, dt=0.5.
        identity_phase(readout)
        z_half(readout, state)
        hadamard_transform(state)
        x_full_in_z_basis(readout, state)
        hadamard_transform(state)
        z_half(readout, state)
        # Strang step 5/8, dt=0.5.
        identity_phase(readout)
        z_half(readout, state)
        hadamard_transform(state)
        x_full_in_z_basis(readout, state)
        hadamard_transform(state)
        z_half(readout, state)
        # Strang step 6/8, dt=0.5.
        identity_phase(readout)
        z_half(readout, state)
        hadamard_transform(state)
        x_full_in_z_basis(readout, state)
        hadamard_transform(state)
        z_half(readout, state)
        # Strang step 7/8, dt=0.5.
        identity_phase(readout)
        z_half(readout, state)
        hadamard_transform(state)
        x_full_in_z_basis(readout, state)
        hadamard_transform(state)
        z_half(readout, state)
        # Strang step 8/8, dt=0.5.
        identity_phase(readout)
        z_half(readout, state)
        hadamard_transform(state)
        x_full_in_z_basis(readout, state)
        hadamard_transform(state)
        z_half(readout, state)
        phase(readout, phase_angle)
        hadamard_transform(readout)

    return synth.synthesize(main, memory_limit_bytes=64 * 1024**2)


def reference(delta: float) -> fq.Circuit:
    z = torch.diag(torch.tensor([1.0, -1.0], dtype=torch.complex128))
    x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)
    matrix = torch.zeros((4, 4), dtype=torch.complex128)
    for bit, (scalar, cz, cx) in enumerate((H0, H1)):
        half_z = torch.matrix_exp(-1j * 0.25 * cz * z)
        step = (half_z @ torch.matrix_exp(-1j * 0.5 * cx * x) @ half_z) * complex(
            math.cos(-0.5 * scalar), math.sin(-0.5 * scalar)
        )
        matrix[2 * bit : 2 * bit + 2, 2 * bit : 2 * bit + 2] = step.matrix_power(8)
    return (
        fq.Circuit(2, dtype=torch.complex128)
        .x(1)
        .h(0)
        .any(0, 1, matrix=matrix)
        .phase(0, theta=4 * delta)
        .h(0)
    )


def run_example() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "tests/fixtures/classiq_qmod/bpde.qmod"
    )
    for delta in (0.37, -0.81):
        expected = fq.run(reference(delta)).statevector().reshape(-1)
        for program in (
            _synthesize(
                source,
                classical_inputs={"delta_e": delta},
                memory_limit_bytes=64 * 1024**2,
            ),
            native_program(delta),
        ):
            result = fq.run(program)
            actual = result.statevector().reshape(4, -1)
            torch.testing.assert_close(actual[:, 0], expected, atol=1e-12, rtol=0)
            assert float(actual[:, 1:].abs().square().sum()) < 1e-24
            print(delta, dict(program.probabilities(result, registers=("readout",))))
            print(program.report["resources"])
    print(
        "Original Qmod, native phase composition and Circuit agree, including identity-term interference."
    )


if __name__ == "__main__":
    run_example()
