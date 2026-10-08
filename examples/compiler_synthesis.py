"""Rewrite a circuit into a target's own basis with the synthesis entry points.

Run from the repository root:

    python -m examples.compiler_synthesis

Target-independent canonicalization lives in `examples/compiler_optimize.py`.
This example is the other half: it takes gates a target cannot run and spells
them in gates the target publishes, then proves the spelling with the shipped
statevector runtime. The three entry points are reached by module path rather
than through `fq.compiler`, because none of them is a stable export.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import flagquantum as fq
from flagquantum.compiler.one_qubit_synthesis import synthesize_one_qubit
from flagquantum.compiler.state_preparation_synthesis import (
    synthesize_state_preparation,
)
from flagquantum.compiler.two_qubit_synthesis import synthesize_two_qubit
from flagquantum.core.ir import Instruction

#: The target this example compiles for: a z-rotation, a pi/2 pulse about x, and
#: one entangler. Every synthesis entry point has to be told which opcode plays
#: each role, because a target may publish `phase` instead of `rz` or `rx`
#: instead of `sx`.
Z_ROTATION = "rz"
PULSE = "sx"
ENTANGLER = "cx"

#: Every leaf the three entry points may emit for this target.
TARGET_BASIS = frozenset({Z_ROTATION, PULSE, ENTANGLER})

OPTIONS = fq.ExecutionOptions(
    mode="statevector",
    backend="pytorch",
    device="cpu",
    precision="complex128",
    allow_backend_fallback=False,
)


def apply(circuit: fq.Circuit, leaves: Sequence[Instruction]) -> fq.Circuit:
    """Append synthesized leaves to a circuit already holding their input."""

    for leaf in leaves:
        circuit.gate(leaf.name, leaf.wires, params=dict(leaf.params))
    return circuit


def compare(
    reference: torch.Tensor, candidate: torch.Tensor
) -> tuple[float, float, float]:
    """Return (|overlap|, global phase, residual) between two statevectors.

    A synthesis result carries a global phase that FlagQuantum IR cannot record,
    so the two states cannot be compared entrywise. Dividing the candidate by
    the overlap's own phase removes exactly that one degree of freedom, and the
    residual is then the part of the difference the rewrite is responsible for.
    """

    left = reference.reshape(-1)
    right = candidate.reshape(-1)
    overlap = torch.vdot(left, right)
    phase = torch.angle(overlap)
    adjusted = right * torch.exp(-1j * phase.to(right.dtype))
    return (
        float(torch.abs(overlap)),
        float(phase),
        float((left - adjusted).abs().max()),
    )


def state_of(circuit: fq.Circuit) -> torch.Tensor:
    return fq.run(circuit.to_ir(), options=OPTIONS).state


def report(
    label: str,
    leaves: Sequence[Instruction],
    measured: tuple[float, float, float],
) -> None:
    magnitude, phase, residual = measured
    names = [leaf.name for leaf in leaves]
    assert {*names} <= TARGET_BASIS, sorted({*names} - TARGET_BASIS)
    assert magnitude > 1 - 1e-12, magnitude
    assert residual < 1e-12, residual

    print(label)
    print(f"  leaves: {len(leaves)}  entanglers: {names.count(ENTANGLER)}")
    print(f"  |overlap| = {magnitude:.15f}, global phase = {phase:+.15f} rad")
    print(f"  residual after removing the phase = {residual:.3e}")
    print("  every leaf is in the target basis: True")


def one_qubit_case() -> None:
    """Spell a Hadamard as the target's z-rotation and pi/2 pulses."""

    source = fq.Circuit(n_qubits=1, dtype=torch.complex128).h(0)
    leaves = synthesize_one_qubit(
        source.to_ir().instructions[0], z_rotation=Z_ROTATION, pulse_opcode=PULSE
    )
    assert leaves is not None
    print(f"one-qubit synthesis: h(0) -> {[leaf.name for leaf in leaves]}")
    report(
        "  h(0) on |0>",
        leaves,
        compare(
            state_of(source),
            state_of(apply(fq.Circuit(1, dtype=torch.complex128), leaves)),
        ),
    )
    print(
        f"  angles: {[float(leaf.params['theta']) for leaf in leaves if leaf.params]}"
    )


def two_qubit_case() -> None:
    """Spell a gate outside the entangler's class as entanglers and leaves."""

    # `swap` is neither a `cx` nor a local factor, so it exercises the whole KAK
    # path instead of the entangler-alone shortcut. It is read with `qubits[0]` on
    # the most significant index bit, the order `fq.Circuit` itself uses.
    matrix = [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ]

    def input_state() -> fq.Circuit:
        """A product state that is not an eigenstate of `swap`."""

        return fq.Circuit(2, dtype=torch.complex128).h(0).ry(1, theta=0.7)

    leaves = synthesize_two_qubit(
        matrix,
        qubits=(0, 1),
        entangler=ENTANGLER,
        z_rotation=Z_ROTATION,
        pulse_opcode=PULSE,
    )
    assert leaves is not None
    source = input_state().any(0, 1, unitary=matrix)
    print("two-qubit synthesis: swap on qubits (0, 1)")
    report(
        "  swap on h(0) ry(0.7)(1)",
        leaves,
        compare(state_of(source), state_of(apply(input_state(), leaves))),
    )


def state_preparation_case() -> None:
    """Prepare an amplitude vector directly, with no gate-level derivation."""

    amplitudes = [0.5, 0.5, 0.5, 0.5j]
    leaves = synthesize_state_preparation(amplitudes)
    assert leaves is not None
    wanted = torch.tensor(amplitudes, dtype=torch.complex128)
    print("state-preparation synthesis")
    report(
        f"  amplitudes {amplitudes}",
        leaves,
        compare(wanted, state_of(apply(fq.Circuit(2, dtype=torch.complex128), leaves))),
    )


def refusal_case() -> None:
    """Show what the boundary refuses instead of approximating.

    Two entries in this family look alike and are not: `synthesize_one_qubit`
    accepts `phase` and `u1` as its z-rotation, because a single gate may carry
    any global phase, while `synthesize_state_preparation` refuses them, because
    a uniformly controlled ladder is built from a rotation about one qubit and
    `phase` is exactly `exp(i*theta/2)` times `rz` -- a different constant on
    every branch. The ladder therefore requires the entrywise-exact form.
    """

    source = fq.Circuit(n_qubits=1, dtype=torch.complex128).h(0)
    instruction = source.to_ir().instructions[0]

    substituted = synthesize_one_qubit(
        instruction, z_rotation="phase", pulse_opcode=PULSE
    )
    assert substituted is not None
    assert [leaf.name for leaf in substituted] == ["phase", PULSE, "phase"]

    refused_basis = synthesize_one_qubit(
        instruction, z_rotation="u3", pulse_opcode=PULSE
    )
    refused_pulse = synthesize_one_qubit(
        instruction, z_rotation=Z_ROTATION, pulse_opcode="ry"
    )
    refused_ladder = synthesize_state_preparation(
        [0.5, 0.5, 0.5, 0.5j], z_rotation="phase"
    )
    refused_entangler = synthesize_two_qubit(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        qubits=(0, 1),
        entangler="ecr",
        z_rotation=Z_ROTATION,
        pulse_opcode=PULSE,
    )

    assert refused_basis is None
    assert refused_pulse is None
    assert refused_ladder is None
    assert refused_entangler is None

    print("refusals")
    print("  one-qubit z_rotation='u3'      -> None  (not a z-rotation)")
    print("  one-qubit pulse_opcode='ry'    -> None  (not a pi/2 pulse)")
    print("  ladder    z_rotation='phase'   -> None  (needs entrywise-exact `rz`)")
    print("  two-qubit entangler='ecr'      -> None  (not supercontrolled)")
    print("  but the one-qubit path does accept 'phase', for one gate: substituted")


def main() -> None:
    one_qubit_case()
    two_qubit_case()
    state_preparation_case()
    refusal_case()

    print("FlagQuantum compiler synthesis check passed")
    print("  target basis: z-rotation 'rz', pi/2 pulse 'sx', entangler 'cx'")
    print("  next: python -m pytest tests/team/compiler -q")
    print("  contract: flagquantum/compiler/README.md, section 'Start here'")


if __name__ == "__main__":
    main()
