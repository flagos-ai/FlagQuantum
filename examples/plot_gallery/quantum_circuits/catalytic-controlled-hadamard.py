"""Redraw Lee et al., arXiv:2011.03494v3, Fig. 17."""

import torch

from flagquantum.dynamic import DynamicCircuit
from flagquantum.experimental.controlled_pauli import controlled_pauli
from flagquantum.experimental.dynamic import run_dynamic


# Wire order: measured ancilla, T catalyst, Hadamard target, control.
# Initialize the ancilla and target to |0> and the catalyst to T|+>.
t_phase = torch.exp(torch.tensor(1j * torch.pi / 4))
initial = torch.zeros((2, 16), dtype=torch.complex64)
# Run both control values with the same catalyst and target.
initial[0, 0] = initial[1, 1] = 2 ** -0.5
initial[0, 4] = initial[1, 5] = t_phase * 2 ** -0.5
# The catalyst qubit carries |T> and returns to |T>.
ancilla, catalyst, target, control = range(4)
m = 0
program = DynamicCircuit(4, bsz=2, inputs=initial)
program.h(target).s(target).h(target)
name, matrix = controlled_pauli("0", "X")
program.gate(name, (target, catalyst), matrix=matrix)
name, matrix = controlled_pauli("10", "X")
program.gate(name, (catalyst, control, ancilla), matrix=matrix)
program.cx(target, ancilla).s(ancilla).cx(target, ancilla).h(ancilla)
program.measure(ancilla, classical_bit=m)
name, matrix = controlled_pauli("0", "Z")
program.conditional(name, (control, catalyst), classical_bit=m, matrix=matrix)
program.cx(control, target).cx(target, catalyst).h(target).sd(target).h(target)

# Compare each measured branch with the expected target and returned catalyst.
shots = 32
result = run_dynamic(program, shots=shots, seed=17)
measured = result.classical_bits[..., m]
branches = result.final_states.reshape(2, shots, 2, 8)
expected = initial[:, :8].clone()
expected[1, 1] = expected[1, 3] = 0.5
expected[1, 5] = expected[1, 7] = t_phase * 0.5
for control_value in (0, 1):
    outcomes = measured[control_value]
    assert set(outcomes.tolist()) == {0, 1}
    states = branches[control_value, torch.arange(shots), outcomes.long()]
    fidelity = (states @ expected[control_value].conj()).abs().square()
    torch.testing.assert_close(fidelity, torch.ones_like(fidelity))

catalyst_label = r"$|T\rangle$"
fig = program.draw(format="latex", inputs={ancilla: r"$|0\rangle$", catalyst: catalyst_label},
                   outputs={catalyst: catalyst_label})
fig.save_source("catalytic-controlled-hadamard.tex")
for extension in ("png", "pdf", "svg"):
    fig.save(f"catalytic-controlled-hadamard.{extension}")
