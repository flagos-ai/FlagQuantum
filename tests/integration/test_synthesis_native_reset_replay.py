"""Native invariant reset loops retain channels, ownership and expanded budgets."""

import math

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler._construction.frontend_program import capture_program
from flagquantum.compiler._construction.register_profile import verify_register_program
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from flagquantum.simulation.density_matrix import density_matrix_from_ir
from tests.team.compiler.test_construction_python_program import program_definition

pytestmark = pytest.mark.integration
PREFIX = "from flagquantum.synth import RESET,H,CX,R,QArray\n"
BELL = PREFIX + """
@qfunc
def cycle(q: QArray[QBit]) -> None:
    H(q[0])
    CX(q[0],q[1])
    RESET(q[0])

@qfunc
def main(q: Output[QArray[QBit]]) -> None:
    allocate(2,q)
    for iteration in range(50):
        cycle(q)
"""


def test_fifty_bell_reset_cycles_reuse_template_and_preserve_mixed_partner() -> None:
    definition = program_definition(BELL)
    root, functions = capture_program(definition)
    verify_register_program(root, functions)
    assert len(functions) == 1
    assert sum(op.name == "qreg.replay" for op in root.body.blocks[0].operations) == 1
    assert len(functions[0].body.blocks[0].operations) == 6
    assert (root, functions) == capture_program(definition)
    program = synth.synthesize(definition)
    assert program.execution.state_mode == "density_matrix"
    assert len(program.circuit.instructions) == 150
    assert (
        sum(bool(g.metadata.get("is_channel")) for g in program.circuit.instructions)
        == 50
    )
    expected = torch.diag(torch.tensor([0.5, 0, 0.5, 0], dtype=torch.complex128))
    actual = fq.run(program).state
    assert isinstance(actual, torch.Tensor)
    torch.testing.assert_close(actual.reshape(4, 4), expected, atol=1e-12, rtol=0)
    # Independent Kraus evolution uses no synthesized circuit or replay template.
    state = torch.tensor([1, 0, 0, 0], dtype=torch.complex128)
    rho = state[:, None] * state.conj()[None, :]
    h = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / 2**0.5
    identity = torch.eye(2, dtype=torch.complex128)
    # Qfunc element zero is the least-significant physical bit.
    cx = torch.eye(4, dtype=torch.complex128)[torch.tensor([0, 3, 2, 1])]
    unitary = cx @ torch.kron(identity, h)
    k0 = torch.kron(identity, torch.tensor([[1, 0], [0, 0]], dtype=torch.complex128))
    k1 = torch.kron(identity, torch.tensor([[0, 1], [0, 0]], dtype=torch.complex128))
    for _ in range(50):
        rho = unitary @ rho @ unitary.conj().T
        rho = k0 @ rho @ k0.conj().T + k1 @ rho @ k1.conj().T
    torch.testing.assert_close(rho, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("index", ["i", "49-i"])
def test_index_dependent_reset_loops_expand_each_angle(index: str) -> None:
    source = PREFIX + f"""
@qfunc
def main(q: Output[QBit]) -> None:
    allocate(q)
    for i in range(50):
        RESET(q)
        R(0.01*({index}),1.5707963267948966,q)
"""
    definition = program_definition(source)
    root, functions = capture_program(definition)
    assert not functions
    assert not any(op.name == "qreg.replay" for op in root.body.blocks[0].operations)
    program = synth.synthesize(definition)
    last = 0.49 if index == "i" else 0.0
    vector = torch.tensor(
        [math.cos(last / 2), math.sin(last / 2)],
        dtype=torch.complex128,
    )
    actual = fq.run(program).state
    torch.testing.assert_close(
        actual.reshape(2, 2),
        vector[:, None] * vector.conj()[None, :],
        atol=1e-12,
        rtol=0,
    )


def test_unselected_reset_branch_uses_checked_expansion() -> None:
    source = PREFIX + """
@qfunc
def cycle(q: QBit, flag: int) -> None:
    if flag == 1:
        RESET(q)
    H(q)
@qfunc
def main(q: Output[QBit]) -> None:
    allocate(q)
    for i in range(3):
        cycle(q,0)
"""
    definition = program_definition(source)
    root, functions = capture_program(definition)
    assert not functions
    assert sum(op.name == "qreg.gate" for op in root.body.blocks[0].operations) == 3
    assert synth.synthesize(definition).execution.state_mode != "density_matrix"


@pytest.mark.parametrize("operations,memory", [(149, None), (None, 128)])
def test_expanded_resources_are_charged(
    operations: int | None, memory: int | None
) -> None:
    with pytest.raises(
        (PlanningError, HybridVerificationError), match="budget|limit|memory|infeasible"
    ):
        synth.synthesize(
            program_definition(BELL),
            max_operations=operations,
            memory_limit_bytes=memory,
        )


@pytest.mark.parametrize(
    "body",
    [
        "allocate(q); RESET(q)",
        "RESET(q); RESET(other)",
        "cycle(q,q)",
    ],
)
def test_invalid_cycle_storage_and_aliases_refuse(body: str) -> None:
    source = PREFIX + f"""
@qfunc
def cycle(a:QBit,b:QBit) -> None:
    RESET(a)
    CX(a,b)
@qfunc
def main(q:Output[QBit], other:Output[QBit]) -> None:
    allocate(q)
    for i in range(3):
        {body}
"""
    with pytest.raises(HybridVerificationError):
        synth.synthesize(program_definition(source))


def test_helper_budget_cannot_be_evaded_by_reset_replay() -> None:
    with pytest.raises(HybridVerificationError, match="budget"):
        synth.synthesize(
            program_definition(
                BELL.replace(
                    "cycle(q)\n",
                    "cycle(q)\n        cycle(q)\n        cycle(q)\n        cycle(q)\n        cycle(q)\n        cycle(q)\n",
                )
            )
        )


@pytest.mark.parametrize("theta,phi", [(0.7, 0.3), (-1.1, 0.9)])
def test_complex_initial_state_matches_independent_uncompressed_circuit(
    theta: float, phi: float
) -> None:
    source = BELL.replace(
        "allocate(2,q)",
        f"allocate(2,q)\n    R({theta},{phi},q[0])\n    R({phi},{theta},q[1])",
    )
    program = synth.synthesize(program_definition(source))
    circuit = fq.Circuit(2, dtype=torch.complex128)
    for wire, angle, azimuth in ((1, theta, phi), (0, phi, theta)):
        c, s = math.cos(angle / 2), math.sin(angle / 2)
        phase = complex(math.cos(azimuth), math.sin(azimuth))
        matrix = torch.tensor(
            [[c, -1j * s * phase.conjugate()], [-1j * s * phase, c]],
            dtype=torch.complex128,
        )
        circuit.any(wire, matrix=matrix)
    for _ in range(50):
        circuit.h(1).cx(1, 0).amplitude_damping(1, gamma=1)
    expected = density_matrix_from_ir(circuit.to_ir(), dtype=torch.complex128)
    actual = fq.run(program).state
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    assert abs(expected.reshape(4, 4)[0, 2]) > 1e-3


def test_nested_iterations_and_large_cycle_keep_construction_bounds() -> None:
    nested = BELL.replace(
        "cycle(q)\n", "for j in range(100):\n            RESET(q[0])\n"
    )
    with pytest.raises(HybridVerificationError, match="budget"):
        synth.synthesize(program_definition(nested))
    large = BELL.replace(
        "cycle(q)\n",
        "for j in range(256):\n            H(q[0])\n            RESET(q[0])\n        H(q[0])\n",
    )
    with pytest.raises(HybridVerificationError, match="512"):
        synth.synthesize(program_definition(large))
