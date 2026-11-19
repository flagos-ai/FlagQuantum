"""Shared pure borrowed helpers preserve arbitrary input amplitudes and budgets."""

import math
from dataclasses import replace

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler._construction.frontend_program import capture_program
from flagquantum.compiler._construction.register_profile import verify_register_program
from flagquantum.compiler._construction.selection import select_register_program
from flagquantum.compiler._construction.target import Target
from flagquantum.compiler._hybrid.model import HybridProgram, Operation
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.compiler.resource_estimation import ResourceEstimate
from tests.team.compiler.test_construction_python_program import program_definition

pytestmark = pytest.mark.integration
IMPORTS = "from flagquantum.synth import H, X, R, RZ, phase, control, within, apply, adjoint, RESET\n"
KICK = """
@qfunc
def kick(theta: float, q: QBit) -> None:
    for i in range(16):
        RZ(theta, q)
        R(theta, 0, q)
    phase(1, 0.13)
"""


def _kick(circuit: fq.Circuit, wire: int, theta: float) -> None:
    for _ in range(16):
        circuit.rz(wire, theta).rx(wire, theta)
    circuit.any(
        wire,
        matrix=torch.eye(2, dtype=torch.complex128)
        * complex(math.cos(0.13), math.sin(0.13)),
    )


def _replace_operations(
    program: HybridProgram, operations: tuple[Operation, ...]
) -> HybridProgram:
    block = program.body.blocks[0]
    return replace(
        program,
        body=replace(program.body, blocks=(replace(block, operations=operations),)),
    )


def test_shared_helper_remaps_actuals_and_specializes_nonzero_angles() -> None:
    definition = program_definition(
        f"""{IMPORTS}{KICK}
@qfunc
def main(x: Output[QBit], y: Output[QBit]) -> None:
    allocate(x)
    allocate(y)
    H(x)
    H(y)
    kick(0.27, x)
    kick(0.41, y)
    kick(0.27, y)
    kick(0.41, x)
    kick(0.27, x)
""",
    )
    root, functions = capture_program(definition)
    verify_register_program(root, functions)
    assert len(functions) == 2
    assert (
        sum(op.name == "qreg.unitary_call" for op in root.body.blocks[0].operations)
        == 3
    )
    assert (root, functions) == capture_program(definition)
    program = synth.synthesize(definition, max_qubits=2, max_workspace=0)
    reference = fq.Circuit(2, dtype=torch.complex128).h(0).h(1)
    for wire, theta in ((0, 0.27), (1, 0.41), (1, 0.27), (0, 0.41), (0, 0.27)):
        _kick(reference, wire, theta)
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(),
        reference.state().flatten(),
        atol=1e-12,
        rtol=0,
    )
    # Each helper has 32 rotations and four gates for its constant global phase.
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    assert resources.n_operations == 182
    with pytest.raises(ValueError, match="operation|budget"):
        synth.synthesize(definition, max_operations=181)


def test_shared_adjoint_preserves_absolute_phase_and_control_interference() -> None:
    definition = program_definition(
        f"""{IMPORTS}{KICK}
@qfunc
def main(control_bit: Output[QBit], q: Output[QBit]) -> None:
    allocate(control_bit)
    allocate(q)
    H(control_bit)
    H(q)
    kick(0.37, q)
    with adjoint():
        kick(0.37, q)
    control(control_bit, lambda: kick(0.37, q))
    H(control_bit)
""",
    )
    program = synth.synthesize(definition)
    # Build a unitary from independent basis-state Circuit evolution.
    columns = []
    for basis in range(2):
        circuit = fq.Circuit(1, dtype=torch.complex128)
        if basis:
            circuit.x(0)
        _kick(circuit, 0, 0.37)
        columns.append(circuit.state().flatten())
    matrix = torch.zeros((4, 4), dtype=torch.complex128)
    matrix[:2, :2] = torch.eye(2, dtype=torch.complex128)
    matrix[2:, 2:] = torch.stack(columns, dim=1)
    reference = fq.Circuit(2, dtype=torch.complex128).h(0).h(1)
    reference.any(0, 1, matrix=matrix).h(0)
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(),
        reference.state().flatten(),
        atol=1e-12,
        rtol=0,
    )


def test_three_hundred_calls_keep_source_bounded_but_charge_all_gates() -> None:
    definition = program_definition(
        f"""{IMPORTS}{KICK}
@qfunc
def repeat_kick(q: QBit) -> None:
    for index in range(150):
        kick(0.071, q)
        kick(0.071, q)
@qfunc
def main(q: Output[QBit]) -> None:
    allocate(q)
    H(q)
    repeat_kick(q)
""",
    )
    root, functions = capture_program(definition)
    assert len(root.body.blocks[0].operations) < 400
    program = synth.synthesize(definition, max_qubits=1, max_workspace=0)
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    assert resources.n_operations == 10801
    reference = fq.Circuit(1, dtype=torch.complex128).h(0)
    for _ in range(300):
        _kick(reference, 0, 0.071)
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(),
        reference.state().flatten(),
        atol=2e-11,
        rtol=0,
    )
    with pytest.raises(ValueError, match="operation|budget"):
        synth.synthesize(definition, max_operations=10800)


def test_shared_const_predicate_scratch_is_clean_for_nonzero_target() -> None:
    definition = program_definition(
        f"""{IMPORTS}
@qfunc
def marked(x: Const[QNum], q: QBit) -> None:
    for index in range(7):
        control(x == 2, lambda: R(0.21, 0, q))
@qfunc
def main(q: Output[QBit], x: Output[QNum[2]]) -> None:
    allocate(q)
    allocate(x)
    X(q)
    hadamard_transform(x)
    marked(x, q)
    marked(x, q)
    with adjoint():
        marked(x, q)
""",
    )
    root, functions = capture_program(definition)
    assert any(op.name == "qreg.unitary_call" for op in root.body.blocks[0].operations)
    program = synth.synthesize(definition)
    reference = fq.Circuit(3, dtype=torch.complex128).x(0).h(1).h(2)
    # q occupies the most significant wire; x==2 is local basis suffix 10.
    matrix = torch.eye(8, dtype=torch.complex128)
    theta = 7 * 0.21
    matrix[2, 2] = matrix[6, 6] = math.cos(theta / 2)
    matrix[2, 6] = matrix[6, 2] = torch.tensor(
        -1j * math.sin(theta / 2), dtype=torch.complex128
    )
    reference.any(0, 1, 2, matrix=matrix)
    actual = fq.run(program).statevector().reshape(8, -1)
    torch.testing.assert_close(
        actual[:, 0], reference.state().flatten(), atol=1e-12, rtol=0
    )
    assert torch.count_nonzero(actual[:, 1:]) == 0
    with pytest.raises(ValueError, match="workspace|budget"):
        synth.synthesize(definition, max_workspace=0)


def test_cached_body_cannot_launder_const_storage() -> None:
    definition = program_definition(
        f"""{IMPORTS}{KICK}
@qfunc
def misuse(readonly: Const[QBit], writable: QBit) -> None:
    kick(0.27, writable)
    kick(0.27, readonly)
@qfunc
def main(x: Output[QBit], y: Output[QBit]) -> None:
    allocate(x)
    allocate(y)
    misuse(x, y)
""",
    )
    with pytest.raises(ValueError, match="Const cannot be passed as writable"):
        synth.synthesize(definition)


def test_phase_target_checks_inspect_body_instead_of_call_opcode() -> None:
    body = """
@qfunc
def main(q: Output[QBit]) -> None:
    allocate(q)
    H(q)
    kick(0.27, q)
    kick(0.27, q)
"""
    target = Target(basis=("h", "rx", "rz"), optimize=False)
    without_phase = KICK.replace("    phase(1, 0.13)\n", "")
    root, functions = capture_program(
        program_definition(IMPORTS + without_phase + body)
    )
    assert any(op.name == "qreg.unitary_call" for op in root.body.blocks[0].operations)
    assert select_register_program(root, functions, target=target).selected is not None
    root, functions = capture_program(program_definition(IMPORTS + KICK + body))
    with pytest.raises(ValueError, match="global phase"):
        select_register_program(root, functions, target=target)


def test_shared_helper_reuses_released_scratch() -> None:
    region = """
    with within():
        mark: QNum[1] = x == 2
    with apply():
        phase(mark, 0.17)
    marked(x, q)
"""
    definition = program_definition(
        f"""{IMPORTS}
@qfunc
def marked(x: Const[QNum], q: QBit) -> None:
    for index in range(7):
        control(x == 2, lambda: R(0.21, 0, q))
@qfunc
def main(x: Output[QNum[2]], q: Output[QBit]) -> None:
    allocate(x)
    allocate(q)
    hadamard_transform(x)
    X(q)
    marked(x, q)
{region}""",
    )
    root, _ = capture_program(definition)
    assert any(op.name == "qreg.unitary_call" for op in root.body.blocks[0].operations)
    program = synth.synthesize(definition)
    actual = fq.run(program).statevector().reshape(8, -1)
    expected = torch.zeros(8, dtype=torch.complex128)
    expected[1::2] = 0.5
    factor = complex(math.cos(0.17), math.sin(0.17))
    expected[4:6] = torch.tensor(
        [
            -0.5j * math.sin(14 * 0.21 / 2) * factor,
            0.5 * math.cos(14 * 0.21 / 2) * factor,
        ],
        dtype=torch.complex128,
    )
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-12, rtol=0)
    assert torch.count_nonzero(actual[:, 1:]) == 0
    with pytest.raises(ValueError, match="workspace|budget"):
        synth.synthesize(definition, max_workspace=0)


@pytest.mark.parametrize(
    "violation", ["alias", "uninitialized", "inverse", "reset", "cycle"]
)
def test_tampered_unitary_calls_cannot_bypass_borrow_or_unitary_proofs(
    violation: str,
) -> None:
    definition = program_definition(
        f"""{IMPORTS}{KICK}
@qfunc
def main(q: Output[QBit]) -> None:
    allocate(q)
    kick(0.27, q)
    kick(0.27, q)
""",
    )
    root, functions = capture_program(definition)
    operations = list(root.body.blocks[0].operations)
    index = next(i for i, op in enumerate(operations) if op.name == "qreg.unitary_call")
    call = operations[index]
    helpers = list(functions)
    if violation == "alias":
        operations[index] = replace(call, operands=(*call.operands, call.operands[-1]))
    elif violation == "uninitialized":
        allocation = next(
            i for i, op in enumerate(operations) if op.name == "qreg.allocate"
        )
        op = operations.pop(allocation)
        operations[index - 1] = replace(
            call, operands=(op.operands[0], *call.operands[1:])
        )
        # Remove prior effects too: the call is the first operation on fresh q.
        operations = [
            operations[0],
            operations[index - 1],
            replace(operations[-1], operands=(call.results[0], call.operands[1])),
        ]
    elif violation == "inverse":
        operations[index] = replace(
            call, attributes={**call.attributes, "inverse": False}
        )
    else:
        body = list(helpers[0].body.blocks[0].operations)
        target = next(i for i, op in enumerate(body) if op.name == "qreg.gate")
        gate = body[target]
        body[target] = replace(
            gate,
            name="qreg.reset" if violation == "reset" else "qreg.unitary_call",
            attributes=(
                {"index": 0} if violation == "reset" else {"callee": helpers[0].name}
            ),
        )
        helpers[0] = _replace_operations(helpers[0], tuple(body))
    with pytest.raises(HybridVerificationError):
        verify_register_program(
            _replace_operations(root, tuple(operations)), tuple(helpers)
        )
