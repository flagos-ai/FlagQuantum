"""``Circuit.compose`` places a program on chosen qubits without changing semantics."""

import math

import pytest
import torch

import flagquantum as fq
from flagquantum.errors import ValidationError

pytestmark = pytest.mark.unit


def _recorded(circuit):
    return [
        (instruction.name, tuple(instruction.wires), dict(instruction.params))
        for instruction in circuit.to_ir().instructions
    ]


def test_compose_is_end_to_end_the_hand_built_circuit():
    """The composed circuit equals the circuit built by appending the same gates.

    This is the whole contract of composition: a rewrite of qubit labels, not a new
    way to describe a program.  Both circuits are compared through ``to_ir()``, so an
    equality here is instruction-by-instruction equality of the versioned IR rather
    than of a printer's output.
    """

    block = fq.Circuit(3).h(0).cphase(2, 0, theta=math.pi / 2).cx(1, 2).rz(2, theta=0.4)
    composed = fq.Circuit(6).x(5).compose(block, qubits=(1, 2, 3))

    hand_built = (
        fq.Circuit(6)
        .x(5)
        .h(1)
        .cphase(3, 1, theta=math.pi / 2)
        .cx(2, 3)
        .rz(3, theta=0.4)
    )

    assert _recorded(composed) == _recorded(hand_built)


def test_compose_returns_the_receiving_circuit_so_calls_chain():
    circuit = fq.Circuit(4)
    returned = circuit.compose(fq.Circuit(1).h(0), qubits=2)

    assert returned is circuit
    assert _recorded(circuit) == [("h", (2,), {})]


def test_compose_defaults_to_the_identity_placement():
    """With no map the identity is the rule, and it names every local qubit."""

    composed = fq.Circuit(2).h(0).compose(fq.Circuit(2).cx(0, 1))

    assert _recorded(composed) == [("h", (0,), {}), ("cx", (0, 1), {})]


def test_compose_accepts_a_mapping_as_well_as_a_sequence():
    """``qubit_map`` and ``qubits`` are two spellings of one total map.

    The mapping form is the one a caller reaches for when the source labels are not
    dense, so it is read into the same dense target sequence and the two forms are
    asserted to agree rather than merely to both work.
    """

    block = fq.Circuit(2).h(0).cx(0, 1)
    by_sequence = fq.Circuit(5).compose(block, qubits=(3, 4))
    by_mapping = fq.Circuit(5).compose(block, qubit_map={0: 3, 1: 4})

    assert _recorded(by_sequence) == _recorded(by_mapping)


def test_compose_accepts_a_circuit_ir():
    """A ``CircuitIR`` is the same program, so it composes to the same circuit."""

    block = fq.Circuit(2).h(0).cx(0, 1)
    from_circuit = fq.Circuit(4).compose(block, qubits=(1, 2))
    from_ir = fq.Circuit(4).compose(block.to_ir(), qubits=(1, 2))

    assert _recorded(from_circuit) == _recorded(from_ir)


def test_compose_snapshots_the_source_so_a_circuit_composes_onto_itself():
    """Appending a program onto itself must not iterate the list it grows.

    The source instructions are read into a tuple before the first append, so
    ``circuit.compose(circuit)`` doubles the program once instead of looping.
    """

    circuit = fq.Circuit(2).h(0)
    circuit.compose(circuit, qubits=(1, 0))

    assert _recorded(circuit) == [("h", (0,), {}), ("h", (1,), {})]


def test_compose_keeps_parameters_and_the_batch_dimension():
    """A batched angle stays one batched angle on its new qubit."""

    block = (
        fq.Circuit(2, bsz=2).rx(0, theta=torch.tensor([0.1, 0.2])).crx(0, 1, theta=0.3)
    )
    composed = fq.Circuit(3, bsz=2).compose(block, qubits=(1, 2))

    recorded = _recorded(composed)
    assert recorded[0][0] == "rx"
    assert recorded[0][1] == (1,)
    assert torch.equal(recorded[0][2]["theta"], torch.tensor([0.1, 0.2]))
    assert recorded[1] == ("crx", (1, 2), {"theta": 0.3})


def test_compose_preserves_execution():
    """The composed program runs, and runs to the hand-built state."""

    block = fq.Circuit(2).h(0).cx(0, 1)
    composed = fq.Circuit(4).compose(block, qubits=(1, 2))
    hand_built = fq.Circuit(4).h(1).cx(1, 2)

    options = fq.ExecutionOptions(precision="complex128")
    assert torch.allclose(
        composed.run(options=options).state,
        hand_built.run(options=options).state,
    )


def test_compose_invalidates_the_cached_ir():
    """A composed circuit must not report the IR it had before the append."""

    circuit = fq.Circuit(2)
    assert len(circuit.to_ir().instructions) == 0
    circuit.compose(fq.Circuit(1).h(0), qubits=0)

    assert len(circuit.to_ir().instructions) == 1


def test_compose_carries_the_unitary_of_an_arbitrary_operation():
    """An arbitrary-unitary instruction keeps its matrix through the rewrite.

    Re-walking a circuit by hand through ``gate(name, qubits, params=...)`` drops the
    matrix, and the rewritten instruction is then refused with ``unknown opcode
    'any'`` because the unitary was the only thing that made the opcode legal.
    """

    unitary = torch.eye(2, dtype=torch.complex64)
    block = fq.Circuit(1).any(0, unitary=unitary)
    composed = fq.Circuit(2).compose(block, qubits=1)

    recorded = composed.to_ir().instructions[0]
    assert recorded.name == "any"
    assert recorded.wires == (1,)
    assert recorded.matrix is unitary


def test_compose_carries_instruction_metadata():
    """A channel stays a channel after it is placed on another qubit.

    ``is_channel`` is free-form metadata, so nothing but the rewrite itself can keep
    it: it is read by the planner, the noise selector, and five executors, and an
    instruction that loses it is silently treated as a unitary.
    """

    channel = fq.Instruction(
        "depolarizing",
        (0,),
        params={"probability": 0.1},
        metadata={"is_channel": True},
    )
    composed = fq.Circuit(2).compose(fq.CircuitIR(1, (channel,)), qubits=1)

    recorded = composed.to_ir().instructions[0]
    assert recorded.wires == (1,)
    assert dict(recorded.metadata) == {"is_channel": True}
    assert dict(recorded.params) == {"probability": 0.1}


@pytest.mark.parametrize(
    "qubits, message",
    [
        ((1,), "must name all 2 qubit"),
        ((1, 2), "outside circuit range"),
        ((1, 1), "cannot place two local qubits on qubit 1"),
        ((-1, 0), "must be non-negative"),
    ],
)
def test_compose_refuses_a_placement_that_is_not_a_total_map(qubits, message):
    with pytest.raises(ValidationError, match=message):
        fq.Circuit(2).compose(fq.Circuit(2).h(0).cx(0, 1), qubits=qubits)


def test_compose_refuses_a_local_qubit_the_source_does_not_have():
    with pytest.raises(ValidationError, match="names local qubit 2"):
        fq.Circuit(4).compose(fq.Circuit(2).h(0), qubit_map={0: 1, 2: 3})


def test_compose_refuses_a_mapping_that_skips_a_local_qubit():
    with pytest.raises(ValidationError, match="must name every local qubit"):
        fq.Circuit(4).compose(fq.Circuit(2).h(0), qubit_map={0: 1})


def test_compose_refuses_two_spellings_of_one_placement():
    with pytest.raises(TypeError, match="either qubits or qubit_map, not both"):
        fq.Circuit(4).compose(fq.Circuit(1).h(0), qubits=(1,), qubit_map={0: 1})


@pytest.mark.parametrize("label", [1.5, "1", True])
def test_compose_refuses_a_target_that_is_not_an_integer(label):
    """``int(0.5)`` is qubit ``0``, so the label rule refuses rather than rewrites."""

    with pytest.raises(TypeError, match="must be an integer"):
        fq.Circuit(4).compose(fq.Circuit(1).h(0), qubits=(label,))


def test_compose_refuses_a_source_that_is_not_a_program():
    with pytest.raises(TypeError, match="accepts a Circuit or a CircuitIR"):
        fq.Circuit(4).compose([("h", (0,))])


def test_compose_refuses_mismatched_batch_sizes():
    """Mixing batch dimensions is refused instead of broadcast or truncated."""

    with pytest.raises(ValidationError, match="cannot mix batch sizes"):
        fq.Circuit(4, bsz=2).compose(fq.Circuit(1).h(0), qubits=0)


def test_compose_reads_a_hand_built_ir_without_a_batch_size_as_one():
    """A ``CircuitIR`` with no batch dimension means one, and refuses ``bsz=2``."""

    block = fq.CircuitIR(1, (fq.Instruction("h", (0,)),))

    assert _recorded(fq.Circuit(2).compose(block, qubits=1)) == [("h", (1,), {})]
    with pytest.raises(ValidationError, match="cannot mix batch sizes"):
        fq.Circuit(2, bsz=2).compose(block, qubits=1)


def test_compose_onto_an_empty_program_changes_nothing():
    circuit = fq.Circuit(3).h(0)
    circuit.compose(fq.Circuit(2))

    assert _recorded(circuit) == [("h", (0,), {})]


def test_compose_of_a_single_qubit_program_accepts_a_bare_qubit():
    """A one-qubit program reads ``qubits=3`` as the length-one map it is."""

    assert _recorded(fq.Circuit(4).compose(fq.Circuit(1).h(0), qubits=3)) == [
        ("h", (3,), {})
    ]
