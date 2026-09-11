import pytest

from flagquantum.compiler import schedule_layers
from flagquantum.core.ir import CircuitIR, Instruction

pytestmark = pytest.mark.unit


def test_scheduler_does_not_move_gate_before_earlier_shared_wire_dependency() -> None:
    first = Instruction("h", (0,))
    parallel = Instruction("h", (1,))
    dependency = Instruction("cx", (0, 2))
    follower = Instruction("x", (2,))
    ir = CircuitIR(3, (first, parallel, dependency, follower))

    layers = schedule_layers(ir)

    assert layers == [[first, parallel], [dependency], [follower]]


def test_scheduler_preserves_program_order_on_every_wire() -> None:
    instructions = (
        Instruction("h", (0,)),
        Instruction("h", (1,)),
        Instruction("cx", (0, 2)),
        Instruction("x", (2,)),
        Instruction("cx", (1, 2)),
        Instruction("z", (0,)),
    )
    layers = schedule_layers(CircuitIR(3, instructions))
    layer_by_instruction = {
        id(instruction): layer_index
        for layer_index, layer in enumerate(layers)
        for instruction in layer
    }

    for wire in range(3):
        wire_layers = [
            layer_by_instruction[id(instruction)]
            for instruction in instructions
            if wire in instruction.wires
        ]
        assert wire_layers == sorted(wire_layers)
        assert len(wire_layers) == len(set(wire_layers))
