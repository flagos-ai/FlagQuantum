"""Scratch storage preserves guarded dataflow and every quantum branch."""

from copy import deepcopy
from itertools import product
from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.layout.classical_storage import (
    bind_classical_scratch,
)
from flagquantum.compiler.registers.register_binding import BoundRegisterProgram
from flagquantum.core.classical_control import validate_classical_control
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.errors import ValidationError
from tests.unit.measurement_checks import forced_branch

pytestmark = pytest.mark.unit


def _measure(wire: int, bit: int) -> Instruction:
    return Instruction(
        "measure", (wire,), metadata={"is_dynamic": True, "classical_bit": bit}
    )


def _bound(gates: tuple[Instruction, ...], width: int = 3) -> BoundRegisterProgram:
    return BoundRegisterProgram(CircuitIR(width, gates), (tuple(range(width)),), ())


@pytest.mark.parametrize("slots", [(0, 1, 2, 3), (41, 13, 7, 23)])
def test_conditional_writes_snapshots_dnf_redefinitions_and_guarded_effects(
    slots: tuple[int, ...],
) -> None:
    conditional = ClassicalXor((10,), 20).instruction()
    conditional = Instruction(
        conditional.name,
        (),
        params=conditional.params,
        metadata={**conditional.metadata, "conditions": ((11, 1),)},
    )
    guarded = {"is_dynamic": True, "guarded_effect": True, "conditions": ((21, 1),)}
    gates = (
        Instruction("h", (0,)),
        _measure(0, 10),
        Instruction("reset", (0,), metadata={"is_dynamic": True}),
        Instruction("h", (1,)),
        _measure(1, 11),
        Instruction("reset", (1,), metadata={"is_dynamic": True}),
        ClassicalXor((), 20, True).instruction(),
        conditional,
        ClassicalXor((20, 10, 10), 21).instruction(),
        Instruction(
            "z",
            (2,),
            metadata={"condition_clauses": (((10, 0), (11, 1)), ((10, 1), (21, 1)))},
        ),
        ClassicalXor((), 12).instruction(),
        Instruction("h", (0,)),
        Instruction("measure", (0,), metadata={**guarded, "classical_bit": 12}),
        Instruction("reset", (0,), metadata=guarded),
        Instruction("x", (2,), metadata={"conditions": ((12, 1),)}),
        Instruction("h", (1,)),
        _measure(1, 10),
        Instruction("z", (2,), metadata={"conditions": ((10, 1),)}),
    )
    original = _bound(gates)
    saved = deepcopy(original)
    mapped = bind_classical_scratch(original, slots)
    assert original == saved
    assert mapped.outputs == original.outputs and mapped.events == original.events
    assert [(g.name, g.wires) for g in mapped.circuit.instructions] == [
        (g.name, g.wires) for g in gates
    ]
    assert {
        g.metadata["classical_bit"]
        for g in mapped.circuit.instructions
        if "classical_bit" in g.metadata
    } <= set(slots)
    validate_classical_control(mapped.circuit.instructions, n_qubits=3)
    state = torch.tensor(
        [complex(i + 1, 3 - 2 * i) for i in range(8)], dtype=torch.complex128
    )
    state /= state.norm()
    for history in product((0, 1), repeat=4):
        expected = forced_branch(state.clone(), gates, history)
        actual = forced_branch(state.clone(), mapped.circuit.instructions, history)
        torch.testing.assert_close(actual, expected, atol=2e-14, rtol=0)


def test_storage_cycles_in_supplied_order_and_keeps_latest_assignments() -> None:
    original = _bound(tuple(_measure(0, bit) for bit in range(8)), 1)
    mapped = bind_classical_scratch(original, (8, 4, 11, 6))
    assert tuple(g.metadata["classical_bit"] for g in mapped.circuit.instructions) == (
        8,
        4,
        11,
        6,
        8,
        4,
        11,
        6,
    )
    assert len(mapped.circuit.instructions) == len(original.circuit.instructions)


def test_live_reads_cannot_spill_and_exported_values_cannot_be_recycled() -> None:
    original = _bound(
        (
            _measure(0, 10),
            _measure(1, 11),
            Instruction("z", (2,), metadata={"conditions": ((10, 1), (11, 1))}),
        )
    )
    saved = deepcopy(original)
    with pytest.raises(ValueError, match="Insufficient"):
        bind_classical_scratch(original, (0,))
    assert original == saved
    exported = BoundRegisterProgram(original.circuit, original.outputs, (), ((10,),))
    with pytest.raises(ValueError, match="exported"):
        bind_classical_scratch(exported, (0, 1))


@pytest.mark.parametrize("slots", [(True,), (-1,), (1, 1), (1.5,), [1, 2]])
def test_malformed_storage_rejected(slots: Any) -> None:
    with pytest.raises(ValueError, match="distinct nonnegative"):
        bind_classical_scratch(_bound((_measure(0, 10),)), slots)


def test_read_before_definition_keeps_core_validation() -> None:
    source = _bound((Instruction("z", (0,), metadata={"conditions": ((10, 1),)}),), 1)
    with pytest.raises(ValidationError, match="before"):
        bind_classical_scratch(source, (0,))
