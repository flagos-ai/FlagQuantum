"""Native SDK definitions retain their actual reachable helper modules."""

from dataclasses import replace
from itertools import product
from math import pi, sqrt
from typing import Any

import pytest
import torch

from flagquantum.compiler.formats.legacy_word_factories import _rename_program
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.primitives.controlled_rotation_component import (
    controlled_z_power_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program import FunctionRecord
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.ecosystem.cirq._register_import import import_cirq_register_optree
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_cirq_decomposition_file import locked_source as locked_source

pytestmark = [
    pytest.mark.unit,
    pytest.mark.ftqc_reference,
    pytest.mark.usefixtures("locked_source"),
]
cirq = pytest.importorskip("cirq")


@pytest.mark.parametrize("measured", (False, True))
def test_complete_czpow_definition_module_and_unrelated_declaration(
    measured: bool,
) -> None:
    program = _rename_program(controlled_z_power_program(0.375), "known_phase")
    gate = cirq.CZPowGate(exponent=0.375)
    qubits = tuple(cirq.LineQubit.range(3 if measured else 2))
    operations = [gate.on(*qubits[:2])]
    if measured:
        operations.insert(0, cirq.measure(qubits[2], key="external"))
    definitions: dict[Any, FunctionRecord] = {
        gate: program.functions[0],
        **{function.name: function for function in program.functions[1:]},
        "unused": FunctionRecord(
            "unused", (), registers=program.functions[0].registers
        ),
    }
    if measured:
        record = import_cirq_register_optree(
            cirq.Circuit(operations), gate_definitions=definitions
        )
        inputs = tuple((i,) for i in range(3))
    else:
        placement = {"q": [[qubits[0]], [qubits[1]]]}
        record = import_cirq_register_optree(
            cirq.Circuit(operations),
            signature=Signature(program.functions[0].registers or ()),
            in_quregs=placement,
            out_quregs=placement,
            gate_definitions=definitions,
        )
        inputs = ((0,), (1,))
    assert "unused" not in {function.name for function in record.functions}
    logical = 3 if measured else 2
    bound = bind_register_program(record, inputs, clean=(logical,))
    assert sum(g.name == "measure" for g in bound.circuit.instructions) == 1 + measured
    assert all(g.name != "cphase" for g in bound.circuit.instructions)
    if not measured:
        expected = bind_register_program(
            program, inputs, clean=(logical,), entry="known_phase"
        )
        assert bound == expected
    else:
        data = torch.arange(1, 9, dtype=torch.float64)
        amplitudes = torch.complex(data, 1 + data * data / 7)
        amplitudes /= amplitudes.norm()
        initial = torch.zeros(16, dtype=torch.complex128)
        initial[::2] = amplitudes
        reference = (
            Instruction(
                "measure", (2,), metadata={"is_dynamic": True, "classical_bit": 0}
            ),
            Instruction("cphase", (0, 1), params={"theta": pi * 0.375}),
        )
        for external, erasure in product((0, 1), repeat=2):
            actual = forced_branch(
                initial, bound.circuit.instructions, (external, erasure)
            )
            expected_state = forced_branch(initial, reference, (external,)) / sqrt(2)
            torch.testing.assert_close(actual, expected_state, atol=1e-11, rtol=0)
            assert torch.count_nonzero(actual[1::2]) == 0


def test_missing_and_conflicting_reachable_helper_definitions() -> None:
    program = _rename_program(controlled_z_power_program(0.375), "known_phase")
    gate = cirq.CZPowGate(exponent=0.375)
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(gate.on(*qubits))
    with pytest.raises(ValueError, match="Missing Cirq native helper"):
        import_cirq_register_optree(
            circuit, gate_definitions={gate: program.functions[0]}
        )
    definitions = {
        gate: program.functions[0],
        **{f.name: f for f in program.functions[1:]},
    }
    helper = next(f for f in program.functions if f.name.endswith("_compute"))
    definitions["conflicting"] = replace(helper, blocks=())
    with pytest.raises(ValueError, match="Conflicting Cirq gate definition"):
        import_cirq_register_optree(circuit, gate_definitions=definitions)


@pytest.mark.parametrize("measured", (False, True))
def test_native_definition_precedes_sdk_eigenphase_normalization(
    measured: bool,
) -> None:
    builder = ComponentProgramBuilder(Signature.build(q=1), name="identity")
    definition = builder.finalize(builder.inputs).functions[0]
    gate = cirq.XPowGate(exponent=0.375, global_shift=0.2)
    qubits = cirq.LineQubit.range(2 if measured else 1)
    operations = [gate.on(qubits[0])]
    if measured:
        operations.insert(0, cirq.measure(qubits[1], key="external"))
    record = import_cirq_register_optree(
        cirq.Circuit(operations), gate_definitions={gate: definition}
    )
    bound = bind_register_program(record, tuple((i,) for i in range(len(qubits))))
    assert tuple(g.name for g in bound.circuit.instructions) == (
        ("measure",) if measured else ()
    )


def test_default_main_module_uses_existing_import_namespace() -> None:
    program = controlled_z_power_program(0.375)
    gate = cirq.CZPowGate(exponent=0.375)
    qubits = cirq.LineQubit.range(2)
    definitions = {
        gate: program.functions[0],
        **{f.name: f for f in program.functions[1:]},
    }
    placement = {"q": [[qubits[0]], [qubits[1]]]}
    record = import_cirq_register_optree(
        cirq.Circuit(gate.on(*qubits)),
        signature=Signature(program.functions[0].registers or ()),
        in_quregs=placement,
        out_quregs=placement,
        gate_definitions=definitions,
    )
    actual = bind_register_program(record, ((0,), (1,)), clean=(2,))
    expected = bind_register_program(program, ((0,), (1,)), clean=(2,))
    assert actual == expected
    assert program.functions[0].name == "main"
