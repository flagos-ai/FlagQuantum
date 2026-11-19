"""Original complete composite examples through native Program construction."""

from typing import Any

import numpy as np
import pytest

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.ecosystem.cirq._register_program_export import (
    export_bound_register_cirq,
)
from tests.test_cirq_register_export import source_export as source_export

pytestmark = [
    pytest.mark.unit,
    pytest.mark.ftqc_reference,
    pytest.mark.usefixtures("source_export"),
]
pytest.importorskip("numpy")
pytest.importorskip("cirq")
pytest.importorskip("qualtran")


@pytest.mark.parametrize("shaped", [False, True])
def test_original_composite_cnot_examples_preserve_diagram_and_full_operator(
    shaped: bool,
    pytestconfig: pytest.Config,
) -> None:
    pytestconfig.addinivalue_line("markers", "notebook: upstream notebook witness")
    from qualtran._infra.composite_bloq_test import TestMultiCNOT, TestTwoCNOT
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    bit = RegisterType("bit", 1)
    if shaped:
        signature = Signature(
            (Register("control", bit), Register("target", bit, shape=(2, 3)))
        )
        original: Any = TestMultiCNOT().decompose_bloq()
    else:
        signature = Signature.build(q1=1, q2=1)
        original = TestTwoCNOT().decompose_bloq()
    cnot = bit_gate_function(
        "CNOT",
        (Register("ctrl", bit), Register("target", bit)),
        CircuitIR(2, (Instruction("cx", (0, 1)),)),
    )
    builder = ComponentProgramBuilder(signature, (cnot,))
    if shaped:
        control = builder.inputs["control"]
        targets = np.asarray(builder.inputs["target"], dtype=object).copy()
        for index in np.ndindex(2, 3):
            result = builder.call_dict(
                "CNOT", {"ctrl": control, "target": targets[index]}
            )
            control, targets[index] = result["ctrl"], result["target"]
        outputs = {"control": control, "target": targets}
        width, count = 7, 6
    else:
        result = builder.call_dict(
            "CNOT", {"ctrl": builder.inputs["q1"], "target": builder.inputs["q2"]}
        )
        result = builder.call_dict(
            "CNOT", {"ctrl": result["target"], "target": result["ctrl"]}
        )
        outputs = {"q1": result["ctrl"], "q2": result["target"]}
        width, count = 2, 2
    record = builder.finalize(outputs)
    bound = bind_register_program(record, tuple((i,) for i in range(width)))
    actual, returned = export_bound_register_cirq(bound, signature)
    from qualtran._infra.gate_with_registers import get_named_qubits

    expected, source_returned = original.to_cirq_circuit_and_quregs(
        **get_named_qubits(original.signature.lefts())
    )
    assert tuple(actual.circuit.all_operations()) == tuple(expected.all_operations())
    assert actual.circuit.to_text_diagram() == expected.to_text_diagram()
    for name in returned:
        np.testing.assert_array_equal(returned[name], source_returned[name])
    order = tuple(q for reg in signature for q in returned[reg.name].flat)
    np.testing.assert_allclose(
        actual.circuit.unitary(qubit_order=order),
        expected.unitary(qubit_order=order),
        atol=0,
        rtol=0,
    )
    assert len(bound.circuit.instructions) == count
    assert all(gate.name == "cx" for gate in bound.circuit.instructions)
    assert get_cost_value(original, QECGatesCost()).clifford == count
