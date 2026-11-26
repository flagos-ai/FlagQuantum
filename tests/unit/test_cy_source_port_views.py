"""CY scalar/array trace views preserve exact gates and all following wires."""

from contextlib import nullcontext

import pytest

try:
    import numpy as np
except ModuleNotFoundError:
    pytest.skip("Reference trace checks require NumPy", allow_module_level=True)

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_component_hierarchy_control import _cy_source_port_views

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("fault", ("none", "cx", "reverse", "following_wire"))
def test_cy_view_proves_exact_body_and_preserves_successor_operands(fault: str) -> None:
    from qualtran.bloqs.basic_gates import CYGate, Hadamard

    gate = "cx" if fault == "cx" else "cy"
    wires = (1, 0) if fault == "reverse" else (0, 1)
    cy = bit_gate_function(
        "cy",
        (Register("q", quantum_bit_type(), (2,)),),
        CircuitIR(2, (Instruction(gate, wires),)),
    )
    h = bit_gate_function(
        "h",
        (Register("q", quantum_bit_type()),),
        CircuitIR(1, (Instruction("h", (0,)),)),
    )
    signature = Signature.build(a=1, b=1)
    native = ComponentProgramBuilder(signature, (cy, h))
    source = _source(signature)
    values = native.call_dict(
        "cy", {"q": np.array([native.inputs["a"], native.inputs["b"]], dtype=object)}
    )["q"]
    assert isinstance(values, np.ndarray)
    output = native.call_dict("h", {"q": values[0 if fault == "following_wire" else 1]})
    record = native.finalize(
        {"a": output["q"], "b": values[1]}
        if fault == "following_wire"
        else {"a": values[0], "b": output["q"]}
    )
    source_values = source.builder.add_d(
        CYGate(), ctrl=source.inputs["a"], target=source.inputs["b"]
    )
    b = source.builder.add(Hadamard(), q=source_values["target"])
    original = source.builder.finalize(a=source_values["ctrl"], b=b)
    with nullcontext() if fault == "none" else pytest.raises(AssertionError):
        views = _cy_source_port_views(record, "main", original)
        assert _native_trace(record) == _source_trace(
            source, original, bit_port_views=views
        )
