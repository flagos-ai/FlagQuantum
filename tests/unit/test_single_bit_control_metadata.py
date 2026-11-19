"""One physical control bit retains native word metadata and singleton shape."""

from collections import Counter
from itertools import product

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from examples.algorithms.controlled_programs import controlled_x_program
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program.registers import Register, RegisterType

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
q = pytest.importorskip("qualtran")
cirq = pytest.importorskip("cirq")
np = pytest.importorskip("numpy")
CASES = [
    (kind, shape, polarity)
    for kind, shape, polarity in product(
        ("QBit", "QAny", "QUInt", "QInt"), ((), (1,), (1, 1)), (0, 1)
    )
]


@pytest.mark.parametrize("kind,shape,polarity", CASES)
def test_single_bit_metadata_and_physical_control(
    kind: str, shape: tuple[int, ...], polarity: int
) -> None:
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from tests.unit.test_register_control_tutorial import _source_gates
    from tests.unit.test_register_subtraction import assert_source

    signed = kind == "QInt"
    value = -polarity if signed else polarity
    source_type = getattr(q, kind)() if kind == "QBit" else getattr(q, kind)(1)
    spec = q.CtrlSpec(source_type, np.full(shape, value))
    assert spec.get_single_ctrl_val() == polarity
    dtype = RegisterType(
        {"QBit": "bit", "QAny": "opaque", "QUInt": "uint", "QInt": "signed"}[kind], 1
    )
    port = Register("ctrl", dtype, shape)
    pattern = ControlRegister(1, (value,), "twos_complement" if signed else "unsigned")
    record = controlled_x_program((port,), (pattern,))
    registers = record.functions[0].registers
    assert registers is not None and registers[0] == port
    bound = bind_register_program(record, ((0,), (1,)))
    assert bound.outputs == ((0,), (1,))
    observed = Counter(g.name for g in bound.circuit.instructions)
    assert observed == ({"cx": 1} if polarity else {"x": 2, "cx": 1})
    source = ControlledViaAnd(XGate(), spec)
    if shape == (1, 1):
        # Frozen source fails to extract a scalar Soquet from this valid shape.
        with pytest.raises(AssertionError):
            source.decompose_bloq()
    else:
        assert_source(
            bound.circuit.instructions,
            _source_gates(source, {"ctrl": (0,), "q": (1,)}, (), {}),
        )
    state = torch.complex(torch.arange(1, 9).double(), torch.arange(8).double())
    state /= state.norm()
    if shape == (1, 1):
        with pytest.raises(AssertionError):
            source.tensor_contract()
        with pytest.raises(AssertionError):
            cirq.unitary(source)
        matrix_array = source._tensor_data()
    else:
        matrix_array = source.tensor_contract()
    matrix = torch.as_tensor(matrix_array.copy(), dtype=torch.complex128).reshape(4, 4)
    expected = (matrix @ state.reshape(4, 2)).flatten()
    circuit = Circuit(3, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(3, bound.circuit.instructions))
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=3e-14)
    # A controlled X with unchanged activation is its exact inverse.
    circuit.append(FieldOperation(3, bound.circuit.instructions))
    torch.testing.assert_close(circuit.state().flatten(), state, rtol=0, atol=3e-14)
    if shape != (1, 1):
        model = get_cost_value(source, QECGatesCost())
        assert model.clifford == (1 if polarity else 3)
        assert model.total_t_count() == 0
