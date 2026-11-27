"""Check typed polynomial coefficient views under shared control passthrough."""

from hashlib import sha1
from pathlib import Path

from tests._reference_dependencies import unavailable_reference

try:
    import numpy as np
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)
import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.control.component_bookkeeping import (
    bookkeeping_component_control_system,
)
from flagquantum.compiler.registers.finite_field.field_polynomial_component import (
    field_polynomial_component_program,
)
from flagquantum.compiler.registers.finite_field.field_polynomial_layout import (
    FieldPolynomialLayout,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.partition_symbols import split_join_wire_symbol
from flagquantum.drawer.register_score_data import register_score_data
from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding
from tests.unit.register_source_graph import source_line_qubit_index
from tests.unit.test_drawer_component_score import assert_source_geometry

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    from qualtran.bloqs.gf_poly_arithmetic import gf_poly_split_and_join

    raw = Path(gf_poly_split_and_join.__file__).read_bytes()
    assert sha1(b"blob %d\0" % len(raw) + raw).hexdigest() == (
        "966e38d6070cedf8c99f83dd2bd7fc5eb1743854"
    )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("degree,field_degree", [(0, 1), (2, 2)])
def test_polynomial_control_keeps_typed_coefficient_order_and_state(
    reverse: bool, degree: int, field_degree: int
) -> None:
    try:
        from cirq import LineQubit
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)
    from qualtran._infra.bloq import Bloq
    from qualtran._infra.composite_bloq import BloqBuilder
    from qualtran._infra.controlled import CtrlSpec
    from qualtran._infra.data_types import QGF, QBit, QGFPoly
    from qualtran._infra.registers import Register as ForeignRegister
    from qualtran._infra.registers import Signature as ForeignSignature
    from qualtran.bloqs.gf_poly_arithmetic.gf_poly_split_and_join import GFPolySplit
    from qualtran.drawing.musical_score import get_musical_score_data
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value
    from qualtran.resource_counting._qubit_counts import QubitCount

    source: Bloq = GFPolySplit(QGFPoly(degree, QGF(2, field_degree)))
    if reverse:
        source = source.adjoint()
    from qualtran.bloqs.gf_poly_arithmetic.gf_poly_split_and_join import GFPolyJoin

    assert isinstance(source, (GFPolySplit, GFPolyJoin))
    field = FiniteFieldEncoding(2, field_degree).register_type()
    layout = FieldPolynomialLayout(degree, field_degree)
    record = field_polynomial_component_program(layout, field, "leaf", reverse=reverse)
    registers = record.functions[0].registers
    assert registers is not None
    assert tuple((r.name, r.shape, r.bitsize, r.side.name) for r in registers) == tuple(
        (r.name, r.shape, r.bitsize, r.side.name) for r in source.signature
    )
    coefficients = registers[0 if reverse else 1]
    assert coefficients.dtype == field
    selected, adder = bookkeeping_component_control_system(record, "leaf")
    assert selected is record

    controls_port = Register("ctrls", quantum_bit_type(), (2,))
    builder = ComponentProgramBuilder(
        Signature((controls_port, *registers)), record.functions
    )
    controls = (builder.inputs["ctrls"],)
    returned, (result,) = adder(builder, controls, {"reg": builder.inputs["reg"]})
    assert returned is controls
    controlled = builder.finalize({"ctrls": returned[0], "reg": result})
    assert tuple(op.callee for op in controlled.functions[0].blocks[0].operations) == (
        "leaf",
    )

    source_builder, src_inputs = BloqBuilder.from_signature(
        ForeignSignature([ForeignRegister("ctrls", QBit(), (2,)), *source.signature])
    )
    src_controls = (src_inputs["ctrls"],)
    wrapped, callback = source.get_ctrl_system(CtrlSpec(cvs=(0, 1)))
    src_returned, (src_result,) = callback(
        source_builder, src_controls, {"reg": src_inputs["reg"]}
    )
    assert wrapped is source and src_returned is src_controls
    composite = source_builder.finalize(ctrls=src_returned[0], reg=src_result)
    assert tuple(instance.bloq for instance in composite.bloq_instances) == (source,)

    assert_source_geometry(
        register_score_data(
            component_register_score(controlled),
            register_wire_symbol=split_join_wire_symbol,
        ),
        get_musical_score_data(composite),
        ordered_ports=True,
    )

    packed = tuple(reversed(range(2, 2 + layout.logical_width)))
    words = layout.split(packed)
    wires = np.asarray([LineQubit(i) for i in packed])
    op, source_ports = source.as_cirq_op(
        None, reg=wires.reshape(degree + 1, field_degree) if reverse else wires
    )
    assert op is None
    original = source_ports["reg"].reshape(
        -1, layout.logical_width if reverse else field_degree
    )
    expected = tuple(
        tuple(source_line_qubit_index(q) for q in word) for word in original
    )
    bound = bind_register_program(
        controlled, ((0,), (1,), *(words if reverse else (packed,)))
    )
    assert bound.outputs == ((0,), (1,), *expected)
    assert bound.events == () and bound.circuit.instructions == ()
    assert get_cost_value(composite, QECGatesCost()).asdict() == {}
    assert get_cost_value(composite, QubitCount()) == layout.logical_width + 2

    width = bound.circuit.n_wires
    real = torch.arange(1, 2**width + 1, dtype=torch.float64)
    state = torch.complex(real, (real % 13) - 6)
    state /= state.norm()
    circuit = Circuit(width, inputs=state, dtype=state.dtype)
    circuit.append(FieldOperation(width, bound.circuit.instructions))
    torch.testing.assert_close(circuit.state().flatten(), state, rtol=0, atol=2e-12)


def test_polynomial_field_port_must_have_a_matching_executable_encoding() -> None:
    field = FiniteFieldEncoding(2, 2).register_type()
    with pytest.raises(ValueError, match="matching executable field"):
        field_polynomial_component_program(FieldPolynomialLayout(1, 3), field)
    odd_field = FiniteFieldEncoding(3, 2).register_type()
    record = field_polynomial_component_program(FieldPolynomialLayout(1, 4), odd_field)
    bound = bind_register_program(record, (tuple(range(8)),))
    assert bound.outputs == ((0, 1, 2, 3), (4, 5, 6, 7))
    assert bound.events == () and bound.circuit.instructions == ()
