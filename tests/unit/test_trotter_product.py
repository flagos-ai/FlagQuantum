"""Exercise the product schedule independently of any chemistry term."""

from dataclasses import dataclass

import pytest

from flagquantum.algorithms.primitives.trotter_product import TrotterProduct
from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.resource_model import ComponentCost
from flagquantum.core.ir import Instruction
from flagquantum.core.parameters import Parameter

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class AxisTerm:
    axis: str
    layout = (("system", 1),)

    def emit(self, ports, angle):
        return (Instruction(self.axis, ports["system"], {"theta": angle}),)

    def resource_model(self, angle):
        return ComponentCost(self.axis, 1)


def test_truncated_schedule_negative_indices_and_symbolic_adjoint():
    dt, coefficient = Parameter("dt"), Parameter("coefficient")
    product = TrotterProduct(
        (AxisTerm("rx"), AxisTerm("rz")), (-1, 0, 99), (coefficient, -0.5), dt
    )
    assert [edge.component.name for edge in product.resource_model().calls] == [
        "rz",
        "rx",
    ]
    forward = product.make_on(system=(1,))
    inverse = product.make_on(system=(1,), adjoint=True)
    bound = {"coefficient": 2, "dt": 0.125}
    circuit = Circuit(2).append(forward).bind_parameters(bound)
    assert circuit.to_ir().instructions == (
        Instruction("rz", (1,), {"theta": 0.5}),
        Instruction("rx", (1,), {"theta": -0.125}),
    )
    assert (
        Circuit(2).append(inverse).bind_parameters(bound).to_ir().instructions
        == circuit.adjoint().to_ir().instructions
    )


def test_empty_product_and_binding_failures():
    term = AxisTerm("rx")
    empty = TrotterProduct((term,), (), (1,), 1)
    assert empty.resource_model() == ComponentCost("TrotterizedUnitary", 1)
    assert not Circuit(2).append(empty.make_on(system=(1,))).to_ir().instructions
    with pytest.raises(ValueError, match="width"):
        empty.make_on(system=(0, 1))
    with pytest.raises(ValueError, match="exceeds circuit width"):
        Circuit(2).append(empty.make_on(system=(2,)))
    with pytest.raises(ValueError, match="at least one"):
        TrotterProduct((), (), (), 1)
    with pytest.raises(IndexError):
        TrotterProduct((term,), (1,), (1,), 1).resource_model()
    with pytest.raises(ValueError, match="finite real"):
        TrotterProduct((term,), (0,), (1,), float("inf")).resource_model()
