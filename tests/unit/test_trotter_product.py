"""Exercise the product schedule independently of any chemistry term."""

from dataclasses import dataclass

import pytest

from flagquantum.algorithms.primitives.trotter_product import TrotterProduct
from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.resource_model import ComponentCost
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program.register_types import (
    quantum_bit_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
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
    template = Circuit(2).append(forward)
    symbolic_ir = template.to_ir()
    assert (
        CircuitIR.from_json(symbolic_ir.to_json()).instructions
        == symbolic_ir.instructions
    )
    circuit = template.bind_parameters(bound)
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


@dataclass(frozen=True)
class TypedTerm:
    signature: Signature
    layout = (("system", 2),)

    def emit(self, ports, angle):
        return tuple(
            Instruction("rx", (wire,), {"theta": angle}) for wire in ports["system"]
        )

    def resource_model(self, angle):
        return ComponentCost("TypedTerm", 2)


def test_product_preserves_shaped_through_metadata_and_rejects_mismatches():
    pair = Signature((Register("system", quantum_bit_type(), shape=(2,)),))
    flat = Signature.build(system=2)
    unsigned = Signature((Register("system", unsigned_register_type(2)),))
    term = TypedTerm(pair)
    product = TrotterProduct((term, term), (0, 1), (1, 0.5), 0.1)
    assert product.signature == pair
    assert product.resource_model().qubits == 2
    circuit = Circuit(3).append(product.make_on(system=(2, 0)))
    assert tuple(gate.wires for gate in circuit.to_ir().instructions) == (
        (2,),
        (0,),
        (2,),
        (0,),
    )
    for left, right in ((pair, flat), (flat, unsigned)):
        with pytest.raises(ValueError, match="identical register signatures"):
            TrotterProduct((TypedTerm(left), TypedTerm(right)), (), (), 1)
    with pytest.raises(ValueError, match="quantum through layout"):
        TrotterProduct((TypedTerm(Signature.build(system=3)),), (), (), 1)
    right_only = Signature(
        (Register("system", quantum_bit_type(), shape=(2,), side=RegisterSide.RIGHT),)
    )
    with pytest.raises(ValueError, match="quantum through layout"):
        TrotterProduct((TypedTerm(right_only),), (), (), 1)


@pytest.mark.ftqc_reference
def test_locked_trotter_tutorial_shared_signature_and_calls():
    pytest.importorskip("qualtran")
    import torch
    from qualtran import QAny, Side
    from qualtran.bloqs.chemistry.trotter.trotterized_unitary import _trott_unitary
    from qualtran.resource_counting.generalizers import ignore_split_join

    from flagquantum.algorithms.primitives.ising_evolution import (
        IsingXUnitary,
        IsingZZUnitary,
    )
    from flagquantum.compiler._component_trace.call_graph import modeled_call_graph
    from flagquantum.simulation.unitary import get_unitary

    source = _trott_unitary()
    product = TrotterProduct(
        (IsingXUnitary(3), IsingZZUnitary(3)), (0, 1, 0), (0.05, 2, 0.05), 0.01
    )
    (port,) = product.signature
    (expected_port,) = source.signature
    assert port.dtype.encoding == "opaque" and expected_port.dtype == QAny(3)
    assert product.signature.thru_registers_only and expected_port.side == Side.THRU
    assert (port.name, port.bitsize, port.shape) == (
        expected_port.name,
        expected_port.bitsize,
        expected_port.shape,
    )
    native = modeled_call_graph(product.resource_model(), max_depth=1)
    _, original = source.call_graph(max_depth=1, generalizer=ignore_split_join)
    assert {node.name: count for node, count in native.leaf_counts.items()} == {
        type(node).__name__: count for node, count in original.items()
    }
    circuit = Circuit(3).append(product.make_on(system=(0, 1, 2)))
    torch.testing.assert_close(
        get_unitary(circuit, dtype=torch.complex128),
        torch.from_numpy(source.tensor_contract()),
        atol=1e-12,
        rtol=0,
    )
