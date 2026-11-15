"""Verify the frozen Product tutorial, nested block and seeded dense fixtures."""

from collections.abc import Mapping

import numpy as np
import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from examples.algorithms.product_block_catalog import (
    Binder,
    product_child,
    product_child_binder,
    product_example_signature,
    product_symbolic_example,
)
from flagquantum import Circuit
from flagquantum.algorithms.primitives.product_block_encoding import (
    ProductBlockEncoding,
)
from flagquantum.algorithms.primitives.unitary_block_encoding import (
    UnitaryBlockEncoding,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.simulation.unitary import get_unitary
from tests.unit.test_first_quantization_resource_model import assert_source_model

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("auxiliary", [False, True])
def test_original_product_examples_metadata_model_and_full_matrix(
    auxiliary: bool,
) -> None:
    from qualtran.bloqs.block_encoding.product import (
        _product_block_encoding,
        _product_block_encoding_properties,
    )

    source = (
        _product_block_encoding_properties if auxiliary else _product_block_encoding
    )()
    children = (
        (
            product_child("t", alpha=0.5, ancilla=2, resource=1, epsilon=0.01),
            product_child("h", alpha=0.5, ancilla=1, resource=1, epsilon=0.1),
        )
        if auxiliary
        else (product_child("t"), product_child("h"))
    )
    block = ProductBlockEncoding(children)
    shape = product_example_signature(block)
    assert tuple((r.name, r.total_bits()) for r in shape) == tuple(
        (r.name, r.total_bits()) for r in source.signature
    )
    assert (block.alpha, block.epsilon) == (source.alpha, source.epsilon)
    ancillary = tuple(range(1, 1 + block.ancilla_bitsize))
    resource = tuple(
        range(1 + len(ancillary), 1 + len(ancillary) + block.resource_bitsize)
    )
    required = 1 + len(ancillary) + len(resource)
    assert block.signal_state.emit(ancillary, (), (), ()) == ()
    cost = (
        ComponentCost("T", 1 + (3 if auxiliary else 0), GateCost(t=1)),
        ComponentCost("Hadamard", 1 + (2 if auxiliary else 0), GateCost(clifford=1)),
    )
    assert_source_model(block.resource_model(cost), source)
    for inverse in (False, True):
        operation = block.make_on(
            tuple(product_child_binder(child) for child in children),
            system=(0,),
            ancilla=ancillary,
            resource=resource,
            workspaces=((), ()),
            classical_bits=((), ()),
            adjoint=inverse,
        )
        circuit = Circuit(required, dtype=torch.complex128).append(operation)
        ir = circuit.to_ir()
        assert (
            CircuitIR.from_json(ir.to_json()).validate().instructions == ir.instructions
        )
        original = source.adjoint() if inverse else source
        torch.testing.assert_close(
            get_unitary(circuit),
            torch.as_tensor(original.tensor_contract(), dtype=torch.complex128),
            rtol=0,
            atol=2e-12,
        )


def test_original_product_symbolic_signature_and_metadata() -> None:
    from qualtran import DecomposeTypeError
    from qualtran.bloqs.block_encoding.product import _product_block_encoding_symb

    reference = _product_block_encoding_symb()
    signature, alpha, epsilon = product_symbolic_example()
    assert tuple((r.name, r.total_bits()) for r in signature) == tuple(
        (r.name, r.total_bits()) for r in reference.signature
    )
    assert (alpha, epsilon) == (reference.alpha, reference.epsilon)
    with pytest.raises(DecomposeTypeError, match="Cannot decompose symbolic"):
        reference.decompose_bloq()


def test_original_nested_three_x_product_wires_and_full_matrix() -> None:
    import cirq
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.block_encoding import Product, Unitary

    native_x = product_child("x")
    bind_x = product_child_binder(native_x)
    inner = ProductBlockEncoding((native_x, native_x))
    outer = ProductBlockEncoding((inner, native_x))

    def inner_binder(
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        slots: tuple[int, ...],
        inverse: bool,
    ) -> FieldOperation:
        bound = inner.make_on(
            (bind_x, bind_x),
            system=ports["system"],
            ancilla=ports["ancilla"],
            workspaces=((), ()),
            classical_bits=((), ()),
            adjoint=inverse,
        )
        return FieldOperation(bound.required_qubits, bound.instructions)

    original = Product(
        (Product((Unitary(XGate()), Unitary(XGate()))), Unitary(XGate()))
    )
    qubits = cirq.LineQubit.range(3)
    reference = cirq.Circuit(
        cirq.decompose_once(
            original.on_registers(system=qubits[:1], ancilla=qubits[1:]),
            context=cirq.DecompositionContext(
                cirq.GreedyQubitManager(prefix="_a", maximize_reuse=True)
            ),
        )
    )
    operation = outer.make_on(
        (inner_binder, bind_x),
        system=(0,),
        ancilla=(1, 2),
        workspaces=((), ()),
        classical_bits=((), ()),
    )
    circuit = Circuit(3, dtype=torch.complex128).append(operation)
    assert circuit.to_ir().instructions == (Instruction("x", (0,)),) * 3
    torch.testing.assert_close(
        get_unitary(circuit),
        torch.as_tensor(reference.unitary(qubit_order=qubits), dtype=torch.complex128),
        rtol=0,
        atol=2e-12,
    )


def _dense_binder(matrix: torch.Tensor) -> Binder:
    def bind(
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        slots: tuple[int, ...],
        inverse: bool,
    ) -> FieldOperation:
        wires = ports["system"]
        gate = Instruction(
            "dense_child", wires, matrix=matrix.mH if inverse else matrix
        )
        return FieldOperation(max(wires) + 1, (gate,))

    return bind


def _unused_emitter(
    ports: Mapping[str, tuple[int, ...]],
    work: tuple[int, ...],
    slots: tuple[int, ...],
    inverse: bool,
) -> tuple[Instruction, ...]:
    raise AssertionError("Dense test bodies are supplied by the directional binders")


def test_all_original_seeded_dense_product_fixtures() -> None:
    from qualtran.bloqs.block_encoding import Product, Unitary
    from qualtran.bloqs.for_testing.matrix_gate import MatrixGate

    random_state = np.random.RandomState(1234)
    for _ in range(10):
        count = int(random_state.randint(3, 6))
        width = int(random_state.randint(1, 3))
        gates = tuple(
            MatrixGate.random(width, random_state=random_state) for _ in range(count)
        )
        source = Product(tuple(Unitary(gate) for gate in gates))
        source_matrix = torch.as_tensor(
            source.tensor_contract(), dtype=torch.complex128
        )
        children = (UnitaryBlockEncoding((("q", width),), _unused_emitter),) * count
        native = ProductBlockEncoding(children)
        binders = tuple(
            _dense_binder(
                torch.as_tensor(gate.tensor_contract(), dtype=torch.complex128)
            )
            for gate in gates
        )
        ancilla = tuple(range(width, width + native.ancilla_bitsize))
        for inverse in (False, True):
            op = native.make_on(
                binders,
                system=tuple(range(width)),
                ancilla=ancilla,
                workspaces=((),) * count,
                classical_bits=((),) * count,
                adjoint=inverse,
            )
            circuit = Circuit(width + len(ancilla), dtype=torch.complex128).append(op)
            torch.testing.assert_close(
                get_unitary(circuit),
                source_matrix.mH if inverse else source_matrix,
                rtol=0,
                atol=2e-12,
            )
