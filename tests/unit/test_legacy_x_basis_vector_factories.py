"""Frozen X-vector constructors retain source emission and covector boundaries."""

from collections import Counter

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.formats.legacy_basis_vector_factories import (
    x_basis_vector_factories,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program import verify_program_record
from flagquantum.errors import SerializationError
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_legacy_arithmetic_factories import _library

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("minus", (False, True))
def test_source_ket_calls_amplitudes_resources_and_append(minus: bool) -> None:
    import cirq
    import numpy as np
    from qualtran.bloqs.basic_gates import MinusState, PlusState
    from qualtran.resource_counting import GateCounts, QECGatesCost, get_cost_value

    source = MinusState() if minus else PlusState()
    programs = tuple(
        resolve_component_program(
            _library(source, depth), 0, x_basis_vector_factories()
        )
        for depth in (0, 1, 2)
    )
    assert programs[0] == programs[1] == programs[2]
    assert verify_program_record(programs[0]) == ()
    manager = cirq.SimpleQubitManager()
    op, ports = source.as_cirq_op(manager)
    assert op is not None
    source_ops = tuple(cirq.Circuit(op).all_operations())
    # The source Minus container retains its literal X then H leaves.
    source_ops = tuple(
        leaf
        for operation in source_ops
        for leaf in (
            operation.circuit.all_operations()
            if isinstance(operation, cirq.CircuitOperation)
            else (operation,)
        )
    )
    bound = bind_register_program(
        programs[0], (), clean=(1,), borrowed=(0,), allow_workspace_outputs=True
    )
    names = ["x" if operation.gate == cirq.X else "h" for operation in source_ops]
    assert all(operation.gate in (cirq.X, cirq.H) for operation in source_ops)
    assert bound.circuit.instructions == tuple(
        Instruction(name, (1,)) for name in names
    )
    assert bound.outputs == ((1,),)
    assert Counter(g.name for g in bound.circuit.instructions) == (
        {"x": 1, "h": 1} if minus else {"h": 1}
    )
    assert get_cost_value(source, QECGatesCost()) == GateCounts()
    initial = torch.tensor([1, 0, 1j, 0], dtype=torch.complex128) / (2**0.5)
    vector = source.tensor_contract()
    np.testing.assert_allclose(
        cirq.final_state_vector(
            cirq.Circuit(op), qubit_order=list(ports["q"]), dtype=np.complex128
        ),
        vector,
    )
    expected = torch.kron(
        torch.tensor([1, 1j], dtype=torch.complex128) / (2**0.5),
        torch.as_tensor(vector, dtype=torch.complex128),
    )
    torch.testing.assert_close(
        forced_branch(initial, bound.circuit.instructions, ()), expected
    )
    circuit = Circuit(2, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(2, bound.circuit.instructions))
    torch.testing.assert_close(circuit.state().reshape(-1), expected)


@pytest.mark.parametrize("minus", (False, True))
def test_source_effects_remain_logical_and_tensor_owned(minus: bool) -> None:
    import cirq
    import numpy as np
    from qualtran.bloqs.basic_gates import (
        MinusEffect,
        MinusState,
        PlusEffect,
        PlusState,
    )

    source = MinusEffect() if minus else PlusEffect()
    for depth in (0, 1, 2):
        program = resolve_component_program(
            _library(source, depth), 0, x_basis_vector_factories()
        )
        assert verify_program_record(program) == ()
        assert not program.functions[0].blocks
        with pytest.raises(ValueError):
            bind_register_program(program, ((0,),), input_ownership=((0, False),))
    with pytest.raises(ValueError, match="no Cirq equivalent"):
        source.as_cirq_op(cirq.SimpleQubitManager())
    ket = MinusState() if minus else PlusState()
    np.testing.assert_allclose(source.tensor_contract(), ket.tensor_contract().conj())


@pytest.mark.parametrize(
    "kind", ("PlusState", "MinusState", "PlusEffect", "MinusEffect")
)
def test_constructor_domain_and_signature(kind: str) -> None:
    from qualtran.bloqs.basic_gates import x_basis

    factory = x_basis_vector_factories()["qualtran.bloqs.basic_gates.x_basis." + kind]
    with pytest.raises(NotImplementedError, match="Come back"):
        getattr(x_basis, kind)(2)
    with pytest.raises(NotImplementedError, match="Come back"):
        factory("bad", None, {"n": 2})
    with pytest.raises(SerializationError, match="signature"):
        factory("bad", (), {})
    with pytest.raises(SerializationError, match="only accepts"):
        factory("bad", None, {"extra": 1})
