"""Original inverse primitives, retained junk and explicit clean-free boundary."""

from collections import Counter
from typing import Any

import numpy as np
import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.finite_field.addition_component import (
    field_addition_program,
)
from flagquantum.compiler.registers.finite_field.field_multiplication import (
    FieldDecomposition,
    bind_field_outputs,
)
from flagquantum.compiler.registers.finite_field.inverse_decomposition import (
    field_inverse_decomposition_program,
)
from flagquantum.compiler.registers.finite_field.karatsuba_component import (
    karatsuba_decomposition_program,
)
from flagquantum.compiler.registers.finite_field.square_component import (
    field_square_decomposition_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.field_product_checks import product_source_stream
from tests.unit.test_field_frobenius_component import dtype
from tests.unit.test_field_linear import _source_type

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
FIELDS = ((2, 7), (3, 11), (5, 37), (9, 529))


@pytest.mark.parametrize("width,modulus", FIELDS)
def test_complete_untouched_root_and_literal_dependencies(
    width: int, modulus: int
) -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_addition import GF2Addition
    from qualtran.bloqs.gf_arithmetic.gf2_inverse import GF2Inverse
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import GF2MulViaKaratsuba
    from qualtran.bloqs.gf_arithmetic.gf2_square import GF2Square

    field_type = dtype(width, modulus)
    native = field_inverse_decomposition_program(field_type, modulus=modulus)
    registers = native.functions[0].registers
    assert registers is not None
    builder = ComponentProgramBuilder(Signature(registers))

    class OriginalCalls:
        def allocate(self, n: int | None = None, dtype: Any = None) -> Any:
            assert dtype == _source_type(width, modulus)
            assert n in (None, width)
            return builder.allocate(dtype=field_type)

        def free(self, value: Any) -> None:
            builder.free(value)

        def add(self, bloq: Any, **values: Any) -> Any:
            if isinstance(bloq, GF2Addition):
                child = field_addition_program(field_type)
            elif isinstance(bloq, GF2Square):
                child = field_square_decomposition_program(
                    field_type, int(bloq.k), modulus=modulus, uncompute=bloq.uncompute
                )
            else:
                assert isinstance(bloq, GF2MulViaKaratsuba)
                child = karatsuba_decomposition_program(field_type, modulus=modulus)
            outputs = builder.call_program(child, values)
            result = tuple(outputs[r.name] for r in bloq.signature.rights())
            return result[0] if len(result) == 1 else result

    source: Any = GF2Inverse(_source_type(width, modulus))
    outputs = source.build_composite_bloq(OriginalCalls(), **builder.inputs)
    assert builder.finalize(outputs) == native


@pytest.mark.parametrize("width,modulus", FIELDS)
def test_every_original_primitive_alias_input_and_private_free(
    width: int, modulus: int
) -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_inverse import GF2Inverse
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value

    program = field_inverse_decomposition_program(
        dtype(width, modulus), modulus=modulus
    )
    bound = bind_register_program(
        program,
        (tuple(range(width)),),
        clean=tuple(range(width, width * 12)),
        input_ownership=tuple((wire, False) for wire in range(width)),
        allow_workspace_outputs=True,
    )
    original = GF2Inverse(_source_type(width, modulus))
    gates, outputs = product_source_stream(original, bound, {"x": tuple(range(width))})
    assert bound.circuit.instructions == gates
    assert tuple(w for word in bound.outputs for w in word) == tuple(
        w for r in original.signature.rights() for w in outputs[r.name]
    )
    measured = Counter(g.name for g in gates)
    model = get_cost_value(original, QECGatesCost())
    assert (measured["cx"], measured["ccx"]) == (model.clifford, model.toffoli)
    assert set(measured) <= {"cx", "ccx"}
    assert (
        bound.classical_outputs == ()
    )  # No measurement branches in this forward file.
    active = (
        max(
            *(w for g in gates for w in g.wires),
            *(w for event in bound.events for w in event.wires),
        )
        + 1
    )
    columns = np.zeros((active, 1 << width), dtype=np.bool_)
    for wire in range(width):
        columns[wire] = (np.arange(1 << width) >> (width - 1 - wire)) & 1
    initial = columns.copy()
    for offset in range(len(gates) + 1):
        for event in bound.events:
            if event.kind == "free" and event.instruction_offset == offset:
                assert not np.any(columns[list(event.wires)])
        if offset == len(gates):
            break
        gate = gates[offset]
        columns[gate.wires[-1]] ^= np.all(columns[list(gate.wires[:-1])], axis=0)
    # CNOT/Toffoli are exact phase-free permutations. The all-column equality
    # therefore preserves arbitrary complex weights and an entangled spectator.
    logical = {wire for word in bound.outputs for wire in word}
    assert not np.any(columns[[w for w in range(active) if w not in logical]])
    for gate in reversed(gates):
        columns[gate.wires[-1]] ^= np.all(columns[list(gate.wires[:-1])], axis=0)
    np.testing.assert_array_equal(columns, initial)

    if width <= 3:
        count = active + 2
        desired = tuple(
            tuple(range(i * width, (i + 1) * width)) for i in range(len(bound.outputs))
        )
        instructions = bind_field_outputs(
            FieldDecomposition(gates, bound.outputs, ()), desired
        )
        operator = torch.as_tensor(
            original.decompose_bloq().flatten().tensor_contract(),
            dtype=torch.complex128,
        ).reshape(1 << (width * len(bound.outputs)), 1 << width)
        weights = torch.randn(
            1 << width,
            2,
            dtype=torch.complex128,
            generator=torch.Generator().manual_seed(516),
        )
        weights /= weights.norm()
        state = torch.zeros(1 << count, dtype=torch.complex128)
        state.reshape(1 << width, -1)[:, (0, 2)] = weights
        expected = torch.zeros_like(state)
        expected.reshape(operator.shape[0], -1)[:, (0, 2)] = operator @ weights
        operation = FieldOperation(count, instructions)
        circuit = Circuit(count, inputs=state, dtype=torch.complex128).append(operation)
        torch.testing.assert_close(
            circuit.state().flatten(), expected, atol=2e-12, rtol=0
        )
        inverse = Circuit(count).append(operation).adjoint().to_ir()
        torch.testing.assert_close(
            circuit.compose(inverse).state().flatten(), state, atol=2e-12, rtol=0
        )


@pytest.mark.parametrize("width,modulus", ((4, 19), (6, 67), (8, 283), (8, 285)))
def test_unproved_root_release_refuses_without_claiming_clean_work(
    width: int, modulus: int
) -> None:
    with pytest.raises(
        ValueError, match="root temporary release has no clean-work proof"
    ):
        field_inverse_decomposition_program(dtype(width, modulus), modulus=modulus)
