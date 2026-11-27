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
def test_actual_dirty_root_release_refuses_execution(width: int, modulus: int) -> None:
    from examples.algorithms.register_field_inverse import FieldInverse

    operation = FieldInverse(
        dtype(width, modulus),
        tuple(range(width)),
        tuple(range(width, width * 12)),
        modulus,
        literal=True,
    )
    with pytest.raises(ValueError, match="Nonzero clean release"):
        operation.bind()


@pytest.mark.parametrize("width,modulus,bad_inputs", ((4, 19, 12), (8, 283, 250)))
def test_static_source_graph_diagnoses_dirty_free_without_resetting(
    width: int, modulus: int, bad_inputs: int
) -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_inverse import GF2Inverse

    from examples.algorithms.register_field_inverse import FieldInverse
    from flagquantum.compiler.registers.finite_field.inverse_decomposition import (
        field_inverse_decomposition_program,
    )

    field_type = dtype(width, modulus)
    graph = field_inverse_decomposition_program(field_type, modulus=modulus)
    bound = bind_register_program(
        graph,
        (tuple(range(width)),),
        clean=tuple(range(width, width * 12)),
        allow_workspace_outputs=True,
    )
    gates, _ = product_source_stream(
        GF2Inverse(_source_type(width, modulus)), bound, {"x": tuple(range(width))}
    )
    assert gates == bound.circuit.instructions
    active = max(w for event in bound.events for w in event.wires) + 1
    columns = np.zeros((active, 1 << width), dtype=np.bool_)
    for wire in range(width):
        columns[wire] = (np.arange(1 << width) >> (width - 1 - wire)) & 1
    violations = []
    dirty_reuse = False
    for offset in range(len(gates) + 1):
        for event in bound.events:
            if event.instruction_offset != offset:
                continue
            if event.kind == "free":
                bad = np.any(columns[list(event.wires)], axis=0)
                if np.any(bad):
                    violations.append(int(np.count_nonzero(bad)))
            elif event.kind == "allocate" and np.any(columns[list(event.wires)]):
                dirty_reuse = True
        if offset == len(gates):
            break
        gate = gates[offset]
        columns[gate.wires[-1]] ^= np.all(columns[list(gate.wires[:-1])], axis=0)
    assert bad_inputs in violations
    assert dirty_reuse
    operation = FieldInverse(
        field_type,
        tuple(range(width)),
        tuple(range(width, width * 12)),
        modulus,
        literal=True,
    )
    with pytest.raises(ValueError, match="Nonzero clean release"):
        Circuit(width).append(operation)


@pytest.mark.parametrize("width,modulus", FIELDS[:2])
def test_literal_append_uses_source_recipe_without_mathematical_owner(
    width: int, modulus: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from examples.algorithms import register_field_inverse as example

    def unavailable(*args: object, **kwargs: object) -> None:
        raise AssertionError("Literal append must not call the mathematical owner")

    monkeypatch.setattr(example, "field_inverse_program", unavailable)
    count = width * 3
    operation = example.FieldInverse(
        dtype(width, modulus),
        tuple(range(width)),
        tuple(range(width, count)),
        modulus,
        literal=True,
    )
    bound = operation.bind()
    weights = torch.randn(
        1 << width,
        2,
        dtype=torch.complex128,
        generator=torch.Generator().manual_seed(5162),
    )
    weights /= weights.norm()
    state = torch.zeros(1 << (count + 1), dtype=torch.complex128)
    state.reshape(1 << width, -1)[:, (0, 1)] = weights
    expected = (
        Circuit(count + 1, inputs=state, dtype=torch.complex128)
        .append(FieldOperation(count + 1, bound.circuit.instructions))
        .state()
    )
    actual = Circuit(count + 1, inputs=state, dtype=torch.complex128).append(operation)
    torch.testing.assert_close(actual.state(), expected, atol=2e-12, rtol=0)
    assert bound.circuit.instructions != ()


@pytest.mark.parametrize(
    "width,modulus,clean",
    (
        (1, 3, True),
        (2, 7, True),
        (3, 11, True),
        (4, 19, False),
        (5, 37, True),
        (6, 91, False),
        (7, 131, False),
        (8, 285, False),
        (9, 529, True),
        (10, 1135, False),
        (11, 2053, False),
        (12, 4331, False),
        (13, 8219, False),
        (14, 16553, False),
        (15, 32821, False),
        (16, 65581, False),
        (17, 131081, True),
    ),
)
def test_full_source_domain_uses_actual_polynomial_proof_not_degree_refusal(
    width: int, modulus: int, clean: bool
) -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_inverse import GF2Inverse

    from examples.algorithms.register_field_inverse import FieldInverse

    program = field_inverse_decomposition_program(
        dtype(width, modulus), modulus=modulus
    )
    source = tuple(range(width))
    pool = tuple(range(width, width * 12))
    diagnostic = bind_register_program(
        program, (source,), clean=pool, allow_workspace_outputs=True
    )
    gates, outputs = product_source_stream(
        GF2Inverse(_source_type(width, modulus)), diagnostic, {"x": source}
    )
    assert gates == diagnostic.circuit.instructions
    assert tuple(w for word in diagnostic.outputs for w in word) == tuple(
        w for name in ("x", "result", "junk") for w in outputs.get(name, ())
    )
    operation = FieldInverse(dtype(width, modulus), source, pool, modulus, literal=True)
    if clean:
        assert operation.bind() == diagnostic
    else:
        with pytest.raises(ValueError, match="exact Boolean polynomial is not zero"):
            operation.bind()


def test_failed_analysis_provider_prevents_literal_append(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from examples.algorithms import register_field_inverse as example

    def unavailable(*args: object, **kwargs: object) -> None:
        raise NotImplementedError("Replacement analysis cannot prove this release")

    monkeypatch.setattr(example, "require_zero_releases", unavailable)
    operation = example.FieldInverse(
        dtype(3, 11), (0, 1, 2), tuple(range(3, 9)), 11, literal=True
    )
    circuit = Circuit(9)
    with pytest.raises(NotImplementedError, match="Replacement analysis"):
        circuit.append(operation)
    assert circuit.to_ir().instructions == ()


def test_valid_degree_two_original_adjoint_and_native_reverse_agree() -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_inverse import GF2Inverse

    from examples.algorithms.register_field_inverse import FieldInverse

    original = GF2Inverse(_source_type(2, 7))
    forward = torch.as_tensor(
        original.decompose_bloq().flatten().tensor_contract(), dtype=torch.complex128
    ).reshape(16, 4)
    reverse = torch.as_tensor(
        original.adjoint().decompose_bloq().flatten().tensor_contract(),
        dtype=torch.complex128,
    ).reshape(4, 16)
    torch.testing.assert_close(
        reverse @ forward, torch.eye(4, dtype=torch.complex128), atol=1e-12, rtol=0
    )
    operation = FieldInverse(dtype(2, 7), (0, 1), (2, 3), 7, literal=True)
    weights = torch.randn(
        4, 2, dtype=torch.complex128, generator=torch.Generator().manual_seed(5164)
    )
    weights /= weights.norm()
    state = torch.zeros(32, dtype=torch.complex128)
    state.reshape(4, 8)[:, :2] = weights
    circuit = Circuit(5, inputs=state, dtype=torch.complex128).append(operation)
    inverse = Circuit(5).append(operation).adjoint().to_ir()
    torch.testing.assert_close(
        circuit.compose(inverse).state().flatten(), state, atol=1e-12, rtol=0
    )
