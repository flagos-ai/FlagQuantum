"""Literal square decomposition parity is separate from mathematical Frobenius."""

from collections import Counter
from dataclasses import replace
from itertools import product
from typing import Any, SupportsIndex

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from examples.algorithms.field_power_decomposition import (
    field_power_child,
    field_power_program,
)
from flagquantum import Circuit
from flagquantum.compiler.registers.finite_field.field_multiplication import (
    FieldDecomposition,
    bind_field_outputs,
)
from flagquantum.compiler.registers.finite_field.frobenius_component import (
    field_frobenius_program,
)
from flagquantum.compiler.registers.finite_field.linear_resource_model import (
    source_binary_linear_resource_formula,
)
from flagquantum.compiler.registers.finite_field.square_component import (
    field_square_decomposition_matrix,
    field_square_decomposition_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.unitary import get_unitary
from tests.unit.test_binary_linear_component import SourceCalls
from tests.unit.test_field_frobenius_component import dtype
from tests.unit.test_field_linear import _source_type
from tests.unit.test_field_multiplication import assert_gate_streams, reference_gates

np = pytest.importorskip("numpy")
pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
FIELDS = ((1, 3), (2, 7), (3, 11), (4, 19), (4, 25), (8, 283))
COUNTS = (-3, 0, 1, 2, 3, 4, 5, 7, 8)


@pytest.mark.parametrize("width,modulus", FIELDS)
@pytest.mark.parametrize("count,uncompute", tuple(product(COUNTS, (False, True))))
def test_full_source_matrix_calls_gates_outputs_and_independent_resource(
    width: int, modulus: int, count: int, uncompute: bool
) -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_square import GF2Square
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value

    source: Any = GF2Square(_source_type(width, modulus), k=count, uncompute=uncompute)
    matrix = field_square_decomposition_matrix(width, count, modulus=modulus)
    np.testing.assert_array_equal(matrix, source.squaring_matrix)
    register_type = dtype(width, modulus)
    program = field_square_decomposition_program(
        register_type, count, modulus=modulus, uncompute=uncompute
    )
    assert verify_program_record(program) == ()
    wires = tuple(reversed(range(width)))
    bound = bind_register_program(program, (wires,))
    reader = SourceCalls()
    returned = source.build_composite_bloq(reader, x=np.asarray(wires))
    assert [(g.name, g.wires) for g in bound.circuit.instructions] == reader.gates
    assert bound.outputs == (tuple(returned["x"]),)
    exported, exported_outputs = reference_gates(
        source, {"x": wires}, (), repair_adjoint=False
    )
    assert_gate_streams(reader.gates, exported)
    assert bound.outputs == (exported_outputs["x"],)
    assert bound.events == () and bound.classical_outputs == ()
    root = program.functions[0]
    assert len(root.blocks[0].operations) == 3
    assert root.registers is not None and root.registers[0].dtype == register_type
    model = get_cost_value(source, QECGatesCost())
    counts = Counter(g.name for g in bound.circuit.instructions)
    independent = source_binary_linear_resource_formula(matrix)
    assert counts["cx"] == independent.clifford == model.clifford
    assert (
        counts["t"]
        == counts["measure"]
        == model.measurement
        == model.total_t_count()
        == 0
    )
    # The actual source catalog consumer must select this decomposition rather
    # than silently substituting the corrected mathematical power/inverse.
    assert (
        field_power_program(
            "GF2Square", Signature(root.registers), {"k": count, "uncompute": uncompute}
        )
        == program
    )
    linear_name = root.blocks[0].operations[1].callee
    assert linear_name is not None
    qualified = field_power_child(
        program,
        linear_name,
        Signature(root.registers),
        {"k": count, "uncompute": uncompute},
    )
    assert qualified is not None
    assert qualified[0] == "SynthesizeLRCircuit" + ("†" if uncompute else "")


@pytest.mark.parametrize("width,modulus", FIELDS)
@pytest.mark.parametrize(
    "count,uncompute", tuple(product((-3, 0, 1, 3, 5), (False, True)))
)
def test_append_complete_complex_operator_spectator_and_exact_native_inverse(
    width: int, modulus: int, count: int, uncompute: bool
) -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_square import GF2Square

    source = GF2Square(_source_type(width, modulus), k=count, uncompute=uncompute)
    inputs = tuple(range(width))
    bound = bind_register_program(
        field_square_decomposition_program(
            dtype(width, modulus), count, modulus=modulus, uncompute=uncompute
        ),
        (inputs,),
    )
    emitted = bind_field_outputs(
        FieldDecomposition(bound.circuit.instructions, bound.outputs, ()), (inputs,)
    )
    actual = get_unitary(
        Circuit(width, dtype=torch.complex128).append(FieldOperation(width, emitted))
    )
    expected = torch.as_tensor(
        source.decompose_bloq().flatten().tensor_contract(), dtype=torch.complex128
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-12)
    # One spectator is coherently entangled; the other starts and stays zero.
    state = torch.zeros(1 << (width + 2), dtype=torch.complex128)
    state[::2] = torch.randn(
        1 << (width + 1),
        dtype=torch.complex128,
        generator=torch.Generator().manual_seed(522),
    )
    state /= state.norm()
    circuit = Circuit(width + 2, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(width + 2, emitted))
    wanted = torch.zeros_like(state)
    wanted[::2] = (expected @ state[::2].reshape(1 << width, 2)).flatten()
    torch.testing.assert_close(circuit.state().flatten(), wanted, rtol=0, atol=2e-12)
    assert torch.count_nonzero(circuit.state().flatten()[1::2]) == 0
    # This is an explicitly owned inverse of captured primitives. The source
    # uncompute flag is not claimed to restore the input.
    inverse = (
        Circuit(width + 2, dtype=torch.complex128)
        .append(FieldOperation(width + 2, emitted))
        .adjoint()
    )
    circuit.append(FieldOperation(width + 2, inverse.to_ir().instructions))
    torch.testing.assert_close(circuit.state().flatten(), state, rtol=0, atol=2e-12)


def test_reproduced_source_defects_remain_distinct_from_mathematical_execution() -> (
    None
):
    register_type = dtype(3, 11)
    for count in (0, 3):
        literal = bind_register_program(
            field_square_decomposition_program(register_type, count, modulus=11),
            ((0, 1, 2),),
        )
        exact = bind_register_program(
            field_frobenius_program(register_type, count, modulus=11), ((0, 1, 2),)
        )
        assert (literal.circuit.instructions, literal.outputs) != (
            exact.circuit.instructions,
            exact.outputs,
        )
        mathematical = field_frobenius_program(register_type, count, modulus=11)
        assert (
            field_power_child(
                mathematical,
                "main",
                Signature(mathematical.functions[0].registers or ()),
                {"k": count, "uncompute": False},
            )
            is None
        )
    assert field_square_decomposition_program(
        register_type, modulus=11, uncompute=True
    ) == field_square_decomposition_program(register_type, modulus=11)
    assert field_frobenius_program(
        register_type, modulus=11, adjoint=True
    ) != field_frobenius_program(register_type, modulus=11)


def test_invalid_matrix_dimensions_field_modulus_and_unresolved_count_refuse() -> None:
    import sympy

    for width, modulus in ((0, 1), (3, 15), (3, 7)):
        with pytest.raises(ValueError):
            field_square_decomposition_matrix(width, 1, modulus=modulus)
    with pytest.raises(ValueError):
        field_square_decomposition_program(
            replace(dtype(3, 11), encoding="opaque"), modulus=11
        )
    with pytest.raises(ValueError):
        field_square_decomposition_program(dtype(3, 11), modulus=13)
    with pytest.raises(ValueError):
        field_square_decomposition_matrix(3, sympy.Symbol("k"), modulus=11)


@pytest.mark.parametrize(
    "kind", ("bool_zero", "bool_one", "numpy", "symbolic_one", "symbolic_three")
)
def test_exact_integral_source_counts_preserve_the_original_schedule(kind: str) -> None:
    import sympy
    from qualtran.bloqs.gf_arithmetic.gf2_square import GF2Square

    counts: dict[str, Any] = {
        "bool_zero": False,
        "bool_one": True,
        "numpy": np.int64(3),
        "symbolic_one": sympy.Integer(1),
        "symbolic_three": sympy.Integer(3),
    }
    count = counts[kind]
    assert isinstance(count, SupportsIndex)
    source = GF2Square(_source_type(3, 11), k=counts[kind])
    assert source.k is count
    np.testing.assert_array_equal(
        field_square_decomposition_matrix(3, count, modulus=11), source.squaring_matrix
    )
    program = field_square_decomposition_program(dtype(3, 11), count, modulus=11)
    root = program.functions[0]
    assert root.registers is not None
    assert (
        field_power_program(
            "GF2Square", Signature(root.registers), {"k": count, "uncompute": False}
        )
        == program
    )


def test_literal_compile_and_append_are_independent_of_source_sdks_and_math_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    import flagquantum.compiler.registers.finite_field.field_linear as mathematical

    original_import = builtins.__import__
    register_type = dtype(3, 11)

    def blocked(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"qualtran", "cirq", "galois", "kickmix"}:
            raise AssertionError(f"Literal execution imported {name}")
        return original_import(name, *args, **kwargs)

    def unavailable(*args: object, **kwargs: object) -> None:
        raise AssertionError("Literal decomposition used the mathematical emitter")

    monkeypatch.setattr(mathematical, "gf2_frobenius", unavailable)
    monkeypatch.setattr(builtins, "__import__", blocked)
    bound = bind_register_program(
        field_square_decomposition_program(
            register_type, 3, modulus=11, uncompute=True
        ),
        ((2, 0, 1),),
    )
    state = torch.randn(
        32, dtype=torch.complex128, generator=torch.Generator().manual_seed(522)
    )
    state[1::2] = 0
    state /= state.norm()
    gates = bind_field_outputs(
        FieldDecomposition(bound.circuit.instructions, bound.outputs, ()), ((2, 0, 1),)
    )
    circuit = Circuit(5, inputs=state, dtype=torch.complex128).append(
        FieldOperation(5, gates)
    )
    assert torch.count_nonzero(circuit.state().flatten()[1::2]) == 0
    inverse = (
        Circuit(5, dtype=torch.complex128).append(FieldOperation(5, gates)).adjoint()
    )
    circuit.append(FieldOperation(5, inverse.to_ir().instructions))
    torch.testing.assert_close(circuit.state().flatten(), state, rtol=0, atol=2e-12)
