"""Complete stored quadratic and Karatsuba chains, exact branches and append use."""

from collections import Counter
from dataclasses import replace
from itertools import product
from typing import Literal

import pytest
import torch

from examples.algorithms.register_field_multiplication import FieldProduct
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.finite_field.field_multiplication import (
    gf2_multiplication,
)
from flagquantum.compiler.registers.finite_field.karatsuba_component import (
    karatsuba_multiplication_program,
)
from flagquantum.compiler.registers.finite_field.multiplication_component import (
    field_multiplication_program,
)
from flagquantum.compiler.registers.finite_field.product_erasure_program import (
    product_erasure_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.field_product_checks import bind_product, product_source_stream
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import assert_source_wire_streams
from tests.unit.test_field_frobenius_component import dtype
from tests.unit.test_field_linear import _source_type
from tests.unit.test_field_multiplication import reference_gates

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
FIELDS = ((1, 3), (2, 7), (3, 11), (4, 19), (5, 37), (8, 283))


@pytest.mark.parametrize("width,modulus", FIELDS)
@pytest.mark.parametrize("accumulate", [False, True])
@pytest.mark.parametrize("adjoint", [False, True])
def test_quadratic_original_complete_graph(
    width: int, modulus: int, accumulate: bool, adjoint: bool
) -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import GF2Multiplication

    program = field_multiplication_program(
        dtype(width, modulus),
        modulus=modulus,
        plus_equal_prod=accumulate,
        adjoint=adjoint,
    )
    bound, ports = bind_product(program)
    source = GF2Multiplication(_source_type(width, modulus), plus_equal_prod=accumulate)
    if adjoint:
        source = source.adjoint()
    gates, outputs = product_source_stream(source, bound, ports)
    assert_source_wire_streams(bound.circuit.instructions, gates)
    assert bound.outputs == tuple(outputs.values())
    assert bound.classical_outputs == ()
    assert Counter(g.name for g in gates)["ccx"] == width**2


@pytest.mark.parametrize("width,modulus", FIELDS)
@pytest.mark.parametrize("algorithm", ["quadratic", "karatsuba"])
def test_full_graph_matches_accepted_math_owner_and_independent_source_model(
    width: int, modulus: int, algorithm: Literal["quadratic", "karatsuba"]
) -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import (
        GF2Multiplication,
        GF2MulViaKaratsuba,
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    if algorithm == "karatsuba" and width == 1:
        with pytest.raises(ValueError):
            karatsuba_multiplication_program(dtype(width, modulus), modulus=modulus)
        return
    words = tuple(tuple(range(i * width, (i + 1) * width)) for i in range(3))
    work = tuple(range(3 * width, 3 * width + 8))
    bound = FieldProduct(
        dtype(width, modulus), words[0], words[1], words[2], work, modulus, algorithm
    ).bind()
    plan = gf2_multiplication(
        words[0], words[1], words[2], work, modulus=modulus, algorithm=algorithm
    )
    assert bound.circuit.instructions == plan.instructions
    assert bound.outputs == plan.outputs
    assert not bound.classical_outputs
    source = (GF2Multiplication if algorithm == "quadratic" else GF2MulViaKaratsuba)(
        _source_type(width, modulus)
    )
    counts = Counter(g.name for g in plan.instructions)
    model = get_cost_value(source, QECGatesCost())
    assert (counts["ccx"], counts["cx"]) == (model.toffoli, model.clifford)


@pytest.mark.parametrize("width,modulus", FIELDS[:3])
@pytest.mark.parametrize("algorithm", ["quadratic", "karatsuba"])
def test_append_full_complex_product_spectator_work_and_exact_restore(
    width: int,
    modulus: int,
    algorithm: Literal["quadratic", "karatsuba"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if width == 1 and algorithm == "karatsuba":
        return
    import flagquantum.compiler.registers.finite_field.field_multiplication as physical

    def replaced(*args: object, **kwargs: object) -> None:
        raise AssertionError("Append must follow the stored graph")

    monkeypatch.setattr(physical, "gf2_multiplication", replaced)
    words = tuple(tuple(range(i * width, (i + 1) * width)) for i in range(3))
    operation = FieldProduct(
        dtype(width, modulus),
        words[0],
        words[1],
        words[2],
        tuple(range(3 * width, 4 * width)),
        modulus,
        algorithm,
    )
    bound = operation.bind()
    count = 4 * width + 2
    state = torch.zeros(1 << count, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    field = _source_type(width, modulus).gf_type
    for x, y, spectator in product(range(1 << width), range(1 << width), (0, 1)):
        index = x << (count - width) | y << (count - 2 * width) | spectator << 1
        amplitude = complex(x + 1 + spectator, 2 * y - 3 - spectator)
        state[index] = torch.tensor(amplitude, dtype=torch.complex128)
        value = int(field(x) * field(y))
        output = sum(
            ((value >> (width - 1 - i)) & 1) << (count - 1 - w)
            for i, w in enumerate(bound.outputs[2])
        )
        expected[index | output] = torch.tensor(amplitude, dtype=torch.complex128)
    norm = state.norm()
    state /= norm
    expected /= norm
    circuit = Circuit(count, inputs=state, dtype=torch.complex128).append(operation)
    torch.testing.assert_close(circuit.state().flatten(), expected, atol=1e-12, rtol=0)
    if algorithm == "quadratic":
        inverse = replace(
            operation,
            x=bound.outputs[0],
            y=bound.outputs[1],
            target=bound.outputs[2],
            adjoint=True,
        )
        restored = circuit.append(inverse).state().flatten()
        torch.testing.assert_close(restored, state, atol=1e-12, rtol=0)


@pytest.mark.parametrize("width,modulus", FIELDS[:3])
def test_arbitrary_target_accumulation_and_exact_inverse(
    width: int, modulus: int
) -> None:
    words = tuple(tuple(range(i * width, (i + 1) * width)) for i in range(3))
    operation = FieldProduct(
        dtype(width, modulus),
        words[0],
        words[1],
        words[2],
        (),
        modulus,
        "quadratic",
        plus_equal_prod=True,
    )
    bound = operation.bind()
    count = 3 * width + 2
    state = torch.zeros(1 << count, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    field = _source_type(width, modulus).gf_type
    for x, y, z, spectator in product(
        range(1 << width), range(1 << width), range(1 << width), (0, 1)
    ):
        index = (x << 2 * width | y << width | z) << 2 | spectator << 1
        state[index] = torch.tensor(
            complex(x + 1 + z, y - 3 - spectator), dtype=torch.complex128
        )
        value = z ^ int(field(x) * field(y))
        out = x << (count - width) | y << (count - 2 * width) | spectator << 1
        out |= sum(
            ((value >> (width - 1 - i)) & 1) << (count - 1 - w)
            for i, w in enumerate(bound.outputs[2])
        )
        expected[out] = state[index]
    norm = state.norm()
    state /= norm
    expected /= norm
    circuit = Circuit(count, inputs=state, dtype=torch.complex128).append(operation)
    torch.testing.assert_close(circuit.state().flatten(), expected, atol=1e-12, rtol=0)
    inverse = replace(operation, target=bound.outputs[2], adjoint=True)
    torch.testing.assert_close(
        circuit.append(inverse).state().flatten(), state, atol=1e-12, rtol=0
    )


@pytest.mark.parametrize("width,modulus", FIELDS[1:3])
@pytest.mark.parametrize("algorithm", ["quadratic", "karatsuba"])
@pytest.mark.parametrize("clear", [False, True])
def test_compute_then_erase_every_unnormalized_complex_branch(
    width: int, modulus: int, algorithm: Literal["quadratic", "karatsuba"], clear: bool
) -> None:
    from examples.algorithms.binary_field_arithmetic import FieldOperation

    field_type = dtype(width, modulus)
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(x=field_type, y=field_type)
    )
    program = (
        field_multiplication_program(field_type, modulus=modulus)
        if algorithm == "quadratic"
        else karatsuba_multiplication_program(field_type, modulus=modulus)
    )
    values = builder.call_program(program, builder.inputs)
    outputs = builder.call_program(product_erasure_program(field_type), values)
    combined = builder.finalize(outputs)
    bound = bind_register_program(
        combined,
        (tuple(range(width)), tuple(range(width, 2 * width))),
        clean=tuple(range(2 * width, 4 * width)),
        measured_workspace="reset" if clear else "retire",
    )
    count = 4 * width + 2
    state = torch.zeros(1 << count, dtype=torch.complex128)
    for x, y, spectator in product(range(1 << width), range(1 << width), (0, 1)):
        index = x << (count - width) | y << (count - 2 * width) | spectator << 1
        state[index] = torch.tensor(
            complex(2 * x + 1 + spectator, y - 3), dtype=torch.complex128
        )
    state /= state.norm()
    circuit = Circuit(count, inputs=state, dtype=torch.complex128).append(
        FieldOperation(count, bound.circuit.instructions)
    )
    gates = circuit.to_ir().instructions
    assert gates == bound.circuit.instructions
    measured = [g.wires[0] for g in gates if g.name == "measure"]
    total = 0.0
    for history in product((0, 1), repeat=width):
        branch = forced_branch(state, gates, history)
        expected = state.clone() / (2 ** (width / 2))
        if not clear:
            mask = sum(
                value << (count - 1 - wire)
                for wire, value in zip(measured, history, strict=True)
            )
            expected = expected[torch.arange(len(expected)) ^ mask]
        torch.testing.assert_close(branch, expected, atol=2e-14, rtol=0)
        total += float(branch.abs().square().sum())
    assert total == pytest.approx(1)
    emitted = Counter(g.name for g in gates)
    assert emitted["cz"] == width**2 and emitted["measure"] == width
    assert emitted["reset"] == (width if clear else 0)


def test_unchanged_source_ignored_adjoint_counterexample_retained() -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import GF2MulViaKaratsuba

    source, outputs = reference_gates(
        GF2MulViaKaratsuba(_source_type(2, 7)), {"x": (0, 1), "y": (2, 3)}, (4, 5)
    )
    bits = [0, 1, 0, 1, 0, 0]
    for _, wires in source:
        bits[wires[-1]] ^= all(bits[w] for w in wires[:-1])
    assert sum(bits[w] << (1 - i) for i, w in enumerate(outputs["result"])) == 0
    assert int(_source_type(2, 7).gf_type(1) ** 2) == 1


@pytest.mark.parametrize("width,modulus", FIELDS[1:])
def test_original_karatsuba_root_calls_and_operands_with_owned_exact_child(
    width: int, modulus: int
) -> None:
    """Read original root construction; exact scale-adjoint is a disclosed adapter."""
    from typing import Any

    from qualtran.bloqs.basic_gates import CNOT
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import (
        BinaryPolynomialMultiplication,
        GF2MulK,
        GF2ShiftRight,
        _GF2MulViaKaratsubaImpl,
    )

    from flagquantum.compiler.registers.finite_field._polynomial_component_calls import (
        product_bit_functions,
    )
    from flagquantum.compiler.registers.finite_field.constant_multiplication_component import (
        constant_multiplication_program,
    )
    from flagquantum.compiler.registers.finite_field.karatsuba_component import (
        karatsuba_implementation_program,
    )
    from flagquantum.compiler.registers.finite_field.monomial_component import (
        monomial_multiplication_program,
    )
    from flagquantum.compiler.registers.finite_field.polynomial_product_component import (
        binary_polynomial_program,
    )

    field_type = dtype(width, modulus)
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(f=field_type, g=field_type, h=field_type),
        product_bit_functions(),
    )

    class OriginalCalls:
        def split(self, word: Any) -> Any:
            return builder.split(word)

        def join(self, bits: Any, dtype: Any = None) -> Any:
            assert dtype is not None
            assert int(dtype.degree) == width
            return builder.join(bits, field_type)

        def add(self, bloq: Any, **values: Any) -> Any:
            if isinstance(bloq, CNOT):
                return builder.call_tuple("product_cx", values)
            adjoint = isinstance(getattr(bloq, "subbloq", None), GF2MulK)
            child = bloq.subbloq if adjoint else bloq
            if isinstance(child, GF2MulK):
                program = constant_multiplication_program(
                    field_type, int(child._const), modulus=modulus, adjoint=adjoint
                )
            elif isinstance(child, GF2ShiftRight):
                program = monomial_multiplication_program(
                    field_type, int(child.k), modulus=modulus
                )
            else:
                assert isinstance(child, BinaryPolynomialMultiplication)
                program = binary_polynomial_program(int(child.n))
            result = builder.call_program(program, values)
            outputs = tuple(result[r.name] for r in bloq.signature.rights())
            return outputs[0] if len(outputs) == 1 else outputs

    original = _GF2MulViaKaratsubaImpl(
        _source_type(width, modulus).gf_type.irreducible_poly
    )
    outputs = original.build_composite_bloq(OriginalCalls(), **builder.inputs)
    assert builder.finalize(outputs) == karatsuba_implementation_program(
        field_type, modulus=modulus
    )
