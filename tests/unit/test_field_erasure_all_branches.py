"""Full eight-bit erasure channels use the shared exact phase certificate."""

from collections import Counter
from typing import Literal

import pytest
import torch

from examples.algorithms.register_field_multiplication import FieldProduct
from flagquantum.compiler.registers.finite_field.field_erasure import (
    gf2_product_erasure,
)
from flagquantum.compiler.registers.finite_field.karatsuba_component import (
    karatsuba_decomposition_program,
)
from tests.unit.field_product_checks import bind_product
from tests.unit.measurement_checks import certified_and_monomial_map
from tests.unit.test_field_frobenius_component import dtype

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def columns(width: int, wires: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple((value >> (2 * width - 1 - i)) & 1 for i in range(2 * width))
        + (0,) * (wires - 2 * width)
        for value in range(1 << (2 * width))
    )


@pytest.mark.parametrize(
    "width,modulus", ((2, 7), (3, 11), (5, 37), (8, 283), (8, 285))
)
@pytest.mark.parametrize("algorithm", ("quadratic", "karatsuba"))
def test_complete_channel_all_inputs_all_raw_histories_and_clean_work(
    width: int, modulus: int, algorithm: Literal["quadratic", "karatsuba"]
) -> None:
    x, y, target, scratch = tuple(
        tuple(range(i * width, (i + 1) * width)) for i in range(4)
    )
    forward = FieldProduct(
        dtype(width, modulus), x, y, target, scratch, modulus, algorithm
    ).bind()
    erasure = gf2_product_erasure(
        forward.outputs[0],
        forward.outputs[1],
        forward.outputs[2],
        tuple(range(7, 7 + width)),
        modulus=modulus,
    )
    erase = erasure.lower(parity_bit=7 + width)
    n_wires = 4 * width + 1
    inputs = columns(width, n_wires)
    actual, phases, effects = certified_and_monomial_map(
        (*forward.circuit.instructions, *erase), inputs
    )
    assert actual == inputs
    torch.testing.assert_close(phases, torch.ones_like(phases), atol=0, rtol=0)
    assert effects == width
    # Every actual X-effect has the same positive 1/sqrt(2) Kraus scalar.
    # All its outcome coefficients vanished on every complete input column,
    # proving each of 2**width histories, absolute complex phases, external
    # entanglement and the exact unnormalized 2**(-width/2) branch amplitude.
    assert (1 << effects) * 2 ** (-effects) == 1
    counts = Counter(g.name for g in erase)
    assert counts["cz"] == width**2
    assert counts["classical_xor"] == effects
    assert counts["measure"] == counts["h"] == counts["x"] == effects
    assert counts["t"] == counts["ccx"] == 0


@pytest.mark.parametrize("width,modulus", ((2, 7), (8, 283)))
def test_original_literal_karatsuba_plus_same_erase_retains_source_phase_defect(
    width: int, modulus: int
) -> None:
    literal, _ = bind_product(
        karatsuba_decomposition_program(dtype(width, modulus), modulus=modulus)
    )
    erasure = gf2_product_erasure(
        literal.outputs[0],
        literal.outputs[1],
        literal.outputs[2],
        tuple(range(width)),
        modulus=modulus,
    )
    gates = (*literal.circuit.instructions, *erasure.lower(parity_bit=width))
    n_wires = max(w for gate in gates for w in gate.wires) + 2
    with pytest.raises(AssertionError, match="Uncancelled measured lookup phase"):
        certified_and_monomial_map(gates, columns(width, n_wires))


@pytest.mark.parametrize("width", (2, 3))
def test_actual_append_measured_adjoint_preserves_prior_classical_snapshot(
    width: int,
) -> None:
    from dataclasses import replace
    from itertools import product

    from examples.algorithms.field_product_definitions import (
        field_product_catalog_program,
    )
    from flagquantum import Circuit
    from flagquantum.compiler.registers.finite_field.product_erasure_program import (
        product_erasure_program,
    )
    from flagquantum.core.classical_dataflow import ClassicalXor
    from flagquantum.core.ir import CircuitIR, Instruction
    from flagquantum.core.ir.program.signatures import Signature
    from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding
    from tests.unit.measurement_checks import forced_branch

    encoding = FiniteFieldEncoding(2, width)
    resolved_type = encoding.register_type()
    word_type = replace(
        resolved_type,
        parameters=tuple(
            (name, value)
            for name, value in resolved_type.parameters
            if not name.startswith("polynomial")
        ),
    )
    assert "polynomial" not in dict(word_type.parameters)
    modulus = int(encoding.gf_type.irreducible_poly)
    x, y, target, scratch = tuple(
        tuple(range(i * width, (i + 1) * width)) for i in range(4)
    )
    operation = FieldProduct(word_type, x, y, target, scratch, modulus)
    forward = operation.bind()
    inverse = replace(
        operation,
        x=forward.outputs[0],
        y=forward.outputs[1],
        target=forward.outputs[2],
        adjoint=True,
    )
    inverse_bound = inverse.bind()
    assert inverse_bound.outputs == forward.outputs[:2]
    assert len(inverse_bound.events) == width
    assert all(e.kind == "retire" for e in inverse_bound.events)
    program = product_erasure_program(resolved_type)
    assert (
        field_product_catalog_program(
            "GF2MulMBUC", Signature(program.functions[0].registers or ()), {}
        )
        == program
    )
    count = 4 * width + 2
    state = torch.zeros(1 << count, dtype=torch.complex128)
    slots = (torch.arange(1 << (2 * width)) << (count - 2 * width)).repeat_interleave(2)
    slots |= torch.arange(slots.numel()).remainder(2) << 1
    state[slots] = torch.randn(
        slots.numel(),
        dtype=torch.complex128,
        generator=torch.Generator().manual_seed(1084),
    )
    state /= state.norm()
    circuit = Circuit(count, inputs=state, dtype=torch.complex128)
    circuit.compose(
        CircuitIR(count, (ClassicalXor((), 100, constant=True).instruction(),))
    )
    circuit.append(operation)
    circuit.append(inverse)
    circuit.compose(
        CircuitIR(
            count,
            (Instruction("x", (count - 2,), metadata={"conditions": ((100, 1),)}),),
        )
    )
    gates = circuit.to_ir().instructions
    measurements = [g for g in gates if g.name == "measure"]
    assert len(measurements) == width
    assert all(int(g.metadata["classical_bit"]) > 100 for g in measurements)
    for history in product((0, 1), repeat=width):
        actual = forced_branch(state, gates, history)
        mask = 2 | sum(
            outcome << (count - 1 - g.wires[0])
            for g, outcome in zip(measurements, history, strict=True)
        )
        expected = state[torch.arange(state.numel()) ^ mask] / 2 ** (width / 2)
        torch.testing.assert_close(actual, expected, atol=2e-14, rtol=0)


def test_erasure_keeps_nominal_descriptor_and_rejects_conflicting_modulus() -> None:
    from dataclasses import replace

    from flagquantum.compiler.registers.finite_field.product_erasure_program import (
        product_erasure_program,
    )
    from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding

    full = FiniteFieldEncoding(2, 3).register_type()
    word_type = replace(
        full,
        parameters=tuple(
            (name, value)
            for name, value in full.parameters
            if not name.startswith("polynomial")
        ),
    )
    with pytest.raises(ValueError, match="resolved binary polynomial"):
        product_erasure_program(word_type)
    resolved = product_erasure_program(word_type, modulus=11)
    assert all(r.dtype is word_type for r in resolved.functions[0].registers or ())
    with pytest.raises(ValueError):
        product_erasure_program(dtype(3, 11), modulus=13)
