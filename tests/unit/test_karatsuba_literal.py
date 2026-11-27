"""Frozen Karatsuba physical defects have an executable literal owner."""

from collections import Counter
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.registers.finite_field.field_multiplication import (
    FieldDecomposition,
    bind_field_outputs,
)
from flagquantum.compiler.registers.finite_field.karatsuba_component import (
    karatsuba_decomposition_implementation,
    karatsuba_decomposition_program,
    karatsuba_multiplication_program,
)
from tests.unit.field_product_checks import bind_product, product_source_stream
from tests.unit.test_field_frobenius_component import dtype
from tests.unit.test_field_linear import _source_type

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
FIELDS = ((2, 7), (3, 11), (4, 19), (5, 37), (8, 283))


@pytest.mark.parametrize("width,modulus", FIELDS)
@pytest.mark.parametrize("public", (False, True))
def test_original_calls_output_ownership_and_independent_model(
    width: int, modulus: int, public: bool
) -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import (
        GF2MulViaKaratsuba,
        _GF2MulViaKaratsubaImpl,
    )
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value

    original: Any = (GF2MulViaKaratsuba if public else _GF2MulViaKaratsubaImpl)(
        _source_type(width, modulus)
    )
    program = (
        karatsuba_decomposition_program
        if public
        else karatsuba_decomposition_implementation
    )(dtype(width, modulus), modulus=modulus)
    bound, ports = bind_product(program)
    gates, outputs = product_source_stream(original, bound, ports)
    # The observer walks the untouched builder graph, not a repaired adjoint
    # or the source scalar product callback.
    assert bound.circuit.instructions == gates
    assert bound.outputs == tuple(outputs.values())
    assert bound.classical_outputs == ()
    histogram = Counter(g.name for g in bound.circuit.instructions)
    model = get_cost_value(original, QECGatesCost())
    assert (histogram["cx"], histogram["ccx"]) == (model.clifford, model.toffoli)
    assert histogram["measure"] == histogram["reset"] == 0
    if public:
        exact, _ = bind_product(
            karatsuba_multiplication_program(dtype(width, modulus), modulus=modulus)
        )
        assert (exact.circuit.instructions, exact.outputs) != (
            bound.circuit.instructions,
            bound.outputs,
        )


@pytest.mark.parametrize("width,modulus", FIELDS[:2])
@pytest.mark.parametrize("public", (False, True))
def test_full_original_complex_operator_clean_padding_and_native_inverse(
    width: int, modulus: int, public: bool
) -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import (
        GF2MulViaKaratsuba,
        _GF2MulViaKaratsubaImpl,
    )

    source: Any = (GF2MulViaKaratsuba if public else _GF2MulViaKaratsubaImpl)(
        _source_type(width, modulus)
    )
    bound, _ = bind_product(
        (
            karatsuba_decomposition_program
            if public
            else karatsuba_decomposition_implementation
        )(dtype(width, modulus), modulus=modulus)
    )
    data = 3 * width
    active = max([data, *(wire + 1 for event in bound.events for wire in event.wires)])
    count = active + 2
    desired = tuple(tuple(range(i * width, (i + 1) * width)) for i in range(3))
    instructions = bind_field_outputs(
        FieldDecomposition(bound.circuit.instructions, bound.outputs, ()), desired
    )
    operator = torch.as_tensor(
        source.decompose_bloq().flatten().tensor_contract(), dtype=torch.complex128
    )
    input_width = 2 * width if public else data
    amplitudes = torch.randn(
        1 << input_width,
        2,
        dtype=torch.complex128,
        generator=torch.Generator().manual_seed(519),
    )
    amplitudes /= amplitudes.norm()
    state = torch.zeros(1 << count, dtype=torch.complex128)
    state.reshape(1 << input_width, -1)[:, (0, 2)] = amplitudes
    expected = torch.zeros_like(state)
    expected.reshape(1 << data, -1)[:, (0, 2)] = operator @ amplitudes
    operation = FieldOperation(count, instructions)
    circuit = Circuit(count, inputs=state, dtype=torch.complex128).append(operation)
    torch.testing.assert_close(circuit.state().flatten(), expected, atol=2e-12, rtol=0)
    # Every workspace/unused bit is zero in the complete expected vector; the
    # only non-data degree of freedom is the entangled spectator at bit1.
    exact_inverse = Circuit(count).append(operation).adjoint().to_ir()
    torch.testing.assert_close(
        circuit.compose(exact_inverse).state().flatten(), state, atol=2e-12, rtol=0
    )


def test_original_public_degree_one_refusal() -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import GF2MulViaKaratsuba

    with pytest.raises(ValueError):
        GF2MulViaKaratsuba(_source_type(1, 3))
    with pytest.raises(ValueError):
        karatsuba_decomposition_program(dtype(1, 3), modulus=3)


def test_odd_width_every_raw_target_and_source_padding_free_points() -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import _GF2MulViaKaratsubaImpl

    width, modulus = 5, 37
    bound, ports = bind_product(
        karatsuba_decomposition_implementation(dtype(width, modulus), modulus=modulus)
    )
    gates, _ = product_source_stream(
        _GF2MulViaKaratsubaImpl(_source_type(width, modulus)), bound, ports
    )
    data = 3 * width
    active = max(wire + 1 for event in bound.events for wire in event.wires)
    count = active + 2
    # Enumerate every f/g/h word and both spectator values, independently
    # propagating the original primitive permutation without a dense operator.
    starts = (
        torch.arange(1 << data, dtype=torch.int64) << (count - data)
    ).repeat_interleave(2)
    starts |= torch.arange(starts.numel(), dtype=torch.int64).remainder(2) << 1
    destinations = starts.clone()
    amplitudes = torch.randn(
        starts.numel(),
        dtype=torch.complex128,
        generator=torch.Generator().manual_seed(5195),
    )
    amplitudes /= amplitudes.norm()
    for offset in range(len(gates) + 1):
        for event in bound.events:
            if event.kind == "free" and event.instruction_offset == offset:
                for wire in event.wires:
                    assert (
                        torch.count_nonzero((destinations >> (count - 1 - wire)) & 1)
                        == 0
                    )
        if offset == len(gates):
            break
        gate = gates[offset]
        assert gate.name in {"cx", "ccx"}
        control = torch.ones_like(destinations)
        for wire in gate.wires[:-1]:
            control &= (destinations >> (count - 1 - wire)) & 1
        destinations ^= control << (count - 1 - gate.wires[-1])
    state = torch.zeros(1 << count, dtype=torch.complex128)
    state[starts] = amplitudes
    expected = torch.zeros_like(state)
    expected[destinations] = amplitudes
    operation = FieldOperation(count, bound.circuit.instructions)
    actual = Circuit(count, inputs=state, dtype=torch.complex128).append(operation)
    torch.testing.assert_close(actual.state().flatten(), expected, atol=2e-12, rtol=0)
    inverse = Circuit(count).append(operation).adjoint().to_ir()
    torch.testing.assert_close(
        actual.compose(inverse).state().flatten(), state, atol=2e-12, rtol=0
    )
