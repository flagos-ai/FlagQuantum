"""Stored accumulation uses the live product and preserves occupied outputs."""

from collections import Counter
from itertools import product

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.arithmetic.integer_product import (
    _product_terms,
    integer_product_requirements,
)
from flagquantum.compiler.registers.arithmetic.product_accumulation import (
    product_accumulation_ir,
)
from flagquantum.compiler.registers.arithmetic.product_accumulation_component import (
    product_accumulation_program,
)
from flagquantum.compiler.registers.bits.bit_compression import (
    weighted_column_components,
)
from flagquantum.compiler.registers.component_expansion import expand_component_calls
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import verify_program_record
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_weighted_columns_component import normalized_feedback

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "n,m,r", ((1, 1, 2), (1, 3, 4), (3, 1, 5), (2, 2, 4), (2, 3, 6), (4, 6, 12))
)
@pytest.mark.parametrize("expanded", (False, True))
def test_complete_source_ports_live_word_and_primitives(
    n: int, m: int, r: int, expanded: bool
) -> None:
    from qualtran._infra.data_types import QUInt
    from qualtran.bloqs.arithmetic.addition import Add
    from qualtran.bloqs.arithmetic.multiplication import PlusEqualProduct
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value

    a, b, output = (
        tuple(range(n)),
        tuple(range(n, n + m)),
        tuple(range(n + m, n + m + r)),
    )
    _, count = integer_product_requirements(n, m)
    work = tuple(range(n + m + r, n + m + r + count))
    carries = tuple(range(n + m + r + count, n + m + r + count + r - 1))
    for inverse in (False, True):
        source = PlusEqualProduct(n, m, r, inverse)
        record = product_accumulation_program(n, m, r, adjoint=inverse)
        assert verify_program_record(record) == ()
        ports = record.functions[0].registers
        assert ports is not None
        assert [
            (reg.name, reg.dtype.encoding, reg.dtype.quantum_bits, reg.side.name)
            for reg in ports
        ] == [
            (reg.name, "uint", reg.bitsize, reg.side.name) for reg in source.signature
        ]
        functions = {f.name: f for f in record.functions}
        nested = [
            functions[op.callee or ""]
            for op in record.functions[0].blocks[0].operations
            if (op.callee or "").startswith("__fq_component_")
        ]
        assert len(nested) == 1
        assert tuple(reg.name for reg in nested[0].registers or ()) == ("a", "b")
        if expanded:
            record, _ = expand_component_calls(record)
        bound = bind_register_program(record, (a, b, output), clean=(*work, *carries))
        assert bound.outputs == (a, b, output)
        expected = product_accumulation_ir(
            a, b, output, work, carries, tuple(range(r - 1)), adjoint=inverse
        )
        assert_source(
            normalized_feedback(bound.circuit.instructions),
            normalized_feedback(expected),
        )
        compute, word, erase = weighted_column_components(
            n + m, work, _product_terms(a, b)
        )
        # Reuse the source Add gate stream on the actual live column addresses.
        live_word = tuple(index for index in word if index is not None)
        original_add = Add(QUInt(len(live_word)), QUInt(r))
        child = reference_arithmetic_gates(
            original_add.adjoint() if inverse else original_add,
            {"a": live_word, "b": output},
            carries,
            dict(zip(carries, range(r - 1), strict=True)),
        )
        from flagquantum.compiler.registers.bits.bit_compression import (
            _expand_column_components,
        )

        source_children = (
            *_expand_column_components(compute),
            *child,
            *_expand_column_components(erase, erase_bit=0),
        )
        assert_source(
            normalized_feedback(bound.circuit.instructions),
            normalized_feedback(source_children),
        )
        counts = Counter(g.name for g in bound.circuit.instructions)
        assert counts["t"] + counts["tdg"] == 4 * (count + r - 1)
        assert counts["measure"] == counts["reset"] == count + r - 1
        assert (
            sum(len(event.wires) for event in bound.events if event.kind == "allocate")
            == count + r - 1
        )
        assert (
            sum(len(event.wires) for event in bound.events if event.kind == "free")
            == count + r - 1
        )
        assert (
            get_cost_value(source, QECGatesCost()).total_t_count() == 8 * max(n, m) ** 2
        )


@pytest.mark.parametrize("n,m,r", ((1, 1, 2), (1, 2, 3), (2, 2, 4)))
@pytest.mark.parametrize("expanded", (False, True))
def test_full_coherent_occupied_accumulator_and_inverse(
    n: int, m: int, r: int, expanded: bool
) -> None:
    a, b, output = (
        tuple(range(n)),
        tuple(range(n, n + m)),
        tuple(range(n + m, n + m + r)),
    )
    _, count = integer_product_requirements(n, m)
    work = tuple(range(n + m + r, n + m + r + count + r - 1))
    size = n + m + r + len(work) + 1
    before: dict[int, complex] = {}
    after: dict[int, complex] = {}
    for left, right, occupied, spectator in product(
        range(1 << n), range(1 << m), range(1 << r), (0, 1)
    ):
        amplitude = complex(
            left + right + occupied + 1, left - right + occupied + spectator
        )
        before[
            ((((left << m) | right) << r) | occupied) << (len(work) + 1) | spectator
        ] = amplitude
        image = (occupied + left * right) % (1 << r)
        after[((((left << m) | right) << r) | image) << (len(work) + 1) | spectator] = (
            amplitude
        )
    norm = sum(abs(value) ** 2 for value in before.values()) ** 0.5
    before, after = (
        {i: value / norm for i, value in state.items()} for state in (before, after)
    )
    bound_directions = []
    for inverse, initial, expected in ((False, before, after), (True, after, before)):
        record = product_accumulation_program(n, m, r, adjoint=inverse)
        if expanded:
            record, _ = expand_component_calls(record)
        bound = bind_register_program(record, (a, b, output), clean=work)
        bound_directions.append(bound)
        for seed in (2, 31):
            actual = execute_sparse_register(
                CircuitIR(size, bound.circuit.instructions), initial, seed=seed
            )
            assert all(
                abs(actual.get(i, 0) - expected.get(i, 0)) < 6e-12
                for i in actual.keys() | expected.keys()
            )
    if n == m == 1:
        initial_vector = torch.zeros(1 << size, dtype=torch.complex128)
        expected_vector = torch.zeros_like(initial_vector)
        for i, value in before.items():
            initial_vector[i] = torch.tensor(value, dtype=torch.complex128)
        for i, value in after.items():
            expected_vector[i] = torch.tensor(value, dtype=torch.complex128)
        forward, inverse_bound = bound_directions
        histories = tuple(product((0, 1), repeat=count + r - 1))
        for first, second in product(histories, repeat=2):
            state = forced_branch(initial_vector, forward.circuit.instructions, first)
            torch.testing.assert_close(
                state, expected_vector / 2 ** ((count + r - 1) / 2), atol=6e-12, rtol=0
            )
            restored = forced_branch(state, inverse_bound.circuit.instructions, second)
            torch.testing.assert_close(
                restored, initial_vector / 2 ** (count + r - 1), atol=6e-12, rtol=0
            )


def test_normal_append_replaces_flat_accumulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flagquantum.compiler.registers.arithmetic import product_accumulation

    def refuse(*args: object, **kwargs: object) -> object:
        raise AssertionError("Flat accumulation must be replaceable")

    monkeypatch.setattr(product_accumulation, "product_accumulation_ir", refuse)
    words = ((0,), (1,), (2, 3))
    forward = bind_register_program(
        product_accumulation_program(1, 1, 2), words, clean=(4, 5)
    )
    inverse = bind_register_program(
        product_accumulation_program(1, 1, 2, adjoint=True), words, clean=(4, 5)
    )
    state = torch.zeros(1 << 7, dtype=torch.complex128)
    state[0b1111001] = torch.tensor((1 + 2j) / 5**0.5, dtype=torch.complex128)
    circuit = DynamicCircuit(7, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(7, forward.circuit.instructions))
    circuit.append(FieldOperation(7, inverse.circuit.instructions))
    actual = run_dynamic(circuit, shots=4, seed=11, strategy="trajectory")
    torch.testing.assert_close(
        actual.final_states.reshape(4, -1), state.expand(4, -1), atol=6e-12, rtol=0
    )


@pytest.mark.parametrize(
    "n,m,r", ((True, 1, 2), (1, 0, 2), (0, 1, 2), (1, 1, True), (2, 2, 3))
)
def test_invalid_widths(n: int, m: int, r: int) -> None:
    with pytest.raises(ValueError):
        product_accumulation_program(n, m, r)


def test_invalid_direction() -> None:
    with pytest.raises(ValueError, match="boolean"):
        product_accumulation_program(1, 1, 2, adjoint=1)  # type: ignore[arg-type]
