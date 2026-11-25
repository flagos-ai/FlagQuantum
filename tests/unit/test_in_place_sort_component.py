"""Stored sort composes shared owners and restores coherent distinct words."""

from collections import Counter
from collections.abc import Iterator
from itertools import permutations, product

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.compiler.registers.sorting.in_place_sort import in_place_sort_ir
from flagquantum.compiler.registers.sorting.in_place_sort_component import (
    in_place_sort_program,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import verify_program_record
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.test_in_place_sort import expected_budget, layout, pack, samples
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


def integer_samples(
    count: int, width: int, work: int, signed: bool
) -> Iterator[tuple[tuple[int, ...], tuple[int, ...]]]:
    if not signed:
        yield from samples(count, width, work)
        return
    limit = 1 << (width - 1)
    rank_width = (count - 1).bit_length()
    for words in permutations(range(-limit, limit), count):
        ordered = tuple(sorted(words))
        ranks = tuple(ordered.index(word) for word in words)
        yield (
            pack(words, (0,) * count, width, rank_width, work),
            pack(ordered, ranks, width, rank_width, work),
        )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("count,width", ((1, 1), (2, 1), (2, 2), (3, 2), (4, 2)))
@pytest.mark.parametrize("adjoint", (False, True))
@pytest.mark.parametrize("signed", (False, True))
def test_full_typed_sort_signature_resources_and_all_small_distinct_inputs(
    count: int, width: int, adjoint: bool, signed: bool
) -> None:
    from qualtran import QInt, QUInt
    from qualtran.bloqs.arithmetic.lists.sort_in_place import SortInPlace
    from qualtran.serialization.registers import register_to_proto

    from flagquantum.compiler.formats.legacy_registers import deserialize_register

    words, ranks, clean, slots = layout(count, width)
    graph = in_place_sort_program(count, width, adjoint=adjoint, signed=signed)
    assert graph.functions[0].registers == tuple(
        deserialize_register(register_to_proto(r).SerializeToString(deterministic=True))
        for r in SortInPlace(count, (QInt if signed else QUInt)(width)).signature
    )
    assert verify_program_record(graph) == ()
    bound = bind_register_program(graph, (*words, *ranks), clean=clean)
    assert bound.outputs == (*words, *ranks)
    assert CircuitIR.from_json(bound.circuit.to_json()) == bound.circuit
    reference = in_place_sort_ir(words, ranks, clean, slots, adjoint=adjoint)
    if signed and count > 1:
        bias = tuple(Instruction("x", (word[0],)) for word in words)
        reference = (*bias, *reference, *bias)
    counts = Counter(g.name for g in bound.circuit.instructions)
    assert counts == Counter(g.name for g in reference)
    if signed:
        child = bind_register_program(
            in_place_sort_program(count, width, adjoint=adjoint),
            (*words, *ranks),
            clean=clean,
        )
        # The stored and flat owners have different temporary-wire lifetimes.
        # Compare this wrapper to the unchanged stored child it actually calls.
        bias = (
            tuple(Instruction("x", (word[0],)) for word in words) if count > 1 else ()
        )
        assert_source(
            bound.circuit.instructions, (*bias, *child.circuit.instructions, *bias)
        )
    before, after = zip(*integer_samples(count, width, len(clean), signed), strict=True)
    inputs, expected = (after, before) if adjoint else (before, after)
    t_count, measurements = expected_budget(count, width)
    assert counts["t"] + counts["tdg"] == t_count
    assert counts["measure"] == measurements
    assert certified_and_channel_map(bound.circuit.instructions, inputs) == (
        expected,
        measurements,
    )
    live: set[int] = set()
    for event in bound.events:
        if event.kind == "allocate":
            live.update(event.wires)
        else:
            live.difference_update(event.wires)
    assert not live


@pytest.mark.parametrize("adjoint", (False, True))
@pytest.mark.parametrize("signed", (False, True))
def test_complete_complex_spectator_every_measurement_branch_and_append(
    adjoint: bool,
    signed: bool,
) -> None:
    words, ranks, clean, _ = layout(2, 1)
    logical = 4
    pool = tuple(q + 1 for q in clean)
    bound = bind_register_program(
        in_place_sort_program(2, 1, adjoint=adjoint, signed=signed),
        (*words, *ranks),
        clean=pool,
    )
    total = logical + 1 + len(pool)
    initial, target = (
        torch.zeros(1 << total, dtype=torch.complex128) for _ in range(2)
    )
    rows = tuple(integer_samples(2, 1, len(pool), signed))
    raw = torch.arange(1, len(rows) * 2 + 1, dtype=torch.float64)
    amplitudes = torch.complex(raw, 2 - 7 * raw)
    amplitudes /= amplitudes.norm()
    for index, ((before, after), spectator) in enumerate(product(rows, (0, 1))):
        source, destination = (after, before) if adjoint else (before, after)
        a = int("".join(map(str, (*source[:logical], spectator, *source[logical:]))), 2)
        b = int(
            "".join(
                map(str, (*destination[:logical], spectator, *destination[logical:]))
            ),
            2,
        )
        initial[a], target[b] = amplitudes[index], amplitudes[index]
    circuit = DynamicCircuit(total, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(total, bound.circuit.instructions))
    events = sum(g.name == "measure" for g in bound.circuit.instructions)
    for history in product((0, 1), repeat=events):
        branch = forced_branch(initial, circuit.to_ir().instructions, history)
        torch.testing.assert_close(
            branch, target / 2 ** (events / 2), rtol=0, atol=3e-12
        )
        assert (
            torch.linalg.vector_norm(branch.reshape(-1, 1 << len(pool))[:, 1:]) < 3e-12
        )


@pytest.mark.parametrize("count,width", ((0, 1), (5, 2), (2, 0), (True, 2)))
def test_invalid_shapes_preserve_existing_validation(count: int, width: int) -> None:
    with pytest.raises(ValueError):
        in_place_sort_program(count, width)


@pytest.mark.parametrize("adjoint", (False, True))
@pytest.mark.parametrize("signed", (False, True))
def test_three_word_padded_complex_channel_and_clean_workspace(
    adjoint: bool, signed: bool
) -> None:
    from tests.unit.sparse_register_execution import execute_sparse_register

    words, ranks, clean, _ = layout(3, 2)
    bound = bind_register_program(
        in_place_sort_program(3, 2, adjoint=adjoint, signed=signed),
        (*words, *ranks),
        clean=clean,
    )
    rows = tuple(integer_samples(3, 2, len(clean), signed))
    raw = torch.arange(1, 2 * len(rows) + 1, dtype=torch.float64)
    amplitudes = torch.complex(raw, 1 - raw * 3)
    amplitudes /= amplitudes.norm()
    before: dict[int, complex] = {}
    expected: dict[int, complex] = {}
    for index, ((a, b), spectator) in enumerate(product(rows, (0, 1))):
        source, target = (b, a) if adjoint else (a, b)
        before[int("".join(map(str, (*source, spectator))), 2)] = complex(
            amplitudes[index]
        )
        expected[int("".join(map(str, (*target, spectator))), 2)] = complex(
            amplitudes[index]
        )
    size = len(rows[0][0]) + 1
    circuit = DynamicCircuit(size)
    circuit.append(FieldOperation(size, bound.circuit.instructions))
    result = execute_sparse_register(circuit.to_ir(), before, seed=81)
    assert all(
        abs(result.get(i, 0) - expected.get(i, 0)) < 3e-12
        for i in result.keys() | expected.keys()
    )
    work_mask = ((1 << len(clean)) - 1) << 1
    assert (
        sum(abs(value) ** 2 for address, value in result.items() if address & work_mask)
        < 1e-24
    )
