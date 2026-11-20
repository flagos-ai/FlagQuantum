"""Stored permutation indexing retains primitive calls and occupied XOR targets."""

from collections import Counter
from itertools import permutations, product
from typing import Literal

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.component_expansion import expand_component_calls
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.compiler.registers.sorting.permutation_index import (
    permutation_index_requirements,
    permutation_rank_xor_ir,
    permutation_words_xor_ir,
)
from flagquantum.compiler.registers.sorting.permutation_index_component import (
    permutation_index_program,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import verify_program_record
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_weighted_columns_component import normalized_feedback

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("count,width", ((1, 1), (2, 1), (2, 3), (3, 2), (4, 3)))
@pytest.mark.parametrize("operation", ("rank", "words"))
def test_complete_existing_primitive_sequence_and_ownership(
    count: int, width: int, operation: Literal["rank", "words"]
) -> None:
    rank_width, work, slots = permutation_index_requirements(count, width)
    widths = (width,) * (2 * count) + (rank_width,) * count
    cursor = 0
    ports = []
    for size in widths:
        ports.append(tuple(range(cursor, cursor + size)))
        cursor += size
    clean = tuple(range(cursor, cursor + work))
    record = permutation_index_program(count, width, operation)
    assert verify_program_record(record) == ()
    table, words, ranks = (
        tuple(ports[:count]),
        tuple(ports[count : 2 * count]),
        tuple(ports[2 * count :]),
    )
    reference = (
        permutation_rank_xor_ir(table, words, ranks, clean, tuple(range(slots)))
        if operation == "rank"
        else permutation_words_xor_ir(table, ranks, words, clean, tuple(range(slots)))
    )
    bound = bind_register_program(record, tuple(ports), clean=clean)
    actual = normalized_feedback(bound.circuit.instructions)
    expected = normalized_feedback(reference)

    def history(gates: tuple[Instruction, ...], wire: int) -> tuple[object, ...]:
        return tuple(
            (
                gate.name,
                gate.wires.index(wire),
                tuple((i, q) for i, q in enumerate(gate.wires) if q < cursor),
            )
            for gate in gates
            if wire in gate.wires
        )

    mapping = dict(zip(range(cursor), range(cursor), strict=True))
    for wire in clean:
        candidates = [
            q
            for q in clean
            if q not in mapping.values()
            and history(actual, wire) == history(expected, q)
        ]
        assert candidates, wire
        mapping[wire] = candidates[0]
    renamed = tuple(
        Instruction(
            g.name,
            tuple(mapping[q] for q in g.wires),
            g.params,
            metadata=g.metadata,
        )
        for g in actual
    )
    assert_source(renamed, expected)
    assert bound.outputs == tuple(ports)
    assert Counter(g.name for g in bound.circuit.instructions) == Counter(
        g.name for g in reference
    )
    assert CircuitIR.from_json(bound.circuit.to_json()) == bound.circuit
    assert (
        sum(len(e.wires) * (1 if e.kind == "allocate" else -1) for e in bound.events)
        == 0
    )


@pytest.mark.parametrize("expanded", (False, True))
@pytest.mark.parametrize("operation", ("rank", "words"))
def test_all_small_complex_inputs_occupied_targets_every_branch_and_append(
    operation: Literal["rank", "words"],
    expanded: bool,
) -> None:
    ports = tuple((i,) for i in range(6))
    record = permutation_index_program(2, 1, operation)
    if expanded:
        record, _ = expand_component_calls(record)
    total = 11 if expanded else 9
    bound = bind_register_program(record, ports, clean=tuple(range(7, total)))
    initial = torch.zeros(1 << total, dtype=torch.complex128)
    target = torch.zeros_like(initial)
    addresses, outputs = [], []
    for table in permutations((0, 1)):
        for words in product((0, 1), repeat=2):
            for ranks in product((0, 1), repeat=2):
                for spectator in (0, 1):
                    if operation == "rank":
                        changed_words = words
                        changed_ranks = tuple(
                            old ^ table.index(word)
                            for old, word in zip(ranks, words, strict=True)
                        )
                    else:
                        changed_words = tuple(
                            old ^ table[rank]
                            for old, rank in zip(words, ranks, strict=True)
                        )
                        changed_ranks = ranks
                    before = (*table, *words, *ranks, spectator, *((0,) * (total - 7)))
                    after = (
                        *table,
                        *changed_words,
                        *changed_ranks,
                        spectator,
                        *((0,) * (total - 7)),
                    )
                    addresses.append(int("".join(map(str, before)), 2))
                    outputs.append(int("".join(map(str, after)), 2))
    raw = torch.arange(1, len(addresses) + 1, dtype=torch.float64)
    amplitudes = torch.complex(raw, 3 - raw * 7)
    amplitudes /= amplitudes.norm()
    initial[addresses], target[outputs] = amplitudes, amplitudes
    circuit = DynamicCircuit(total, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(total, bound.circuit.instructions))
    events = sum(g.name == "measure" for g in bound.circuit.instructions)
    for history in product((0, 1), repeat=events):
        branch = forced_branch(initial, circuit.to_ir().instructions, history)
        torch.testing.assert_close(
            branch, target / 2 ** (events / 2), rtol=0, atol=3e-12
        )
        assert torch.count_nonzero(branch.reshape(-1, 1 << (total - 7))[:, 1:]) == 0


@pytest.mark.parametrize("count,width", ((0, 1), (2, 0), (5, 2), (True, 2)))
def test_invalid_dimensions_keep_existing_refusals(count: int, width: int) -> None:
    with pytest.raises(ValueError):
        permutation_index_program(count, width, "rank")


def test_three_entry_copy_includes_every_out_of_range_rank_and_dirty_target() -> None:
    from tests.unit.measurement_checks import certified_and_channel_map

    ports = tuple(tuple(range(2 * i, 2 * i + 2)) for i in range(9))
    bound = bind_register_program(
        permutation_index_program(3, 2, "words"), ports, clean=(18, 19)
    )
    inputs: list[tuple[int, ...]] = []
    outputs: list[tuple[int, ...]] = []
    table, words = (0, 1, 2), (1, 2, 3)
    for ranks in product(range(4), repeat=3):
        copied = tuple(
            word ^ (table[rank] if rank < 3 else 0)
            for word, rank in zip(words, ranks, strict=True)
        )
        for rows, values in (
            (inputs, (*table, *words, *ranks)),
            (outputs, (*table, *copied, *ranks)),
        ):
            rows.append(
                tuple((value >> bit) & 1 for value in values for bit in (1, 0)) + (0, 0)
            )
    events = sum(g.name == "measure" for g in bound.circuit.instructions)
    assert certified_and_channel_map(bound.circuit.instructions, tuple(inputs)) == (
        tuple(outputs),
        events,
    )
