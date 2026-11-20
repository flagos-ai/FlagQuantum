"""Stored set operations preserve occupied outputs, phases and owned histories."""

from collections import Counter
from itertools import product

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.compiler.registers.sorting.symmetric_difference import (
    symmetric_difference_ir,
)
from flagquantum.compiler.registers.sorting.symmetric_difference_component import (
    symmetric_difference_program,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import verify_program_record
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_symmetric_difference import layout, rows

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "lhs,rhs,width,size",
    (
        (0, 0, 1, 0),
        (1, 0, 2, 1),
        (2, 0, 2, 1),
        (1, 1, 1, 0),
        (1, 1, 1, 2),
        (2, 1, 2, 2),
        (2, 1, 2, 0),
        (2, 1, 2, 4),
        (2, 1, 2, 1),
        (2, 2, 2, 2),
        (3, 2, 2, 1),
        (3, 2, 2, 3),
        (3, 3, 2, 4),
    ),
)
def test_complete_signature_resources_all_small_sets_and_self_inverse(
    lhs: int, rhs: int, width: int, size: int
) -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.lists.symmetric_difference import SymmetricDifference
    from qualtran.serialization.registers import register_to_proto

    from flagquantum.compiler.formats.legacy_registers import deserialize_register

    a, b, out, flag, clean, slots = layout(lhs, rhs, width, size)
    graph = symmetric_difference_program(lhs, rhs, width, size)
    assert verify_program_record(graph) == ()
    assert graph.functions[0].registers == tuple(
        deserialize_register(register_to_proto(r).SerializeToString(deterministic=True))
        for r in SymmetricDifference(lhs, rhs, size, QUInt(width)).signature
    )
    ports = (*a, *b, *out, (flag,))
    bound = bind_register_program(graph, ports, clean=clean)
    assert bound.outputs == ports
    assert CircuitIR.from_json(bound.circuit.to_json()) == bound.circuit
    counts = Counter(g.name for g in bound.circuit.instructions)
    assert counts == Counter(
        g.name for g in symmetric_difference_ir(a, b, out, flag, clean, slots)
    )
    before, after = zip(*rows(lhs, rhs, width, size, len(clean)), strict=True)
    assert certified_and_channel_map(bound.circuit.instructions, before) == (
        after,
        counts["measure"],
    )
    assert certified_and_channel_map(bound.circuit.instructions, after) == (
        before,
        counts["measure"],
    )
    live: set[int] = set()
    for event in bound.events:
        if event.kind == "allocate":
            live.update(event.wires)
        else:
            live.difference_update(event.wires)
    assert not live


@pytest.mark.parametrize("size", (0, 2))
def test_small_complex_occupied_targets_every_raw_history_and_append(size: int) -> None:
    a, b, out, flag, clean, _ = layout(1, 1, 1, size)
    logical = flag + 1
    pool = tuple(q + 1 for q in clean)
    bound = bind_register_program(
        symmetric_difference_program(1, 1, 1, size), (*a, *b, *out, (flag,)), clean=pool
    )
    total = logical + 1 + len(pool)
    initial = torch.zeros(1 << total, dtype=torch.complex128)
    target = torch.zeros_like(initial)
    entries = tuple(rows(1, 1, 1, size, len(pool)))
    raw = torch.arange(1, len(entries) * 2 + 1, dtype=torch.float64)
    amplitudes = torch.complex(raw, 2 - 7 * raw)
    amplitudes /= amplitudes.norm()
    for i, ((before, after), spectator) in enumerate(product(entries, (0, 1))):
        for vector, state in ((initial, before), (target, after)):
            address = int(
                "".join(map(str, (*state[:logical], spectator, *state[logical:]))), 2
            )
            vector[address] = amplitudes[i]
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


@pytest.mark.parametrize("shape", ((2, 2, 2, 2), (3, 2, 2, 3)))
def test_compaction_and_padding_complex_trajectories_and_clean_work(
    shape: tuple[int, int, int, int],
) -> None:
    a, b, out, flag, clean, _ = layout(*shape)
    bound = bind_register_program(
        symmetric_difference_program(*shape), (*a, *b, *out, (flag,)), clean=clean
    )
    entries = tuple(rows(*shape, len(clean)))
    raw = torch.arange(1, 2 * len(entries) + 1, dtype=torch.float64)
    amplitudes = torch.complex(raw, 1 - 3 * raw)
    amplitudes /= amplitudes.norm()
    initial: dict[int, complex] = {}
    expected: dict[int, complex] = {}
    for i, ((before, after), spectator) in enumerate(product(entries, (0, 1))):
        initial[int("".join(map(str, (*before, spectator))), 2)] = complex(
            amplitudes[i]
        )
        expected[int("".join(map(str, (*after, spectator))), 2)] = complex(
            amplitudes[i]
        )
    total = len(entries[0][0]) + 1
    circuit = DynamicCircuit(total)
    circuit.append(FieldOperation(total, bound.circuit.instructions))
    work_mask = ((1 << len(clean)) - 1) << 1
    for seed in (0, 9, 81):
        result = execute_sparse_register(circuit.to_ir(), initial, seed=seed)
        assert all(
            abs(result.get(i, 0) - expected.get(i, 0)) < 3e-12
            for i in result.keys() | expected.keys()
        )
        assert (
            sum(abs(v) ** 2 for address, v in result.items() if address & work_mask)
            < 1e-24
        )


@pytest.mark.parametrize(
    "shape", ((0, 1, 1, 0), (3, 2, 1, 1), (2, 1, 0, 1), (2, 1, 2, -1), (True, 0, 1, 1))
)
def test_invalid_shapes_keep_shared_validation(
    shape: tuple[int, int, int, int],
) -> None:
    with pytest.raises(ValueError):
        symmetric_difference_program(*shape)
