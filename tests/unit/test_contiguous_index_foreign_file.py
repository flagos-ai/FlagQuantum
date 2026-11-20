"""Roundtrip the complete native index graph through existing SDK ports."""

from itertools import product
from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.register_binding import (
    BoundRegisterProgram,
    bind_register_program,
)
from flagquantum.compiler.registers.sorting.contiguous_index import (
    contiguous_index_requirements,
)
from flagquantum.compiler.registers.sorting.contiguous_index_component import (
    contiguous_index_program,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.ecosystem.cirq._register_import import import_cirq_register_optree
from flagquantum.ecosystem.cirq._register_program_export import (
    export_bound_register_cirq,
)
from tests.unit.measurement_checks import forced_branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
np = pytest.importorskip("numpy")
cirq = pytest.importorskip("cirq")


def _assert_lifetime_instructions(
    bound: BoundRegisterProgram,
    restored: BoundRegisterProgram,
    inputs: tuple[Any, ...],
    allocations: list[tuple[Any, ...]],
) -> None:
    """Map each original lease to its independently observed SDK Qids."""
    live = dict(enumerate(inputs))
    physical = {qid: wire for wire, qid in enumerate(inputs)}
    supplied = iter(allocations)
    cursor = 0

    def compare(stop: int) -> None:
        nonlocal cursor
        for expected, actual in zip(
            bound.circuit.instructions[cursor:stop],
            restored.circuit.instructions[cursor:stop],
            strict=True,
        ):
            assert (actual.name, actual.params, actual.metadata) == (
                expected.name,
                expected.params,
                expected.metadata,
            )
            expected_qids = tuple(live[wire] for wire in expected.wires)
            for qid, wire in zip(expected_qids, actual.wires, strict=True):
                assert physical.setdefault(qid, wire) == wire
            assert tuple(physical[qid] for qid in expected_qids) == actual.wires
        cursor = stop

    assert len(restored.circuit.instructions) == len(bound.circuit.instructions)
    for event in bound.events:
        assert event.instruction_offset is not None
        compare(event.instruction_offset)
        if event.kind == "allocate":
            qids = next(supplied)
            assert len(qids) == len(event.wires)
            assert not set(qids) & set(live.values())
            live.update(zip(event.wires, qids, strict=True))
        else:
            assert event.kind == "free"
            for wire in event.wires:
                del live[wire]
    compare(len(bound.circuit.instructions))
    assert next(supplied, None) is None
    assert live == dict(enumerate(inputs))
    assert restored.circuit.n_wires == bound.circuit.n_wires


@pytest.mark.parametrize("width,result_width", [(1, 1), (1, 3), (2, 1), (2, 4), (4, 8)])
def test_original_ports_complete_instruction_feedback_and_alias_roundtrip(
    width: int, result_width: int
) -> None:
    from cirq.ops.greedy_qubit_manager import GreedyQubitManager

    class RecordingManager(GreedyQubitManager):
        def __init__(self) -> None:
            super().__init__("index_work", maximize_reuse=True)
            self.allocations: list[tuple[Any, ...]] = []

        def qalloc(self, n: int, dim: int = 2) -> list[Any]:
            values = super().qalloc(n, dim)
            self.allocations.append(tuple(values))
            return values

    manager = RecordingManager()
    graph = contiguous_index_program(width, result_width)
    signature = Signature(graph.functions[0].registers or ())
    logical = 2 * width + result_width
    count, _ = contiguous_index_requirements(width)
    words = (
        tuple(range(width)),
        tuple(range(width, 2 * width)),
        tuple(range(2 * width, logical)),
    )
    work = tuple(range(logical, logical + count))
    bound = bind_register_program(graph, words, clean=work)
    qids = tuple(reversed(cirq.LineQubit.range(logical)))
    arrays = {
        register.name: np.asarray([qids[i] for i in word], dtype=object)
        for register, word in zip(signature, words, strict=True)
    }
    exported, outputs = export_bound_register_cirq(
        bound,
        signature,
        input_quregs=arrays,
        feedback_key_prefix="index",
        qubit_manager=manager,
    )
    for name in arrays:
        np.testing.assert_array_equal(outputs[name], arrays[name])
    imported = import_cirq_register_optree(
        exported.circuit, signature=signature, in_quregs=arrays, out_quregs=outputs
    )
    restored = bind_register_program(imported, words, clean=work)
    assert restored.outputs == bound.outputs == words
    _assert_lifetime_instructions(bound, restored, qids, manager.allocations)
    assert CircuitIR.from_json(restored.circuit.to_json()) == restored.circuit
    if width != 1:
        return
    total = restored.circuit.n_wires + 1
    initial = torch.zeros(1 << total, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    coefficients = torch.arange(1, (1 << (logical + 1)) + 1).double()
    values = torch.complex(coefficients, 3 - coefficients * 2)
    values /= values.norm()
    mask = (1 << result_width) - 1
    for raw in range(1 << logical):
        mu = raw >> (width + result_width)
        nu = (raw >> result_width) & ((1 << width) - 1)
        image = (raw & ~mask) | ((raw & mask) ^ ((nu * (nu + 1) // 2 + mu) & mask))
        for spectator in (0, 1):
            initial[(raw << (count + 1)) | spectator] = values[2 * raw + spectator]
            expected[(image << (count + 1)) | spectator] = values[2 * raw + spectator]
    measured = sum(g.name == "measure" for g in restored.circuit.instructions)
    for history in product((0, 1), repeat=measured):
        actual = forced_branch(initial, restored.circuit.instructions, history)
        torch.testing.assert_close(
            actual, expected * 2 ** (-measured / 2), rtol=0, atol=5e-14
        )


def test_original_factory_serialization() -> None:
    from qualtran.bloqs.arithmetic.conversions.contiguous_index import _to_contg_index
    from qualtran.testing import assert_bloq_example_serializes

    assert_bloq_example_serializes(_to_contg_index)
