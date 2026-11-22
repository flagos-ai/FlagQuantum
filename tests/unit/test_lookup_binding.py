"""Exercise complete table wrapper streams, feedback and owned clean ports."""

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from flagquantum.compiler._component_trace.trace import ComponentCall
from flagquantum.compiler.artifacts.reversible_phase import ReversiblePhaseProgram
from flagquantum.compiler.formats.kmx_block_marks import (
    decode_kmx_block_marks,
    encode_kmx_block_marks,
)
from flagquantum.compiler.registers.lookup.lookup_binding import lookup_binding
from tests.unit.fixtures.generate_lookup_binding import Case

pytestmark = pytest.mark.unit


def native_trace(
    case: Case,
) -> tuple[ReversiblePhaseProgram, tuple[ComponentCall, ...]]:
    n, w = case["address_width"], case["target_width"]
    address = tuple(reversed(range(n)))
    target = tuple(reversed(range(n, n + w)))
    work = tuple(range(n + w, 2 * n + w))
    if case["reuse_pool"]:
        work = work[::-1]
    results = []
    for erase in (
        (False, True) if case["route"] == "joined" else (case["route"] == "erase",)
    ):
        results.append(
            lookup_binding(
                case["table"],
                address,
                target,
                workspace=work,
                erase=erase,
                classical_width=case["classical_width"],
                reused_phase_bits=(
                    (case["classical_width"],)
                    if erase and case["route"] == "joined" and n and w
                    else ()
                ),
            )
        )
    program, calls = results[0]
    if len(results) == 1:
        return program, calls
    tail, tail_calls = results[1]

    def shift(call: ComponentCall) -> ComponentCall:
        return replace(
            call,
            start=call.start + len(program.operations),
            stop=call.stop + len(program.operations),
            children=tuple(shift(child) for child in call.children),
        )

    return replace(tail, operations=(*program.operations, *tail.operations)), (
        *calls,
        *(shift(call) for call in tail_calls),
    )


def test_complete_frozen_wrapper_streams_and_marked_roundtrip() -> None:
    data = json.loads(
        Path(__file__).with_name("fixtures").joinpath("lookup_binding.json").read_text()
    )
    assert data["source_blob"] == "227392313663d4c2c06987e47d93c226869bf4be"
    for row in data["cases"]:
        result = native_trace(row["input"])
        text = encode_kmx_block_marks(*result)
        body = "\n".join(text.splitlines()[1:])
        assert hashlib.sha256(body.encode()).hexdigest() == row["sha256"], row["input"]
        assert len(body.splitlines()) == row["records"]
        assert (
            dict(Counter(line.split()[0] for line in body.splitlines()))
            == row["counts"]
        )
        assert decode_kmx_block_marks(text) == result


@pytest.mark.parametrize("n", [0, 1, 2])
@pytest.mark.parametrize("w", [1, 2])
@pytest.mark.parametrize("permuted", [False, True])
def test_joined_all_raw_complex_branches_with_dirty_target(
    n: int, w: int, permuted: bool
) -> None:
    from itertools import product

    import torch

    from flagquantum.compiler.lowering.reversible_phase_execution import (
        bind_reversible_phase_execution,
    )
    from flagquantum.core.classical_dataflow import ClassicalXor
    from tests.unit.measurement_checks import expand_reset_branches, forced_branch

    count = 2 * n + w + 1
    wires = tuple(reversed(range(count))) if permuted else tuple(range(count))
    address, target, work = wires[:n], wires[n : n + w], wires[n + w : -1]
    spectator = wires[-1]
    table = [(5 * a + 1) % (1 << w) for a in range(1 << n)]
    front, _ = lookup_binding(table, address, target, workspace=work, classical_width=3)
    back, _ = lookup_binding(
        table,
        address,
        target,
        workspace=work,
        erase=True,
        classical_width=3,
        reused_phase_bits=(3,) if n else (),
    )
    program = replace(back, operations=(*front.operations, *back.operations))
    gates = expand_reset_branches(
        (
            *(ClassicalXor((), bit, True).instruction() for bit in program.input_bits),
            *bind_reversible_phase_execution(program),
        ),
        first_slot=100,
    )
    initial = torch.zeros(1 << count, dtype=torch.complex128)
    expected = torch.zeros_like(initial)

    def index(value: int, ports: tuple[int, ...]) -> int:
        return sum(
            ((value >> bit) & 1) << (count - port - 1)
            for bit, port in enumerate(reversed(ports))
        )

    for a, s in product(range(1 << n), (0, 1)):
        amplitude = torch.tensor(
            complex(a + 1, 2 * s - 1) / (1 << n), dtype=torch.complex128
        )
        base = index(a, address) | index(s, (spectator,))
        initial[base | index((1 << w) - 1, target)] = amplitude
        expected[base] = amplitude
    events = [gate for gate in gates if gate.name == "measure"]
    norm = initial.norm()
    initial /= norm
    expected /= norm
    total = torch.zeros_like(initial.real)
    for outcomes in product((0, 1), repeat=len(events)):
        sign = -1 if sum(outcomes[:w]) % 2 else 1
        actual = forced_branch(initial, gates, outcomes)
        torch.testing.assert_close(
            actual, expected * sign / 2 ** (len(events) / 2), atol=6e-14, rtol=6e-14
        )
        total += actual.abs().square()
    torch.testing.assert_close(total, expected.abs().square(), atol=6e-14, rtol=6e-14)
    clear_mask = sum(1 << (count - q - 1) for q in (*target, *work))
    assert torch.count_nonzero(total[(torch.arange(1 << count) & clear_mask) != 0]) == 0


@pytest.mark.parametrize(
    "table", [[], [0], [0, 1, 0], [-1, 0], [0, 2], [0.0, 2], ["0", "2"]]
)
def test_invalid_tables_are_rejected_before_any_program_is_returned(
    table: list[int],
) -> None:
    with pytest.raises(ValueError):
        lookup_binding(table, (0,), (1,), workspace=(2,))


def test_empty_target_and_large_address_guard_preserve_source_admission() -> None:
    for n in (0, 1, 31, 32):
        for table in ([], [0], [0, 0, 0]):
            program, calls = lookup_binding(
                table, tuple(range(n)), (), workspace=tuple(range(n, 2 * n)), erase=True
            )
            assert not program.operations and not calls
    with pytest.raises(ValueError, match="32 address"):
        lookup_binding([], tuple(range(33)), (), workspace=tuple(range(33, 66)))


def test_original_six_address_one_output_erasure_work_shortfall_is_retained() -> None:
    with pytest.raises(ValueError, match="workspace"):
        lookup_binding(
            [0] * 64, tuple(range(6)), (6,), workspace=tuple(range(7, 13)), erase=True
        )


def test_owned_ports_and_classical_storage_are_validated() -> None:
    with pytest.raises(ValueError):
        lookup_binding([0, 1], (0,), (0,), workspace=(1,))
    with pytest.raises(ValueError, match="clean lease"):
        lookup_binding([0, 1], (0,), (1,))
    with pytest.raises(ValueError, match="Classical storage"):
        lookup_binding([0, 1], (0,), (1,), workspace=(2,), classical_width=-1)
    with pytest.raises(ValueError, match="quantum-only"):
        lookup_binding([0, 1], (True,), (1,), workspace=(2,))


def test_normal_append_executes_initialized_lookup_and_measured_cleanup() -> None:
    import torch

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.compiler.lowering.reversible_phase_execution import (
        bind_reversible_phase_execution,
    )
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    front, _ = lookup_binding([0, 1], (0,), (1,), workspace=(2,))
    back, _ = lookup_binding(
        [0, 1], (0,), (1,), workspace=(2,), erase=True, reused_phase_bits=(0,)
    )
    circuit = DynamicCircuit(3, dtype=torch.complex128)
    circuit.h(0)
    for program in (front, back):
        circuit.append(FieldOperation(3, bind_reversible_phase_execution(program)))
    result = run_dynamic(circuit, shots=8, seed=23, strategy="batched")
    expected = torch.zeros(8, dtype=torch.complex128)
    expected[0] = expected[4] = 2**-0.5
    torch.testing.assert_close(
        result.final_states.reshape(8, -1), expected.expand(8, -1), atol=1e-12, rtol=0
    )


def test_source_integer_conversion_and_indexed_table_access() -> None:
    class IntOnly:
        def __int__(self) -> int:
            return 1

    class IndexedTable:
        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int) -> object:
            return ("0", IntOnly())[index]

    for table in (
        [False, True],
        [-0.9, 1.9],
        [b"0", b"1"],
        [0, IntOnly()],
        IndexedTable(),
    ):
        program, calls = lookup_binding(
            cast(Sequence[object], table), (0,), (1,), workspace=(2,)
        )
        assert (program, calls) == lookup_binding([0, 1], (0,), (1,), workspace=(2,))
