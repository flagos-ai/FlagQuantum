"""Blocked XOR calls preserve source flow and restore coherent borrowed banks."""

from hashlib import sha1
from importlib.metadata import version
from itertools import product
from math import prod
from pathlib import Path

import pytest
import torch

from flagquantum.compiler.registers.lookup.select_swap_qrom_component import (
    select_swap_qrom_program,
)
from flagquantum.compiler.registers.lookup.source_qroam_clean import QroamCleanTable
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import Instruction
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_multi_control_component import MeasurementSlots
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_select_swap_qrom import original_select_swap_gates

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
np = pytest.importorskip("numpy")


@pytest.fixture(scope="module", autouse=True)
def pinned_select_swap() -> None:
    import kickmix
    from qualtran.bloqs.data_loading import select_swap_qrom

    assert (
        version("qualtran") == "0.7.0"
        and kickmix.__version__ == "0.1"
        and hasattr(kickmix, "Simulator")
    )
    folder = Path(select_swap_qrom.__file__).parent
    for name, digest in (
        ("select_swap_qrom.py", "0bfc92876f90b2bacca5f876b63a289a7f9eb9fc"),
        ("qrom_base.py", "efe1c5e2300642b8a15d282695c1f73402eb084d"),
        ("select_swap_qrom_test.py", "028fa9ec0cc66698fe727988723be1cd8b8610de"),
    ):
        raw = (folder / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest


@pytest.mark.parametrize("dirty", (False, True))
@pytest.mark.parametrize("adjoint", (False, True))
@pytest.mark.parametrize("shape,logs", (((4,), (1,)), ((5,), (1,))))
def test_source_operands_feedback_all_small_branches_and_borrowed_entanglement(
    dirty: bool, adjoint: bool, shape: tuple[int, ...], logs: tuple[int, ...]
) -> None:
    from qualtran.bloqs.data_loading.select_swap_qrom import SelectSwapQROM
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    table = QroamCleanTable(shape, logs, (tuple(i % 2 for i in range(prod(shape))),))
    source = SelectSwapQROM.build_from_data(
        np.array(table.datasets[0]),
        target_bitsizes=(1,),
        log_block_sizes=logs,
        use_dirty_ancilla=dirty,
    )
    record = select_swap_qrom_program(table, (1,), dirty=dirty, adjoint=adjoint)
    selection_width = (shape[0] - 1).bit_length()
    target_wire = selection_width
    bank_start = target_wire + 1
    borrowed = tuple(range(bank_start, bank_start + 2)) if dirty and not adjoint else ()
    clean_start = bank_start + 2 if borrowed else bank_start
    bound = bind_register_program(
        record,
        (tuple(range(selection_width)), (target_wire,)),
        clean=tuple(range(clean_start, clean_start + 5)),
        borrowed=borrowed,
    )
    bank_events = [e for e in bound.events if e.kind == "allocate"][:2]
    banks = (tuple(e.wires for e in bank_events),)
    assert all(event.dirty is bool(borrowed) for event in bank_events)
    bank_wires = {w for bank in banks for word in bank for w in word}
    work = tuple(
        dict.fromkeys(
            w
            for e in bound.events
            if e.kind == "allocate"
            for w in e.wires
            if w not in bank_wires
        )
    )
    reference, _ = original_select_swap_gates(
        source,
        {"selection": tuple(range(selection_width)), "target0_": (target_wire,)},
        banks,
        work,
        tuple(range(len(work))),
        adjoint=adjoint,
    )
    slots = MeasurementSlots(bound.circuit.instructions)
    active = {}
    remapped = []
    for gate in reference:
        metadata = dict(gate.metadata)
        if gate.name == "measure":
            old = int(metadata["classical_bit"])
            active[old] = slots[gate.wires[0]]
            metadata["classical_bit"] = active[old]
        if "conditions" in metadata:
            metadata["conditions"] = tuple(
                (active[bit], value) for bit, value in metadata["conditions"]
            )
        remapped.append(
            Instruction(gate.name, gate.wires, gate.params, metadata=metadata)
        )
    assert_source(bound.circuit.instructions, tuple(remapped))
    model = get_cost_value(source.adjoint() if adjoint else source, QECGatesCost())
    count = sum(op.name == "measure" for op in bound.circuit.instructions)
    assert count == model.measurement
    assert (
        sum(op.name in ("t", "tdg") for op in bound.circuit.instructions)
        == model.total_t_count()
    )
    n = max(w for g in bound.circuit.instructions for w in g.wires) + 2
    spectator = n - 1
    before = torch.zeros(1 << n, dtype=torch.complex128)
    expected = torch.zeros_like(before)
    for address, target, bank in product(
        range(shape[0]), range(2), range(4 if borrowed else 1)
    ):
        values = {
            **{
                w: address >> (selection_width - 1 - w) & 1
                for w in range(selection_width)
            },
            target_wire: target,
            spectator: (address + target + bank) % 2,
        }
        values.update(
            {w: bank >> (1 - i) & 1 for i, w in enumerate(sorted(bank_wires))}
        )
        index = sum(bit << (n - w - 1) for w, bit in values.items())
        coefficient = torch.tensor(
            complex(1 + address + bank, target - bank - 1), dtype=torch.complex128
        )
        before[index] = coefficient
        expected[index ^ ((table.datasets[0][address]) << (n - target_wire - 1))] = (
            coefficient
        )
    norm = before.norm()
    before /= norm
    expected /= norm
    for history in product((0, 1), repeat=count):
        torch.testing.assert_close(
            forced_branch(before, bound.circuit.instructions, history),
            expected / (2 ** (count / 2)),
            atol=4e-12,
            rtol=0,
        )
    assert bound.outputs == (tuple(range(selection_width)), (target_wire,))
    live: set[int] = set()
    for event in bound.events:
        if event.kind == "allocate":
            live.update(event.wires)
        else:
            live.difference_update(event.wires)
    assert not live


@pytest.mark.parametrize("example", ("_qroam_multi_data", "_qroam_multi_dim"))
@pytest.mark.parametrize("dirty", (False, True))
@pytest.mark.parametrize("adjoint", (False, True))
@pytest.mark.parametrize("blocked", (False, True))
def test_blocked_original_multidata_examples_source_stream_and_complex_bank_channel(
    example: str, dirty: bool, adjoint: bool, blocked: bool
) -> None:
    import itertools

    import attrs
    from qualtran.bloqs.data_loading import select_swap_qrom

    from tests.unit.sparse_register_execution import execute_sparse_register

    source = getattr(select_swap_qrom, example).make()
    source = attrs.evolve(
        source,
        use_dirty_ancilla=dirty,
        log_block_sizes=(
            (1,) * len(source.data_shape) if blocked else source.log_block_sizes
        ),
    )
    table = QroamCleanTable(
        tuple(source.data_shape),
        tuple(source.log_block_sizes),
        tuple(tuple(int(v) for v in data.flat) for data in source.data),
    )
    widths = tuple(source.target_bitsizes)
    record = select_swap_qrom_program(table, widths, dirty=dirty, adjoint=adjoint)
    incoming: dict[str, tuple[int, ...]] = {}
    cursor = 0
    for register in source.signature:
        width = int(register.total_bits())
        incoming[register.name] = tuple(range(cursor, cursor + width))
        cursor += width
    bank_count = prod(table.block_shape) * sum(widths)
    borrowed = (
        tuple(range(cursor, cursor + bank_count)) if dirty and not adjoint else ()
    )
    clean_start = cursor + len(borrowed)
    bound = bind_register_program(
        record,
        tuple(incoming.values()),
        clean=tuple(range(clean_start, clean_start + bank_count + 8)),
        borrowed=borrowed,
    )
    cells = prod(table.block_shape)
    allocated = [event for event in bound.events if event.kind == "allocate"]
    # The inverse dirty graph starts routing the last dataset before the
    # first load. Allocation events therefore follow reversed dataset order;
    # the source's bank arguments below still use semantic dataset order.
    allocation_order = (
        tuple(reversed(range(len(widths))))
        if dirty and adjoint
        else tuple(range(len(widths)))
    )
    by_dataset = {
        dataset: tuple(
            event.wires
            for event in allocated[position * cells : (position + 1) * cells]
        )
        for position, dataset in enumerate(allocation_order)
    }
    banks = tuple(by_dataset[i] for i in range(len(widths)))
    bank_wires = {w for bank in banks for word in bank for w in word}
    work = tuple(
        dict.fromkeys(
            w for event in allocated for w in event.wires if w not in bank_wires
        )
    )
    reference, _ = original_select_swap_gates(
        source, incoming, banks, work, tuple(range(len(work))), adjoint=adjoint
    )
    slots = MeasurementSlots(bound.circuit.instructions)
    active: dict[int, int] = {}
    remapped = []
    for gate in reference:
        metadata = dict(gate.metadata)
        if gate.name == "measure":
            old = int(metadata["classical_bit"])
            active[old] = slots[gate.wires[0]]
            metadata["classical_bit"] = active[old]
        if "conditions" in metadata:
            metadata["conditions"] = tuple(
                (active[bit], value) for bit, value in metadata["conditions"]
            )
        remapped.append(
            Instruction(gate.name, gate.wires, gate.params, metadata=metadata)
        )
    assert_source(bound.circuit.instructions, tuple(remapped))
    n = bound.circuit.n_wires
    from tests.unit.measurement_certificate import certify_erasure_histories

    events = certify_erasure_histories(
        bound.circuit.instructions, n_wires=n, zero_wires=work
    )
    assert events == sum(gate.name == "measure" for gate in reference)
    initial: dict[int, complex] = {}
    expected: dict[int, complex] = {}
    selectors = [
        wires for name, wires in incoming.items() if name.startswith("selection")
    ]
    targets = [wires for name, wires in incoming.items() if name.startswith("target")]
    for row, address in enumerate(itertools.product(*(range(v) for v in table.shape))):
        for branch in (0, 1):
            values: dict[int, int] = {}
            for wires, value in zip(selectors, address, strict=True):
                values.update(
                    {wire: value >> bit & 1 for bit, wire in enumerate(reversed(wires))}
                )
            for i, wires in enumerate(targets):
                value = (row + branch + i) % (1 << len(wires))
                values.update(
                    {wire: value >> bit & 1 for bit, wire in enumerate(reversed(wires))}
                )
            if borrowed:
                values.update(dict.fromkeys(bank_wires, branch))
            index = sum(value << (n - wire - 1) for wire, value in values.items())
            amplitude = complex(row + branch + 1, row - branch - 3)
            initial[index] = initial.get(index, 0j) + amplitude
            output = index
            for i, wires in enumerate(targets):
                data = table.datasets[i][row]
                output ^= sum(
                    ((data >> bit) & 1) << (n - wire - 1)
                    for bit, wire in enumerate(reversed(wires))
                )
            expected[output] = expected.get(output, 0j) + amplitude
    norm = sum(abs(v) ** 2 for v in initial.values()) ** 0.5
    initial = {i: v / norm for i, v in initial.items()}
    expected = {i: v / norm for i, v in expected.items()}
    for seed in (3, 17):
        actual = execute_sparse_register(bound.circuit, dict(initial), seed=seed)
        for index in actual.keys() | expected.keys():
            assert actual.get(index, 0j) == pytest.approx(
                expected.get(index, 0j), abs=4e-12
            )
    assert bound.outputs == tuple(incoming.values())
