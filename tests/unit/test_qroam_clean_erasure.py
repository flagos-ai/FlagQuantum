"""Branch evidence for measurement erasure of source clean-bank lookups."""

from itertools import product
from math import prod

import pytest
import torch

from flagquantum.compiler._component_trace.resource_comparison import (
    count_emitted_instructions,
)
from flagquantum.compiler.registers.lookup.qroam_clean_erasure import (
    qroam_clean_measurement_erase_ir,
)
from flagquantum.compiler.registers.lookup.qroam_resource_model import (
    source_qroam_measurement_erase_cost,
)
from flagquantum.compiler.registers.lookup.source_qroam_clean import (
    QroamCleanTable,
    source_qroam_clean_ir,
)
from flagquantum.core.ir import CircuitIR
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_relative_controlled_swap import source_relative_swap
from tests.unit.test_source_qroam_clean import source_clean_graph

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "shape,logs,n_controls,datasets,widths",
    [
        ((1,), (0,), 0, ((1,),), (1,)),
        ((2,), (0,), 0, ((3, 3), (1, 1)), (2, 1)),
        ((4,), (1,), 0, ((1, 1, 1, 1),), (1,)),
        ((5,), (1,), 1, ((1, 0, 1, 1, 0),), (1,)),
        ((3, 4), (1, 1), 0, (tuple(i % 2 for i in range(12)),), (1,)),
        ((2, 3), (0, 0), 2, ((1, 0, 1, 0, 0, 1),), (1,)),
    ],
)
def test_erase_locked_forward_output(
    shape: tuple[int, ...],
    logs: tuple[int, ...],
    n_controls: int,
    datasets: tuple[tuple[int, ...], ...],
    widths: tuple[int, ...],
) -> None:
    pytest.importorskip("qualtran")
    np = pytest.importorskip("numpy")
    from qualtran.bloqs.data_loading.qroam_clean import QROAMClean

    table = QroamCleanTable(shape, logs, datasets)
    source = QROAMClean.build_from_data(
        *(np.array(data).reshape(shape) for data in datasets),
        target_bitsizes=widths,
        num_controls=n_controls,
        log_block_sizes=logs,
    )
    selection_widths = tuple((extent - 1).bit_length() for extent in shape)
    live = n_controls + sum(selection_widths)
    n_banks = prod(table.block_shape) * sum(widths)
    n_work = table.workspace_size(n_controls, selection_widths)
    n_qubits = live + n_banks + n_work + 1  # The last wire is a spectator.
    wires = iter(range(n_qubits))

    def take(count: int) -> tuple[int, ...]:
        return tuple(next(wires) for _ in range(count))

    controls = take(n_controls)
    selections = tuple(take(width) for width in selection_widths)
    banks = tuple(
        tuple(take(width) for _ in range(prod(table.block_shape))) for width in widths
    )
    scratch = take(n_work)
    bits = tuple(range(n_work))
    bank_bits = tuple(range(n_work, n_work + n_banks))
    ports = {"control": controls} if controls else {}
    ports.update(
        zip(
            (reg.name for reg in source.selection_registers),
            filter(None, selections),
            strict=True,
        )
    )
    outputs = dict(ports)
    for i, bank in enumerate(banks):
        outputs[f"target{i}_"] = bank[0]
        if len(bank) > 1:
            outputs[f"junk_target{i}_"] = tuple(
                wire for word in bank[1:] for wire in word
            )
    forward, _, swaps = source_clean_graph(source, ports, outputs, scratch, bits)
    assert_source(
        source_qroam_clean_ir(table, controls, selections, banks, scratch, bits),
        forward,
    )
    erase = qroam_clean_measurement_erase_ir(
        table, controls, selections, banks, scratch, bits, bank_bits
    )
    reverse_route = tuple(
        gate
        for swap in reversed(swaps)
        for gate in source_relative_swap(swap["ctrl"][0], swap["x"], swap["y"])
    )
    assert_source(erase[: len(reverse_route)], reverse_route)

    forward_counts = count_emitted_instructions(CircuitIR(n_qubits, forward).validate())
    erase_counts = count_emitted_instructions(CircuitIR(n_qubits, erase).validate())
    forward_events = forward_counts.unconditional.get("measure", 0)
    erase_events = erase_counts.unconditional["measure"]
    assert erase_counts.t_count == forward_counts.t_count
    assert erase_events == n_banks + forward_events
    assert (
        tuple(
            gate.metadata["classical_bit"]
            for gate in erase[len(reverse_route) : len(reverse_route) + 3 * n_banks]
            if gate.name == "measure"
        )
        == bank_bits
    )

    # Populate all physical addresses, every control word and both spectator
    # states. Comparing the full vector also proves that all banks are zero.
    state = torch.zeros(1 << n_qubits, dtype=torch.complex128)
    weights = torch.arange(1, (2 << live) + 1, dtype=torch.float64)
    coefficients = torch.complex(weights, weights * (weights % 5 - 2))
    coefficients /= coefficients.norm()
    index = torch.arange(2 << live)
    state[((index // 2) << (n_qubits - live)) | (index % 2)] = coefficients
    expected = state * 2 ** (-(forward_events + erase_events) / 2)
    for history in product((0, 1), repeat=forward_events):
        prepared = forced_branch(state, forward, history)
        for outcomes in product((0, 1), repeat=erase_events):
            recovered = forced_branch(prepared, erase, outcomes)
            torch.testing.assert_close(recovered, expected, atol=2e-12, rtol=0)


@pytest.mark.parametrize("bank_bits", [(), (1,), (-1, 3), (3, 3), (0, 3), (True, 3)])
def test_erase_requires_distinct_bank_slots(bank_bits: tuple[int, ...]) -> None:
    table = QroamCleanTable((3,), (0,), ((0, 1, 3),))
    with pytest.raises(ValueError, match="distinct slots separate from scratch"):
        qroam_clean_measurement_erase_ir(
            table, (), ((0, 1),), (((2, 3),),), (4,), (0,), bank_bits
        )


@pytest.mark.ftqc_reference
def test_measured_instructions_and_abstract_erase_model_are_distinct() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.data_loading.qroam_clean import QROAMClean
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    data = (1, 0, 1, 1, 0)
    table = QroamCleanTable((5,), (1,), (data,))
    erase = qroam_clean_measurement_erase_ir(
        table, (0,), ((1, 2, 3),), (((4,), (5,)),), (6, 7), (0, 1), (2, 3)
    )
    emitted = count_emitted_instructions(CircuitIR(8, erase).validate())
    assert emitted.t_count == 12
    assert emitted.unconditional["measure"] == 4
    assert emitted.conditional == {"cz": 2, "x": 2, "z": 3}
    assert table.workspace_size(1, (3,)) == 2

    source = QROAMClean.build_from_data(
        data, target_bitsizes=(1,), num_controls=1, log_block_sizes=(1,)
    ).adjoint()
    model = source_qroam_measurement_erase_cost(table, 1, (1,)).wrapper
    source_gates = get_cost_value(source, QECGatesCost())
    assert model.gates.toffoli == source_gates.toffoli == 2
    assert model.gates.measurement == source_gates.measurement == 0
    assert model.qubits == get_cost_value(source, QubitCount()) == 6
