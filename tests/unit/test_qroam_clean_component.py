"""Clean multi-bank typed calls preserve the complete original forward lookup."""

from hashlib import sha1
from importlib.metadata import version
from math import prod
from pathlib import Path
from typing import Any

import pytest

from flagquantum.compiler.registers.lookup.qroam_clean_component import (
    qroam_clean_program,
    qroam_clean_unblocked_program,
)
from flagquantum.compiler.registers.lookup.source_qroam_clean import QroamCleanTable
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import verify_program_record
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_multi_control_component import MeasurementSlots
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_source_qroam_clean import source_clean_graph

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_unblocked_template_replaces_finite_lookup_without_changing_execution() -> None:
    from qualtran.bloqs.data_loading.qroam_clean import QROAMClean

    from flagquantum.compiler.formats.legacy_component_resolution import (
        resolve_component_program,
    )
    from flagquantum.compiler.formats.legacy_qrom_factories import qrom_factories
    from flagquantum.compiler.registers.lookup.qroam_clean_metadata import (
        qroam_clean_signature,
    )
    from tests.unit.test_legacy_arithmetic_factories import _library

    source = QROAMClean.build_from_data(
        [0, 1, 2, 3], [3, 2, 1, 0], target_bitsizes=(2, 2), log_block_sizes=(0,)
    )
    lookup = resolve_component_program(
        _library(source.qrom_bloq, 0), 0, qrom_factories()
    )
    signature = qroam_clean_signature((4,), (2, 2), (0,), selection_widths=(2,))
    actual = qroam_clean_unblocked_program(signature, lookup)
    expected = qroam_clean_program(
        QroamCleanTable((4,), (0,), ((0, 1, 2, 3), (3, 2, 1, 0))),
        (2, 2),
        reserve_workspace=False,
    )
    left = bind_register_program(
        actual,
        ((0, 1),),
        clean=tuple(range(2, 24)),
        allow_workspace_outputs=True,
        measured_workspace="reset",
    )
    right = bind_register_program(
        expected,
        ((0, 1),),
        clean=tuple(range(2, 24)),
        allow_workspace_outputs=True,
        measured_workspace="reset",
    )
    assert left.circuit == right.circuit and left.outputs == right.outputs
    assert tuple(
        (e.kind, e.wires, e.dirty, e.instruction_offset) for e in left.events
    ) == tuple((e.kind, e.wires, e.dirty, e.instruction_offset) for e in right.events)


@pytest.fixture(scope="module", autouse=True)
def pinned_source() -> None:
    import kickmix
    from qualtran.bloqs.data_loading import qroam_clean

    assert (
        version("qualtran") == "0.7.0"
        and kickmix.__version__ == "0.1"
        and hasattr(kickmix, "Simulator")
    )
    folder = Path(qroam_clean.__file__).parent
    for name, expected in (
        ("qroam_clean.py", "ab53dc58b9a488ca4e0c95d1d8c8c65fd2242e63"),
        ("qroam_clean_test.py", "c4e5ce6d7ef3c648098bd7de1e98e50a718d29ae"),
        ("select_swap_qrom.py", "0bfc92876f90b2bacca5f876b63a289a7f9eb9fc"),
        ("qrom_base.py", "efe1c5e2300642b8a15d282695c1f73402eb084d"),
    ):
        raw = (folder / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == expected


def _ports(
    registers: Any, words: tuple[tuple[int, ...], ...]
) -> dict[str, tuple[int, ...]]:
    offset = 0
    result = {}
    for register in registers:
        count = prod(register.shape)
        result[register.name] = tuple(
            w for word in words[offset : offset + count] for w in word
        )
        offset += count
    assert offset == len(words)
    return result


@pytest.mark.parametrize(
    "example", ("_qroam_clean_multi_data", "_qroam_clean_multi_dim")
)
def test_original_examples_forward_source_calls_complex_sparse_and_clean_work(
    example: str,
) -> None:
    import kickmix
    from qualtran.bloqs.data_loading import qroam_clean

    assert (
        version("qualtran") == "0.7.0"
        and kickmix.__version__ == "0.1"
        and hasattr(kickmix, "Simulator")
    )
    source = getattr(qroam_clean, example).make()
    _check_original_forward(source)


def _check_original_forward(source: Any) -> None:
    table = QroamCleanTable(
        tuple(source.data_shape),
        tuple(source.log_block_sizes),
        tuple(tuple(int(v) for v in values.flat) for values in source.data),
    )
    record = qroam_clean_program(
        table,
        tuple(source.target_bitsizes),
        selection_widths=tuple(source.selection_bitsizes),
        num_controls=int(source.num_controls),
    )
    assert verify_program_record(record) == ()
    signature = record.functions[0].registers
    assert signature is not None
    assert tuple(
        (reg.name, reg.shape, str(reg.bitsize), reg.side.name) for reg in signature
    ) == tuple(
        (reg.name, reg.shape, str(reg.bitsize), reg.side.name)
        for reg in source.signature
    )
    offset = 0
    inputs = []
    for register in source.signature.lefts():
        inputs.append(tuple(range(offset, offset + int(register.bitsize))))
        offset += int(register.bitsize)
    bound = bind_register_program(
        record,
        tuple(inputs),
        clean=tuple(range(offset, offset + 96)),
        allow_workspace_outputs=True,
    )
    incoming = _ports(source.signature.lefts(), tuple(inputs))
    outgoing = _ports(source.signature.rights(), bound.outputs)
    bank = {
        w
        for name, wires in outgoing.items()
        if name.startswith(("target", "junk_"))
        for w in wires
    }
    work = tuple(
        dict.fromkeys(
            w
            for event in bound.events
            if event.kind == "allocate"
            for w in event.wires
            if w not in bank
        )
    )
    reference, _, _ = source_clean_graph(
        source, incoming, outgoing, work, tuple(range(len(work)))
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
    from tests.unit.measurement_certificate import certify_erasure_histories

    events = certify_erasure_histories(
        bound.circuit.instructions, n_wires=bound.circuit.n_wires, zero_wires=work
    )
    assert events == sum(g.name == "measure" for g in reference)
    n = bound.circuit.n_wires
    state = {}
    # Cover every valid address coherently without allocating a dense bank state.
    import itertools

    domains = (
        *((range(1 << int(source.num_controls)),) if source.num_controls else ()),
        *(range(v) for v in table.shape),
    )
    for position, address in enumerate(itertools.product(*domains)):
        index = 0
        for (name, wires), value in zip(incoming.items(), address, strict=True):
            for bit, wire in enumerate(reversed(wires)):
                index |= ((value >> bit) & 1) << (n - wire - 1)
        state[index] = complex(position + 1, 2 - position)
    norm = sum(abs(value) ** 2 for value in state.values()) ** 0.5
    state = {index: value / norm for index, value in state.items()}
    for seed in (3, 17):
        actual = execute_sparse_register(bound.circuit, dict(state), seed=seed)
        expected = execute_sparse_register(
            CircuitIR(n, tuple(remapped)), dict(state), seed=seed
        )
        for index in actual.keys() | expected.keys():
            assert actual.get(index, 0j) == pytest.approx(
                expected.get(index, 0j), abs=2e-12
            )
        for index, amplitude in actual.items():
            if any(index & (1 << (n - wire - 1)) for wire in work):
                assert abs(amplitude) < 2e-12


@pytest.mark.parametrize(
    "shape,logs,controls",
    (((3, 4), (1, 1), 1), ((2, 3), (0, 0), 2), ((4,), (1,), 2)),
)
def test_block_control_and_full_bank_boundaries_keep_source_physical_layout(
    shape: tuple[int, ...], logs: tuple[int, ...], controls: int
) -> None:
    np = pytest.importorskip("numpy")
    from qualtran.bloqs.data_loading.qroam_clean import QROAMClean

    data = np.arange(prod(shape)).reshape(shape) % 2
    source = QROAMClean.build_from_data(
        data,
        (data + 1) % 4,
        target_bitsizes=(1, 2),
        log_block_sizes=logs,
        num_controls=controls,
    )
    _check_original_forward(source)


def test_full_table_bank_keeps_original_finalization_failure_explicit() -> None:
    np = pytest.importorskip("numpy")
    from qualtran import BloqError
    from qualtran.bloqs.data_loading.qroam_clean import QROAMClean

    from flagquantum.compiler.registers.lookup.source_qroam_clean import (
        source_qroam_clean_ir,
    )

    source = QROAMClean.build_from_data(
        np.arange(4), target_bitsizes=(2,), log_block_sizes=(2,)
    )
    with pytest.raises(BloqError, match="Soquet named `selection`"):
        source.decompose_bloq()
    table = QroamCleanTable((4,), (2,), ((0, 1, 2, 3),))
    record = qroam_clean_program(table, (2,))
    bound = bind_register_program(
        record, ((0, 1),), clean=tuple(range(2, 10)), allow_workspace_outputs=True
    )
    outputs = _ports(source.signature.rights(), bound.outputs)
    words = (
        outputs["target0_"],
        *(outputs["junk_target0_"][j : j + 2] for j in range(0, 6, 2)),
    )
    native = source_qroam_clean_ir(table, (), ((0, 1),), (words,), (), ())
    assert_source(bound.circuit.instructions, native)


@pytest.mark.parametrize("log", (0, 1))
def test_new_forward_and_existing_native_erase_all_raw_branches(log: int) -> None:
    from itertools import product

    import torch

    from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
    from flagquantum.compiler.registers.lookup.clean_lookup_component import (
        clean_lookup_program,
    )
    from flagquantum.core.ir.program.register_types import unsigned_register_type
    from flagquantum.core.ir.program.signatures import Signature
    from tests.unit.measurement_checks import forced_branch

    table = QroamCleanTable((4,), (log,), ((1, 0, 1, 0),))
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(selection=unsigned_register_type(2))
    )
    loaded = builder.call_program(
        qroam_clean_program(table, (1,)), {"selection": builder.inputs["selection"]}
    )
    restored = builder.call_program(clean_lookup_program(table, 1, erase=True), loaded)
    bound = bind_register_program(
        builder.finalize(restored), ((0, 1),), clean=tuple(range(3, 13))
    )
    n = max(w for gate in bound.circuit.instructions for w in gate.wires) + 1
    initial = torch.zeros(1 << n, dtype=torch.complex128)
    amplitudes = torch.complex(
        torch.arange(1, 9, dtype=torch.float64),
        torch.arange(9, 1, -1, dtype=torch.float64),
    )
    amplitudes /= amplitudes.norm()
    initial[torch.arange(8) << (n - 3)] = amplitudes
    count = sum(g.name == "measure" for g in bound.circuit.instructions)
    assert count <= 7
    for history in product((0, 1), repeat=count):
        torch.testing.assert_close(
            forced_branch(initial, bound.circuit.instructions, history),
            initial / (2 ** (count / 2)),
            atol=3e-12,
            rtol=0,
        )
    assert bound.outputs == ((0, 1),)
    live: set[int] = set()
    for event in bound.events:
        if event.kind == "allocate":
            live.update(event.wires)
        else:
            live.difference_update(event.wires)
    assert not live
