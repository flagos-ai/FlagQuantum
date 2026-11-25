"""Compare sparse PREPARE's available source calls and explicit native inverse."""

from hashlib import sha1
from itertools import product
from pathlib import Path
from typing import Any

import pytest

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.lookup.qroam_clean_component import (
    qroam_clean_program,
)
from flagquantum.compiler.registers.lookup.qroam_erasure_component import (
    qroam_measurement_erasure_program,
)
from flagquantum.compiler.registers.lookup.source_qroam_clean import QroamCleanTable
from flagquantum.compiler.registers.preparation.sparse_chemistry_prepare_component import (
    sparse_prepare_component_pair,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.registers import RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.ecosystem._default_linear_combination_oracles import _signature
from tests.unit.preparation_reference import preparation_terminals
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_hubbard_hamiltonian_gqsp import _source_sparse_branch
from tests.unit.test_qubitization_walk_component import (
    reference_runtime as reference_runtime,
)
from tests.unit.test_sparse_chemistry_prepare import _branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_sparse_preparation_source() -> None:
    from qualtran.bloqs.chemistry.sparse import prepare

    for name, blob in (
        ("prepare.py", "02e4b381091a6677d68b299305a9b32c0bf5e04f"),
        ("prepare_test.py", "4d3fa0ec18ef9abf0aef050231f3147b27bb078f"),
    ):
        raw = (Path(prepare.__file__).parent / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


def native_pair(source: Any) -> tuple[ProgramRecord, ProgramRecord]:
    lookup = source.build_qrom_bloq()
    table = QroamCleanTable(
        (source.num_non_zero,),
        (int(source.log_block_size),),
        tuple(tuple(int(x) for x in data.flat) for data in lookup.data),
    )
    widths = tuple(int(width) for width in lookup.target_bitsizes)
    return sparse_prepare_component_pair(
        Signature(_signature(source.signature)),
        source.num_non_zero,
        (
            qroam_clean_program(table, widths, reserve_workspace=False),
            qroam_measurement_erasure_program(table, widths),
        ),
    )


def bind_used_workspace(
    record: ProgramRecord, *, allow_workspace_outputs: bool = False
) -> tuple[Any, int]:
    signature = Signature(record.functions[0].registers or ())
    inputs = []
    owned: list[tuple[int, bool]] = []
    offset = 0
    for reg in signature.lefts():
        for _ in reg.all_idxs():
            width = int(reg.bitsize)
            inputs.append(tuple(range(offset, offset + width)))
            if reg.side is RegisterSide.LEFT:
                owned.extend((wire, False) for wire in inputs[-1])
            offset += width
    provisional = bind_register_program(
        record,
        tuple(inputs),
        clean=tuple(range(offset, offset + 128)),
        allow_workspace_outputs=allow_workspace_outputs,
        input_ownership=tuple(owned),
    )
    used = (
        {wire for event in provisional.events for wire in event.wires}
        | {wire for word in provisional.outputs for wire in word}
        | set(range(offset))
    )
    end = max(used, default=-1) + 1
    bound = bind_register_program(
        record,
        tuple(inputs),
        clean=tuple(range(offset, end)),
        allow_workspace_outputs=allow_workspace_outputs,
        input_ownership=tuple(owned),
    )
    return bound, offset


@pytest.mark.parametrize("mu,log", ((1, 0), (2, 1)))
def test_small_original_forward_complex_output_and_native_joint_roundtrip(
    mu: int, log: int
) -> None:
    cirq = pytest.importorskip("cirq")
    from qualtran._infra.gate_with_registers import get_named_qubits
    from qualtran.bloqs.chemistry.sparse.prepare import PrepareSparse

    rows = ((0, 0, 1, 1), (0, 1, 0, 1), (0, 1, 1, 0), (0, 0, 1, 1))
    source = PrepareSparse(
        4,
        4,
        mu,
        tuple(tuple(reversed(row)) for row in rows),
        (1, 0, 0, 1),
        (0, 1, 1, 0),
        rows,
        (0, 1, 1, 0),
        (1, 0, 0, 1),
        (0, (1 << mu) - 1, 1, 0),
        log_block_size=log,
    )
    pair = native_pair(source)
    for inverse, record in enumerate(pair):
        assert not verify_program_record(record)
        terminals = preparation_terminals(
            record,
            native_boundaries=(
                f.name
                for f in record.functions
                if f.name.endswith("native_qroam_erase_leaf")
            ),
        )
        compare_layers(record, source.adjoint() if inverse else source, terminals)
    forward, left = bind_used_workspace(pair[0], allow_workspace_outputs=True)
    named = get_named_qubits(source.signature)
    visible = tuple(q for reg in source.signature for q in named[reg.name].flat)
    ops = list(
        cirq.decompose(
            source.on_registers(**named),
            keep=lambda op: cirq.is_measurement(op)
            or isinstance(op, cirq.ClassicallyControlledOperation)
            or isinstance(op.gate, cirq.ResetChannel)
            or (len(op.qubits) <= 3 and cirq.has_unitary(op)),
        )
    )
    initial = {0: 0.6 + 0j, 1 << (left - 1): 0.8j}
    source_input = {
        key << (len(visible) - left): value for key, value in initial.items()
    }
    outputs = tuple(wire for word in forward.outputs for wire in word)
    hidden = set(range(forward.circuit.n_wires)) - set(outputs)
    hidden_mask = sum(1 << (forward.circuit.n_wires - 1 - wire) for wire in hidden)
    forward_count = sum(g.name == "measure" for g in forward.circuit.instructions)
    assert forward_count <= 10
    expected = _source_sparse_branch(ops, visible, source_input, 0)
    for history in product((0, 1), repeat=forward_count):
        actual = _branch(
            forward.circuit.instructions,
            forward.circuit.n_wires,
            history,
            {
                key << (forward.circuit.n_wires - left): value
                for key, value in initial.items()
            },
            normalize_measurements=True,
        )
        assert all(not key & hidden_mask for key in actual)
        decoded = {
            sum(
                1 << (len(outputs) - 1 - i)
                for i, wire in enumerate(outputs)
                if key & (1 << (forward.circuit.n_wires - 1 - wire))
            ): value
            for key, value in actual.items()
        }
        for key in expected.keys() | decoded.keys():
            assert decoded.get(key, 0) == pytest.approx(expected.get(key, 0), abs=3e-10)
    builder = ComponentProgramBuilder(
        Signature(tuple(Signature(pair[0].functions[0].registers or ()).lefts()))
    )
    loaded = builder.call_program(pair[0], builder.inputs)
    restored = builder.call_program(pair[1], loaded)
    joint = builder.finalize(restored)
    bound, logical = bind_used_workspace(joint)
    assert logical == left
    count = sum(g.name == "measure" for g in bound.circuit.instructions)
    target = {
        key << (bound.circuit.n_wires - logical): value
        for key, value in initial.items()
    }
    histories = (
        (0,) * count,
        (1,) * count,
        tuple(i % 2 for i in range(count)),
        tuple((i + 1) % 2 for i in range(count)),
    )
    for history in histories:
        recovered = _branch(
            bound.circuit.instructions,
            bound.circuit.n_wires,
            history,
            target,
            normalize_measurements=True,
        )
        for key in recovered.keys() | target.keys():
            assert recovered.get(key, 0) == pytest.approx(target.get(key, 0), abs=3e-10)
