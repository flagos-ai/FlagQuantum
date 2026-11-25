"""Check the sparse PREPARE extension with arbitrary caller-owned lookup banks."""

from itertools import product
from math import pi
from typing import Any

import pytest

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.lookup.qroam_bank_unitary_component import (
    qroam_bank_unitary_pair,
)
from flagquantum.compiler.registers.lookup.source_qroam_clean import QroamCleanTable
from flagquantum.compiler.registers.preparation.prepared_reflection_component import (
    prepared_reflection_program,
)
from flagquantum.compiler.registers.preparation.sparse_chemistry_prepare_component import (
    sparse_prepare_component_pair,
)
from flagquantum.compiler.registers.primitives.global_phase_component import (
    scalar_phase_program,
)
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.ecosystem._default_linear_combination_oracles import _signature
from tests.unit.measurement_certificate import certify_erasure_histories
from tests.unit.test_qubitization_walk_component import (
    reference_runtime as reference_runtime,
)
from tests.unit.test_sparse_chemistry_prepare import _branch
from tests.unit.test_sparse_chemistry_prepare_component import (
    bind_used_workspace,
)
from tests.unit.test_sparse_chemistry_prepare_component import (
    frozen_sparse_preparation_source as frozen_sparse_preparation_source,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def bank_preparation_fixture(
    log: int,
) -> tuple[Any, tuple[ProgramRecord, ProgramRecord], Signature]:
    from qualtran.bloqs.chemistry.sparse.prepare import PrepareSparse

    rows = ((0, 0, 1, 1), (0, 1, 0, 1), (0, 1, 1, 0), (0, 0, 1, 1))
    original = PrepareSparse(
        4,
        4,
        1,
        tuple(tuple(reversed(row)) for row in rows),
        (1, 0, 0, 1),
        (0, 1, 1, 0),
        rows,
        (0, 1, 1, 0),
        (1, 0, 0, 1),
        (0, 1, 1, 0),
        log_block_size=log,
    )
    lookup = original.build_qrom_bloq()
    table = QroamCleanTable(
        (4,),
        (log,),
        tuple(tuple(int(value) for value in data.flat) for data in lookup.data),
    )
    signature = Signature(
        tuple(
            Register(reg.name, reg.dtype, shape=reg.shape)
            for reg in _signature(original.signature)
        )
    )
    pair = sparse_prepare_component_pair(
        signature,
        4,
        qroam_bank_unitary_pair(
            table, tuple(int(width) for width in lookup.target_bitsizes)
        ),
    )
    return original, pair, signature


@pytest.mark.parametrize("log", (0, 1))
def test_arbitrary_bank_complex_state_all_joint_measurement_histories(log: int) -> None:
    _, pair, signature = bank_preparation_fixture(log)
    assert all(not verify_program_record(record) for record in pair)
    builder = ComponentProgramBuilder(signature)
    prepared = builder.call_program(pair[0], builder.inputs)
    restored = builder.call_program(pair[1], prepared)
    bound, logical = bind_used_workspace(builder.finalize(restored))
    work = bound.circuit.n_wires - logical
    initial = {
        0: 0.5 + 0j,
        ((1 << logical) - 1) << work: 0.5j,
        (1 << (logical - 1)) << work: -0.5 + 0j,
        1 << work: -0.5j,
    }
    count = sum(g.name == "measure" for g in bound.circuit.instructions)
    assert count == (6 if log == 0 else 2)
    for history in product((0, 1), repeat=count):
        actual = _branch(
            bound.circuit.instructions,
            bound.circuit.n_wires,
            history,
            initial,
            normalize_measurements=True,
        )
        for key in actual.keys() | initial.keys():
            assert actual.get(key, 0) == pytest.approx(initial.get(key, 0), abs=3e-10)


@pytest.mark.parametrize("log", (0, 1))
@pytest.mark.parametrize("control", (None, 0, 1))
@pytest.mark.parametrize("phase", (1, -1))
def test_retained_signal_reflection_conjugates_selection_zero_projector(
    log: int, control: int | None, phase: int
) -> None:
    original, pair, signature = bank_preparation_fixture(log)
    selection_names = tuple(reg.name for reg in original.selection_registers)
    junk = tuple(reg for reg in signature if reg.name not in selection_names)
    retained = tuple("signal_" + reg.name for reg in junk)
    reflection = prepared_reflection_program(
        *pair,
        selection_names,
        control_value=control,
        signal_junk_names=retained,
        global_phase=phase,
        scalar_phase=(
            scalar_phase_program(pi) if control is None and phase == -1 else None
        ),
    )
    assert not verify_program_record(reflection)
    outer = Signature(
        (
            *(
                (Register("control", quantum_bit_type()),)
                if control is not None
                else ()
            ),
            *signature,
        )
    )
    builder = ComponentProgramBuilder(outer)
    values = dict(builder.inputs)
    ctrl = values.pop("control", None)
    prepared = builder.call_program(pair[0], values)
    reflected = builder.call_program(
        reflection,
        {
            **({"control": ctrl} if ctrl is not None else {}),
            **{name: prepared[name] for name in selection_names},
            **{
                kept: prepared[reg.name]
                for reg, kept in zip(junk, retained, strict=True)
            },
        },
    )
    after = {
        **{name: reflected[name] for name in selection_names},
        **{reg.name: reflected[kept] for reg, kept in zip(junk, retained, strict=True)},
    }
    restored = builder.call_program(pair[1], after)
    if control is not None:
        restored["control"] = reflected["control"]
    bound, logical = bind_used_workspace(builder.finalize(restored))
    work = bound.circuit.n_wires - logical
    selection_width = sum(int(reg.total_bits()) for reg in original.selection_registers)
    bank_width = logical - selection_width - int(control is not None)
    selection_mask = ((1 << selection_width) - 1) << bank_width
    # Conjugating the actual reflection by its PREPARE leaves a known diagonal
    # operator on arbitrary bank words, including coherent control sectors.
    witness = {
        0: 0.5 + 0j,
        (1 << bank_width) - 1: 0.5j,
        1 << bank_width: -0.5 + 0j,
        (1 << (logical - 1)) | 1: -0.5j,
    }
    assert len(witness) == 4
    expected = {
        key
        << work: value
        * (
            -1
            if not key & selection_mask
            and (control is None or (key >> (logical - 1)) == control)
            else 1
        )
        for key, value in witness.items()
    }
    expected = {
        key: value
        * (phase if control is None or (key >> (work + logical - 1)) == control else 1)
        for key, value in expected.items()
    }
    initial = {key << work: value for key, value in witness.items()}
    count = sum(g.name == "measure" for g in bound.circuit.instructions)
    # The blocked relative-phase network is tested separately on its complete
    # small operator. The unblocked composition also admits this support proof.
    if log == 0:
        assert (
            certify_erasure_histories(
                bound.circuit.instructions,
                n_wires=bound.circuit.n_wires,
                zero_wires=tuple(range(logical, bound.circuit.n_wires)),
            )
            == count
        )
    for outcome in (0, 1):
        actual = _branch(
            bound.circuit.instructions,
            bound.circuit.n_wires,
            (outcome,) * count,
            initial,
            normalize_measurements=True,
        )
        for key in actual.keys() | expected.keys():
            assert actual.get(key, 0) == pytest.approx(expected.get(key, 0), abs=3e-10)
