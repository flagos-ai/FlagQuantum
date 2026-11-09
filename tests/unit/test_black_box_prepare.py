"""Source-packed PREPARE ports and real alias/reflection consumption."""

from math import sqrt

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.algorithms.primitives.alias_preparation import (
    alias_black_box_prepare,
    alias_preparation_ir,
)
from flagquantum.algorithms.primitives.alias_sampling import (
    alias_sampling_table,
    sparse_alias_sampling_table,
)
from flagquantum.algorithms.primitives.black_box_prepare import BlackBoxPrepare
from flagquantum.compiler.registers.preparation.prepare_reflection import (
    prepare_reflection_ir,
    prepare_reflection_requirements,
)
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_alias_preparation import expected_state, layout

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("sparse", [False, True])
def test_locked_black_box_partition_and_alias_complex_state(sparse):
    pytest.importorskip("qualtran")
    import cirq
    import numpy as np
    from qualtran.bloqs.bookkeeping.auto_partition import AutoPartition
    from qualtran.bloqs.state_preparation.black_box_prepare import (
        BlackBoxPrepare as SourceBlackBox,
    )
    from qualtran.bloqs.state_preparation.state_preparation_alias_sampling import (
        SparseStatePreparationAliasSampling,
        StatePreparationAliasSampling,
    )
    from qualtran.cirq_interop.testing import GateHelper

    if sparse:
        weights = {3: 1, 0: 3}
        table = sparse_alias_sampling_table(weights, 4, precision=0.25)
        child = SparseStatePreparationAliasSampling.from_sparse_dict(
            weights, 4, precision=0.25
        )
    else:
        table = alias_sampling_table((1, 3), precision=1)
        child = StatePreparationAliasSampling.from_probabilities([1, 3], precision=1)
    source = SourceBlackBox(child)
    (stage,) = source.decompose_bloq().bloq_instances
    assert isinstance(stage.bloq, AutoPartition)
    assert stage.bloq.bloq == child

    native = alias_black_box_prepare(table)
    assert native.selection_width == source.selection_bitsize
    assert native.junk_width == source.junk_bitsize
    assert native.normalization == source.l1_norm_of_coeffs
    assert tuple(
        tuple(name for name, _ in ports)
        for ports in (native.selection_layout, native.junk_layout)
    ) == tuple(tuple(parts) for _, parts in stage.bloq.partitions)

    bound, row = layout(table)
    selection, sigma, alt, keep, flag, work, bits = bound
    junk = (*sigma, *row, *alt, *keep, flag)
    gates = native.emit(selection, junk, work, bits)
    inverse = native.emit(selection, junk, work, bits, adjoint=True)
    assert gates == alias_preparation_ir(table, *bound, sparse_index=row)
    assert inverse == alias_preparation_ir(
        table, *bound, sparse_index=row, adjoint=True
    )

    expected = expected_state(table, bound, row)
    helper = GateHelper(source)
    reference = (
        cirq.Simulator(dtype=np.complex128)
        .simulate(helper.circuit, qubit_order=helper.operation.qubits)
        .final_state_vector
    )
    torch.testing.assert_close(
        expected.reshape(len(reference), -1)[:, 0],
        torch.from_numpy(reference),
        atol=3e-12,
        rtol=0,
    )
    initial = torch.zeros_like(expected)
    initial[0] = 1
    for stream, a, b in ((gates, initial, expected), (inverse, expected, initial)):
        events = sum(g.name == "measure" for g in stream)
        for history in ((0,) * events, (1,) * events):
            torch.testing.assert_close(
                forced_branch(a, stream, history),
                b / sqrt(2**events),
                atol=3e-12,
                rtol=0,
            )


def test_bound_alias_preparation_consumed_by_reflection():
    table = alias_sampling_table((1, 3), precision=1)
    prepare = alias_black_box_prepare(table)
    bound, row = layout(table)
    selection, sigma, alt, keep, flag, work, bits = bound
    junk = (*sigma, *row, *alt, *keep, flag)
    forward = prepare.emit(selection, junk, work, bits)
    inverse = prepare.emit(selection, junk, work, bits, adjoint=True)
    reflection_work, reflection_slots = prepare_reflection_requirements(
        len(selection), controlled=False
    )
    size = flag + 1 + len(work) + reflection_work
    reflection = prepare_reflection_ir(
        selection,
        tuple(range(size - reflection_work, size)),
        tuple(range(reflection_slots)),
        prepare=forward,
        unprepare=inverse,
        global_phase=1j,
    )
    circuit = DynamicCircuit(size, dtype=torch.complex128)
    circuit.append(FieldOperation(size, (*forward, *reflection, *inverse)))
    final = run_dynamic(
        circuit, shots=4, seed=4, strategy="trajectory"
    ).final_states.reshape(4, -1)
    expected = torch.zeros_like(final)
    expected[:, 0] = -1j
    torch.testing.assert_close(final, expected, atol=3e-12, rtol=0)


def test_zero_selection_shortcut_and_bad_bindings():
    def unreachable(*args):
        pytest.fail("Zero-selection source bypasses the child")

    empty = BlackBoxPrepare((("empty", 0),), (("junk", 1),), unreachable)
    assert (
        empty.emit((), (5,), (), ()) == empty.emit((), (5,), (), (), adjoint=True) == ()
    )
    with pytest.raises(ValueError, match="direction must be boolean"):
        empty.emit((), (5,), (), (), adjoint=1)
    with pytest.raises(ValueError, match="names must be distinct"):
        BlackBoxPrepare((("dup", 1),), (("dup", 1),), unreachable).emit(
            (0,), (1,), (), ()
        )
    with pytest.raises(ValueError, match="outside the bound"):
        BlackBoxPrepare((("data", 1),), (), lambda *_: (Instruction("x", (4,)),)).emit(
            (0,), (), (), ()
        )
