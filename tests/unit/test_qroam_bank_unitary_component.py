"""Prove the explicit caller-bank extension on dirty entangled bank inputs."""

from itertools import product
from math import sqrt

import pytest
import torch

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.lookup.qroam_bank_unitary_component import (
    qroam_bank_unitary_pair,
)
from flagquantum.compiler.registers.lookup.source_qroam_clean import (
    QroamCleanTable,
    source_qroam_clean_ir,
)
from flagquantum.core.ir.program import verify_program_record
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.measurement_certificate import certify_erasure_histories
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_qroam_clean_component import _ports
from tests.unit.test_qroam_clean_component import pinned_source as pinned_source
from tests.unit.test_qubitization_walk_component import (
    reference_runtime as reference_runtime,
)
from tests.unit.test_source_qroam_clean import source_clean_graph
from tests.unit.test_sparse_chemistry_prepare_component import bind_used_workspace

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "table,widths,controls",
    (
        (QroamCleanTable((4,), (1,), ((1, 0, 1, 1),)), (1,), 0),
        (QroamCleanTable((3,), (0,), ((1, 0, 1),)), (1,), 1),
        (QroamCleanTable((2,), (1,), ((1, 0), (0, 1))), (1, 1), 0),
        (QroamCleanTable((4,), (1,), ((1, 0, 1, 1), (0, 1, 1, 0))), (1, 1), 0),
        (QroamCleanTable((3,), (0,), ((2, 1, 3),)), (2,), 1),
        (QroamCleanTable((2, 2), (0, 0), ((0, 1, 1, 0),)), (1,), 0),
        (QroamCleanTable((2, 2), (1, 1), ((0, 1, 1, 0),)), (1,), 0),
    ),
)
def test_dirty_bank_exact_joint_operator_and_all_small_histories(
    table: QroamCleanTable, widths: tuple[int, ...], controls: int
) -> None:
    np = pytest.importorskip("numpy")
    from qualtran._infra.composite_bloq import BloqError
    from qualtran.bloqs.data_loading.qroam_clean import QROAMClean

    pair = qroam_bank_unitary_pair(table, widths, num_controls=controls)
    assert all(not verify_program_record(record) for record in pair)
    signature = Signature(pair[0].functions[0].registers or ())
    assert pair[0].functions[0].registers == pair[1].functions[0].registers
    forward_bound, forward_logical = bind_used_workspace(pair[0])
    source = QROAMClean.build_from_data(
        *(np.array(data).reshape(table.shape) for data in table.datasets),
        target_bitsizes=widths,
        num_controls=controls,
        log_block_sizes=table.log_block_sizes,
    )
    incoming_words = []
    offset = 0
    for reg in signature:
        for _ in reg.all_idxs():
            width = int(reg.bitsize)
            incoming_words.append(tuple(range(offset, offset + width)))
            offset += width
    inputs = _ports(signature, tuple(incoming_words))
    outputs = _ports(signature, forward_bound.outputs)
    # Extract original allocated-bank gate actions onto explicitly supplied
    # wires. The upstream RIGHT contract itself still assumes zero banks.
    work = tuple(range(offset, forward_bound.circuit.n_wires))
    if all(extent == 1 for extent in table.batch_shape):
        # The unchanged source root drops selection for this full-block case.
        # Keep that failure explicit; compare native extension to the accepted
        # physical batch/routing owner, rather than claiming a source root graph.
        selection_names = [
            reg.name for reg in signature if reg.name.startswith("selection")
        ]
        with pytest.raises(BloqError, match=f"Soquet named `{selection_names[0]}`"):
            source.decompose_bloq()
        reference = source_qroam_clean_ir(
            table,
            inputs.get("control", ()),
            tuple(inputs[name] for name in selection_names),
            tuple(
                (
                    inputs[f"target{i}_"],
                    *(
                        inputs[f"junk_target{i}_"][start : start + width]
                        for start in range(
                            0, len(inputs.get(f"junk_target{i}_", ())), width
                        )
                    ),
                )
                for i, width in enumerate(widths)
                if width
            ),
            work,
            tuple(range(len(work))),
        )
    else:
        reference, _, _ = source_clean_graph(
            source,
            {reg.name: inputs[reg.name] for reg in source.signature.lefts()},
            outputs,
            work,
            tuple(range(len(work))),
        )
    forward_events = sum(
        g.name == "measure" for g in forward_bound.circuit.instructions
    )
    source_events = sum(g.name == "measure" for g in reference)
    assert source_events == forward_events
    builder = ComponentProgramBuilder(signature)
    forward = builder.call_program(pair[0], builder.inputs)
    backward = builder.call_program(pair[1], forward)
    bound, logical = bind_used_workspace(builder.finalize(backward))
    events = certify_erasure_histories(
        bound.circuit.instructions,
        n_wires=bound.circuit.n_wires,
        zero_wires=tuple(range(logical, bound.circuit.n_wires)),
    )
    assert events <= 8
    # Every computational column includes arbitrary bank words. A complex
    # coherent witness also checks their relative phases without discarding work.
    stride = 1 << (bound.circuit.n_wires - logical)
    columns = range(1 << logical)
    for column in columns:
        assert logical == forward_logical
        forward_input = torch.zeros(
            1 << forward_bound.circuit.n_wires, dtype=torch.complex128
        )
        forward_input[column << (forward_bound.circuit.n_wires - logical)] = 1
        expected = forced_branch(forward_input, reference, (0,) * source_events)
        prepared = forced_branch(
            forward_input, forward_bound.circuit.instructions, (0,) * forward_events
        )
        torch.testing.assert_close(prepared, expected, atol=3e-11, rtol=0)
        initial = torch.zeros(1 << bound.circuit.n_wires, dtype=torch.complex128)
        initial[column * stride] = 1
        actual = forced_branch(initial, bound.circuit.instructions, (0,) * events)
        torch.testing.assert_close(
            actual * sqrt(2**events), initial, atol=3e-11, rtol=0
        )
    weights = torch.arange(1, (1 << logical) + 1, dtype=torch.float64).to(
        torch.complex128
    )
    weights[1::2] *= 1j
    weights /= weights.norm()
    initial = torch.zeros(1 << bound.circuit.n_wires, dtype=torch.complex128)
    initial[::stride] = weights
    for history in product((0, 1), repeat=events):
        actual = forced_branch(initial, bound.circuit.instructions, history)
        torch.testing.assert_close(
            actual * sqrt(2**events), initial, atol=3e-11, rtol=0
        )
