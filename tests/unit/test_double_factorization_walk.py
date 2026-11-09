"""Check the native DF walk and the locked analytical call graph."""

from dataclasses import asdict

import numpy as np
import pytest

from flagquantum.algorithms.primitives.double_factorization_block_encoding import (
    DoubleFactorizationBlockEncoding,
)
from flagquantum.algorithms.primitives.double_factorization_walk import (
    DoubleFactorizationWalk,
)
from flagquantum.compiler._component_trace.resource_comparison import (
    count_emitted_instructions,
)
from flagquantum.compiler.registers.layout.partition import partition_register
from flagquantum.dynamic import DynamicCircuit
from tests.unit.test_sparse_chemistry_prepare import _branch

pytestmark = pytest.mark.unit


def test_df_walk_phase_and_two_step_chebyshev_projection():
    block = DoubleFactorizationBlockEncoding.from_eigenvalues(
        (1.0,), ((2.0,),), ((), ()), outer_bits=1, inner_bits=1, rank_rotation_bits=2
    )
    walk = DoubleFactorizationWalk(block)
    logical = sum(size for _, size in walk.layout)
    work, _ = walk.requirements()
    width = logical + work
    ports = partition_register(tuple(range(logical)), walk.layout)
    workspace = tuple(range(logical, width))
    forward = walk.make_on(ports, workspace=workspace)
    backward = walk.make_on(ports, workspace=workspace, adjoint=True)
    ir = DynamicCircuit(width).append(forward).to_ir()
    counts = count_emitted_instructions(ir)
    events = counts.unconditional["measure"]
    assert (counts.t_count, events, work) == (166, 24, 5)
    assert walk.alpha == 2

    block_ir = (
        DynamicCircuit(width)
        .append(block.make_on(ports, workspace=workspace[: block.requirements()[0]]))
        .to_ir()
    )
    mask = sum(
        1 << (width - 1 - wire)
        for name, _ in block.signal_layout
        for wire in ports[name]
    )
    vector = np.array([1, 1j, -1, -1j]) / 2
    initial = {i << work: amp for i, amp in enumerate(vector)}
    state = _branch(
        block_ir.instructions, width, (0,) * 20, initial, normalize_measurements=True
    )
    reflected = {index: -amp if index & mask else amp for index, amp in state.items()}
    # Each rank-one prepare succeeds with probability 1/2 at two bits.
    z_average = np.array([1.0, 0.0, 0.0, -1.0])
    a = (1 - z_average) / 2
    b = (1 + z_average) / 2
    projected_block = (a + 2 * b**2 - 1) / 2
    expected_square = (2 * projected_block**2 - 1) * vector
    squared = DynamicCircuit(width).append(forward).append(forward).to_ir()
    roundtrip = DynamicCircuit(width).append(forward).append(backward).to_ir()

    def run(circuit, history):
        result = _branch(
            circuit.instructions, width, history, initial, normalize_measurements=True
        )
        assert all(index % (1 << work) == 0 for index in result)
        return result

    for pattern in ((0,), (1,), (0, 1), (1, 0)):
        history = (pattern * events)[:events]
        opposite = tuple(1 - bit for bit in history)
        assert run(ir, history) == pytest.approx(reflected, abs=8e-12)
        result = run(squared, history + opposite)
        projected = [result.get(i << work, 0) for i in range(4)]
        np.testing.assert_allclose(projected, expected_square, atol=8e-12, rtol=0)
        assert run(roundtrip, history + opposite) == pytest.approx(initial, abs=8e-12)


@pytest.mark.ftqc_reference
def test_df_walk_locked_gate_cost_and_missing_selection_binding():
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.df.double_factorization import (
        DoubleFactorizationBlockEncoding as SourceBlock,
    )
    from qualtran.bloqs.qubitization.qubitization_walk_operator import (
        QubitizationWalkOperator,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    block = DoubleFactorizationBlockEncoding.from_eigenvalues(
        (1.0,) * 5,
        ((1.0,) * 5,) * 50,
        ((0,) * 4,) * 255,
        outer_bits=12,
        inner_bits=12,
        basis_rotation_bits=7,
        rank_rotation_bits=7,
    )
    source = QubitizationWalkOperator(
        SourceBlock(
            num_spin_orb=10,
            num_aux=50,
            num_eig=250,
            num_bits_state_prep=12,
            num_bits_rot=7,
            num_bits_rot_aa_inner=7,
        )
    )
    counts = DoubleFactorizationWalk(block).source_gate_cost()
    assert {key: value for key, value in asdict(counts).items() if value} == (
        get_cost_value(source, QECGatesCost()).asdict()
    )
    with pytest.raises(KeyError, match="selection"):
        source.decompose_bloq()
    with pytest.raises(RuntimeError, match="selection"):
        get_cost_value(source, QubitCount())
