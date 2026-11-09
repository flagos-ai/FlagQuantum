"""Locked THC resource graph, its source limits, and native uniform amplitudes."""

from itertools import product
from math import sqrt

import pytest

from flagquantum.algorithms.primitives.thc_uniform_preparation import (
    THCUniformPreparation,
)
from flagquantum.compiler._component_trace.resource_comparison import (
    count_emitted_instructions,
)
from flagquantum.compiler.registers.layout.partition import partition_register
from flagquantum.dynamic import DynamicCircuit
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_sparse_chemistry_prepare import _branch

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("m,n", [(2, 4), (4, 6), (4, 8), (6, 6), (10, 4)])
def test_locked_thc_uniform_cost(m, n):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.thc.prepare import UniformSuperpositionTHC

    source = UniformSuperpositionTHC(m, n)
    native = THCUniformPreparation(m, n)
    assert native.layout == tuple(
        (reg.name, reg.total_bits()) for reg in source.signature
    )
    assert len(source.decompose_bloq().bloq_instances) == 35
    assert_source_model(native.source_resource_model(), source)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("m", [1, 3, 7])
def test_locked_one_body_sentinel_overflows_address(m):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.thc.prepare import UniformSuperpositionTHC
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    with pytest.raises(ValueError, match="Too-large classical value"):
        get_cost_value(UniformSuperpositionTHC(m, 2), QECGatesCost())
    with pytest.raises(ValueError, match="sentinel does not fit"):
        THCUniformPreparation(m, 2).source_resource_model()


@pytest.mark.ftqc_reference
def test_locked_graph_has_no_executable_greater_than_and_disjoint_success_controls():
    pytest.importorskip("qualtran")
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.arithmetic.comparison import (
        EqualsAConstant,
        GreaterThanConstant,
        LessThanConstant,
        LessThanEqual,
    )
    from qualtran.bloqs.basic_gates import Toffoli, XGate
    from qualtran.bloqs.bookkeeping import Free
    from qualtran.bloqs.chemistry.thc.prepare import UniformSuperpositionTHC
    from qualtran.bloqs.mcmt import MultiControlX

    source = UniformSuperpositionTHC(2, 4)
    graph = tuple(source.decompose_bloq().iter_bloqsoqs())
    (success,) = [
        inputs for item, inputs, _ in graph if isinstance(item.bloq, MultiControlX)
    ]
    lt, le, combined = (control.binst.bloq for control in success["controls"])
    assert isinstance(lt, LessThanConstant) and lt.less_than_val == 2
    assert isinstance(le, LessThanEqual)
    assert isinstance(combined, Toffoli)
    (predicate_inputs,) = [
        inputs for item, inputs, _ in graph if item == success["controls"][2].binst
    ]
    equality, greater = (control.binst.bloq for control in predicate_inputs["ctrl"])
    assert isinstance(equality, EqualsAConstant) and equality.val == 3
    assert isinstance(greater, GreaterThanConstant)
    with pytest.raises(DecomposeNotImplementedError):
        greater.decompose_bloq()
    with pytest.raises(NotImplementedError, match="GreaterThanConstant"):
        source.tensor_contract()
    assert not any(nu < lt.less_than_val and nu == equality.val for nu in range(4))
    # The last clean Free receives a word whose final bit is explicitly flipped
    # after the paired Toffolis have undone their conjunction.
    (free_inputs,) = [
        inputs for item, inputs, _ in graph if isinstance(item.bloq, Free)
    ]
    (join_inputs,) = [
        inputs for item, inputs, _ in graph if item == free_inputs["reg"].binst
    ]
    terminal = join_inputs["reg"][-1].binst
    assert isinstance(terminal.bloq, XGate)
    (flip_inputs,) = [inputs for item, inputs, _ in graph if item == terminal]
    assert isinstance(flip_inputs["q"].binst.bloq, Toffoli)


@pytest.mark.parametrize("m,n", [(1, 2), (2, 4), (3, 4), (4, 4)])
def test_native_uniform_domain_branches_and_reverse(m, n):
    prepare = THCUniformPreparation(m, n)
    logical = sum(bits for _, bits in prepare.layout)
    work, slots = prepare.requirements()
    width = logical + work + 1  # One spectator wire.
    ports = partition_register(tuple(range(logical)), prepare.layout)
    scratch = tuple(range(logical, logical + work))
    bits = tuple(range(11, 11 + slots))
    forward = prepare.make_on(ports, workspace=scratch, classical_bits=bits)
    inverse = prepare.make_on(
        ports, workspace=scratch, classical_bits=bits, adjoint=True
    )
    ir = DynamicCircuit(width).append(forward).to_ir()
    inverse_ir = DynamicCircuit(width).append(inverse).to_ir()
    counts = count_emitted_instructions(ir)
    events = counts.unconditional["measure"]
    assert events == sum(gate.name == "measure" for gate in inverse_ir.instructions)
    assert counts.unconditional["ry"] == (5 if m == 4 else 3)
    initial = {0: (1 + 1j) / 2, 1: (1 - 1j) / 2}
    domain = [(mu, nu) for nu in range(m) for mu in range(nu + 1)]
    domain.extend((mu, m) for mu in range(n // 2))
    expected = {}
    for (mu, nu), spectator in product(domain, (0, 1)):
        word = (((mu << m.bit_length()) | nu) << 3) | (4 if nu == m else 0) | 2
        index = (word << (work + 1)) | spectator
        expected[index] = initial[spectator] / sqrt(len(domain))
    for pattern in ((0,), (1,), (0, 1), (1, 0)):
        history = (pattern * events)[:events]
        assert _branch(
            ir.instructions, width, history, initial, normalize_measurements=True
        ) == pytest.approx(
            expected,
            abs=1e-12,
            rel=1e-10,
        )
        assert _branch(
            inverse_ir.instructions,
            width,
            history,
            expected,
            normalize_measurements=True,
        ) == pytest.approx(
            initial,
            abs=1e-12,
            rel=1e-10,
        )
