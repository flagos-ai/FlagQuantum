"""Bounded native ArcTan and the locked arithmetic-gate contract."""

from cmath import exp
from collections import Counter
from itertools import product
from math import atan, pi, sqrt

import pytest

from flagquantum.compiler.registers.lookup.arctan_lookup import (
    _arctan_magnitude_words,
    arctan_lookup_ir,
    arctan_lookup_requirements,
)
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.sparse_register_execution import execute_sparse_register

pytestmark = pytest.mark.unit


def layout(n, p):
    work, slots = arctan_lookup_requirements(n, p)
    return (
        tuple(range(n)),
        n,
        tuple(range(n + 1, n + 1 + p)),
        tuple(range(n + 1 + p, n + 1 + p + work)),
        tuple(range(slots)),
    )


@pytest.mark.parametrize("n,p", [(1, 1), (2, 3), (3, 4), (4, 6)])
def test_all_inputs_and_occupied_outputs(n, p):
    bound = layout(n, p)
    gates = arctan_lookup_ir(*bound)
    count = Counter(gate.name for gate in gates)
    assert count["measure"] == count["reset"]
    assert count["t"] + count["tdg"] == 4 * count["measure"]
    assert count["ccx"] == 0

    def bits(word, width):
        return tuple(word >> j & 1 for j in reversed(range(width)))

    before, after = [], []
    for x, sign, y in product(range(1 << n), (0, 1), range(1 << p)):
        encoded = int((2 * atan(x) / pi) * (1 << p))
        clean = (0,) * len(bound[3])
        before.append(bits(x, n) + (sign,) + bits(y, p) + clean)
        after.append(bits(x, n) + (sign ^ bool(x),) + bits(y ^ encoded, p) + clean)
    events = count["measure"]
    assert certified_and_channel_map(gates, tuple(before)) == (tuple(after), events)
    assert certified_and_channel_map(gates, tuple(after)) == (tuple(before), events)


@pytest.mark.ftqc_reference
def test_maximum_supported_table_words_from_literal_source():
    pytest.importorskip("qualtran")
    from qualtran.bloqs.arithmetic.trigonometric.arctan import ArcTan

    gate = ArcTan(12, 32)
    actual = _arctan_magnitude_words(12, 32)
    assert actual == tuple(gate.apply(x, 0, 0)[2] for x in range(4096))
    assert actual[0] == 0
    assert actual[1] == 1 << 31
    # Check that the binary64 margin does not depend on a near-tie.
    assert (
        min(
            abs(2 * atan(x) / pi * 2**32 - round(2 * atan(x) / pi * 2**32))
            for x in range(2, 4096)
        )
        > 2**-15
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n,p", [(1, 2), (2, 3), (3, 3)])
def test_literal_arctan_matrix_and_source_qrom(n, p):
    cirq = pytest.importorskip("cirq")
    np = pytest.importorskip("numpy")
    pytest.importorskip("qualtran")
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.arithmetic.trigonometric.arctan import ArcTan
    from qualtran.bloqs.data_loading.qrom import QROM
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from tests.unit.test_register_subtraction import assert_source
    from tests.unit.test_source_qrom import source_qrom_gates

    source, sign, target, workspace, slots = layout(n, p)
    gates = arctan_lookup_ir(source, sign, target, workspace, slots)
    ref = ArcTan(n, p)
    rows = tuple(ref.apply(x, 0, 0) for x in range(1 << n))
    qrom = QROM(
        (np.array([row[1] for row in rows]), np.array([row[2] for row in rows])),
        (n,),
        (1, p),
        ((), ()),
    )
    assert_source(
        gates,
        source_qrom_gates(
            qrom,
            {"selection": source, "target0_": (sign,), "target1_": target},
            workspace,
            slots,
        ),
    )
    count = Counter(g.name for g in gates)
    qrom_cost = get_cost_value(qrom, QECGatesCost())
    assert qrom_cost.total_t_count() == count["t"] + count["tdg"]
    assert qrom_cost.measurement == count["measure"]
    with pytest.raises(DecomposeNotImplementedError):
        ref.decompose_bloq()

    width = n + 1 + p
    matrix = cirq.unitary(ref)
    dimension = 1 << width
    values = np.array([exp(0.17j * i * i) / sqrt(dimension) for i in range(dimension)])
    expected = matrix @ values
    qubits = width + len(workspace)
    ir = DynamicCircuit(qubits).append(FieldOperation(qubits, gates)).to_ir()
    before = {i << len(workspace): complex(a) for i, a in enumerate(values)}
    after = {i << len(workspace): complex(a) for i, a in enumerate(expected)}
    assert set(execute_sparse_register(ir, before, seed=57)) == set(after)
    for initial, desired in ((before, after), (after, before)):
        observed = execute_sparse_register(ir, initial, seed=37)
        assert (
            max(
                abs(observed.get(i, 0) - desired.get(i, 0))
                for i in observed.keys() | desired.keys()
            )
            < 2e-12
        )


@pytest.mark.parametrize(
    "n,p,match",
    [
        (0, 4, "source bits"),
        (13, 4, "source bits"),
        (True, 4, "source bits"),
        (3, 0, "target bits"),
        (3, 33, "target bits"),
    ],
)
def test_arctan_rejects_excessive_table_formats(n, p, match):
    with pytest.raises(ValueError, match=match):
        arctan_lookup_requirements(n, p)
