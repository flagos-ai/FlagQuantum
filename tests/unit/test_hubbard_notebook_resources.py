"""Check the concrete Hubbard notebook graph and gate queries."""

from collections import Counter
from dataclasses import asdict

import pytest

from examples.algorithms.hubbard_qubitization_resources import (
    notebook_consumer_signature,
    notebook_controlled_select_score,
    notebook_models,
    notebook_signature,
)
from flagquantum.algorithms.primitives.hubbard_qubitization_resource_model import (
    source_hubbard_majorana_formula,
    source_hubbard_spin_up_formula,
)
from flagquantum.compiler._component_trace.call_graph import modeled_call_graph
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.qec.resources.estimation import estimate_symbolic_ftqc_resources

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _assert_signature(native, source):
    assert [
        (port.name, port.bitsize, port.side.value, port.shape, port.dtype)
        for port in source
    ] == [
        (
            port.name,
            port.bitsize,
            port.side.value,
            port.shape,
            _source_dtype(port),
        )
        for port in native
    ]


def _source_dtype(port):
    from qualtran import BQUInt, QAny, QBit

    if port.dtype.encoding == "bit":
        return QBit()
    if port.dtype.encoding == "uint":
        return BQUInt(port.bitsize, dict(port.dtype.parameters)["bound"])
    assert port.dtype.encoding == "opaque"
    return QAny(port.bitsize)


def test_hubbard_notebook_models_match_locked_child_costs_and_multiplicities():
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.hubbard_model.qubitization import (
        PrepareHubbard,
        SelectHubbard,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from qualtran.resource_counting.generalizers import ignore_split_join

    source_models = (
        (4, SelectHubbard(4, 4)),
        (4, PrepareHubbard(4, 4, t=1.0, u=4.0)),
        (11, SelectHubbard(11, 11, control_val=1)),
    )
    for actual, source in zip(
        (
            notebook_signature(4),
            notebook_signature(4, prepare=True),
            notebook_signature(11, controlled=True),
        ),
        source_models,
        strict=True,
    ):
        _assert_signature(actual, source[1].signature)
    for (side, model), (source_side, original) in zip(
        notebook_models(), source_models, strict=True
    ):
        assert side == source_side
        graph = modeled_call_graph(model, max_depth=1)
        _, expected = original.call_graph(max_depth=1, generalizer=ignore_split_join)
        assert Counter(
            (
                node.name.partition("(")[0],
                tuple(
                    sorted(
                        (key, value)
                        for key, value in asdict(
                            estimate_component_resources(node).gates
                        ).items()
                        if value
                    )
                ),
                count,
            )
            for node, count in graph.leaf_counts.items()
        ) == Counter(
            (
                type(node).__name__,
                tuple(sorted(get_cost_value(node, QECGatesCost()).asdict().items())),
                count,
            )
            for node, count in expected.items()
        )
        actual = estimate_component_resources(model)
        assert {key: value for key, value in asdict(actual.gates).items() if value} == (
            get_cost_value(original, QECGatesCost()).asdict()
        )
        assert actual.peak_logical_qubits == get_cost_value(original, QubitCount())


@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("majorana", [False, True])
def test_hubbard_symbolic_leaf_costs_and_concrete_width(controlled, majorana):
    pytest.importorskip("qualtran")
    sympy = pytest.importorskip("sympy")
    from qualtran.bloqs.chemistry.hubbard_model.qubitization import (
        HubbardMajorannaOperator,
        HubbardSpinUpZ,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    constructor = HubbardMajorannaOperator if majorana else HubbardSpinUpZ
    formula = (
        source_hubbard_majorana_formula if majorana else source_hubbard_spin_up_formula
    )
    control_value = 1 if controlled else None
    n = sympy.Symbol("n")
    symbolic = estimate_symbolic_ftqc_resources(formula(n, controlled=controlled))
    source = constructor(n, n, control_val=control_value)
    _assert_signature(
        notebook_consumer_signature(n, majorana=majorana, controlled=controlled),
        source.signature,
    )
    source_counts = get_cost_value(source, QECGatesCost()).asdict()
    gate_fields = (
        "and_bloq",
        "clifford",
        "measurement",
        "t",
        "toffoli",
        "cswap",
        "rotation",
    )
    for field in gate_fields:
        assert (
            sympy.simplify(
                getattr(symbolic.logical, field) - source_counts.get(field, 0)
            )
            == 0
        )
    assert symbolic.logical.logical_qubits == get_cost_value(source, QubitCount())
    assert sympy.simplify(symbolic.t_equivalents - 4 * source_counts["and_bloq"]) == 0

    for side in (2, 3, 5, 11):
        actual = formula(side, controlled=controlled)
        concrete_source = constructor(side, side, control_val=control_value)
        _assert_signature(
            notebook_consumer_signature(side, majorana=majorana, controlled=controlled),
            concrete_source.signature,
        )
        expected = get_cost_value(concrete_source, QECGatesCost()).asdict()
        for field in gate_fields:
            assert getattr(actual, field) == expected.get(field, 0)
            assert sympy.sympify(getattr(symbolic.logical, field)).subs(
                n, side
            ) == getattr(actual, field)
        assert actual.logical_qubits == get_cost_value(concrete_source, QubitCount())
        assert actual.logical_qubits > symbolic.logical.logical_qubits.subs(n, side)


def test_large_hubbard_select_score_matches_source_operands():
    pytest.importorskip("qualtran")
    import cirq
    from qualtran._infra.gate_with_registers import get_named_qubits
    from qualtran.bloqs.chemistry.hubbard_model.qubitization import SelectHubbard
    from qualtran.cirq_interop import BloqAsCirqGate

    source = SelectHubbard(11, 11, control_val=1)
    registers = get_named_qubits(source.signature)
    names = {wire: name for name, group in registers.items() for wire in group.flat}
    ops = cirq.flatten_op_tree(
        source.decompose_from_registers(context=None, **registers)
    )
    columns = sorted(
        notebook_controlled_select_score().columns, key=lambda column: column.ordinal
    )
    for column, operation in zip(columns, ops, strict=True):
        touched = {names[wire] for wire in operation.qubits}
        assert set(column.step.inputs) == set(column.step.outputs) == touched
        gate = operation.gate
        if isinstance(gate, BloqAsCirqGate):
            name = type(gate.bloq).__name__
            if name == "HubbardMajorannaOperator":
                name += f"({gate.bloq.gate})"
        elif gate == cirq.S**-1:
            name = "SGate"
        elif gate == cirq.CZ:
            name = "CZ"
        else:
            name = type(gate).__name__
        assert column.step.name == name


@pytest.mark.parametrize("side", [2, 3, 10, 11])
@pytest.mark.parametrize("controlled", [False, True])
def test_source_majorana_count_gap_is_the_controlled_y_leaf(side, controlled):
    pytest.importorskip("qualtran")
    import cirq
    from qualtran.bloqs.chemistry.hubbard_model.qubitization import (
        HubbardMajorannaOperator,
    )
    from qualtran.cirq_interop import CirqGateAsBloq
    from qualtran.resource_counting.generalizers import (
        generalize_cvs,
        ignore_alloc_free,
        ignore_cliffords,
        ignore_split_join,
    )

    source = HubbardMajorannaOperator(side, side, control_val=1 if controlled else None)
    generalizer = [
        ignore_cliffords,
        ignore_alloc_free,
        ignore_split_join,
        generalize_cvs,
    ]
    declared = Counter(source.bloq_counts(generalizer=generalizer))
    decomposed = Counter(source.decompose_bloq().bloq_counts(generalizer=generalizer))
    controlled_y = CirqGateAsBloq(cirq.Y.controlled())
    assert cirq.has_stabilizer_effect(controlled_y.gate)
    assert type(controlled_y.gate).__name__ in ("ControlledGate", "CYPowGate")
    # Cirq 1.6 represents CY with a wrapper recognized by the frozen classifier.
    # Cirq 1.7 introduces CYPowGate, which that classifier does not recognize.
    residual = (
        Counter()
        if isinstance(controlled_y.gate, cirq.ControlledGate)
        else Counter({controlled_y: 2 * side**2})
    )
    assert decomposed == declared + residual
