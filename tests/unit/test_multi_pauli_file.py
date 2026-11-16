"""Whole-file parity for locked Pauli controls and independent CNOT fanout."""

from collections import Counter
from collections.abc import Iterable
from hashlib import sha1
from importlib.metadata import version
from itertools import product
from pathlib import Path
from typing import Any

import cirq
import numpy as np
import pytest
from sympy import Symbol

from examples.algorithms.multi_pauli_catalog import (
    ControlledPauliExample,
    FanoutCNOTExample,
    Pauli,
)
from flagquantum.compiler._component_trace.gate_budget import legacy_gate_budget
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import FanoutCall
from flagquantum.compiler._component_trace.resource_width import (
    query_resource_logical_width,
)
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.dynamic import DynamicCircuit
from flagquantum.simulation.classical_controlled_bits import fanout_classical_values
from flagquantum.simulation.tensor_network.controlled_tensor import (
    controlled_register_tensor,
)
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_sparse_chemistry_prepare import _branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
PAULIS: tuple[Pauli, ...] = ("x", "y", "z")


@pytest.fixture(scope="module", autouse=True)
def locked_source() -> None:
    import kickmix
    from qualtran.bloqs.mcmt import multi_control_pauli

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    folder = Path(multi_control_pauli.__file__).parent
    for name, digest in (
        (
            "multi_control_multi_target_pauli.ipynb",
            "65993bfff5d7faa7526b3840e7a4033e05324325",
        ),
        ("multi_control_pauli.py", "2953333a1cdd9408b03cdec844b5907af091d5a5"),
        ("multi_control_pauli_test.py", "eed8610ca4c31125b96638a91cd90b58906fe01e"),
        ("multi_target_cnot.py", "c4df15202f0ab9a76e97132560692989fe28ad90"),
        ("multi_target_cnot_test.py", "5525a34f7aab590a899084f77cf30f2b60a16791"),
    ):
        raw = (folder / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest


def _source(pauli: Pauli, values: tuple[int, ...]) -> Any:
    from qualtran.bloqs.basic_gates import XGate, YGate, ZGate
    from qualtran.bloqs.mcmt.multi_control_pauli import MultiControlPauli

    with pytest.warns(DeprecationWarning, match="MultiControlPauli"):
        return MultiControlPauli(values, {"x": XGate, "y": YGate, "z": ZGate}[pauli]())


def _assert_signature(native: Signature, original: Any) -> None:
    assert [(r.name, r.bitsize, r.shape_symbolic, r.side.value) for r in native] == [
        (r.name, r.bitsize, r.shape_symbolic, r.side.value) for r in original.signature
    ]


def _emit_pauli(
    pauli: Pauli, values: tuple[int, ...]
) -> tuple[int, tuple[Instruction, ...]]:
    n = len(values)
    full_width = n + 1 + max(0, n - 1)
    gates = (
        DynamicCircuit(full_width)
        .append(ControlledPauliExample(pauli, values))
        .to_ir()
        .instructions
    )
    return full_width, gates


@pytest.mark.parametrize("pauli", PAULIS)
@pytest.mark.parametrize("cv", (0, 1))
@pytest.mark.parametrize("n", (0, 1, 2, *range(7, 17)))
def test_original_pauli_decomposition_counts_and_gates(
    pauli: Pauli, cv: int, n: int
) -> None:
    from qualtran.bloqs.mcmt import ControlledViaAnd
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    values = (cv,) * n
    source = _source(pauli, values)
    example = ControlledPauliExample(pauli, values)
    _assert_signature(example.signature, source)
    ((child, count),) = source.build_call_graph(None).items()
    assert count == 1
    assert isinstance(child, ControlledViaAnd) if n else child == source.target_bloq
    graph_key = f"ControlledViaAnd({pauli.upper()})" if n else pauli.upper()
    assert example.call_graph().leaf_counts == {graph_key: 1}
    width, actual = _emit_pauli(pauli, values)
    work = tuple(range(n + 1, width))
    inputs = {"target": (n,)} | ({"controls": tuple(range(n))} if n else {})
    expected = reference_arithmetic_gates(
        source,
        inputs,
        work,
        dict(zip(work, range(len(work)), strict=True)),
        expected_outputs=inputs,
        keep_pauli_leaves=True,
    )
    assert_source(actual, expected)
    resources = example.resource_model()
    model = get_cost_value(source, QECGatesCost())
    assert (resources.and_bloq, resources.clifford, resources.measurement) == (
        model.and_bloq,
        model.clifford,
        model.measurement,
    )
    assert resources.logical_qubits == width == get_cost_value(source, QubitCount())
    emitted = Counter(g.name for g in actual)
    assert emitted["measure"] == max(0, n - 1)
    assert emitted["t"] + emitted["tdg"] == 4 * max(0, n - 1)


def test_original_concrete_and_symbolic_x_factories() -> None:
    from qualtran.bloqs.mcmt.multi_control_pauli import _ccpauli, _ccpauli_symb
    from qualtran.resource_counting import QubitCount, get_cost_value

    n = Symbol("n")
    concrete, source = _ccpauli.make(), _ccpauli_symb.make()
    _assert_signature(ControlledPauliExample("x", (1, 0, 1, 0, 1)).signature, concrete)
    symbolic = ControlledPauliExample("x", n)
    _assert_signature(symbolic.signature, source)
    assert symbolic.call_graph().leaf_counts == {"ControlledViaAnd(X)": 1}
    model = symbolic.resource_model()
    assert legacy_gate_budget(model)["t"] == source.t_complexity().t == 4 * n - 4
    assert model.logical_qubits == get_cost_value(source, QubitCount())
    with pytest.raises(ValueError, match="concrete activation"):
        DynamicCircuit(5).append(symbolic)


@pytest.mark.parametrize("pauli", PAULIS)
@pytest.mark.parametrize("cv", (0, 1))
@pytest.mark.parametrize("n", range(10))
def test_original_pauli_matrix_grid(pauli: Pauli, cv: int, n: int) -> None:
    source = _source(pauli, (cv,) * n)
    signature = ControlledPauliExample(pauli, (cv,) * n).signature
    data = cirq.unitary({"x": cirq.X, "y": cirq.Y, "z": cirq.Z}[pauli])
    if n:
        actual = controlled_register_tensor(
            Signature(tuple(signature)[:1]),
            Signature(tuple(signature)[1:]),
            (np.full(n, cv),),
            lambda: data,
        ).reshape(2 << n, 2 << n)
    else:
        actual = data
    np.testing.assert_allclose(actual, source.tensor_contract(), atol=1e-14, rtol=0)


@pytest.mark.parametrize("pauli", PAULIS)
@pytest.mark.parametrize("values", ((), (0,), (1,), (0, 1), (1, 0, 1, 0, 1), (1,) * 9))
def test_append_all_small_measured_branches_and_clean_workspace(
    pauli: Pauli, values: tuple[int, ...]
) -> None:
    n = len(values)
    width, gates = _emit_pauli(pauli, values)
    source = _source(pauli, values)
    live_size = 2 << n
    work = width - n - 1
    histories: Iterable[tuple[int, ...]]
    if n < 6:
        basis = tuple(range(live_size))
        histories = product((0, 1), repeat=work)
    else:
        basis = (0, 1, live_size - 2, live_size - 1)
        histories = ((0,) * work, (1,) * work)
    vector = np.zeros(live_size, dtype=np.complex128)
    for i in basis:
        vector[i] = complex(i + 1, 1 - i) / live_size
    reference = source.tensor_contract().reshape(live_size, live_size) @ vector
    state = {i << work: vector[i] for i in basis}
    expected = {
        i << work: a * 2 ** (-work / 2)
        for i, a in enumerate(reference)
        if abs(a) > 1e-12
    }
    for outcomes in histories:
        assert _branch(gates, width, outcomes, state) == pytest.approx(
            expected, abs=2e-12
        )


@pytest.mark.parametrize(
    "values,target,controls",
    (
        ((0,), 1, (0,)),
        ((1, 0), 0, (1, 0)),
        ((1, 1, 1), 1, (1, 1, 1)),
        ((1, 0, 1, 0), 1, (1, 0, 1, 0)),
        ((1,), 0, (0,)),
        ((), 0, ()),
    ),
)
def test_original_classical_controlled_x_cases(
    values: tuple[int, ...], target: int, controls: tuple[int, ...]
) -> None:
    from qualtran.bloqs.mcmt.multi_control_pauli import MultiControlX

    original = MultiControlX(values)
    outputs = original.call_classically(
        target=target, **({"controls": controls} if controls else {})
    )
    assert not controls or tuple(outputs[0]) == controls
    width, gates = _emit_pauli("x", values)
    work = max(0, len(values) - 1)
    bits = sum(bit << i for i, bit in enumerate(reversed(controls)))
    initial = ((bits << 1) | target) << work
    expected = ((bits << 1) | int(outputs[-1])) << work
    assert _branch(gates, width, (0,) * work, {initial: 1 + 0j}) == pytest.approx(
        {expected: 2 ** (-work / 2)}, abs=2e-12
    )


@pytest.mark.parametrize("n", (1, 2, 3, 4, 5, 6, 8, 10))
def test_original_fanout_layers_and_dirty_targets(n: int) -> None:
    from qualtran.bloqs.mcmt import MultiTargetCNOT
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = MultiTargetCNOT(n)
    example = FanoutCNOTExample(n)
    _assert_signature(example.signature, source)
    assert example.wire_symbols() == tuple(
        source._circuit_diagram_info_(None).wire_symbols
    )
    actual = DynamicCircuit(n + 1).append(example).to_ir().instructions
    expected = reference_arithmetic_gates(
        source,
        {"control": (0,), "targets": tuple(range(1, n + 1))},
        (),
        {},
        cirq_style=True,
    )
    assert actual == expected
    assert len(actual) == 2 * n - 1
    ready = [0] * (n + 1)
    for g in actual:
        c, t = g.wires
        ready[c] = ready[t] = max(ready[c], ready[t]) + 1
    assert max(ready) == 2 * (n - 1).bit_length() + 1
    assert (
        get_cost_value(source, QECGatesCost()).clifford
        == query_qec_gate_cost(FanoutCall(n)).clifford
        == 1
    )
    assert query_qec_gate_cost(FanoutCall(n), legacy_shims=True).clifford == len(actual)
    assert query_resource_logical_width(FanoutCall(n)) == get_cost_value(
        source, QubitCount()
    )
    mask = (1 << n) - 1
    state = {i: complex(i + 1, -i) for i in range(2 << n)}
    branch = _branch(actual, n + 1, (), state)
    assert branch == {i ^ (mask if i >> n else 0): a for i, a in state.items()}
    if n < 5:
        for i in range(2 << n):
            outputs = source.on_classical_vals(control=i >> n, targets=i & mask)
            control, targets = fanout_classical_values(i >> n, i & mask, n)
            assert (control, targets) == (outputs["control"], outputs["targets"])
            assert branch[(outputs["control"] << n) | outputs["targets"]] == state[i]


def test_fanout_source_graph_and_symbolic_factory() -> None:
    from qualtran.bloqs.mcmt.multi_target_cnot import _c_multi_not, _c_multi_not_symb
    from qualtran.resource_counting import QubitCount, get_cost_value
    from qualtran.resource_counting.generalizers import ignore_split_join

    n = Symbol("n")
    for count, original in ((5, _c_multi_not.make()), (n, _c_multi_not_symb.make())):
        native = FanoutCNOTExample(count)
        _assert_signature(native.signature, original)
        _, leaves = original.call_graph(max_depth=1, generalizer=ignore_split_join)
        assert native.call_graph().leaf_counts == {
            type(node).__name__: times for node, times in leaves.items()
        }
        assert query_resource_logical_width(FanoutCall(count)) == get_cost_value(
            original, QubitCount()
        )
    with pytest.raises(ValueError, match="positive concrete"):
        DynamicCircuit(2).append(FanoutCNOTExample(n))
    with pytest.raises(ValueError, match="Symbolic bitsize"):
        FanoutCNOTExample(n).wire_symbols()
    with pytest.raises(ValueError, match="Symbolic bitsize"):
        _c_multi_not_symb.make()._circuit_diagram_info_(None)
