"""Check frozen And teaching definitions and the native state/effect examples."""

from hashlib import sha1
from importlib.metadata import version
from itertools import product
from math import sqrt
from pathlib import Path

import numpy as np
import pytest
import torch

from examples.algorithms.and_notebook import (
    AndLadder,
    and_signature,
    and_source_graph,
    projected_conjunction,
)
from flagquantum.compiler._component_trace.gate_budget import legacy_gate_budget
from flagquantum.compiler.registers.control.control_predicate_resource_model import (
    control_predicate_resource_formula,
)
from flagquantum.drawer.register_score import multi_and_register_score
from flagquantum.drawer.register_score_data import register_score_data
from flagquantum.dynamic import DynamicCircuit
from flagquantum.simulation.classical_and import multi_and_classical_values
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_and_files() -> None:
    import kickmix
    from qualtran.bloqs.mcmt import and_bloq

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    source = Path(and_bloq.__file__).parent
    for name, digest in (
        ("and_bloq.py", "2e6efeb53938ac43f6f0330d79140f04138861eb"),
        ("and_bloq_test.py", "70d853579f933303cbfdd22c09674aec96067fc0"),
        ("and_bloq.ipynb", "54a99b45230d185fe9c791a2789d426d76aecf3b"),
    ):
        raw = (source / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest


@pytest.mark.parametrize("symbolic", [False, True])
def test_signature_directions_and_generalized_graph(symbolic: bool) -> None:
    from qualtran.bloqs.mcmt import And, MultiAnd
    from qualtran.symbolics import HasLength
    from sympy import Symbol

    controls = (1, 0, 1, 0, 1, 0)
    width = Symbol("n", positive=True, integer=True) if symbolic else len(controls)
    source = MultiAnd(HasLength(width) if symbolic else controls)
    keys = {"And": "And", "XGate": "XGate"}
    _, leaves = source.call_graph(max_depth=1)
    assert and_source_graph(width if symbolic else controls).leaf_counts == {
        keys[type(node).__name__]: count for node, count in leaves.items()
    }
    _, and_leaves = And().call_graph(max_depth=1)
    assert and_source_graph((1, 1)).leaf_counts == {
        type(node).__name__: count for node, count in and_leaves.items()
    }
    for original, count, multi in ((And(), 2, False), (source, width, True)):
        for erase in (False, True):
            reference = original.adjoint() if erase else original
            native = and_signature(count, multi=multi, erase=erase)
            assert [
                (r.name, r.bitsize, r.shape_symbolic, r.side.value) for r in native
            ] == [
                (r.name, r.bitsize, r.shape_symbolic, r.side.value)
                for r in reference.signature
            ]


def test_state_effect_columns_and_and_identity_both_branches() -> None:
    from qualtran.bloqs.mcmt import MultiAnd

    values = (1, 1, 1)
    source = MultiAnd(values).tensor_contract().reshape(8, 4, 8)
    for controls in product((0, 1), repeat=3):
        index = sum(value << (2 - i) for i, value in enumerate(controls))
        np.testing.assert_array_equal(
            projected_conjunction(values, controls).numpy(), source[index, :, index]
        )

    incoming = torch.tensor([1, 1j, -2, 1 - 2j], dtype=torch.complex128)
    incoming /= incoming.norm()
    initial = torch.stack((incoming, torch.zeros_like(incoming)), dim=-1).flatten()
    circuit = DynamicCircuit(3, inputs=initial, dtype=initial.dtype)
    circuit.append(AndLadder((1, 1))).append(AndLadder((1, 1), erase=True))
    gates = circuit.to_ir().instructions
    assert (len(gates), sum(gate.name == "measure" for gate in gates)) == (17, 1)
    for result in (0, 1):
        torch.testing.assert_close(
            forced_branch(initial, gates, (result,)),
            initial / sqrt(2),
            atol=2e-12,
            rtol=0,
        )


def test_all_atomic_polarity_symbols() -> None:
    from qualtran.bloqs.mcmt import And
    from qualtran.drawing import get_musical_score_data

    for values in product((0, 1), repeat=2):
        native = register_score_data(multi_and_register_score(values))["soqs"]
        expected = get_musical_score_data(And(*values)).soqs
        assert [
            {key: val for key, val in port.items() if key != "ident"} for port in native
        ] == [
            {key: val for key, val in port.json_dict().items() if key != "ident"}
            for port in expected
        ]


def test_mixed_six_controls_exact_gates_resources_and_all_feedback_histories() -> None:
    from qualtran.bloqs.mcmt import MultiAnd
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    values = (1, 0, 1, 0, 1, 0)
    source = MultiAnd(values)
    controls, work, slots = tuple(range(6)), tuple(range(6, 11)), tuple(range(5))
    ports = {"ctrl": controls, "junk": work[:-1], "target": work[-1:]}
    initial = torch.zeros(2048, dtype=torch.complex128)
    amps = torch.arange(1, 65, dtype=torch.float64)
    state = torch.complex(amps, amps.flip(0))
    initial[::32] = state / state.norm()
    circuit = DynamicCircuit(11, inputs=initial, dtype=initial.dtype)
    circuit.append(AndLadder(values))
    forward = circuit.to_ir().instructions
    circuit.append(AndLadder(values, erase=True))
    gates = circuit.to_ir().instructions
    assert_source(
        forward, reference_arithmetic_gates(source, ports, (), {}, cirq_style=True)
    )
    assert_source(
        gates[len(forward) :],
        reference_arithmetic_gates(
            source.adjoint(),
            ports,
            (),
            dict(zip(work, slots, strict=True)),
            expected_outputs={"ctrl": controls},
        ),
    )
    assert (
        sum(g.name in {"t", "tdg"} for g in forward)
        == get_cost_value(source, QECGatesCost()).total_t_count()
        == 20
    )
    assert (
        sum(g.name == "measure" for g in gates)
        == get_cost_value(source.adjoint(), QECGatesCost()).measurement
        == 5
    )
    assert get_cost_value(source, QubitCount()) == 11
    for history in product((0, 1), repeat=5):
        torch.testing.assert_close(
            forced_branch(initial, gates, history),
            initial / sqrt(32),
            atol=2e-12,
            rtol=0,
        )


def test_locked_random_classical_projection_and_legacy_costs() -> None:
    from qualtran.bloqs.mcmt import And, MultiAnd
    from qualtran.symbolics import HasLength
    from sympy import Symbol

    random = np.random.RandomState(52)
    patterns = random.choice([0, 1], size=(2, 5))
    inputs = random.choice([0, 1], size=(10, 5))
    for pattern, controls in product(patterns, inputs):
        source = MultiAnd(pattern.tolist())
        expected = source.decompose_bloq().call_classically(ctrl=controls)
        native = multi_and_classical_values(
            controls.astype(np.uint8), tuple(pattern.tolist())
        )
        for name, value in zip(("ctrl", "junk", "target"), expected, strict=True):
            np.testing.assert_array_equal(native[name], value)

    pairs = tuple(product((0, 1), repeat=2))
    instances = [(values, And(*values)) for values in pairs]
    instances += [
        ((0,) * 10, MultiAnd((0,) * 10)),
        ((1, 0) * 10, MultiAnd((1, 0) * 10)),
        ((1,) * 10, MultiAnd(HasLength(10))),
    ]
    for values, source in instances:
        for erase in (False, True):
            original = source.adjoint() if erase else source
            formula = control_predicate_resource_formula(
                values, uncompute=erase, legacy_shims=True
            )
            budget = legacy_gate_budget(formula)
            assert budget["t"] == original.t_complexity().t
            assert budget["clifford"] == original.t_complexity().clifford
    n = Symbol("n", positive=True, integer=True)
    assert legacy_gate_budget(control_predicate_resource_formula(n))["t"] == 4 * n - 4
