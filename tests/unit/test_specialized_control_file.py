"""Audit the unchanged specialized-control tutorial and original fixtures."""

import json
from collections import Counter
from hashlib import sha1
from importlib.metadata import version
from itertools import permutations
from pathlib import Path
from typing import Any

import pytest
import torch
from sympy import Symbol

from examples.algorithms.specialized_control_catalog import (
    SpecializedControlExample,
    controlled_adjoint_phase_conjugated_x,
    doubly_controlled_phase_conjugated_x,
)
from flagquantum.compiler._component_trace.program_call_graph import (
    program_component_call_graph,
)
from flagquantum.compiler._component_trace.resource_generalization import RotationCall
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_control import control_component_hierarchy
from flagquantum.compiler.registers.control.controlled_component_resource_model import (
    controlled_pattern_resource_formula,
)
from flagquantum.compiler.registers.control.specialized_component_control import (
    specialized_component_control_system,
)
from flagquantum.compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
)
from flagquantum.core.ir.program import verify_program_record
from flagquantum.core.ir.program.register_types import (
    opaque_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module")
def tutorial(pytestconfig: pytest.Config) -> dict[str, Any]:
    import kickmix
    from qualtran.bloqs.mcmt import specialized_ctrl

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    root = Path(specialized_ctrl.__file__).parent
    for file, digest in (
        ("specialized_ctrl.ipynb", "67c4664b39c2c1043d59312e76d9453c9a294cea"),
        ("specialized_ctrl.py", "98b64995010da5863218dbf93b39bbffbd600a2f"),
        ("specialized_ctrl_test.py", "8ed8dd515d1646ac5d9bcbb9e0332f3060b127df"),
    ):
        data = (root / file).read_bytes()
        assert sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == digest
    pytestconfig.addinivalue_line(
        "markers", "notebook: original Qualtran notebook test"
    )
    source = json.loads((root / "specialized_ctrl.ipynb").read_text())
    namespace: dict[str, Any] = {}
    for i in (1, 6):
        exec(
            compile("".join(source["cells"][i]["source"]), f"locked_cell_{i}", "exec"),
            namespace,
        )
    return namespace


def test_unchanged_double_control_tutorial(tutorial: dict[str, Any]) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    original = tutorial["BloqWithSpecializedCtrl"]().controlled().controlled()
    module = doubly_controlled_phase_conjugated_x()
    record = program_component_call_graph(module)
    assert (
        record.leaf_counts["cx"]
        == record.leaf_counts["t"]
        == record.leaf_counts["tdg"]
        == 1
    )
    ports = {"ctrl": (0, 1), "q": (2,)}
    expected = reference_arithmetic_gates(
        original, ports, (3,), {3: 0}, expected_outputs=ports
    )
    gates = (
        DynamicCircuit(4)
        .append(SpecializedControlExample(module, 2))
        .to_ir()
        .instructions
    )
    assert_source(gates, expected)
    model = get_cost_value(original, QECGatesCost())
    assert (model.t, model.and_bloq, model.clifford, model.measurement) == (2, 1, 2, 1)
    assert get_cost_value(original, QubitCount()) == 4
    count = Counter(g.name for g in gates)
    assert (count["t"] + count["tdg"], count["cx"], count["measure"]) == (6, 7, 1)
    weights = torch.arange(1, 9, dtype=torch.float64)
    data = torch.complex(weights, weights.flip(0))
    data /= data.norm()
    initial = torch.zeros(16, dtype=data.dtype)
    initial[::2] = data
    matrix = torch.as_tensor(original.tensor_contract().reshape(8, 8), dtype=data.dtype)
    target = torch.zeros_like(initial)
    target[::2] = (matrix @ data) * 2**-0.5
    for history in ((0,), (1,)):
        torch.testing.assert_close(
            forced_branch(initial, gates, history), target, rtol=0, atol=2e-12
        )


def test_unchanged_adjoint_specialization_and_fallback_boundary(
    tutorial: dict[str, Any],
) -> None:
    from qualtran import BloqError, Controlled
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    default = tutorial["BloqWithSpecializedCtrl"]().adjoint().controlled()
    assert isinstance(default, Controlled)
    with pytest.raises(BloqError, match="requires a Soquet named `q`"):
        default.call_graph()
    wrapped = tutorial["BloqWithSpecializedCtrlWithAdjoint"]
    selected = wrapped().adjoint().controlled()
    assert selected == wrapped(is_controlled=True).adjoint()
    ports: dict[str, tuple[int, ...]] = {"ctrl": (0,), "q": (1,)}
    expected = reference_arithmetic_gates(
        selected, ports, (), {}, expected_outputs=ports
    )
    model = get_cost_value(selected, QECGatesCost())
    assert (model.t, model.clifford, model.measurement) == (2, 1, 0)
    weights = torch.arange(1, 5, dtype=torch.float64)
    data = torch.complex(weights, weights.flip(0))
    matrix = torch.as_tensor(selected.tensor_contract().reshape(4, 4), dtype=data.dtype)
    for specialize in (False, True):
        program = controlled_adjoint_phase_conjugated_x(specialize=specialize)
        gates = (
            DynamicCircuit(2)
            .append(SpecializedControlExample(program, 1))
            .to_ir()
            .instructions
        )
        if specialize:
            assert gates == expected
        else:
            assert [gate.name for gate in gates] == ["cphase", "cx", "cphase"]
        torch.testing.assert_close(
            forced_branch(data, gates, ()), matrix @ data, rtol=0, atol=2e-12
        )


@pytest.mark.parametrize(
    "groups",
    (
        ((1,),),
        ((0,),),
        ((0,), (0,)),
        ((0,) * 4,),
        ((0, 0), (0, 0)),
        ((1,), (0,) * 5),
        ((1,), (1,), (1,)),
        ((0,) * 4, (1,) * 3, (0,) * 5),
    ),
)
@pytest.mark.parametrize("port", ("ctrl", "control"))
def test_original_atomic_cost_groups(
    groups: tuple[tuple[int, ...], ...], port: str, tutorial: dict[str, Any]
) -> None:
    from qualtran import CtrlSpec
    from qualtran.bloqs.mcmt.specialized_ctrl_test import AtomWithSpecializedControl
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    original = AtomWithSpecializedControl(ctrl_reg_name=port)
    for group in groups:
        original = original.controlled(CtrlSpec(cvs=group))
    values = tuple(bit for group in groups for bit in group)
    child = SymbolicResourceFormula(
        0,
        0,
        0,
        3,
        rotation=Symbol("_c_ctrl_") + Symbol("_c_target_"),
    )
    native = (
        child
        if len(values) == 1
        else controlled_pattern_resource_formula(
            RotationCall("zpow", 1), child, values, target_width=2
        )
    )
    source = get_cost_value(original, QECGatesCost())
    assert (native.and_bloq, native.clifford, native.measurement, native.rotation) == (
        source.and_bloq,
        source.clifford,
        source.measurement,
        source.rotation,
    )
    assert native.logical_qubits == get_cost_value(original, QubitCount())


@pytest.mark.parametrize(
    "control,target", tuple(permutations(("ctrl", "control", "a", "b"), 2))
)
def test_original_named_decomposable_control(
    control: str, target: str, tutorial: dict[str, Any]
) -> None:
    from qualtran.bloqs.mcmt.specialized_ctrl_test import TestBloqWithDecompose

    original = TestBloqWithDecompose(control, target).controlled()
    ports = Signature(
        (
            Register(control, quantum_bit_type()),
            Register(target, opaque_register_type(2)),
        )
    )
    atom = ComponentProgramBuilder(ports, (), name="atom")
    leaf = atom.finalize(dict(atom.inputs))
    positive = ComponentProgramBuilder(ports, leaf.functions, name="packed")
    selected, adder = specialized_component_control_system(
        positive.finalize(positive.call_dict("atom", positive.inputs)),
        control,
        1,
        1,
        entry="packed",
    )
    body = ComponentProgramBuilder(ports, selected.functions)
    values = body.call_dict("atom", body.inputs)
    record = body.finalize(body.call_dict("atom", values))
    controlled = control_component_hierarchy(record, {"atom": adder})
    assert verify_program_record(controlled) == ()
    registers = controlled.functions[0].registers
    assert registers is not None
    assert [(p.name, p.bitsize, p.shape) for p in registers] == [
        (p.name, p.bitsize, p.shape) for p in original.signature
    ]
    graph = program_component_call_graph(
        controlled,
        keep=lambda name: name
        in {
            "__fq_specialized_0_positive",
            "__fq_specialized_0_and",
            "__fq_specialized_0_erase",
        },
    )
    source = original.decompose_bloq().flatten()
    assert Counter(str(inst.bloq) for inst, _, _ in source.iter_bloqsoqs()) == {
        "And": 2,
        "AtomWithSpecializedControl": 2,
        "And†": 2,
    }
    assert graph.leaf_counts == {
        "__fq_specialized_0_and": 2,
        "__fq_specialized_0_positive": 2,
        "__fq_specialized_0_erase": 2,
    }


@pytest.mark.parametrize("values", ((1, 1), (0, 1), (0, 0)))
def test_original_atomic_control_and_adjoint_leaves(
    values: tuple[int, int], tutorial: dict[str, Any]
) -> None:
    from qualtran import CtrlSpec
    from qualtran.bloqs.mcmt.and_bloq import And
    from qualtran.bloqs.mcmt.specialized_ctrl_test import CTestAtom, TestAtom
    from qualtran.resource_counting import GateCounts, QECGatesCost, get_cost_value

    atom = TestAtom("atom")
    selected = atom.controlled()
    assert selected == CTestAtom("atom")
    original = (
        selected.controlled(CtrlSpec(cvs=values[0]))
        if values[1]
        else atom.controlled(CtrlSpec(cvs=values))
    )
    predicate = And(*values)
    child = CTestAtom("atom")
    _, sigma = original.call_graph()
    assert sigma == {predicate: 1, child: 1, predicate.adjoint(): 1}
    if values == (1, 1):
        assert atom.adjoint().controlled() == child.adjoint()
        for inverse in (
            atom.adjoint().controlled(CtrlSpec(cvs=values)),
            child.adjoint().controlled(),
        ):
            _, reversed_sigma = inverse.call_graph()
            assert reversed_sigma == {
                predicate: 1,
                child.adjoint(): 1,
                predicate.adjoint(): 1,
            }

    child_cost = SymbolicResourceFormula(0, 1, 0, 3)
    model = controlled_pattern_resource_formula(
        RotationCall("zpow", 1), child_cost, values, target_width=2
    )
    source = get_cost_value(
        original,
        QECGatesCost(),
        costs_cache={
            child: GateCounts(clifford=1),
        },
    )
    assert (model.and_bloq, model.clifford, model.measurement) == (
        source.and_bloq,
        source.clifford,
        source.measurement,
    )
