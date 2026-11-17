"""Frozen uniform preparation factories, source grid and append behavior."""

from collections import Counter
from dataclasses import fields
from hashlib import sha1
from importlib.metadata import version
from math import acos, sin, sqrt
from pathlib import Path
from typing import Any

import pytest
from sympy import Symbol

from examples.algorithms.uniform_preparation_catalog import UniformPreparationExample
from flagquantum.compiler._component_trace.resource_model import GateCost
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.dynamic import DynamicCircuit
from tests.unit.sparse_register_execution import execute_sparse_register

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module")
def source_module() -> Any:
    import kickmix
    from qualtran.bloqs.state_preparation import prepare_uniform_superposition

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    folder = Path(prepare_uniform_superposition.__file__).parent
    for name, digest in (
        ("__init__.py", "024a57c221ae73f95645dbaa4247f423b037a932"),
        (
            "prepare_uniform_superposition.ipynb",
            "90f6b5c08b7834551e2a3a6d091a66f43dbdaff4",
        ),
        (
            "prepare_uniform_superposition.py",
            "e567c913bff5777e899f38afae09e4c861c98d0a",
        ),
        (
            "prepare_uniform_superposition_test.py",
            "25f19b8539880ef9f0ce3ebe954cd885562780d4",
        ),
    ):
        raw = (folder / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest
    return prepare_uniform_superposition


def child_key(child: Any) -> str:
    kind = type(child).__name__
    if kind == "Adjoint":
        return child_key(child.subbloq) + "†"
    if kind == "LessThanConstant":
        return f"{kind}({child.bitsize}, {child.less_than_val})"
    if kind == "Rz":
        return f"{kind}({child.angle}, {child.eps})"
    if kind == "MultiAnd":
        if type(child.cvs).__name__ == "HasLength":
            return f"{kind}(HasLength({child.cvs.n}))"
        return f"{kind}{tuple(child.cvs)}"
    assert kind in {"Allocate", "Free", "Hadamard", "CHadamard"}
    return kind


def assert_resource_model(op: UniformPreparationExample, source: Any) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    expected = get_cost_value(source, QECGatesCost())
    actual = op.resource_model()
    assert actual.logical_qubits == get_cost_value(source, QubitCount())
    for field in fields(GateCost):
        assert getattr(actual, field.name) == getattr(expected, field.name)


@pytest.mark.parametrize("kind", ["plain", "controlled", "symbolic"])
def test_original_factories(kind: str, source_module: Any) -> None:
    from qualtran.resource_counting.generalizers import ignore_split_join

    if kind == "plain":
        op, source = UniformPreparationExample(), source_module._prep_uniform.make()
    elif kind == "controlled":
        op, source = (
            UniformPreparationExample(controls=(1,)),
            source_module._c_prep_uniform.make(),
        )
    else:
        op = UniformPreparationExample(Symbol("n", positive=True, integer=True))
        source = source_module._prep_uniform_symb.make()
    assert [(r.name, r.dtype.num_qubits, r.side.name) for r in op.signature] == [
        (r.name, r.bitsize, r.side.name) for r in source.signature
    ]
    assert op.call_graph().leaf_counts == {
        child_key(k): n
        for k, n in source.bloq_counts(generalizer=ignore_split_join).items()
    }
    assert_resource_model(op, source)
    if kind == "symbolic":
        with pytest.raises(ValueError, match="Bind a concrete"):
            DynamicCircuit(8).append(op)
    else:
        circuit = DynamicCircuit(op.qubit_count)
        circuit.append(op)
        emitted = Counter(g.name for g in circuit.to_ir().instructions)
        expected = (9, 36, 70, 9) if kind == "controlled" else (8, 32, 64, 8)
        assert (
            op.qubit_count,
            emitted["t"] + emitted["tdg"],
            emitted["cx"],
            emitted["measure"],
        ) == expected


def test_original_wire_labels_and_control_identity(source_module: Any) -> None:
    import cirq

    op = UniformPreparationExample(5, (1, 0))
    source = source_module.PrepareUniformSuperposition(5, cvs=[1, 0])
    assert op.diagram_symbols() == cirq.circuit_diagram_info(source).wire_symbols
    assert source == source_module.PrepareUniformSuperposition(5, cvs=(1, 0))
    assert op != UniformPreparationExample(5, (0, 1))
    assert UniformPreparationExample() == UniformPreparationExample(5, ())


@pytest.mark.parametrize("count", [*range(3, 20), 25, 41])
@pytest.mark.parametrize("values", [(), (1,)])
def test_original_state_grid(
    count: int, values: tuple[int, ...], source_module: Any
) -> None:
    import cirq

    op = UniformPreparationExample(count, values)
    circuit = DynamicCircuit(op.qubit_count)
    circuit.append(op)
    ir = circuit.to_ir()
    # The sparse test interpreter uses the same exact CH matrix for this opcode.
    controlled_h = cirq.unitary(cirq.H.controlled()).tolist()
    gates = tuple(
        Instruction("any", g.wires, matrix=controlled_h) if g.name == "ch" else g
        for g in ir.instructions
    )
    width = op.qubit_count
    clean = 1 << (width - 1) if values else 0
    inactive = {0: 1j / sqrt(2)} if values else {}
    branch = 1 / sqrt(2) if values else 1
    initial = inactive | {clean: complex(branch)}
    odd = count // (count & -count)
    prefix = (odd - 1).bit_length()
    phase = 1 + 0j
    if prefix:
        probability = odd / (1 << prefix)
        angle = acos(1 - 1 / (2 * probability))
        phase = sqrt(probability) * complex(1 / (2 * probability), sin(angle))
    work = width - len(values) - (count - 1).bit_length()
    expected = inactive | {
        clean | i << work: branch * phase / sqrt(count) for i in range(count)
    }
    actual = execute_sparse_register(CircuitIR(width, gates), dict(initial), seed=17)
    assert all(
        abs(actual.get(i, 0) - expected.get(i, 0)) < 3e-12
        for i in actual.keys() | expected.keys()
    )

    source = source_module.PrepareUniformSuperposition(count, cvs=values)
    assert_resource_model(op, source)


@pytest.mark.parametrize("count", range(3, 41, 3))
def test_original_t_complexity_bounds(count: int, source_module: Any) -> None:
    from flagquantum.compiler._component_trace.gate_budget import legacy_gate_budget
    from flagquantum.compiler.registers.preparation.uniform_preparation_resource_model import (
        source_uniform_preparation_resource_formula,
    )

    for values in ((), (1,)):
        model = source_uniform_preparation_resource_formula(
            count, control_values=values, legacy_shims=True
        )
        budget = legacy_gate_budget(model)
        width = (count - 1).bit_length()
        assert budget["t"] <= 12 * width
        assert budget["rotations"] <= (2 + 2 * (width + 1) if values else 2)
        assert source_module.PrepareUniformSuperposition(count, cvs=values).n == count


@pytest.mark.parametrize("count", [10, 18, Symbol("n", positive=True, integer=True)])
def test_abstract_control_width_preserves_source_dispatch(
    count: Any, source_module: Any
) -> None:
    from qualtran.resource_counting import (
        QECGatesCost,
        SympySymbolAllocator,
        get_cost_value,
    )
    from qualtran.symbolics import HasLength

    from flagquantum.compiler.registers.preparation.uniform_preparation_resource_model import (
        source_uniform_preparation_resource_formula,
    )

    c = Symbol("c", positive=True, integer=True)
    source = source_module.PrepareUniformSuperposition(count, HasLength(c))
    native = UniformPreparationExample(count, abstract_control_width=c)
    assert [(r.name, r.bitsize, r.side.name) for r in native.signature] == [
        (r.name, r.bitsize, r.side.name) for r in source.signature
    ]
    assert native.call_graph().leaf_counts == {
        child_key(key): multiplicity
        for key, multiplicity in source.build_call_graph(SympySymbolAllocator()).items()
    }
    assert_resource_model(native, source)
    legacy = source_uniform_preparation_resource_formula(
        count, abstract_control_width=c, legacy_shims=True
    )
    expected = get_cost_value(source, QECGatesCost(legacy_shims=True))
    for field in fields(GateCost):
        assert getattr(legacy, field.name) == getattr(expected, field.name)
    circuit = DynamicCircuit(8)
    with pytest.raises(ValueError, match="Bind a concrete"):
        circuit.append(native)
    assert circuit.to_ir().instructions == ()


@pytest.mark.parametrize("count", [1, 6, 12, 16])
def test_short_factored_predicate_retains_source_failure(
    count: int, source_module: Any
) -> None:
    from qualtran.resource_counting import SympySymbolAllocator
    from qualtran.symbolics import HasLength

    c = Symbol("c", positive=True, integer=True)
    source = source_module.PrepareUniformSuperposition(count, HasLength(c))
    native = UniformPreparationExample(count, abstract_control_width=c)
    with pytest.raises(ValueError, match="at least 3 control values"):
        source.build_call_graph(SympySymbolAllocator())
    with pytest.raises(ValueError, match="at least 3 MultiAnd controls|at least two"):
        native.call_graph()
    with pytest.raises(ValueError, match="at least 3 MultiAnd controls|at least two"):
        native.resource_model()


def test_literal_control_length_resolves_before_append(source_module: Any) -> None:
    from qualtran.symbolics import HasLength

    source = source_module.PrepareUniformSuperposition(5, HasLength(1))
    native = UniformPreparationExample(5, abstract_control_width=1)
    assert native.controls == source.concrete_cvs == (1,)
    assert_resource_model(native, source)
    circuit = DynamicCircuit(native.qubit_count)
    circuit.append(native)
    expected = DynamicCircuit(native.qubit_count)
    expected.append(UniformPreparationExample(5, controls=(1,)))
    assert circuit.to_ir().instructions == expected.to_ir().instructions
