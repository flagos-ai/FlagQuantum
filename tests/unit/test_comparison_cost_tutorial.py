"""Check the frozen tutorial's literal circuits and every small measurement branch."""

from __future__ import annotations

import ast
import json
from collections import Counter
from collections.abc import Callable
from functools import partial
from hashlib import sha1
from importlib.metadata import version
from itertools import product
from pathlib import Path
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from examples.algorithms.comparison_cost_tutorial import (
    classical_mixer,
    comparison_tree,
    constant_equality,
    linear_constant_comparison,
    one_bit_comparison,
    register_equality,
    simple_mixer,
    two_bit_classical_comparison,
    two_bit_comparison,
)
from flagquantum.core.ir import Instruction
from tests._reference_dependencies import unavailable_reference
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
SOURCE = Path(__file__).resolve().parents[3] / (
    "ftqc-reference-env/qualtran/bloqs/arithmetic/t_complexity_of_comparison_gates.ipynb"
)


@pytest.fixture(scope="module")
def tutorial() -> dict[str, Any]:
    try:
        import kickmix
        import qualtran
    except ModuleNotFoundError as error:
        unavailable_reference(error)

    assert version(qualtran.__name__) == "0.7.0"
    assert getattr(kickmix, "__version__") == "0.1" and hasattr(kickmix, "Simulator")
    data = SOURCE.read_bytes()
    assert (
        sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        == "9ff6e0d10cc925f687fee6e2edbe0fe26a6b4151"
    )
    cells = json.loads(data)["cells"]
    namespace: dict[str, Any] = {}
    for index in (4, 5, 15, 25, 31, 33, 37, 43, 45, 51):
        code = ast.parse("".join(cells[index]["source"]))
        code.body = [
            node
            for node in code.body
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
        ]
        try:
            exec(compile(code, str(SOURCE), "exec"), namespace)
        except ModuleNotFoundError as error:
            unavailable_reference(error)
    return namespace


def _reference_gates(
    operations: Any, qubits: dict[Any, int], slots: dict[int, int]
) -> tuple[Instruction, ...]:
    try:
        import cirq
    except ModuleNotFoundError as error:
        unavailable_reference(error)

    names = {
        cirq.X: "x",
        cirq.H: "h",
        cirq.CNOT: "cx",
        cirq.CZ: "cz",
        cirq.S: "s",
        cirq.T: "t",
        cirq.T**-1: "tdg",
        cirq.TOFFOLI: "ccx",
        cirq.CSWAP: "cswap",
    }
    records: dict[cirq.MeasurementKey, int] = {}
    gates = []
    for op in cirq.decompose(
        operations,
        keep=lambda op: op.gate in names
        or cirq.is_measurement(op)
        or isinstance(op.gate, cirq.ResetChannel)
        or isinstance(op, cirq.ClassicallyControlledOperation),
    ):
        wires = tuple(qubits[q] for q in op.qubits)
        metadata: dict[str, Any] = {}
        if isinstance(op, cirq.ClassicallyControlledOperation):
            (key,) = cirq.control_keys(op)
            metadata["conditions"] = ((records[key], 1),)
            op = op.without_classical_controls()
        if cirq.is_measurement(op):
            bit = slots[wires[0]]
            records[cirq.measurement_key_obj(op)] = bit
            name = "measure"
            metadata = {"is_dynamic": True, "classical_bit": bit}
        elif isinstance(op.gate, cirq.ResetChannel):
            name = "reset"
            metadata = {"is_dynamic": True}
        else:
            assert op.gate is not None
            name = names[op.gate]
        gates.append(Instruction(name, wires, metadata=metadata))
    return tuple(gates)


@pytest.mark.parametrize(
    "case", ("constant_eq", "register_eq", "one", "two", "tree", "linear")
)
def test_literal_source_operations_and_legacy_estimates(
    tutorial: dict[str, Any], case: str
) -> None:
    try:
        import cirq
    except ModuleNotFoundError as error:
        unavailable_reference(error)

    line = cirq.LineQubit.range(17)
    binding: dict[cirq.Qid, int] = {q: q.x for q in line}
    source_t = None
    extra: dict[str, int]
    if case == "constant_eq":
        operations = tutorial["equality_oracle_quantum_classical"](5, line[:3], line[3])
        extra = {"ancilla0": 4, "ancilla1": 5}
        operation, source_t = constant_equality(), 8
    elif case == "register_eq":
        operations = tutorial["equality_oracle_quantum_quantum"](
            line[:2], line[2:4], line[4]
        )
        extra = {"ancilla0": 5}
        operation, source_t = register_equality(), 4
    elif case == "one":
        # The original function closes over the notebook's global z.
        tutorial["z"] = line[3]
        operations = tutorial["single_qubit_compare"](
            line[0], line[1], line[3], line[2], line[4]
        )
        extra = {"anc0": 5, "anc1": 6}
        operation = one_bit_comparison()
    elif case == "two":
        operations = tutorial["compare_two"](
            line[:2], line[2:4], line[5], line[4], line[6]
        )
        extra = {"xor": 7, "anc0": 8, "anc1": 9}
        operation = two_bit_comparison()
    elif case == "tree":
        operations = tutorial["quantum_quantum_comparator"](
            line[:3], line[3:6], line[7], line[6], line[8]
        )
        extra = {f"anc_{i}": i + 9 for i in range(8)}
        operation, source_t = comparison_tree(), 20
    else:
        operations = tutorial["less_than"](5, line[:3], line[3])
        extra = {f"e_{i}": i + 4 for i in range(4)}
        operation, source_t = linear_constant_comparison(), 12
    binding.update({cirq.NamedQubit(name): wire for name, wire in extra.items()})
    operations = tuple(cirq.flatten_op_tree(operations))
    counts = {
        g.wires[0]: int(g.metadata["classical_bit"])
        for g in operation.instructions
        if g.name == "measure"
    }
    assert_source(operation.instructions, _reference_gates(operations, binding, counts))
    if source_t is not None:
        model = tutorial["t_complexity_compat"](cirq.Circuit(operations))
        actual = Counter(g.name for g in operation.instructions)
        assert actual["t"] + actual["tdg"] == source_t == model.t


@pytest.mark.parametrize("value", range(4))
def test_classical_mixer_and_comparator_source(
    tutorial: dict[str, Any], value: int
) -> None:
    try:
        import cirq
    except ModuleNotFoundError as error:
        unavailable_reference(error)

    line = cirq.LineQubit.range(10)
    binding: dict[cirq.Qid, int] = {q: q.x for q in line}
    names = {"ancilla": 5, "q": 6, "cswap_ancilla": 7, "anc0": 8, "anc1": 9}
    binding.update({cirq.NamedQubit(name): wire for name, wire in names.items()})
    mixer, _ = tutorial["quantum_classical_mixer"](line[:2], value)
    forward, _ = classical_mixer(value, work=(5, 6, 7))
    assert_source(forward, _reference_gates(mixer, binding, {}))
    inputs = tutorial["quantum_classical_comparator_logn"](
        line[:2], value, line[3], line[2], line[4]
    )
    actual = two_bit_classical_comparison(value).instructions
    ops = tuple(cirq.flatten_op_tree(inputs))
    assert_source(actual, _reference_gates(ops, binding, {7: 0, 8: 1}))
    assert tutorial["t_complexity_compat"](cirq.Circuit(ops)).t == 8


def test_simple_mixer_source(tutorial: dict[str, Any]) -> None:
    try:
        import cirq
    except ModuleNotFoundError as error:
        unavailable_reference(error)

    line = cirq.LineQubit.range(4)
    binding: dict[cirq.Qid, int] = {q: q.x for q in line}
    binding[cirq.NamedQubit("anc")] = 4
    ops = tutorial["bi_qubit_mixer"](line[:2], line[2:])
    assert simple_mixer() == _reference_gates(ops, binding, {})


def _flags(a: int, b: int) -> int:
    return 4 * (a < b) + 2 * (a == b) + (a > b)


@pytest.mark.parametrize("case", range(11))
def test_every_measurement_history_preserves_phase_arbitrary_target_and_clean_work(
    case: int,
) -> None:
    cases: tuple[tuple[FieldOperation, int, int, Callable[[int], int]], ...] = (
        (constant_equality(), 3, 1, lambda v: int(v == 5)),
        (register_equality(), 4, 1, lambda v: int(v >> 2 == v & 3)),
        (one_bit_comparison(), 2, 3, lambda v: _flags(v >> 1, v & 1)),
        (two_bit_comparison(), 4, 3, lambda v: _flags(v >> 2, v & 3)),
        (comparison_tree(), 6, 3, lambda v: _flags(v >> 3, v & 7)),
        (linear_constant_comparison(), 3, 1, lambda v: int(v < 5)),
        *(
            (two_bit_classical_comparison(k), 2, 3, partial(_flags, b=k))
            for k in range(4)
        ),
        (comparison_tree(1), 2, 3, lambda v: _flags(v >> 1, v & 1)),
    )
    op, inputs, outputs, predicate = cases[case]
    weights = torch.arange(1 << (inputs + outputs), dtype=torch.float64)
    amplitudes = torch.complex(weights + 1, 3 * weights - 5)
    amplitudes /= amplitudes.norm()
    shift = op.n_qubits - inputs - outputs
    initial = torch.zeros(1 << op.n_qubits, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for i in range(len(weights)):
        initial[i << shift] = amplitudes[i]
        expected[(i ^ predicate(i >> outputs)) << shift] = amplitudes[i]
    events = sum(g.name == "measure" for g in op.instructions)
    for history in product((0, 1), repeat=events):
        torch.testing.assert_close(
            forced_branch(initial, op.instructions, history),
            expected / (2 ** (events / 2)),
            atol=1e-12,
            rtol=0,
        )
