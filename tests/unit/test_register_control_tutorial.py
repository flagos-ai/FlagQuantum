"""Execute every control tutorial cell and compare its complete physical action."""

import builtins
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from hashlib import sha1
from itertools import product
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    GateCost,
    estimate_component_resources,
)
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.compiler.registers.control.controlled_component_resource_model import (
    controlled_component_cost,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    source_control_predicate_leaf,
    walk_source_graph,
)
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_register_wiring_notebook import execute

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
pytest.importorskip("qualtran")
pytest.importorskip("IPython")
pytest.importorskip("graphviz")
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parent / "ftqc-reference-env/qualtran/Controlled.ipynb"
NATIVE = ROOT / "examples/notebooks/register_controls.ipynb"


@pytest.fixture(scope="module")
def original() -> dict[str, Any]:
    data = SOURCE.read_bytes()
    assert sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == (
        "069c6045a1e3c7f971e039bf67ced9e5d3389f9f"
    )
    return execute(SOURCE)


@pytest.fixture(scope="module")
def native() -> dict[str, Any]:
    return execute(NATIVE)


def test_whole_notebook_and_independent_native_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = json.loads(SOURCE.read_text())
    notebook = json.loads(NATIVE.read_text())
    assert len(source["cells"]) == len(notebook["cells"]) == 19
    assert [c["cell_type"] for c in source["cells"]] == [
        c["cell_type"] for c in notebook["cells"]
    ]
    assert [c["metadata"]["frozen_source_cell"] for c in notebook["cells"]] == list(
        range(19)
    )
    original_import = builtins.__import__

    def native_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        if name.split(".")[0] in {"qualtran", "cirq", "kickmix"}:
            raise AssertionError("Native control tutorial must use native owners")
        return original_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", native_import)
        actual = execute(NATIVE)
    assert actual["positive_counts"] == {"cx": 1}
    assert actual["signed_metadata"] == ("TextBox", {"text": "255"})
    assert "range" in actual["signed_refusal"]


def _source_case(kind: str, original: dict[str, Any]) -> Any:
    from qualtran import CtrlSpec, QBit, QUInt
    from qualtran.bloqs.basic_gates import XGate

    if kind == "positive":
        return XGate().controlled()
    if kind == "word":
        return XGate().controlled(CtrlSpec(QUInt(4), 6))
    if kind == "array":
        return XGate().controlled(CtrlSpec(cvs=[0, 1, 1, 0]))
    if kind == "grouped":
        return XGate().controlled(CtrlSpec([QBit(), QBit()], [[0, 1], [1, 0]]))
    return original[{"cserial": "cserial", "catom": "catom", "each": "ccx3"}[kind]]


def _source_gates(
    source: Any, ports: SourcePorts, work: tuple[int, ...], measurements: dict[int, int]
) -> tuple[Instruction, ...]:
    from qualtran import Controlled
    from qualtran.bloqs.basic_gates import CNOT, XGate
    from qualtran.bloqs.for_testing import TestAtom

    def leaf(
        component: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        predicate = source_control_predicate_leaf(
            component, inputs, work, tuple(measurements[wire] for wire in work)
        )
        if predicate is not None:
            return predicate
        if isinstance(component, CNOT):
            return (Instruction("cx", (*inputs["ctrl"], *inputs["target"])),), inputs
        if isinstance(component, XGate):
            return (Instruction("x", inputs["q"]),), inputs
        if isinstance(component, Controlled) and isinstance(
            component.subbloq, TestAtom
        ):
            # Source TestAtom is an atomic X matrix with a synthetic model.
            # Compare the original wrapper tensor independently below.
            return (Instruction("cx", (*inputs["ctrl"], *inputs["q"])),), inputs
        return None

    gates, outputs = walk_source_graph(source, ports, ports, leaf)
    assert outputs == ports
    return gates


@pytest.mark.parametrize(
    "kind", ["positive", "word", "array", "grouped", "cserial", "catom", "each"]
)
def test_exact_gate_stream_every_branch_complex_spectator_and_clean_work(
    kind: str, original: dict[str, Any], native: dict[str, Any]
) -> None:
    source = _source_case(kind, original)
    record: ProgramRecord = native[kind]
    signature = Signature(record.functions[0].registers or ())
    inputs = []
    cursor = 0
    for register in signature:
        for _ in register.all_idxs():
            width = int(register.bitsize)
            inputs.append(tuple(range(cursor, cursor + width)))
            cursor += width
    control_count = cursor - (3 if kind == "each" else 1)
    work = tuple(range(cursor, cursor + control_count - 1))
    bound = bind_register_program(record, tuple(inputs), clean=work)
    assert bound.outputs == tuple(inputs)
    ports: SourcePorts = {}
    offset = 0
    for register in source.signature:
        width = int(register.total_bits())
        ports[register.name] = tuple(range(offset, offset + width))
        offset += width
    assert offset == cursor
    gates = bound.circuit.instructions
    measurements = {
        gate.wires[0]: gate.metadata["classical_bit"]
        for gate in gates
        if gate.name == "measure"
    }
    assert_source(gates, _source_gates(source, ports, work, measurements))
    logical_size = 1 << cursor
    vector = torch.complex(
        torch.arange(1, 2 * logical_size + 1, dtype=torch.float64),
        torch.arange(2 * logical_size, dtype=torch.float64).flip(0),
    )
    vector /= vector.norm()
    matrix = torch.as_tensor(source.tensor_contract().copy(), dtype=torch.complex128)
    expected_logical = (
        matrix.reshape(logical_size, logical_size) @ vector.reshape(logical_size, 2)
    ).flatten()
    width = cursor + len(work) + 1
    initial = torch.zeros(1 << width, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    initial.reshape(logical_size, 1 << len(work), 2)[:, 0, :] = vector.reshape(
        logical_size, 2
    )
    expected.reshape(logical_size, 1 << len(work), 2)[:, 0, :] = (
        expected_logical.reshape(logical_size, 2)
    )
    assert len(measurements) == len(work)
    for history in product((0, 1), repeat=len(work)):
        actual = forced_branch(initial, gates, history)
        torch.testing.assert_close(
            actual, expected * 2 ** (-len(work) / 2), rtol=0, atol=3e-14
        )
    if not work:
        circuit = Circuit(width, inputs=initial, dtype=torch.complex128)
        circuit.append(FieldOperation(width, gates))
        torch.testing.assert_close(
            circuit.state().flatten(), expected, rtol=0, atol=3e-14
        )
    else:
        from flagquantum.dynamic import DynamicCircuit
        from flagquantum.runtime.dynamic.execution import run_dynamic

        dynamic = DynamicCircuit(width, inputs=initial, dtype=torch.complex128)
        dynamic.append(FieldOperation(width, gates))
        result = run_dynamic(dynamic, shots=2, seed=29, strategy="trajectory")
        torch.testing.assert_close(
            result.final_states.reshape(2, -1),
            expected.expand(2, -1),
            rtol=0,
            atol=3e-14,
        )
    counts = Counter(gate.name for gate in gates)
    assert counts["measure"] == len(work)
    assert counts["t"] + counts["tdg"] == 4 * len(work)


@pytest.mark.parametrize("kind", ["positive", "word", "array", "grouped", "each"])
def test_declared_model_is_separate_from_emitted_gates(
    kind: str, original: dict[str, Any]
) -> None:
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source = _source_case(kind, original)
    count, targets = (
        (2, 3) if kind == "each" else (4, 1) if kind != "positive" else (1, 1)
    )
    bits = (1, 1) if kind == "each" else (0, 1, 1, 0) if count == 4 else (1,)
    positive = ComponentCost(
        "controlled child", targets + 1, GateCost(clifford=targets)
    )
    model = controlled_component_cost(
        positive, (ControlRegister(1, bits),), target_width=targets
    )
    actual = estimate_component_resources(model).gates
    expected = get_cost_value(source, QECGatesCost())
    assert asdict(actual) == {key: getattr(expected, key) for key in asdict(actual)}


def test_serial_default_wrappers_keep_atom_identity_and_atomic_cost_refusal(
    original: dict[str, Any], native: dict[str, Any]
) -> None:
    from qualtran import DecomposeTypeError
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    record: ProgramRecord = native["cserial"]
    positive = next(
        function for function in record.functions if function.name.endswith("positive")
    )
    calls = positive.blocks[0].operations
    assert [call.callee for call in calls] == [f"controlled_atom{i}" for i in range(3)]
    instances = sorted(
        original["cserial"].decompose_bloq().bloq_instances,
        key=lambda instance: instance.i,
    )
    assert [instance.bloq.subbloq.tag for instance in instances] == [
        f"atom{i}" for i in range(3)
    ]
    for kind in ("catom", "cserial"):
        with pytest.raises(DecomposeTypeError):
            get_cost_value(original[kind], QECGatesCost())
