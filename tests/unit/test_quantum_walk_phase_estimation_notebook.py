"""Complete quantum-walk notebook costs and its distinct physical wire order."""

import builtins
import json
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from hashlib import sha1
from itertools import product
from math import sqrt
from pathlib import Path
from types import ModuleType
from typing import Any

import cirq
import numpy as np
import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from examples.algorithms.quantum_walk_phase_estimation import (
    get_resource_state,
    phase_estimation,
    phase_estimation_model,
)
from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.resource_model import ComponentCost
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_addition_file_audit import execute
from tests.unit.test_qubitization_phase_estimation import native_qpe
from tests.unit.test_qubitization_phase_estimation import source_walk as source_walk

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module")
def frozen_notebook() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3] / "ftqc-reference-env"
    raw = (
        root / "qualtran/bloqs/phase_estimation/phase_estimation_of_quantum_walk.ipynb"
    ).read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "6937510fa2d03e83463fbd9c1443c77827c3a981"
    )
    loaded = json.loads(raw)
    assert isinstance(loaded, dict)
    return loaded


def test_complete_source_notebook_costs_native_imports_and_flame(
    frozen_notebook: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    import IPython.display

    root = Path(__file__).resolve().parents[2]
    reference = root.parent / "ftqc-reference-env"
    source_path = (
        reference
        / "qualtran/bloqs/phase_estimation/phase_estimation_of_quantum_walk.ipynb"
    )
    script = r"""
import contextlib, io, json, sys
from pathlib import Path
from attrs import asdict
import torch
from IPython.core.interactiveshell import InteractiveShell
import qualtran.drawing as drawing
from qualtran.resource_counting import get_cost_value, QECGatesCost
original = json.loads(Path(sys.argv[1]).read_text())
drawing.show_flame_graph = lambda *args, **kwargs: None
shell = InteractiveShell.instance()
ns = shell.user_ns
costs = {}
with contextlib.redirect_stdout(io.StringIO()):
    for index, cell in enumerate(original["cells"]):
        if cell["cell_type"] == "code":
            result = shell.run_cell("".join(cell["source"]), store_history=False)
            if not result.success:
                raise AssertionError(result.error_in_exec or result.error_before_exec)
            if index in (7, 9, 11, 12):
                value = ns["get_qec_gates_cost_for_circuit"](ns["circuit"]) if index == 7 else ns["result"]
                costs[str(index)] = {k:int(v) for k,v in asdict(value).items() if v}
q = ns["qpe_small"]
gates = get_cost_value(q, QECGatesCost())
actual, converted = q.t_complexity(), gates.to_legacy_t_complexity()
def legacy(cost):
    return dict(t=int(cost.t), rotations=int(cost.rotations), clifford=int(cost.clifford))
print(json.dumps(dict(costs=costs, m=ns["m_bits"], legacy=legacy(actual), converted=legacy(converted),
    resource=ns["get_resource_state"](4).tolist())))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(source_path)],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        text=True,
        capture_output=True,
        check=False,
        timeout=90,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    expected = json.loads(result.stdout)
    notebook = json.loads(
        (root / "examples/notebooks/quantum_walk_phase_estimation.ipynb").read_text()
    )
    assert len(notebook["cells"]) == 16 and len(frozen_notebook["cells"]) == 15
    assert [c["cell_type"] for c in frozen_notebook["cells"]] == [
        c["cell_type"] for c in notebook["cells"][:15]
    ]
    assert [
        c["metadata"]["frozen_source_cell"] for c in notebook["cells"][:15]
    ] == list(range(15))
    monkeypatch.setenv(
        "FLAMEGRAPH_EXECUTABLE",
        str(reference / "qualtran/third_party/flamegraph/flamegraph.pl"),
    )
    displays: list[object] = []
    monkeypatch.setattr(IPython.display, "display", displays.append)
    original_import = builtins.__import__

    def native_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        if name.split(".")[0] in {"qualtran", "cirq", "openfermion"}:
            raise AssertionError("Native notebook must use owned implementations")
        return original_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", native_import)
        native = execute(notebook)
    assert len(displays) == 3
    for cell, key in (
        ("7", "small_ising_cost"),
        ("9", "large_ising_cost"),
        ("11", "large_hubbard_cost"),
        ("12", "full_qpe_cost"),
    ):
        assert native[key] == expected["costs"][cell]
    assert native["m_bits"] == expected["m"] == 21
    np.testing.assert_array_equal(native["resource_state"], expected["resource"])
    assert native["small_qpe_legacy"] == expected["converted"]
    assert expected["legacy"]["t"] == expected["converted"]["t"] == 3542140
    assert (
        expected["legacy"]["rotations"] == expected["converted"]["rotations"] == 131428
    )
    # This discrepancy exists between two untouched upstream APIs. Retain both
    # values rather than modifying the reference or relabeling the conversion.
    assert expected["legacy"]["clifford"] == 9167472
    assert expected["converted"]["clifford"] == 8314766
    assert len(native["flame_rows"]) < 1000
    assert sum(int(row.rsplit("\t", 1)[1]) for row in native["flame_rows"]) == 3542140
    assert b"<svg" in native["flame_svg"]


@pytest.mark.parametrize("width", [1, 2, 3])
def test_first_wire_schedule_full_complex_branches_work_and_inverse(
    source_walk: Any, frozen_notebook: dict[str, Any], width: int
) -> None:
    from qualtran._infra.gate_with_registers import get_named_qubits
    from qualtran.bloqs.basic_gates import Power
    from qualtran.bloqs.qubitization import QubitizationWalkOperator

    namespace: dict[str, Any] = {
        "cirq": cirq,
        "np": np,
        "get_named_qubits": get_named_qubits,
        "QubitizationWalkOperator": QubitizationWalkOperator,
    }
    exec("".join(frozen_notebook["cells"][4]["source"]), namespace)
    stages = tuple(
        cirq.flatten_to_ops(namespace["phase_estimation"](source_walk, width))
    )
    phase = tuple(range(width))
    target = (width, width + 1)
    work = (width + 3,)
    wires: dict[cirq.Qid, int] = {cirq.q(f"m_{i}"): i for i in phase}
    wires.update({cirq.q("selection"): target[0], cirq.q("target"): target[1]})
    # Inspect the original stage operands, including the first phase control.
    assert stages[1].qubits[0] == cirq.q("m_0")
    for index in range(1, width):
        reflection, power, after = stages[2 + 3 * (index - 1) : 5 + 3 * (index - 1)]
        assert reflection.qubits[0] == after.qubits[0] == cirq.q(f"m_{index}")
        assert isinstance(power.gate, Power)
        assert power.gate.power == 2**index
    operation = native_qpe(source_walk, width)
    forward = phase_estimation(
        operation, phase=phase, target=target, workspace=work, classical_bits=(0,)
    )
    inverse = phase_estimation(
        operation,
        phase=phase,
        target=target,
        workspace=work,
        classical_bits=(0,),
        adjoint=True,
    )
    rng = np.random.default_rng(47)
    logical = rng.normal(size=4) + 1j * rng.normal(size=4)
    logical /= np.linalg.norm(logical)
    initial = torch.tensor(
        np.kron(
            np.kron(
                np.kron(get_resource_state(width), logical), [sqrt(0.3), 1j * sqrt(0.7)]
            ),
            [1, 0],
        ),
        dtype=torch.complex128,
    )
    expected = Circuit(width + 4, inputs=initial, dtype=torch.complex128)
    # Source matrices contract the actual original stages on their original
    # operand order. Ideal initialization is supplied explicitly on both sides.
    for stage in stages[1:]:
        gate: Any = stage.gate
        assert gate is not None
        if not hasattr(gate, "tensor_contract"):
            gate = getattr(gate, "bloq", gate)
        matrix = (
            gate.tensor_contract()
            if hasattr(gate, "tensor_contract")
            else cirq.unitary(stage)
        )
        expected.append(
            FieldOperation(
                width + 4,
                (
                    Instruction(
                        "any", tuple(wires[q] for q in stage.qubits), matrix=matrix
                    ),
                ),
            )
        )
    expected_state = expected.state().flatten()
    circuit = DynamicCircuit(width + 4, inputs=initial, dtype=torch.complex128).append(
        forward
    )
    gates = circuit.to_ir().instructions
    assert Counter(g.name for g in gates)["measure"] == 1
    for outcome in (0, 1):
        actual = forced_branch(initial, gates, (outcome,))
        torch.testing.assert_close(actual * sqrt(2), expected_state, atol=3e-12, rtol=0)
        assert torch.count_nonzero(actual.reshape(-1, 2)[:, 1]) == 0
    for first, second in product((0, 1), repeat=2):
        actual = forced_branch(
            initial, (*gates, *inverse.instructions), (first, second)
        )
        torch.testing.assert_close(actual * 2, initial, atol=4e-12, rtol=0)


def test_notebook_helpers_validate_width_and_work_contract() -> None:
    np.testing.assert_array_equal(get_resource_state(0), [1.0])
    with pytest.raises(ValueError, match="nonnegative integer"):
        get_resource_state(-1)
    with pytest.raises(ValueError, match="positive integer"):
        phase_estimation_model((ComponentCost("empty", 1),) * 3, 0)
