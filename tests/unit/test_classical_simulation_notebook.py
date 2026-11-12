"""Every locked classical-simulation cell and native cell executes independently."""

import ast
import builtins
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from examples.algorithms.classical_simulation import addition_state, addition_trace

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ROOT = Path(__file__).resolve().parents[2]
CODE_CELLS = (1, 3, 6, 8, 10, 12, 14, 15, 17, 20, 22, 24)


def execute(
    notebook: dict[str, Any], label: str
) -> tuple[dict[str, Any], dict[int, Any]]:
    namespace: dict[str, Any] = {}
    results: dict[int, Any] = {}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]), filename=f"{label}:{index}")
        # Capture the same final expression a notebook kernel displays. The
        # reference source bytes and numerical statements remain unchanged.
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            statement = tree.body.pop()
            assert isinstance(statement, ast.Expr)
            expression = statement.value
            exec(compile(tree, f"{label}:{index}", "exec"), namespace)
            results[index] = eval(
                compile(ast.Expression(expression), f"{label}:{index}", "eval"),
                namespace,
            )
        else:
            exec(compile(tree, f"{label}:{index}", "exec"), namespace)
            results[index] = None
    assert tuple(results) == CODE_CELLS
    return namespace, results


def test_complete_original_and_native_notebooks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import qualtran
    from matplotlib import pyplot as plt
    from qualtran.drawing import ClassicalSimGraphDrawer
    from qualtran.simulation.classical_sim import (
        format_classical_truth_table,
        get_classical_truth_table,
    )

    raw = (
        Path(qualtran.__file__).parent / "simulation/classical_sim.ipynb"
    ).read_bytes()
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "9f99d7e8d86996c40fddf84c275993571a9ed883"
    )
    original = json.loads(raw)
    native = json.loads(
        (ROOT / "examples/notebooks/classical_simulation.ipynb").read_text()
    )
    for notebook in (original, native):
        assert len(notebook["cells"]) == 25
        assert sum(c["cell_type"] == "markdown" for c in notebook["cells"]) == 13
        assert not any(
            c.get("outputs") or c.get("attachments") for c in notebook["cells"]
        )
        assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    presentation: list[object] = []
    with monkeypatch.context() as patch:
        patch.setattr(
            ClassicalSimGraphDrawer, "get_svg", lambda self: presentation.append(self)
        )
        source, source_results = execute(original, "original-classical")
    assert len(presentation) == 1
    import_function = builtins.__import__

    def without_reference(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"qualtran", "cirq", "quimb"}:
            raise AssertionError(f"native notebook imported reference package {name}")
        return import_function(name, *args, **kwargs)

    plt.close("all")
    try:
        with monkeypatch.context() as patch:
            patch.setattr(builtins, "__import__", without_reference)
            actual, native_results = execute(native, "native-classical")
        assert actual["cnot_result"] == source_results[1] == (1, 1)
        assert actual["add_result"] == source_results[6] == (5, 12)
        np.testing.assert_array_equal(actual["encoded_254"], source_results[22])
        assert actual["x_as_int"] == source["x_as_int"] == 4
        assert actual["x_out_as_int"] == source["x_out_as_int"] == 8
        assert actual["x_out_float"] == source["x_out_float"] == 0.5
        assert actual["add_text"] == format_classical_truth_table(
            *get_classical_truth_table(source["add"])
        )
        assert actual["custom_text"] == format_classical_truth_table(
            *get_classical_truth_table(source["my_cnot"])
        )
        assert actual["exhaustive_rows"] == actual["add_table"][2]
        rng = np.random.RandomState(52)
        for pair, outputs in actual["sampled_pairs"]:
            expected_pair = rng.randint(256), rng.randint(256)
            assert pair == expected_pair
            assert outputs == (pair[0], (pair[0] + pair[1]) % 256)
        assert (
            len(plt.get_fignums()) == 1 and native_results[20] is actual["trace_figure"]
        )
        figure = actual["trace_figure"]
        figure.canvas.draw()
        output = tmp_path / "classical-addition-values.png"
        figure.savefig(output, dpi=110)
        pixels = plt.imread(output)
        assert np.std(pixels[..., :3]) > 0.01
        assert (
            len(figure.axes[0].tables[0].get_celld()) == (len(actual["trace"]) + 1) * 3
        )
    finally:
        plt.close("all")


@pytest.mark.parametrize("width", [1, 2, 3, 8])
def test_actual_decomposition_values_and_source_call_prefix(width: int) -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add
    from qualtran.simulation.classical_sim import ClassicalSimState, _get_in_vals

    values = {"a": (1 << width) - 1, "b": 2 % (1 << width)}
    source = Add(QUInt(width)).decompose_bloq()
    original = ClassicalSimState.from_cbloq(source, values)
    original.simulate()
    source_calls = []
    for instance, _, _ in source.iter_bloqsoqs():
        outputs = {
            r.name: _get_in_vals(instance, r, original.soq_assign)
            for r in instance.bloq.signature.rights()
        }
        source_calls.append((type(instance.bloq).__name__, outputs))
    native = addition_trace(width, values)
    assert len(native) == len(source_calls)
    for (name, actual), (kind, expected) in zip(native, source_calls, strict=True):
        assert kind == (
            "CNOT"
            if name == "add_cx"
            else (
                "And"
                if name in {"add_and", "add_and_adjoint"}
                else "Split" if kind == "Split" else "Join"
            )
        )
        assert actual.keys() == expected.keys()
        for key in actual:
            np.testing.assert_array_equal(actual[key], expected[key])
    assert addition_state(width, values).simulate() == {
        "a": values["a"],
        "b": (values["a"] + values["b"]) % (1 << width),
    }


def test_notebook_add_consumer_exact_gates_complex_branches_and_clean_work() -> None:
    from collections import Counter

    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.compiler.registers.arithmetic.addition_component import (
        addition_component_program,
    )
    from flagquantum.compiler.registers.register_binding import bind_register_program
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates

    record = addition_component_program(2)
    bound = bind_register_program(record, ((0, 1), (2, 3)), clean=(4,))
    source = Add(QUInt(2))
    expected_gates = reference_arithmetic_gates(
        source, {"a": (0, 1), "b": (2, 3)}, (4,), {4: 0}
    )
    for emitted, reference in zip(
        bound.circuit.instructions, expected_gates, strict=True
    ):
        assert (
            emitted.name == reference.name
            and emitted.wires == reference.wires
            and emitted.params == reference.params
        )
    counts = Counter(g.name for g in bound.circuit.instructions)
    assert counts == {
        "h": 3,
        "t": 2,
        "tdg": 2,
        "cx": 9,
        "s": 1,
        "measure": 1,
        "cz": 1,
        "reset": 1,
    }
    model = get_cost_value(source, QECGatesCost())
    assert model.and_bloq == 1 and model.measurement == 1 and model.clifford == 4
    grid = torch.arange(1, 17, dtype=torch.float64)
    amplitudes = torch.complex(grid, 3 * grid - 7)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(64, dtype=torch.complex128)
    initial[torch.arange(16) << 2] = amplitudes
    expected = torch.zeros_like(initial)
    for a in range(4):
        for b in range(4):
            output = addition_state(2, {"a": a, "b": b}).simulate()
            out_a, out_b = output["a"], output["b"]
            assert isinstance(out_a, (int, np.integer))
            assert isinstance(out_b, (int, np.integer))
            address = (int(out_a) << 4) | (int(out_b) << 2)
            expected[address] = amplitudes[(a << 2) | b]
    for outcome in (0, 1):
        actual = forced_branch(initial, bound.circuit.instructions, (outcome,))
        torch.testing.assert_close(actual, expected / 2**0.5, atol=2e-14, rtol=0)
        assert torch.count_nonzero(actual.reshape(16, 4)[:, 1:]) == 0
    # Ordinary append accepts this same measured primitive stream. Trajectory
    # execution owns sampling; the exact branches above retain projection factors.
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    circuit = DynamicCircuit(6, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(6, bound.circuit.instructions))
    result = run_dynamic(circuit, shots=4, seed=19, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(4, -1), expected.expand(4, -1), atol=2e-14, rtol=0
    )
