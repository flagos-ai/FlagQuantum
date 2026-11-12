"""Frozen MBUC notebook, numerical phases and physical exact branch acceptance."""

import builtins
import hashlib
import json
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from examples.algorithms.measurement_uncomputation import (
    EraseAnd,
    checked_and_adjoint,
    phase_trace,
    seeded_trials,
    uncomputation_program,
    uncomputation_state,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ROOT = Path(__file__).resolve().parents[2]
CODE_CELLS = (2, 4, 6, 8, 11, 13, 16, 18, 20, 21)


def execute(notebook: dict[str, Any]) -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    observed = []
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), f"MBUC:{i}", "exec"), namespace)
            observed.append(i)
    assert tuple(observed) == CODE_CELLS
    return namespace


@pytest.fixture
def reference(monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], dict[str, Any]]:
    import qualtran
    import qualtran.drawing

    raw = (Path(qualtran.__file__).parent / "simulation/MBUC.ipynb").read_bytes()
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "631fa11a4a9392b8874e1f5956446e7b1b4c2e66"
    )
    notebook = json.loads(raw)
    drawings: list[object] = []
    with monkeypatch.context() as patch:
        patch.setattr(
            qualtran.drawing, "show_bloq", lambda *args, **kwargs: drawings.append(args)
        )
        namespace = execute(notebook)
    assert len(drawings) == 2
    return namespace, notebook


def test_complete_original_and_native_asset(
    reference: tuple[dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    original = builtins.__import__

    def native_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"qualtran", "cirq", "quimb"}:
            raise AssertionError(f"Native workflow imported reference package {name}")
        return original(name, *args, **kwargs)

    notebook = json.loads(
        (ROOT / "examples/notebooks/measurement_uncomputation.ipynb").read_text()
    )
    assert len(notebook["cells"]) == 23
    assert sum(c["cell_type"] == "markdown" for c in notebook["cells"]) == 13
    assert len({c["id"] for c in notebook["cells"]}) == 23
    for cell in notebook["cells"]:
        assert (
            cell["metadata"] == {}
            and not cell.get("outputs")
            and not cell.get("attachments")
        )
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
    plt.close("all")
    try:
        with monkeypatch.context() as patch:
            patch.setattr(builtins, "__import__", native_import)
            native = execute(notebook)
        assert native["measurement_ids"] == (reference[0]["meas_binst_i"],) == (0,)
        assert native["corrected_phases"] == (1,) * 10
        assert native["naive_phases"] == (1, -1, -1, 1, -1, 1, 1, 1, 1, 1)
        assert len(native["exhaustive_rows"]) == len(native["debug_rows"]) == 8
        assert native["debug_trace"] == (-1, 1, 1)
        assert len(plt.get_fignums()) == 2
        corrected_axes = native["corrected_figure"].axes[0]
        texts = [
            cell.get_text().get_text()
            for table in corrected_axes.tables
            for cell in table.get_celld().values()
        ]
        assert "c0=1" in texts and "writes c0" in texts
        assert texts.index("measure") < texts.index("cz") < texts.index("reset")
        for i, number in enumerate(plt.get_fignums()):
            fig = plt.figure(number)
            fig.canvas.draw()
            fig.savefig(tmp_path / f"mbuc-{i}.png")
            assert (tmp_path / f"mbuc-{i}.png").stat().st_size > 5000
    finally:
        plt.close("all")


@pytest.mark.parametrize("feedback", [False, True])
def test_all_basis_inputs_all_outcomes_every_phase_and_rng(
    feedback: bool, reference: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    from qualtran.bloqs.basic_gates import MeasureX
    from qualtran.simulation.classical_sim import (
        PhasedClassicalSimState,
        _get_in_vals,
        do_phased_classical_simulation,
    )

    source = reference[0]["mbuc_target" if feedback else "throw_out_target"]
    instances = [b for b, _, _ in source.iter_bloqsoqs()]
    assert [b.i for b in instances] == list(range(len(instances)))
    for q1, q2, target, outcome in product(range(2), repeat=4):
        values = {"q1": q1, "q2": q2, "target": target}
        native = uncomputation_state(
            values, feedback=feedback, fixed_values={0: outcome}
        )
        original = PhasedClassicalSimState.from_cbloq(
            source, values, fixed_random_vals={0: outcome}
        )
        record = uncomputation_program(feedback=feedback)
        functions = {f.name: f for f in record.functions}
        for position, instance in enumerate(instances):
            native.step()
            while original.last_binst != instance:
                original.step()
            assert native.phase == original.phase
            op = record.functions[0].blocks[0].operations[position]
            function = functions[op.callee or ""]
            from flagquantum.core.ir.program.signatures import Signature

            observed = {
                r.name: native.assignments[v]
                for r, v in zip(
                    Signature(function.registers or ()).rights(),
                    op.results,
                    strict=True,
                )
            }
            expected = {
                r.name: _get_in_vals(instance, r, original.soq_assign)
                for r in instance.bloq.signature.rights()
            }
            assert observed == expected
        assert native.finalize() == original.simulate() == {"q1": q1, "q2": q2}
        assert native.phase == (
            -1 if outcome and (target ^ (int(feedback) & q1 & q2)) else 1
        )
        assert sum(b.bloq_is(MeasureX) for b in instances) == 1
    rng = np.random.default_rng(123)
    original_phases = []
    for _ in range(10):
        outputs, phase = do_phased_classical_simulation(
            source, {"q1": 1, "q2": 1, "target": 1}, rng=rng
        )
        assert outputs == {"q1": 1, "q2": 1}
        original_phases.append(phase)
    assert tuple(original_phases) == seeded_trials(feedback=feedback)


def test_source_notebook_stale_phase_and_global_debug_defects(
    reference: tuple[dict[str, Any], dict[str, Any]], capsys: pytest.CaptureFixture[str]
) -> None:
    namespace, notebook = reference
    from qualtran import DecomposeNotImplementedError
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    # The original CBit-controlled CZ has no cost decomposition. Preserve
    # this unavailable composite estimate rather than reporting zero cost.
    with pytest.raises(DecomposeNotImplementedError):
        get_cost_value(namespace["mbuc_target"], QECGatesCost())
    code = "".join(notebook["cells"][18]["source"])
    namespace["phase"] = -1
    with pytest.raises(AssertionError):
        exec(compile(code, "original-stale-phase", "exec"), namespace)
    assert namespace["sim"].phase == 1
    namespace["phase"] = 1
    namespace["cbloq"] = namespace["throw_out_target"]
    exec(compile(code, "original-false-accept", "exec"), namespace)
    assert namespace["sim"].phase == -1  # The unchanged assertions passed anyway.
    from qualtran.simulation.classical_sim import PhasedClassicalSimState

    namespace["sim"] = PhasedClassicalSimState.from_cbloq(
        namespace["mbuc_target"],
        {"q1": 0, "q2": 0, "target": 0},
        fixed_random_vals={0: 0},
    )
    debug = namespace["DebugPhasedClassicalSim"].from_cbloq(
        namespace["mbuc_target"],
        {"q1": 1, "q2": 1, "target": 1},
        fixed_random_vals={0: 1},
    )
    capsys.readouterr()
    while debug.last_binst is None or getattr(debug.last_binst, "i", None) != 0:
        debug.step()
    assert debug.phase == -1 and namespace["sim"].phase == 1
    assert "-" not in capsys.readouterr().out
    assert phase_trace(
        uncomputation_state({"q1": 1, "q2": 1, "target": 1}, fixed_values={0: 1})
    ) == (-1, 1, 1)


def test_and_adjoint_guard_matches_every_valid_and_invalid_basis() -> None:
    from qualtran.bloqs.mcmt import And

    source = And(uncompute=True)
    for q1, q2, target in product(range(2), repeat=3):
        if target != q1 & q2:
            with pytest.raises(ValueError):
                source.call_classically(ctrl=[q1, q2], target=target)
            with pytest.raises(ValueError):
                checked_and_adjoint(q1, q2, target)
        else:
            (expected,) = source.call_classically(ctrl=[q1, q2], target=target)
            np.testing.assert_array_equal(
                checked_and_adjoint(q1, q2, target)["ctrl"], expected
            )


@pytest.mark.parametrize("feedback", [False, True])
def test_emitted_stream_complete_complex_branches_spectator_and_append(
    feedback: bool,
) -> None:
    from qualtran.bloqs.mcmt import And
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from flagquantum.compiler.registers.register_binding import bind_register_program
    from flagquantum.core.classical_control import instruction_condition_clauses
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates

    op = EraseAnd(feedback=feedback)
    gates = op.instructions()
    original = reference_arithmetic_gates(
        And(uncompute=True),
        {"ctrl": (0, 1), "target": (2,)},
        (),
        {2: 0},
        cirq_style=True,
    )
    expected = original if feedback else tuple(g for g in original if g.name != "cz")
    for actual, reference_gate in zip(gates, expected, strict=True):
        assert (actual.name, actual.wires, actual.params, actual.metadata) == (
            reference_gate.name,
            reference_gate.wires,
            reference_gate.params,
            reference_gate.metadata,
        )
    bound = bind_register_program(
        uncomputation_program(feedback=feedback), ((0,), (1,), (2,))
    )
    assert len(bound.circuit.instructions) == len(gates) - 1
    for actual, prefix in zip(bound.circuit.instructions, gates[:-1], strict=True):
        assert (actual.name, actual.wires, actual.params) == (
            prefix.name,
            prefix.wires,
            prefix.params,
        )
        assert instruction_condition_clauses(actual) == instruction_condition_clauses(
            prefix
        )
    amplitudes = torch.tensor(
        [1 + 2j, 3 - 4j, -2 + 1j, 4 + 3j, 2 - 5j, 1 + 7j, -3 - 2j, 5 + 1j],
        dtype=torch.complex128,
    )
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(16, dtype=torch.complex128)
    expected_clean = torch.zeros_like(initial)
    for q1, q2, spectator in product(range(2), repeat=3):
        i = (q1 << 2) | (q2 << 1) | spectator
        initial[(q1 << 3) | (q2 << 2) | ((q1 & q2) << 1) | spectator] = amplitudes[i]
        expected_clean[(q1 << 3) | (q2 << 2) | spectator] = amplitudes[i]
    for outcome in (0, 1):
        branch = forced_branch(initial, gates, (outcome,))
        expected_branch = expected_clean / 2**0.5
        if not feedback and outcome:
            expected_branch = expected_branch.clone()
            expected_branch[12:14] *= -1
        torch.testing.assert_close(branch, expected_branch, atol=2e-14, rtol=0)
        assert torch.count_nonzero(branch.reshape(4, 2, 2)[:, 1, :]) == 0
    counts = Counter(g.name for g in gates)
    assert counts == (
        {"h": 1, "measure": 1, "cz": 1, "reset": 1}
        if feedback
        else {"h": 1, "measure": 1, "reset": 1}
    )
    model = get_cost_value(And(uncompute=True), QECGatesCost())
    assert model.measurement == 1 and model.clifford == 1 and model.t == 0
    circuit = DynamicCircuit(4, inputs=initial, dtype=torch.complex128)
    circuit.append(op)
    result = run_dynamic(circuit, shots=8, seed=13, strategy="trajectory")
    for state in result.final_states.reshape(-1, 16):
        assert torch.count_nonzero(state.reshape(4, 2, 2)[:, 1, :]) == 0
        good = torch.allclose(state, expected_clean, atol=2e-14, rtol=0)
        if feedback:
            assert good
        else:
            bad = expected_clean.clone()
            bad[12:14] *= -1
            assert good or torch.allclose(state, bad, atol=2e-14, rtol=0)
