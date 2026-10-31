"""One public entry, Program, execution and named results for both producers."""

import math
import subprocess
import sys
from pathlib import Path

import pytest

import flagquantum as fq
from flagquantum import _api, synth
from flagquantum.ecosystem import InteropConversionError
from flagquantum.ecosystem._qmod import source as boundary
from flagquantum.errors import PlanningError
from tests.integration.test_synthesis_register_workflow import check_state

pytestmark = pytest.mark.integration
SOURCE = "qfunc main(output x:qnum<2>, output result:qnum) { allocate(x); hadamard_transform(x); result=x*x+1; }"
ROOT = Path(__file__).resolve().parents[2]


def polynomial(x: int) -> int:
    return x * x + 1


def constant() -> int:
    return 3


@pytest.mark.parametrize("producer", ["python", "text", "file"])
@pytest.mark.parametrize("topology", [None, "line"])
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_shared_public_flow_preserves_phase_cleanup_names_and_actual_costs(
    tmp_path, producer, topology, objective
):
    options = {"topology": topology, "objective": objective}
    if producer == "python":
        program = synth.synthesize(
            function=polynomial,
            inputs={"x": synth.UInt(2)},
            input_state=synth.InputState.uniform(),
            **options,
        )
    elif producer == "text":
        program = synth.synthesize(
            SOURCE, source_format="qmod", source_name="text.qmod", **options
        )
    else:
        path = tmp_path / "source.qmod"
        path.write_bytes(SOURCE.encode())
        program = synth.synthesize(path, **options)
    result = fq.run(program)
    check_state(program, result, {(x, x * x + 1): 0.5 for x in range(4)})
    assert dict(
        program.probabilities(result, registers=("result", "x"))
    ) == pytest.approx({(x * x + 1, x): 0.25 for x in range(4)}, abs=1e-12)
    report = program.report
    assert dict(report["register_widths"]) == {"x": 2, "result": 4}
    assert report["resources"] == program._checked.verify_result(result)
    assert report["scope"] == "actual_cpu_preparation_arithmetic_cleanup"
    assert report["memory_limit_origin"] == "profile_default"
    if producer == "python":
        assert report["source_format"] == "python"
        assert report["memory_limit_bytes"] is None
        assert (
            report["arithmetic_resources"].n_operations
            < report["source_resources"].n_operations
        )
    else:
        assert report["source_format"] == "qmod"
        assert report["memory_limit_bytes"] == 512 * 1024**2
        assert "arithmetic_resources" not in report and "output_widths" not in report
        assert report["source_name"] == (
            "text.qmod" if producer == "text" else str(path)
        )
    with pytest.raises(TypeError):
        report["register_widths"]["x"] = 9
    if topology:
        row = program.selection.selected
        assert any(g.name == "swap" for g in row.target.executable.instructions)
        assert row.target.layout.logical_to_physical != tuple(
            range(program.circuit.n_wires)
        )


def test_source_phase_and_mutation_remain_coherent():
    program = synth.synthesize(
        "qfunc main(output flag:qbit, output n:qnum, output res:qnum) { allocate(flag); H(flag); S(flag); allocate(2,n); hadamard_transform(n); res=flag*n; X(flag); }",
        source_format="qmod",
    )
    result = fq.run(program)
    check_state(
        program,
        result,
        {
            (1 - q, n, q * n): (1j if q else 1) / math.sqrt(8)
            for q in range(2)
            for n in range(4)
        },
    )


@pytest.mark.parametrize(
    "options",
    [
        {"inputs": {}},
        {"input_state": synth.InputState.uniform()},
        {"constants": {}},
        {"functions": (polynomial,)},
        {"preparation": fq.Circuit(1).to_ir()},
        {"preparation_map": {}},
        {"source_format": "python"},
        {"source_format": None},
        {"objective": "unknown"},
        {"topology": "unknown"},
    ],
)
def test_invalid_source_combinations_refuse_before_loading(monkeypatch, options):
    monkeypatch.setattr(
        boundary,
        "load_qmod",
        lambda *a, **k: pytest.fail("source loaded before refusal"),
    )
    with pytest.raises(ValueError):
        synth.synthesize(SOURCE, **({"source_format": "qmod"} | options))


@pytest.mark.parametrize(
    "options",
    [
        {"source_format": "qmod"},
        {"source_name": "x.qmod"},
        {"source_version": "context"},
    ],
)
def test_python_does_not_ignore_native_options(monkeypatch, options):
    from flagquantum.services import synthesis

    monkeypatch.setattr(
        synthesis, "capture", lambda *a, **k: pytest.fail("capture before refusal")
    )
    with pytest.raises(ValueError):
        synth.synthesize(polynomial, inputs={"x": synth.UInt(1)}, **options)


def test_missing_or_empty_python_inputs_preserve_existing_refusal():
    with pytest.raises(TypeError):
        synth.synthesize(polynomial)
    with pytest.raises(ValueError, match="at least one named quantum input"):
        synth.synthesize(constant, inputs={})


@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
def test_invalid_memory_limits_refuse_before_loading(monkeypatch, value):
    monkeypatch.setattr(
        boundary,
        "load_qmod",
        lambda *a, **k: pytest.fail("load before budget validation"),
    )
    with pytest.raises((TypeError, ValueError)):
        synth.synthesize(SOURCE, source_format="qmod", memory_limit_bytes=value)


def test_default_memory_refuses_and_explicit_ceiling_admits_without_execution(
    monkeypatch,
):
    monkeypatch.setattr(_api, "run", lambda *a, **k: pytest.fail("synthesis executed"))
    source = "qfunc main(output q:qnum<26>) { allocate(q); }"
    with pytest.raises(PlanningError, match="memory budget"):
        synth.synthesize(source, source_format="qmod")
    p = synth.synthesize(source, source_format="qmod", memory_limit_bytes=1024**3)
    assert p.execution.state_bytes == 1024**3
    assert p.report["memory_limit_bytes"] == 1024**3
    assert p.report["memory_limit_origin"] == "explicit"
    with pytest.raises(PlanningError):
        synth.synthesize(polynomial, inputs={"x": synth.UInt(1)}, memory_limit_bytes=1)


def test_preparation_only_source_has_whole_cost_without_arithmetic_label():
    p = synth.synthesize(
        "qfunc main(output q:qbit) { allocate(q); H(q); S(q); }", source_format="qmod"
    )
    assert p.report["source_resources"].n_operations == 2
    assert "arithmetic_resources" not in p.report
    assert "construction_resources" not in p.report
    result = fq.run(p)
    check_state(p, result, {(0,): 1 / math.sqrt(2), (1,): 1j / math.sqrt(2)})
    with pytest.raises(ValueError, match="available names.*q"):
        p.probabilities(result)
    with pytest.raises(ValueError, match="budget"):
        p.probabilities(result, registers=("q",), max_amplitudes=1)


def test_filename_strings_are_not_implicitly_read(tmp_path):
    path = tmp_path / "source.qmod"
    path.write_text(SOURCE)
    with pytest.raises(ValueError, match="source_format"):
        synth.synthesize(str(path))
    with pytest.raises(InteropConversionError):
        synth.synthesize(str(path), source_format="qmod")


def test_pathlike_precedes_callable_and_is_resolved_once(tmp_path):
    path = tmp_path / "source.qmod"
    path.write_text("qfunc main(output q:qbit) { allocate(q); X(q); }")

    class CallablePath:
        calls = 0

        def __fspath__(self):
            self.calls += 1
            return str(path)

        def __call__(self):
            pytest.fail("path was executed")

    source = CallablePath()
    p = synth.synthesize(source)
    assert source.calls == 1
    assert dict(p.probabilities(fq.run(p), registers=("q",))) == {(1,): 1.0}


def test_file_changes_do_not_reload_a_checked_program(tmp_path):
    path = tmp_path / "change.qmod"
    path.write_text("qfunc main(output q:qbit) { allocate(q); X(q); }")
    before = synth.synthesize(path)
    path.write_text("qfunc main(output q:qbit) { allocate(q); H(q); }")
    after = synth.synthesize(path)
    r = fq.run(before)
    assert dict(before.probabilities(r, registers=("q",))) == {(1,): 1.0}
    assert dict(after.probabilities(fq.run(after), registers=("q",))) == pytest.approx(
        {(0,): 0.5, (1,): 0.5}
    )
    with pytest.raises(ValueError, match="identity"):
        before.probabilities(fq.run(after), registers=("q",))


def test_constraints_repairs_and_incomplete_search_are_unchanged(monkeypatch):
    with pytest.raises(synth.SynthesisError) as error:
        synth.synthesize(SOURCE, source_format="qmod", max_qubits=2)
    repair = dict(error.value.repairs[0])
    p = synth.synthesize(SOURCE, source_format="qmod", **repair)
    check_state(p, fq.run(p), {(x, x * x + 1): 0.5 for x in range(4)})
    monkeypatch.setattr(
        _api, "plan", lambda *a, **k: pytest.fail("incomplete search planned")
    )
    with pytest.raises(synth.SynthesisError) as error:
        synth.synthesize(
            SOURCE,
            source_format="qmod",
            construction_budget=synth.ConstructionBudget(candidate_checks=1),
        )
    assert not error.value.selection.complete


@pytest.mark.parametrize("body", ["q=missing;", "free(q);", "q=__import__('os');"])
def test_public_source_errors_reuse_structured_conversion_report(body):
    with pytest.raises(InteropConversionError) as error:
        synth.synthesize(
            f"qfunc main(output q:qbit) {{ {body} }}",
            source_format="qmod",
            source_name="diagnostic.qmod",
        )
    report = error.value.report
    assert report.adapter == "qmod" and report.framework_version is None
    assert report.issues[0].code.startswith("qmod.")
    assert "diagnostic.qmod:1:" in report.issues[0].message


def test_python_entry_keeps_native_frontend_lazy(tmp_path):
    code = """import sys
class Block:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("flagquantum.ecosystem._qmod"):
            raise RuntimeError("eager Qmod frontend")
sys.meta_path.insert(0,Block())
import flagquantum as fq
from flagquantum import synth
def identity(x: int) -> int:
    return x
p=synth.synthesize(identity,inputs={"x":synth.UInt(1)},input_state=synth.InputState.basis(x=1))
assert dict(p.probabilities(fq.run(p))) == {(1,):1.0}
"""
    path = tmp_path / "lazy.py"
    path.write_text(code)
    p = subprocess.run(
        [sys.executable, str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert p.returncode == 0, p.stdout + p.stderr
