"""Native scopes produce checked coherent phases without exported temporary flags."""

import math
from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest
import torch

import flagquantum as fq
from flagquantum import _api, synth
from flagquantum.compiler._construction.lowering import Schedule
from flagquantum.compiler._construction.register_lowering import (
    schedule_register_program,
)
from flagquantum.compiler._construction.selection import select_register_program
from flagquantum.ecosystem import InteropConversionError
from flagquantum.ecosystem._qmod.adapter import SUPPORTED_SOURCE_VERSION, parse_qmod
from flagquantum.errors import PlanningError
from flagquantum.services._synthesis_registers import synthesize_registers
from tests.integration.test_construction_register_execution import _check
from tests.register_program_cases import Registers
from tests.synthesis_workflow_helpers import check_state
from tools.synthesis_compiler_examples.local_registers.main import (
    local_composition,
)
from tools.synthesis_import_examples.qmod_within.main import SOURCE

pytestmark = pytest.mark.integration


def marked(x: int, y: int) -> bool:
    return x <= y


@pytest.mark.parametrize("objective", ["width", "depth"])
@pytest.mark.parametrize("x", range(4))
@pytest.mark.parametrize("y", range(4))
def test_public_native_scope_all_sixteen_inputs_keep_absolute_phase_and_zero_work(
    objective: Literal["width", "depth"],
    x: int,
    y: int,
) -> None:
    source = SOURCE.replace("allocate(x); allocate(y);", f"x={x}; y={y};").replace(
        "hadamard_transform(x); hadamard_transform(y);", ""
    )
    program = synth.synthesize(source, source_format="qmod", objective=objective)
    result = fq.run(program)
    check_state(program, result, {(x, y): (-1) ** int(x <= y)})
    assert dict(program.probabilities(result, registers=("x", "y"))) == pytest.approx(
        {(x, y): 1}
    )


@pytest.mark.parametrize("producer", ["native", "python", "owned"])
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_existing_producers_replace_native_scopes_behind_the_same_consumer(
    producer: str,
    objective: Literal["width", "depth"],
) -> None:
    if producer == "native":
        program = synth.synthesize(SOURCE, source_format="qmod", objective=objective)
    elif producer == "python":
        program = synth.synthesize(
            marked,
            inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
            input_state=synth.InputState.uniform(),
            output_mode="phase",
            objective=objective,
        )
    else:
        _, functions = local_composition(helpers=True)
        p = Registers()
        x, y = p.declare("x", 2), p.declare("y", 2)
        for value in (x, y):
            p.emit("qreg.allocate", value)
            for bit in range(2):
                p.emit("qreg.gate", value, gate="h", index=bit)
        for helper in ("mark_less", "mark_equal"):
            p.emit("qreg.call", x, y, callee=helper)
        program = synthesize_registers(
            p.finish(), functions, objective=objective, memory_limit_bytes=64 * 1024**2
        )
    result = fq.run(program)
    check_state(
        program,
        result,
        {(x, y): (-1) ** int(x <= y) / 4 for x in range(4) for y in range(4)},
    )
    assert tuple(program.registers) == ("x", "y")
    assert dict(program.probabilities(result, registers=("x", "y"))) == pytest.approx(
        {(x, y): 1 / 16 for x in range(4) for y in range(4)}
    )
    assert program._checked.verify_result(result) == program.report["resources"]


@pytest.mark.parametrize("kind", ["file", "text"])
def test_scope_source_has_the_same_checked_file_and_text_entry(
    tmp_path: Path, kind: str
) -> None:
    path = tmp_path / "oracle.qmod"
    path.write_text(SOURCE)
    program = (
        synth.synthesize(path)
        if kind == "file"
        else synth.synthesize(SOURCE, source_format="qmod", source_name=str(path))
    )
    result = fq.run(program)
    check_state(
        program,
        result,
        {(x, y): (-1) ** int(x <= y) / 4 for x in range(4) for y in range(4)},
    )
    assert dict(program.report["register_widths"]) == {"x": 2, "y": 2}
    assert path.read_text() == SOURCE


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
def test_raw_source_executes_the_selected_inverse_and_nontrivial_phase(
    strategy: Schedule,
) -> None:
    imported = parse_qmod(
        SOURCE, filename="oracle.qmod", source_version=SUPPORTED_SOURCE_VERSION
    )
    row = schedule_register_program(
        imported.program, imported.functions, n_qubits=24, scheduling=strategy
    )
    raw = replace(row.circuit, n_wires=4 + row.peak_workspace)
    observed = (
        fq.Circuit.from_ir(raw, dtype=torch.complex128, device="cpu")
        .state()
        .reshape(-1)
    )
    expected = torch.zeros_like(observed)
    for x in range(4):
        for y in range(4):
            expected[((x << 2) | y) << row.peak_workspace] = (-1) ** int(x <= y) / 4
    torch.testing.assert_close(observed, expected, atol=1e-12, rtol=0)
    for forward, inverse in (
        (row.segments[0], row.segments[2]),
        (row.segments[3], row.segments[5]),
    ):
        assert inverse.circuit.instructions == forward.circuit.instructions[::-1]


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_const_apply_helpers_preserve_complex_external_entanglement(
    nested: bool,
    objective: Literal["width", "depth"],
) -> None:
    tint = "t:qbit; within {t=b;} apply {phase(t,pi);}" if nested else "phase(b,pi);"
    source = f"""qfunc tint(const b:qbit) {{ {tint} }}
    qfunc mark(const x:qnum<2>,const y:qnum<2>,const enable:qbit) {{
      flag:qbit;
      within {{flag=(x<y)*enable;}} apply {{tint(flag);}}
      within {{flag=(x==y)*enable;}} apply {{tint(flag);}}
    }}
    qfunc main(output x:qnum<2>,output y:qnum<2>,output enable:qbit,output reference:qbit) {{
      allocate(x); allocate(y); hadamard_transform(x); hadamard_transform(y);
      allocate(enable); H(enable); S(enable); reference=enable; mark(x,y,enable);
    }}"""
    program = synth.synthesize(source, source_format="qmod", objective=objective)
    check_state(
        program,
        fq.run(program),
        {
            (x, y, enable, enable): 1j**enable
            * (-1) ** (enable * int(x <= y))
            / math.sqrt(32)
            for x in range(4)
            for y in range(4)
            for enable in range(2)
        },
    )
    assert set(program.registers) == {"x", "y", "enable", "reference"}


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
def test_dependent_local_computations_clean_in_reverse_order(
    strategy: Schedule,
) -> None:
    source = """qfunc mark(const x:qbit,const y:qbit) {
      energy:qnum; flag:qbit;
      within {energy=x*x+y; flag=energy<2;} apply {phase(flag,pi);}
    }
    qfunc main(output x:qbit,output y:qbit,output reference:qbit) {
      allocate(x); H(x); S(x); allocate(y); H(y); reference=x; mark(x,y);
    }"""
    imported = parse_qmod(
        source, filename="chain.qmod", source_version=SUPPORTED_SOURCE_VERSION
    )
    selection = select_register_program(imported.program, imported.functions)
    row = next(r for r in selection.rows if r.strategy == strategy)
    _check(
        replace(selection, selected=row),
        {
            (x, y, x): 1j**x * (-1) ** int(x * x + y < 2) / 2
            for x in range(2)
            for y in range(2)
        },
    )


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_apply_helper_accepts_aliased_const_actuals_and_cleans_private_flag(
    objective: Literal["width", "depth"],
) -> None:
    source = """qfunc tint(const a:qbit,const b:qbit) {phase(a*b,pi);}
    qfunc main(output x:qnum<2>,output y:qnum<2>) {
      allocate(x); allocate(y); hadamard_transform(x); hadamard_transform(y);
      flag:qbit; within {flag=x==y;} apply {tint(flag,flag);}
    }"""
    program = synth.synthesize(source, source_format="qmod", objective=objective)
    check_state(
        program,
        fq.run(program),
        {(x, y): (-1) ** int(x == y) / 4 for x in range(4) for y in range(4)},
    )
    assert tuple(program.registers) == ("x", "y")


@pytest.mark.parametrize("count", [1, 2])
def test_fixed_quantum_dependent_scope_phases_preserve_global_sign(count: int) -> None:
    scopes = "within {flag=x*0+1;} apply {phase(flag,pi);}" * count
    source = f"qfunc main(output x:qnum<2>) {{allocate(x); hadamard_transform(x); flag:qbit; {scopes} }}"
    program = synth.synthesize(source, source_format="qmod")
    check_state(program, fq.run(program), {(x,): (-1) ** count / 2 for x in range(4)})


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_scope_phase_is_observable_by_interference(
    objective: Literal["width", "depth"],
) -> None:
    source = """qfunc main(output q:qbit) {allocate(q); H(q); flag:qbit;
    within {flag=q;} apply {phase(flag,pi);} H(q);}"""
    program = synth.synthesize(source, source_format="qmod", objective=objective)
    check_state(program, fq.run(program), {(1,): 1})


def test_user_constraints_include_hidden_locals_and_source_effect_order() -> None:
    program = synth.synthesize(
        SOURCE, source_format="qmod", max_qubits=10, max_workspace=6
    )
    assert program.report["resources"].n_qubits == 10
    assert program.report["workspace_qubits"] == 6
    main = SOURCE[SOURCE.index("qfunc main") :]
    helper = """qfunc mark(const x:qnum<2>,const y:qnum<2>) {a:qbit; b:qbit;
      within {a=x<y; b=x==y;} apply {phase(a,pi); phase(b,pi);}
    }"""
    overlapping = helper + main
    alternative = synth.synthesize(overlapping, source_format="qmod")
    assert alternative.report["resources"].n_qubits == 11
    assert alternative.report["workspace_qubits"] == 7
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(
            overlapping, source_format="qmod", max_qubits=10, max_workspace=6
        )


def test_scope_admission_keeps_no_execution_memory_and_exact_target_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_api, "run", lambda *a, **k: pytest.fail("synthesis executed"))
    assert synth.synthesize(SOURCE, source_format="qmod").execution is not None
    with pytest.raises(PlanningError, match="memory budget"):
        synth.synthesize(SOURCE, source_format="qmod", memory_limit_bytes=1)
    with pytest.raises(ValueError, match="phase output admits only"):
        synth.synthesize(SOURCE, source_format="qmod", topology="line")


@pytest.mark.parametrize(
    "body",
    [
        "flag:qbit; allocate(flag);",
        "flag:qbit; within {flag=1;} apply {}",
        "flag:qbit; within {flag=x<y;} apply {X(flag);}",
        "flag:qbit; within {flag=x<y;} apply {} phase(flag,pi);",
    ],
)
def test_invalid_scope_errors_precede_any_planning(
    monkeypatch: pytest.MonkeyPatch, body: str
) -> None:
    monkeypatch.setattr(
        _api, "plan", lambda *a, **k: pytest.fail("invalid scope planned")
    )
    with pytest.raises(InteropConversionError) as error:
        synth.synthesize(
            f"qfunc main(output x:qnum<2>,output y:qnum<2>) {{allocate(x); allocate(y); {body}}}",
            source_format="qmod",
            source_name="bad_scope.qmod",
        )
    assert "bad_scope.qmod:1:" in error.value.report.issues[0].message
