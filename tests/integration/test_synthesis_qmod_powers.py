"""Positive literal powers reuse multiplication and checked named results."""

from typing import Literal

import pytest

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError
from flagquantum.ecosystem._qmod.adapter import SUPPORTED_SOURCE_VERSION, parse_qmod
from tests.integration.test_synthesis_register_workflow import check_state

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("power", range(1, 9))
@pytest.mark.parametrize("x", [0, 1])
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_all_literal_powers_on_quantum_inputs(
    power: int, x: int, objective: Literal["width", "depth"]
) -> None:
    source = (
        f"qfunc main(output x:qbit,output result:qnum) {{x={x};result=x**{power};}}"
    )
    p = synth.synthesize(source, source_format="qmod", objective=objective)
    check_state(p, fq.run(p), {(x, x**power): 1})


@pytest.mark.parametrize("objective", ["width", "depth"])
@pytest.mark.parametrize("expression", ["x**3", "(x*x)*x"])
def test_coherent_cube_and_explicit_product_share_consumer(
    objective: Literal["width", "depth"], expression: str
) -> None:
    source = f"qfunc main(output x:qnum<2>,output result:qnum) {{allocate(x);hadamard_transform(x);result={expression};}}"
    p = synth.synthesize(source, source_format="qmod", objective=objective)
    check_state(p, fq.run(p), {(x, x**3): 0.5 for x in range(4)})
    assert dict(p.report["register_widths"]) == {"x": 2, "result": 5}


@pytest.mark.parametrize("power", range(1, 9))
def test_powers_preserve_complex_reference_phase_and_zero_work(power: int) -> None:
    source = f"""qfunc main(output x:qbit,output ref:qbit) {{
      allocate(x);H(x);S(x);ref=x;phase(x**{power},pi);
    }}"""
    p = synth.synthesize(source, source_format="qmod")
    check_state(p, fq.run(p), {(0, 0): 2**-0.5, (1, 1): -1j * 2**-0.5})


@pytest.mark.parametrize(
    "expression", ["x**0", "x**9", "x**y", "x**-1", "x**1.5", "x**(1+1)", "x**2**2"]
)
def test_invalid_powers_fail_before_planning(
    expression: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from flagquantum import _api

    monkeypatch.setattr(
        _api, "plan", lambda *a, **k: pytest.fail("invalid power planned")
    )
    with pytest.raises(InteropConversionError):
        synth.synthesize(
            f"qfunc main(output x:qbit,output y:qbit,output result:qnum) {{allocate(x);allocate(y);result={expression};}}",
            source_format="qmod",
        )


@pytest.mark.parametrize("expression", ["x**8", "(x**2)**2"])
def test_value_powers_preserve_external_entanglement(expression: str) -> None:
    source = f"""qfunc main(output x:qbit,output ref:qbit,output result:qnum) {{
      allocate(x);H(x);S(x);ref=x;result={expression};
    }}"""
    p = synth.synthesize(source, source_format="qmod")
    check_state(p, fq.run(p), {(0, 0, 0): 2**-0.5, (1, 1, 1): 1j * 2**-0.5})


def test_power_width_guard_and_literal_folding() -> None:
    p = synth.synthesize(
        "qfunc main(output result:qnum) {result=2**8;}", source_format="qmod"
    )
    check_state(p, fq.run(p), {(256,): 1})
    with pytest.raises(InteropConversionError):
        synth.synthesize(
            "qfunc main(output x:qnum<256>,output result:qnum) {allocate(x);result=x**8;}",
            source_format="qmod",
        )
    imported = parse_qmod(
        "qfunc main(output x:qnum<2>,output result:qnum) {allocate(x);result=x**8;}",
        source_version=SUPPORTED_SOURCE_VERSION,
    )
    assert (
        sum(
            op.name == "qint.multiply"
            for op in imported.functions[0].body.blocks[0].operations
        )
        == 7
    )
    cube = synth.synthesize(
        "qfunc main(output x:qnum<2>,output result:qnum) {"
        "allocate(x);hadamard_transform(x);result=x**3;}",
        source_format="qmod",
        max_workspace=0,
    )
    assert cube.report["workspace_qubits"] == 0
    check_state(cube, fq.run(cube), {(x, x**3): 0.5 for x in range(4)})


def test_quantum_dependent_constant_power_keeps_initialization_and_coherence() -> None:
    source = """qfunc main(output x:qbit,output result:qnum) {
      allocate(x);H(x);S(x);result=(x*0+1)**8;
    }"""
    p = synth.synthesize(source, source_format="qmod")
    check_state(p, fq.run(p), {(0, 1): 2**-0.5, (1, 1): 1j * 2**-0.5})
    with pytest.raises(InteropConversionError):
        synth.synthesize(
            source.replace("allocate(x);H(x);S(x);", ""), source_format="qmod"
        )
