"""Construction code generation retains import identity and workflow hooks."""

import subprocess
import sys
from importlib import import_module

import pytest

from examples.synthesis_offline_deployment import bell
from flagquantum import synth
from flagquantum.compiler._construction.frontend.python import capture

pytestmark = pytest.mark.unit
CONSTRUCTION = "flagquantum.compiler._construction"
MODULES = (
    ("lowering", "arithmetic"),
    ("register_lowering", "registers"),
    ("boolean_lowering", "boolean"),
    ("gate_tape", "gate_tape"),
)
HELPERS = (
    ("preparation", "_uniform_interval"),
    ("controls", "_array_controlled_x"),
    ("controls", "_bounded_control_group"),
    ("controls", "_modular_exp_instructions"),
    ("transforms", "_pattern_gate"),
    ("transforms", "_type1_instructions"),
    ("transforms", "_type2_instructions"),
    ("transforms", "_real_transform_instructions"),
)


def increment(value: int) -> int:
    return value + 1


@pytest.mark.parametrize(("retained", "implementation"), MODULES)
def test_retained_imports_share_the_implementation_module(
    retained: str, implementation: str
) -> None:
    assert import_module(f"{CONSTRUCTION}.{retained}") is import_module(
        f"{CONSTRUCTION}.codegen.{implementation}"
    )


@pytest.mark.parametrize(("module", "name"), HELPERS)
def test_register_helpers_remain_importable_from_the_retained_path(
    module: str, name: str
) -> None:
    retained = import_module(f"{CONSTRUCTION}.register_lowering")
    implementation = import_module(f"{CONSTRUCTION}.codegen.{module}")
    assert getattr(retained, name) is getattr(implementation, name)


@pytest.mark.parametrize("path", ["register_lowering", "codegen.registers"])
def test_public_synthesis_observes_register_verification_replacements(
    monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    class VerificationStoppedError(Exception):
        pass

    def stop_verification(*args: object, **kwargs: object) -> None:
        raise VerificationStoppedError("register verification reached")

    module = import_module(f"{CONSTRUCTION}.{path}")
    with monkeypatch.context() as patched:
        patched.setattr(module, "verify_register_program", stop_verification)
        with pytest.raises(VerificationStoppedError, match="verification reached"):
            synth.synthesize(bell)

    program = synth.synthesize(bell)
    assert program.circuit.n_wires == 2
    assert tuple(gate.name for gate in program.circuit.instructions) == ("h", "cx")


@pytest.mark.parametrize(
    ("path", "entry"),
    [
        ("lowering", "schedule_arithmetic"),
        ("codegen.arithmetic", "schedule_arithmetic"),
        ("boolean_lowering", "lower_boolean_value"),
        ("codegen.boolean", "lower_boolean_value"),
    ],
)
def test_lowering_observes_binding_replacements(
    monkeypatch: pytest.MonkeyPatch, path: str, entry: str
) -> None:
    class BindingStoppedError(Exception):
        pass

    def stop_bindings(*args: object, **kwargs: object) -> None:
        raise BindingStoppedError("binding verification reached")

    graph = capture(increment, {"value": 1})
    module = import_module(f"{CONSTRUCTION}.{path}")
    with monkeypatch.context() as patched:
        patched.setattr(module, "verify_bindings", stop_bindings)
        with pytest.raises(BindingStoppedError, match="verification reached"):
            getattr(module, entry)(
                (graph,),
                {graph.body.blocks[0].arguments[0].id: (0,)},
                (1, 2),
                entry=graph.name,
                n_qubits=4,
                workspace=(3,),
                workspace_state="fresh_zero",
            )


@pytest.mark.parametrize("implementation_first", [False, True])
def test_both_import_orders_work_in_a_fresh_interpreter(
    implementation_first: bool,
) -> None:
    script = f"""
from importlib import import_module

for retained, implementation in {MODULES!r}:
    paths = [
        f"{CONSTRUCTION}.{{retained}}",
        f"{CONSTRUCTION}.codegen.{{implementation}}",
    ]
    if {implementation_first!r}:
        paths.reverse()
    assert import_module(paths[0]) is import_module(paths[1])
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=30)
