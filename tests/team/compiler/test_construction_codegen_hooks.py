"""Public synthesis uses the authoritative code generation workflow hooks."""

from importlib import import_module

import pytest

from examples.synthesis_offline_deployment import bell
from flagquantum import synth
from flagquantum.compiler._construction.frontend.python import capture

pytestmark = pytest.mark.unit
CONSTRUCTION = "flagquantum.compiler._construction"


def increment(value: int) -> int:
    return value + 1


def test_public_synthesis_observes_register_verification_replacements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class VerificationStoppedError(Exception):
        pass

    def stop_verification(*args: object, **kwargs: object) -> None:
        raise VerificationStoppedError("register verification reached")

    module = import_module(f"{CONSTRUCTION}.codegen.registers")
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
        ("codegen.arithmetic", "schedule_arithmetic"),
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
