"""Retained frontend imports preserve capture identity and replacement hooks."""

import subprocess
import sys
from importlib import import_module
from pathlib import Path

import pytest

from examples.synthesis_offline_deployment import bell
from flagquantum import synth
from flagquantum.compiler._construction.frontend.python import _QuantumFunction
from flagquantum.compiler._hybrid.model import HybridProgram

pytestmark = pytest.mark.unit
CONSTRUCTION = "flagquantum.compiler._construction"
FRONTENDS = tuple(
    path.stem.removeprefix("frontend_")
    for path in sorted(
        (
            Path(__file__).resolve().parents[3] / "flagquantum/compiler/_construction"
        ).glob("frontend_*.py")
    )
)


@pytest.mark.parametrize("name", FRONTENDS)
def test_retained_imports_share_the_implementation_module(name: str) -> None:
    retained = import_module(f"{CONSTRUCTION}.frontend_{name}")
    implementation = import_module(f"{CONSTRUCTION}.frontend.{name}")
    assert retained is implementation


@pytest.mark.parametrize("path", ["frontend_quantized", "frontend.quantized"])
def test_source_capture_observes_replacements_through_either_path(
    monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    class CaptureStoppedError(Exception):
        pass

    def stop_capture(definition: _QuantumFunction) -> HybridProgram | None:
        assert definition is bell
        raise CaptureStoppedError("replacement reached before lowering")

    module = import_module(f"{CONSTRUCTION}.{path}")
    with monkeypatch.context() as patched:
        patched.setattr(module, "try_capture_quantized", stop_capture)
        with pytest.raises(CaptureStoppedError, match="replacement reached"):
            synth.synthesize(bell)

    program = synth.synthesize(bell)
    assert program.circuit.n_wires == 2


@pytest.mark.parametrize("implementation_first", [False, True])
def test_both_import_orders_work_in_a_fresh_interpreter(
    implementation_first: bool,
) -> None:
    script = f"""
from importlib import import_module

for name in {FRONTENDS!r}:
    paths = [
        f"{CONSTRUCTION}.frontend_{{name}}",
        f"{CONSTRUCTION}.frontend.{{name}}",
    ]
    if {implementation_first!r}:
        paths.reverse()
    assert import_module(paths[0]) is import_module(paths[1])
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=30)
