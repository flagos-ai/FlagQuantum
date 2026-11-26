"""Public synthesis uses source capture hooks and imports without cycles."""

import subprocess
import sys
from importlib import import_module

import pytest

from examples.synthesis.synthesis_offline_deployment import bell
from flagquantum import synth
from flagquantum.compiler._construction.frontend.python import _QuantumFunction
from flagquantum.compiler.frontends._structured.model import HybridProgram

pytestmark = pytest.mark.unit
CONSTRUCTION = "flagquantum.compiler._construction"


def test_source_capture_observes_the_implementation_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CaptureStoppedError(Exception):
        pass

    def stop_capture(definition: _QuantumFunction) -> HybridProgram | None:
        assert definition is bell
        raise CaptureStoppedError("replacement reached before lowering")

    module = import_module(f"{CONSTRUCTION}.frontend.quantized")
    with monkeypatch.context() as patched:
        patched.setattr(module, "try_capture_quantized", stop_capture)
        with pytest.raises(CaptureStoppedError, match="replacement reached"):
            synth.synthesize(bell)

    program = synth.synthesize(bell)
    assert program.circuit.n_wires == 2


@pytest.mark.parametrize("reverse", [False, True])
def test_construction_modules_import_in_a_fresh_interpreter(reverse: bool) -> None:
    script = f"""
from importlib import import_module
from pathlib import Path

root = Path("flagquantum/compiler/_construction")
modules = sorted(
    ".".join(path.with_suffix("").parts)
    for path in root.rglob("*.py")
    if path.name != "__init__.py"
)
if {reverse!r}:
    modules.reverse()
for module in modules:
    import_module(module)
from examples.synthesis.synthesis_offline_deployment import bell
from flagquantum import synth
program = synth.synthesize(bell)
assert tuple(gate.name for gate in program.circuit.instructions) == ("h", "cx")
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=30)
