"""Check locked Cirq legacy estimates and notebook utilities against their source."""

from hashlib import sha1
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest

from flagquantum.compiler._component_trace.compound_control_resource import (
    CONTROLLED_HADAMARD_COST,
)
from flagquantum.compiler._component_trace.qec_gate_cost import QECGateCost
from flagquantum.compiler._component_trace.resource_generalization import (
    ConjunctionCall,
    PhaseCall,
    ResourceCall,
)
from flagquantum.core.ir.program.register_types import opaque_register_type
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.ecosystem.cirq._legacy_t_complexity import (
    LegacyTComplexity,
    cirq_t_complexity,
    legacy_compiler_t_complexity,
)
from flagquantum.ecosystem.cirq._notebook import (
    circuit_with_costs,
    display_gate_and_compilation,
    execute_notebook,
    svg_circuit,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def cirq() -> Any:
    return pytest.importorskip("cirq")


@pytest.fixture(scope="module")
def source_protocol() -> Any:
    import kickmix
    from qualtran.cirq_interop import t_complexity_protocol

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    folder = Path(t_complexity_protocol.__file__).parent
    for name, digest in (
        ("jupyter_tools.py", "fab86169ce4e337ef85925c160e48c98b95b020e"),
        ("jupyter_tools_test.py", "a9322cb82bb3916dc62f7d675d68f523e70af626"),
        ("t_complexity.ipynb", "8bdf67ee5b1ad5639ab61e2de807cc035e8cb3f2"),
        ("t_complexity_protocol.py", "64e3da56f373bbcbd8d87b3de6893baf9bb62f2e"),
        ("t_complexity_protocol_test.py", "f6868c5a7f716170ad01e5864970a16d7df8435d"),
    ):
        raw = (folder / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest
    return t_complexity_protocol


@pytest.mark.ftqc_reference
def test_original_legacy_compatibility_grid(cirq: Any, source_protocol: Any) -> None:
    q = cirq.NamedQubit("q")
    circuit = cirq.Circuit(
        cirq.Rz(rads=0.6)(q),
        cirq.T(q),
        cirq.X(q) ** 0.5,
        cirq.Rx(rads=0.1)(q),
        cirq.Ry(rads=0.6)(q),
        cirq.measure(q, key="m"),
    )
    frozen = cirq.FrozenCircuit(
        cirq.T(q) ** -1, cirq.Rx(rads=0.1)(q), cirq.measure(q, key="m")
    )
    cases = [
        cirq.T,
        cirq.T**-1,
        cirq.S,
        cirq.S**-1,
        cirq.H,
        cirq.CNOT,
        cirq.X,
        cirq.Y,
        cirq.Z,
        cirq.rx(2),
        cirq.ry(2),
        cirq.rz(2),
        cirq.FREDKIN,
        cirq.GlobalPhaseGate(1j),
        cirq.global_phase_operation(1j),
        circuit,
        frozen,
        cirq.Moment(cirq.T(q)),
        [cirq.T, cirq.X],
        [cirq.T(q), cirq.X(q)],
        cirq.CircuitOperation(frozen, repetitions=3),
        cirq.CircuitOperation(circuit.freeze(), repetitions=10),
    ]
    for gate in (cirq.X, cirq.T, cirq.rx(0.1), cirq.ry(0.1)):
        cases.extend((gate(q).with_tags("note"), gate(q).with_classical_controls("c")))
    for value in cases:
        assert (
            cirq_t_complexity(value).asdict()
            == source_protocol.t_complexity_compat(value).asdict()
        )
    controlled_h = cirq.H.controlled()
    expected = {"1.6.1": (2, 4, 2), "1.7.0": (1, 3, 1)}
    assert (
        cirq_t_complexity(controlled_h).asdict()
        == source_protocol.t_complexity_compat(controlled_h).asdict()
    )
    if cirq.__version__ in expected:
        assert cirq_t_complexity(controlled_h) == LegacyTComplexity(
            *expected[cirq.__version__]
        )


@pytest.mark.ftqc_reference
def test_annotations_failures_and_original_compiler_conversion(
    cirq: Any, source_protocol: Any
) -> None:
    from qualtran.bloqs.basic_gates import CHadamard
    from qualtran.bloqs.mcmt import And, MultiAnd
    from sympy import Symbol, simplify

    class Annotated(cirq.testing.SingleQubitGate):  # type: ignore[misc]
        def _t_complexity_(self) -> Any:
            return source_protocol.TComplexity(t=1, clifford=2)

    class Unknown(cirq.testing.SingleQubitGate):  # type: ignore[misc]
        pass

    assert cirq_t_complexity(Annotated().on(cirq.NamedQubit("t"))) == LegacyTComplexity(
        1, 2
    )
    for value in (object(), [object()], Unknown()):
        with pytest.raises(TypeError):
            cirq_t_complexity(value)
        with pytest.raises(TypeError):
            source_protocol.t_complexity_compat(value)
    annotation: dict[ResourceCall, QECGateCost] = {
        PhaseCall("t"): QECGateCost(t=1, clifford=2)
    }
    assert legacy_compiler_t_complexity(
        PhaseCall("t"), annotations=annotation
    ) == LegacyTComplexity(1, 2)
    calls: list[tuple[ResourceCall, Any]] = [
        (ConjunctionCall((1, 1)), And()),
        (ConjunctionCall((1, 1), adjoint=True), And().adjoint()),
        (CONTROLLED_HADAMARD_COST, CHadamard()),
    ]
    calls.extend((ConjunctionCall((1,) * n), MultiAnd((1,) * n)) for n in range(3, 12))
    for native, source in calls:
        result = legacy_compiler_t_complexity(native)
        assert result.asdict() == source_protocol.t_complexity(source).asdict()
    precision = Symbol("epsilon", positive=True)
    local = 2 * LegacyTComplexity(t=2, rotations=3) + LegacyTComplexity(t=1)
    original = source_protocol.TComplexity(t=5, rotations=6)
    assert (
        simplify(
            local.t_incl_rotations(precision) - original.t_incl_rotations(precision)
        )
        == 0
    )


@pytest.mark.ftqc_reference
def test_original_svg_order_cost_tag_and_widget_layout(
    cirq: Any, source_protocol: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    ipython_display = pytest.importorskip("IPython.display")
    ipywidgets = pytest.importorskip("ipywidgets")
    from qualtran.bloqs.mcmt import MultiAnd
    from qualtran.cirq_interop import jupyter_tools
    from qualtran.cirq_interop.testing import GateHelper

    fixture = GateHelper(MultiAnd(cvs=(1, 1, 1)))
    signature = Signature(
        (
            Register("ctrl", opaque_register_type(1), shape=(3,)),
            Register(
                "junk", opaque_register_type(1), shape=(1,), side=RegisterSide.RIGHT
            ),
            Register("target", opaque_register_type(1), side=RegisterSide.RIGHT),
        )
    )
    reference = jupyter_tools.svg_circuit(fixture.circuit, fixture.r).data
    actual = svg_circuit(fixture.circuit, signature).data
    assert actual == reference
    assert actual.find("ctrl") < actual.find("junk") < actual.find("target")
    for empty in (cirq.Circuit(), cirq.Circuit(cirq.Moment())):
        with pytest.raises(ValueError):
            svg_circuit(empty)
    assert circuit_with_costs(fixture.circuit) == cirq.Circuit(
        fixture.operation.with_tags("t:8,r:0")
    )
    assert "t:8,r:0" in svg_circuit(fixture.circuit, signature, include_costs=True).data
    displays: list[Any] = []
    monkeypatch.setattr(ipython_display, "display", displays.append)
    display_gate_and_compilation(fixture.operation, registers=signature)
    display_gate_and_compilation(fixture.operation, registers=signature, vertical=True)
    assert [type(item) for item in displays] == [ipywidgets.HBox, ipywidgets.VBox]
    assert all(len(item.children) == 2 for item in displays)


def test_notebook_executor_consumes_explicit_file_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nbformat = pytest.importorskip("nbformat")
    preprocessor = pytest.importorskip("nbconvert.preprocessors").ExecutePreprocessor

    path = tmp_path / "fixture.ipynb"
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("2 + 2")])
    nbformat.write(notebook, path)
    original = path.read_bytes()
    settings = []

    def preprocess(self: Any, nb: Any) -> tuple[Any, dict[str, Any]]:
        settings.append((self.kernel_name, self.timeout))
        nb.cells[0].execution_count = 1
        return nb, {}

    monkeypatch.setattr(preprocessor, "preprocess", preprocess)
    result = execute_notebook(path, kernel_name="test-kernel", timeout=17)
    assert settings == [("test-kernel", 17)]
    assert result.cells[0].execution_count == 1
    assert path.read_bytes() == original
