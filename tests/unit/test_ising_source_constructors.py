"""Exercise complete Ising source constructors through owned Program execution."""

import sys
from collections import Counter
from hashlib import sha1
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest
import sympy
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.formats.legacy_bookkeeping_factories import (
    bookkeeping_factories,
)
from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_fixed_gate_factories import (
    fixed_gate_factories,
)
from flagquantum.compiler.formats.legacy_ising_factories import ising_factories
from flagquantum.compiler.formats.legacy_rotation_factories import rotation_factories
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.errors import SerializationError
from flagquantum.simulation.unitary import get_unitary
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_ising_evolution import _source_instructions
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_multi_control_component import atomic_leaves

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module")
def source_module() -> Any:
    import kickmix
    from qualtran.bloqs.chemistry.trotter.ising import unitaries

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    folder = Path(unitaries.__file__).parent
    for filename, digest in (
        ("unitaries.py", "c9f9b14ce57038fc5e88ffff1d54e4f66a5376fa"),
        ("unitaries_test.py", "542719fdd70750308bcc73dfd537a51b661dacd2"),
    ):
        raw = (folder / filename).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest
    return unitaries


def factories() -> dict[str, ComponentFactory]:
    return {
        **bookkeeping_factories(),
        **fixed_gate_factories(),
        **rotation_factories(),
        **ising_factories(),
    }


@pytest.mark.parametrize(
    "kind,sites", [("X", 1), ("X", 3), ("ZZ", 2), ("ZZ", 3), ("ZZ", 4)]
)
@pytest.mark.parametrize("angle", [0.04, -0.371, 0.0])
@pytest.mark.parametrize("depth", [0, 1, 2])
def test_source_constructor_graph_stream_complex_operator_and_resources(
    source_module: Any, kind: str, sites: int, angle: float, depth: int
) -> None:
    source = getattr(source_module, f"Ising{kind}Unitary")(sites, angle, eps=1e-6)
    library = _library(source, depth)
    wire = serialize_component_library(library)
    assert serialize_component_library(deserialize_component_library(wire)) == wire
    program = resolve_component_program(library, 0, factories())
    assert verify_program_record(program) == ()
    pairs = compare_layers(program, source, atomic_leaves(program))
    for node in pairs.values():
        if type(node).__name__ in {"Rx", "Rz"}:
            assert node.angle == angle
            assert node.eps == (1e-11 if kind == "X" else source.eps)
    wires = tuple(reversed(range(1, sites + 1)))
    bound = bind_register_program(program, (wires,), clean=(sites + 1,))
    assert bound.outputs == (wires,) and not bound.events
    assert bound.circuit.instructions == _source_instructions(source, wires)
    counts = Counter(g.name for g in bound.circuit.instructions)
    assert counts == ({"rx": sites} if kind == "X" else {"cx": 2 * sites, "rz": sites})
    contiguous = bind_register_program(program, (tuple(range(sites)),))
    operation = FieldOperation(sites, contiguous.circuit.instructions)
    torch.testing.assert_close(
        get_unitary(Circuit(sites).append(operation), dtype=torch.complex128),
        torch.from_numpy(source.tensor_contract()),
        atol=1e-12,
        rtol=0,
    )
    reverse = Circuit(sites).append(operation).adjoint()
    assert reverse.to_ir().instructions == _source_instructions(
        source.adjoint(), tuple(range(sites))
    )
    torch.testing.assert_close(
        get_unitary(reverse, dtype=torch.complex128),
        torch.from_numpy(source.tensor_contract()).mH,
        atol=1e-12,
        rtol=0,
    )
    # An entangled complex spectator and an unused clean pool test ownership.
    count = sites + 2
    state = torch.arange(1, (1 << (count - 1)) + 1, dtype=torch.float64)
    state = torch.complex(state, state.flip(0)).to(torch.complex128)
    state = state / torch.linalg.vector_norm(state)
    initial = torch.zeros(1 << count, dtype=torch.complex128)
    initial[::2] = state
    native = (
        Circuit(count, inputs=initial, dtype=torch.complex128)
        .append(FieldOperation(count, bound.circuit.instructions))
        .state()
        .flatten()
    )
    expected = (
        Circuit(count, inputs=initial, dtype=torch.complex128)
        .append(FieldOperation(count, _source_instructions(source, wires)))
        .state()
        .flatten()
    )
    torch.testing.assert_close(native, expected, atol=1e-12, rtol=0)
    assert torch.count_nonzero(native[1::2]) == 0


@pytest.mark.parametrize("kind", ["X", "ZZ"])
def test_original_examples_docs_and_precision(source_module: Any, kind: str) -> None:
    example = getattr(source_module, f"_ising_{kind.lower()}")
    source = example.make()
    assert (source.nsites, source.angle) == (3, 0.04)
    assert getattr(source_module, f"_ISING_{kind}_UNITARY_DOC").examples == (example,)
    program = resolve_component_program(_library(source, 0), 0, factories())
    pairs = compare_layers(program, source, atomic_leaves(program))
    rotations = [node for node in pairs.values() if type(node).__name__ in {"Rx", "Rz"}]
    assert len(rotations) == 1
    assert rotations[0].eps == (1e-11 if kind == "X" else source.eps)


@pytest.mark.parametrize("kind", ["X", "ZZ"])
def test_symbolic_angle_uses_existing_parameter_binding(
    source_module: Any, kind: str
) -> None:
    angle = sympy.Symbol("angle")
    source = getattr(source_module, f"Ising{kind}Unitary")(3, angle)
    program = resolve_component_program(_library(source, 0), 0, factories())
    compare_layers(program, source, atomic_leaves(program))
    bound = bind_register_program(program, ((0, 1, 2),))
    assert len(bound.circuit.instructions) == (3 if kind == "X" else 9)
    circuit = Circuit(3).append(FieldOperation(3, bound.circuit.instructions))
    for value in (0.04, -0.371):
        numeric = getattr(source_module, f"Ising{kind}Unitary")(3, value)
        concrete = circuit.bind_parameters({"angle": value})
        assert concrete.to_ir().instructions == _source_instructions(numeric, (0, 1, 2))


def test_one_site_zz_retains_source_logical_failure(source_module: Any) -> None:
    from qualtran._infra.composite_bloq import BloqError

    source = source_module.IsingZZUnitary(1, 0.04)
    program = resolve_component_program(_library(source, 0), 0, factories())
    assert not program.functions[0].blocks
    with pytest.raises(BloqError, match="not an available Soquet.*CNOT.target"):
        source.decompose_bloq()
    with pytest.raises(ValueError, match="component bodies"):
        bind_register_program(program, ((0,),))


@pytest.mark.parametrize("kind", ["X", "ZZ"])
@pytest.mark.parametrize("sites", [0, sympy.Symbol("n", integer=True, positive=True)])
def test_nonexecutable_source_dimensions_remain_declarations(
    source_module: Any, kind: str, sites: Any
) -> None:
    source = getattr(source_module, f"Ising{kind}Unitary")(sites, 0.04)
    library = _library(source, 0)
    program = resolve_component_program(library, 0, factories())
    assert program.functions[0].registers == library.entries[0].declaration.registers
    assert not program.functions[0].blocks
    assert verify_program_record(program) == ()


def test_native_constructors_do_not_import_optional_sdks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for module in ("qualtran", "cirq", "kickmix", "quimb"):
        monkeypatch.setitem(sys.modules, module, None)
    for factory in ising_factories().values():
        program = factory("main", None, {"nsites": 3, "angle": 0.04})
        assert isinstance(program, ProgramRecord)
        assert verify_program_record(program) == ()
        assert not bind_register_program(program, ((0, 1, 2),)).events


def test_constructor_metadata_validation() -> None:
    factory = next(iter(ising_factories().values()))
    with pytest.raises(SerializationError, match="requires"):
        factory("main", None, {"nsites": 3})
    with pytest.raises(SerializationError, match="signature"):
        factory("main", (), {"nsites": 3, "angle": 0.04})
    with pytest.raises(SerializationError, match="real scalar"):
        factory("main", None, {"nsites": 3, "angle": "invalid"})
