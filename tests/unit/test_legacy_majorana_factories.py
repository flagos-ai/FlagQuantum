"""Majorana constructors preserve source calls, complex phases and clean work."""

from dataclasses import replace
from hashlib import sha1
from importlib.metadata import version
from itertools import product
from math import sqrt
from pathlib import Path

import pytest
import torch

from flagquantum import Circuit
from flagquantum.algorithms.primitives._block_binding import _BoundBlockOperation
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.compiler.registers.register_binding import (
    BoundRegisterProgram,
    bind_register_program,
)
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.ecosystem.cirq._majorana_library import majorana_library_factories
from flagquantum.errors import SerializationError
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_multiplexed_source_binding import logical_operator

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _bind_used_workspace(
    record: ProgramRecord, inputs: tuple[tuple[int, ...], ...]
) -> BoundRegisterProgram:
    """Retain the full allocation trace without simulating untouched pool wires."""
    logical = sum(map(len, inputs))
    profile = bind_register_program(
        record, inputs, clean=tuple(range(logical, logical + 24))
    )
    last = max(
        (wire for event in profile.events for wire in event.wires),
        default=logical - 1,
    )
    bound = bind_register_program(record, inputs, clean=tuple(range(logical, last + 1)))
    assert (
        bound.circuit.to_json()
        == replace(
            profile.circuit,
            n_wires=bound.circuit.n_wires,
            shape=bound.circuit.shape,
        ).to_json()
    )
    assert bound.events == profile.events
    return bound


@pytest.fixture(scope="module", autouse=True)
def pinned_majorana() -> None:
    import kickmix
    from qualtran.bloqs.multiplexers import selected_majorana_fermion

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    for filename, digest in (
        ("selected_majorana_fermion.py", "656373799b28c6838bc3528733c26f0fba0adacf"),
        (
            "selected_majorana_fermion_test.py",
            "d0b964420757f48f6f600d3d8e555855c1b6843c",
        ),
    ):
        raw = (Path(selected_majorana_fermion.__file__).parent / filename).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest


@pytest.mark.parametrize("width,size", ((2, 4), (3, 8), (4, 9)))
@pytest.mark.parametrize("pauli", ("X", "Y"))
@pytest.mark.parametrize("inverse", (False, True))
def test_complete_original_constructor_grid_and_depths(
    width: int, size: int, pauli: str, inverse: bool
) -> None:
    cirq = pytest.importorskip("cirq")
    from qualtran import BQUInt, Register
    from qualtran.bloqs.multiplexers.selected_majorana_fermion import (
        SelectedMajoranaFermion,
    )

    forward = SelectedMajoranaFermion(
        Register("selection", BQUInt(width, size)), target_gate=getattr(cirq, pauli)
    )
    source = forward.adjoint() if inverse else forward
    for depth in (0, 1):
        record = resolve_component_program(
            _library(source, depth), 0, majorana_library_factories()
        )
        assert verify_program_record(record) == () and record.functions[0].blocks
        compare_layers(record, source, atomic_leaves(record))


@pytest.mark.parametrize("gate_name", ("X", "Y", "T", "Rx", "Matrix"))
@pytest.mark.parametrize("controls", (0, 1, 2, 3))
@pytest.mark.parametrize("inverse", (False, True))
def test_complete_complex_operator_and_every_raw_history(
    gate_name: str, controls: int, inverse: bool
) -> None:
    cirq = pytest.importorskip("cirq")
    from qualtran import BQUInt, QAny, Register
    from qualtran.bloqs.multiplexers.selected_majorana_fermion import (
        SelectedMajoranaFermion,
    )

    gate = (
        cirq.rx(0.37)
        if gate_name == "Rx"
        else (
            cirq.MatrixGate(cirq.unitary(cirq.H**0.3))
            if gate_name == "Matrix"
            else getattr(cirq, gate_name)
        )
    )
    forward = SelectedMajoranaFermion(
        Register("address", BQUInt(1, 2)),
        control_regs=(Register("enable", QAny(controls)),) if controls else (),
        target_gate=gate,
    )
    source = forward.adjoint() if inverse else forward
    record = resolve_component_program(
        _library(source, 0), 0, majorana_library_factories()
    )
    composite = source.decompose_bloq()
    assert _native_trace(record) == _source_trace(
        _source(Signature(record.functions[0].registers or ())), composite
    )
    terminals = atomic_leaves(record)
    if gate_name == "Rx" or controls == 3:
        from qualtran import DecomposeNotImplementedError, DecomposeTypeError

        for instance, _, _ in composite.iter_bloqsoqs():
            if type(instance.bloq).__name__ == "Controlled" and type(
                instance.bloq.subbloq
            ).__name__ in ("Rx", "XGate"):
                with pytest.raises((DecomposeTypeError, DecomposeNotImplementedError)):
                    instance.bloq.decompose_bloq()
                callee = record.functions[0].blocks[0].operations[instance.i].callee
                assert callee is not None
                terminals.add(callee)
    if gate_name != "Matrix":
        compare_layers(record, source, terminals)
    cursor = 0
    inputs = []
    for register in record.functions[0].registers or ():
        width = int(register.total_bits())
        inputs.append(tuple(range(cursor, cursor + width)))
        cursor += width
    bound = _bind_used_workspace(record, tuple(inputs))
    assert bound.outputs == tuple(inputs)
    expected = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    torch.testing.assert_close(
        logical_operator(bound.circuit, cursor), expected, atol=5e-11, rtol=0
    )
    index = torch.arange(1 << cursor, dtype=torch.float64)
    ket = torch.complex(index + 1, (index % 7 - 3) * index)
    ket /= ket.norm()
    work = bound.circuit.n_wires - cursor
    initial = torch.zeros(1 << bound.circuit.n_wires, dtype=torch.complex128)
    initial[:: 1 << work] = ket
    final = torch.zeros_like(initial)
    final[:: 1 << work] = expected @ ket
    measurements = sum(g.name == "measure" for g in bound.circuit.instructions)
    for history in product((0, 1), repeat=measurements):
        torch.testing.assert_close(
            forced_branch(initial, bound.circuit.instructions, history)
            * sqrt(2**measurements),
            final,
            atol=6e-11,
            rtol=0,
        )
    appended = (
        Circuit(bound.circuit.n_wires)
        .append(_BoundBlockOperation(bound.circuit.instructions, bound.circuit.n_wires))
        .to_ir()
    )
    assert appended.to_dict()["instructions"] == bound.circuit.to_dict()["instructions"]
    assert appended.from_json(appended.to_json()).to_json() == appended.to_json()


def test_source_free_compilation_and_sdk_free_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    cirq = pytest.importorskip("cirq")
    from qualtran import BQUInt, Register
    from qualtran.bloqs.multiplexers.selected_majorana_fermion import (
        SelectedMajoranaFermion,
    )

    source = SelectedMajoranaFermion(
        Register("selection", BQUInt(1, 2)), target_gate=cirq.T
    )
    library = _library(source.adjoint(), 0)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "qualtran", None)
        record = resolve_component_program(library, 0, majorana_library_factories())
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "qualtran", None)
        patch.setitem(sys.modules, "cirq", None)
        bound = _bind_used_workspace(record, ((0,), (1,), (2, 3)))
        matrix = logical_operator(bound.circuit, 4)
    torch.testing.assert_close(
        matrix,
        torch.as_tensor(source.adjoint().tensor_contract(), dtype=torch.complex128),
        atol=4e-11,
        rtol=0,
    )


def test_source_boundaries_are_not_silently_corrected() -> None:
    cirq = pytest.importorskip("cirq")
    from qualtran import BQUInt, QBit, Register
    from qualtran.bloqs.multiplexers.select_pauli_lcu import _select_pauli_lcu
    from qualtran.bloqs.multiplexers.selected_majorana_fermion import (
        SelectedMajoranaFermion,
    )

    separate = SelectedMajoranaFermion(
        Register("selection", BQUInt(1, 2)),
        control_regs=(Register("first", QBit()), Register("second", QBit())),
        target_gate=cirq.X,
    )
    with pytest.raises(SerializationError, match="dirty accumulator"):
        resolve_component_program(
            _library(separate, 0), 0, majorana_library_factories()
        )
    with pytest.raises(ValueError, match="Object arrays"):
        _library(_select_pauli_lcu.make(), 0)


@pytest.mark.parametrize("inverse", (False, True))
@pytest.mark.parametrize("singleton", (False, True))
def test_nested_axes_and_controlled_singleton_source_graph_and_operator(
    inverse: bool, singleton: bool
) -> None:
    cirq = pytest.importorskip("cirq")
    from qualtran import BQUInt, Register
    from qualtran.bloqs.multiplexers.selected_majorana_fermion import (
        SelectedMajoranaFermion,
    )

    selections = (
        (Register("address", BQUInt(1, 1)),)
        if singleton
        else (Register("row", BQUInt(1, 2)), Register("column", BQUInt(1, 2)))
    )
    forward = SelectedMajoranaFermion(selections, target_gate=cirq.Y)
    source = forward.adjoint() if inverse else forward
    record = resolve_component_program(
        _library(source, 0), 0, majorana_library_factories()
    )
    compare_layers(record, source, atomic_leaves(record))
    inputs = []
    cursor = 0
    for register in record.functions[0].registers or ():
        width = int(register.total_bits())
        inputs.append(tuple(range(cursor, cursor + width)))
        cursor += width
    bound = _bind_used_workspace(record, tuple(inputs))
    torch.testing.assert_close(
        logical_operator(bound.circuit, cursor),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=5e-11,
        rtol=0,
    )


def test_symbolic_address_declaration_and_original_symbolic_extent_refusal() -> None:
    import sympy
    from qualtran import BQUInt, Register
    from qualtran.bloqs.multiplexers.selected_majorana_fermion import (
        SelectedMajoranaFermion,
    )

    symbolic = SelectedMajoranaFermion(
        Register("address", BQUInt(sympy.Symbol("n"), 2))
    )
    for source in (symbolic, symbolic.adjoint()):
        record = resolve_component_program(
            _library(source, 0), 0, majorana_library_factories()
        )
        assert verify_program_record(record) == () and not record.functions[0].blocks
    with pytest.raises(ValueError, match="Symbolic iteration"):
        _library(
            SelectedMajoranaFermion(Register("address", BQUInt(2, sympy.Symbol("N")))),
            0,
        )


def test_omitted_constructor_defaults_match_the_original_explicit_defaults() -> None:
    from qualtran import BQUInt, Register
    from qualtran.bloqs.multiplexers.selected_majorana_fermion import (
        SelectedMajoranaFermion,
    )

    source = SelectedMajoranaFermion(Register("address", BQUInt(1, 2)))
    library = _library(source, 0)
    arguments = {
        argument.name: argument.value
        for argument in library.entries[0].declaration.arguments
    }
    factory = majorana_library_factories()[library.entries[0].declaration.name]
    record = factory("main", None, {"selection_regs": arguments["selection_regs"]})
    assert isinstance(record, ProgramRecord)
    expected = resolve_component_program(library, 0, majorana_library_factories())
    assert matches_component_definition(record, "main", expected)
    with pytest.raises(SerializationError, match="owned gate payload"):
        factory(
            "main",
            None,
            {"selection_regs": arguments["selection_regs"], "target_gate": None},
        )
