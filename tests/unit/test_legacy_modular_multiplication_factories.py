"""Complete modular product constructors reuse certified gates and ownership."""

from collections import Counter
from importlib import import_module
from typing import Any

import pytest
import sympy

from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_modular_multiplication_factories import (
    modular_multiplication_factories,
)
from flagquantum.compiler.formats.legacy_word_factories import _rename_program
from flagquantum.compiler.registers.modular.modular_doubling_component import (
    modular_doubling_program,
)
from flagquantum.compiler.registers.modular.modular_multiplication_component import (
    modular_multiplication_program,
)
from flagquantum.compiler.registers.modular.montgomery_multiplication_component import (
    montgomery_multiplication_program,
)
from flagquantum.compiler.registers.modular.single_window_component import (
    single_window_program,
)
from flagquantum.compiler.registers.modular.windowed_multiplication_component import (
    windowed_multiplication_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.register_types import (
    montgomery_register_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.register_verifier import (
    executable_register_width,
    register_port_types,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.errors import SerializationError
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_modular_multiplication_component import source_stream
from tests.unit.test_multi_control_component import MeasurementSlots
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
_PREFIX = "qualtran.bloqs.mod_arithmetic.mod_multiplication."


def _resolve(source: Any, depth: int = 0) -> ProgramRecord:
    program = resolve_component_program(
        _library(source, depth), 0, modular_multiplication_factories()
    )
    assert verify_program_record(program) == ()
    return program


@pytest.mark.parametrize("doubling", (False, True))
@pytest.mark.parametrize("montgomery", (False, True))
def test_word_constructor_exact_source_stream_output_aliases_and_resources(
    doubling: bool, montgomery: bool
) -> None:
    from qualtran import QMontgomeryUInt, QUInt
    from qualtran.bloqs.mod_arithmetic.mod_multiplication import CModMulK, ModDbl
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    dtype = QMontgomeryUInt(2, 3) if montgomery else QUInt(2)
    source = ModDbl(dtype, 3) if doubling else CModMulK(dtype, 2, 3)
    program = _resolve(source)
    inputs = ((0, 1),) if doubling else ((0,), (1, 2))
    options: dict[str, Any] = (
        {"input_ownership": ((0, False),), "allow_workspace_outputs": True}
        if doubling
        else {}
    )
    bound = bind_register_program(program, inputs, clean=tuple(range(3, 40)), **options)
    registers = program.functions[0].registers
    assert registers is not None
    native_dtype = registers[-1].dtype
    direct = (
        modular_doubling_program(native_dtype, 3)
        if doubling
        else modular_multiplication_program(native_dtype, 3, 2)
    )
    assert bound == bind_register_program(
        direct, inputs, clean=tuple(range(3, 40)), **options
    )
    ports = dict(zip(("x",) if doubling else ("ctrl", "x"), inputs, strict=True))
    if doubling:
        allocations = tuple(
            q for e in bound.events if e.kind == "allocate" for q in e.wires
        )
        expected = reference_arithmetic_gates(
            source,
            ports,
            allocations,
            MeasurementSlots(bound.circuit.instructions),
            reuse_allocations=False,
            preserve_components=True,
            expected_outputs={"x": bound.outputs[0]},
        )
    else:
        expected = source_stream(source, bound, ports)
    assert_source(bound.circuit.instructions, expected)
    counts = Counter(g.name for g in bound.circuit.instructions)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count(ts_per_cswap=7)
    assert counts["measure"] == counts["reset"] == model.measurement


@pytest.mark.parametrize(
    "name", ("_moddbl_small", "_moddbl_large", "_modmul", "_modmul_symb")
)
def test_all_word_examples_resolve_with_original_signature(name: str) -> None:
    source = getattr(import_module(_PREFIX[:-1]), name)()
    assert _resolve(source).functions[0].registers is not None


@pytest.mark.parametrize(
    "kind,inverse,profile",
    (
        ("SingleWindowModMul", False, (4, 2, 5)),
        ("_DirtyOutOfPlaceMontgomeryModMulImpl", False, (4, 2, 5)),
        ("DirtyOutOfPlaceMontgomeryModMul", False, (4, 2, 5)),
        ("DirtyOutOfPlaceMontgomeryModMul", True, (4, 2, 5)),
        ("SingleWindowModMul", False, (1, 1, 1)),
        ("_DirtyOutOfPlaceMontgomeryModMulImpl", False, (3, 1, 3)),
        ("DirtyOutOfPlaceMontgomeryModMul", False, (3, 1, 3)),
        ("DirtyOutOfPlaceMontgomeryModMul", True, (3, 1, 3)),
    ),
)
def test_window_constructor_delegation_and_original_wire_loss(
    kind: str, inverse: bool, profile: tuple[int, int, int]
) -> None:
    module = import_module(_PREFIX[:-1])
    width, window, modulus = profile
    arguments: dict[str, Any] = {
        "bitsize": width,
        "window_size": window,
        "mod": modulus,
    }
    if kind == "DirtyOutOfPlaceMontgomeryModMul":
        arguments["uncompute"] = inverse
    source = getattr(module, kind)(**arguments)
    factory = modular_multiplication_factories()[_PREFIX + kind]
    direct = (
        single_window_program(width, window, modulus)
        if kind == "SingleWindowModMul"
        else (
            windowed_multiplication_program(width, window, modulus)
            if kind.startswith("_")
            else montgomery_multiplication_program(
                width, window, modulus, adjoint=inverse
            )
        )
    )
    program = factory("test_product", None, arguments)
    assert program == _rename_program(direct, "test_product")
    assert isinstance(program, ProgramRecord) and verify_program_record(program) == ()
    library = _library(source, 0)
    declared = library.entries[0].declaration.registers
    assert declared is not None
    assert declared != direct.functions[0].registers
    # The source wire still loses modulus metadata. The outer ports retain
    # that loss; explicit zero-gate views bridge to the unchanged child.
    for register in declared:
        if register.dtype.encoding == "montgomery":
            assert register.dtype.parameters == ()
    resolved = _resolve(source)
    assert resolved.functions[0].registers == declared
    for depth in (1, 2):
        assert _resolve(source, depth) == resolved
    widths = tuple(
        executable_register_width(dtype)
        for dtype in register_port_types(direct.functions[0], RegisterSide.LEFT)
    )
    inputs = []
    logical = 0
    for width in widths:
        inputs.append(tuple(range(logical, logical + width)))
        logical += width
    options: dict[str, Any] = {
        "clean": tuple(range(logical + 1, logical + 100)),
        "borrowed": (logical,),
        "allow_workspace_outputs": kind.startswith("_")
        or kind == "DirtyOutOfPlaceMontgomeryModMul"
        and not inverse,
    }
    if inverse or kind.startswith("_"):
        options["input_ownership"] = tuple(
            (wire, False) for port in inputs[2:] for wire in port
        )
    actual = bind_register_program(resolved, tuple(inputs), **options)
    expected = bind_register_program(direct, tuple(inputs), **options)
    assert actual.circuit == expected.circuit and actual.outputs == expected.outputs
    # Added view operations change expanded logical positions, not physical
    # lifetime order or the emitted instruction offset of any ownership event.
    assert [
        (event.kind, event.wires, event.dirty, event.instruction_offset)
        for event in actual.events
    ] == [
        (event.kind, event.wires, event.dirty, event.instruction_offset)
        for event in expected.events
    ]


def test_all_symbolic_window_declarations_retain_source_shapes_and_directions() -> None:
    from flagquantum.core.ir.program import FunctionRecord

    n, w, p = sympy.symbols("n w p", positive=True, integer=True)
    factories = modular_multiplication_factories()
    for kind in (
        "SingleWindowModMul",
        "_DirtyOutOfPlaceMontgomeryModMulImpl",
        "DirtyOutOfPlaceMontgomeryModMul",
    ):
        arguments: dict[str, Any] = {"bitsize": n, "window_size": w, "mod": p}
        for inverse in (
            (False, True) if kind == "DirtyOutOfPlaceMontgomeryModMul" else (False,)
        ):
            if kind == "DirtyOutOfPlaceMontgomeryModMul":
                arguments["uncompute"] = inverse
            declaration = factories[_PREFIX + kind]("symbolic", None, arguments)
            assert isinstance(declaration, FunctionRecord) and not declaration.blocks
            assert declaration.registers is not None
            assert declaration.registers[0].shape_symbolic == (
                (w,) if kind == "SingleWindowModMul" else ()
            )
            if kind == "DirtyOutOfPlaceMontgomeryModMul":
                assert declaration.registers[-1].side is (
                    RegisterSide.LEFT if inverse else RegisterSide.RIGHT
                )
            source = getattr(import_module(_PREFIX[:-1]), kind)(**arguments)
            resolved = _resolve(source)
            assert (
                resolved.functions[0].registers
                == _library(source, 0).entries[0].declaration.registers
            )
            assert not resolved.functions[0].blocks
            with pytest.raises(ValueError, match="bodies"):
                bind_register_program(resolved, ())


@pytest.mark.parametrize(
    "changed",
    ("modulus", "encoding", "width", "classical", "name", "shape", "side", "count"),
)
def test_source_modulus_bridge_rejects_every_other_signature_change(
    changed: str,
) -> None:
    from qualtran.bloqs.mod_arithmetic.mod_multiplication import SingleWindowModMul

    arguments: dict[str, Any] = {"bitsize": 4, "window_size": 2, "mod": 5}
    declaration = (
        _library(SingleWindowModMul(**arguments), 0).entries[0].declaration.registers
    )
    assert declaration is not None
    registers = list(declaration)
    original = registers[1]
    dtype = {
        "modulus": montgomery_register_type(4, 7),
        "encoding": unsigned_register_type(4),
        "width": montgomery_register_type(5),
        "classical": RegisterType("montgomery", 4, 1),
    }.get(changed, original.dtype)
    registers[1] = Register(
        "other" if changed == "name" else original.name,
        dtype,
        (1,) if changed == "shape" else original.shape,
        RegisterSide.LEFT if changed == "side" else original.side,
    )
    if changed == "count":
        registers.pop()
    with pytest.raises(SerializationError, match="signature"):
        modular_multiplication_factories()[_PREFIX + "SingleWindowModMul"](
            "bad", tuple(registers), arguments
        )


def test_invalid_constructor_metadata_preserves_early_failure() -> None:
    factories = modular_multiplication_factories()
    dtype = unsigned_register_type(2)
    for arguments in (
        {},
        {"dtype": dtype, "mod": 3, "extra": 1},
        {"dtype": "bad", "mod": 3},
        {"dtype": dtype, "mod": "bad"},
    ):
        with pytest.raises(SerializationError):
            factories[_PREFIX + "ModDbl"]("bad", None, arguments)
    with pytest.raises(SerializationError, match="boolean"):
        factories[_PREFIX + "DirtyOutOfPlaceMontgomeryModMul"](
            "bad", None, {"bitsize": 4, "window_size": 2, "mod": 5, "uncompute": 2}
        )
    with pytest.raises(ValueError, match="odd modulus"):
        factories[_PREFIX + "ModDbl"]("bad", None, {"dtype": dtype, "mod": 2})
    with pytest.raises(ValueError, match="coprime"):
        factories[_PREFIX + "CModMulK"]("bad", None, {"dtype": dtype, "mod": 3, "k": 3})
