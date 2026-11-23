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
from flagquantum.core.ir.program.register_types import unsigned_register_type
from flagquantum.core.ir.program.registers import RegisterSide
from flagquantum.errors import SerializationError
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_modular_multiplication_component import source_stream
from tests.unit.test_multi_control_component import MeasurementSlots
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
_PREFIX = "qualtran.bloqs.mod_arithmetic.mod_multiplication."


def _resolve(source: Any) -> ProgramRecord:
    program = resolve_component_program(
        _library(source, 0), 0, modular_multiplication_factories()
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
    "kind,inverse",
    (
        ("SingleWindowModMul", False),
        ("_DirtyOutOfPlaceMontgomeryModMulImpl", False),
        ("DirtyOutOfPlaceMontgomeryModMul", False),
        ("DirtyOutOfPlaceMontgomeryModMul", True),
    ),
)
def test_window_constructor_delegation_and_original_wire_loss(
    kind: str, inverse: bool
) -> None:
    module = import_module(_PREFIX[:-1])
    arguments: dict[str, Any] = {"bitsize": 4, "window_size": 2, "mod": 5}
    if kind == "DirtyOutOfPlaceMontgomeryModMul":
        arguments["uncompute"] = inverse
    source = getattr(module, kind)(**arguments)
    factory = modular_multiplication_factories()[_PREFIX + kind]
    direct = (
        single_window_program(4, 2, 5)
        if kind == "SingleWindowModMul"
        else (
            windowed_multiplication_program(4, 2, 5)
            if kind.startswith("_")
            else montgomery_multiplication_program(4, 2, 5, adjoint=inverse)
        )
    )
    program = factory("test_product", None, arguments)
    assert program == _rename_program(direct, "test_product")
    assert isinstance(program, ProgramRecord) and verify_program_record(program) == ()
    # This is a concrete source codec counterexample, not repaired metadata.
    with pytest.raises(SerializationError, match="signature"):
        _resolve(source)


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
