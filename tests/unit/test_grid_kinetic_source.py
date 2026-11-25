"""Literal source kinetic constructors equal actual stored typed source graphs."""

from dataclasses import replace
from hashlib import sha1
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest
import sympy

from flagquantum.compiler.formats.legacy_bookkeeping_factories import (
    bookkeeping_factories,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_grid_kinetic_factories import (
    grid_kinetic_factories,
)
from flagquantum.compiler.formats.legacy_grid_qvr_factories import grid_qvr_factories
from flagquantum.compiler.formats.legacy_product_model_factories import (
    product_model_factories,
)
from flagquantum.compiler.registers.arithmetic.grid_kinetic_signature import (
    grid_kinetic_signature,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.register_types import unsigned_register_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.errors import SerializationError
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_subtraction_foreign_file import native_signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
_KEY = "qualtran.bloqs.chemistry.trotter.grid_ham.kinetic.KineticEnergy"


def _children() -> dict[str, Any]:
    return {
        **product_model_factories(),
        **grid_qvr_factories(),
        **bookkeeping_factories(),
    }


def _normalized(program: ProgramRecord) -> tuple[FunctionRecord, ...]:
    roles = {}
    for function in program.functions:
        assert function.registers is not None
        names = tuple(port.name for port in function.registers)
        roles[function.name] = {
            ("system",): "root",
            ("input", "result"): "sum",
            ("phi",): "phase",
            ("reg",): "free",
        }[names]
    return tuple(
        sorted(
            (
                replace(
                    function,
                    name=roles[function.name],
                    blocks=tuple(
                        replace(
                            block,
                            operations=tuple(
                                (
                                    replace(operation, callee=roles[operation.callee])
                                    if operation.callee
                                    else operation
                                )
                                for operation in block.operations
                            ),
                        )
                        for block in function.blocks
                    ),
                )
                for function in program.functions
            ),
            key=lambda f: f.name,
        )
    )


@pytest.mark.parametrize("electrons,grid", ((1, 1), (2, 3), (3, 8), (12, 17)))
@pytest.mark.parametrize("depth", (0, 1, 2))
def test_shallow_factory_matches_actual_complete_source_graph(
    electrons: int, grid: int, depth: int
) -> None:
    from qualtran.bloqs.chemistry.trotter.grid_ham.kinetic import KineticEnergy

    source = KineticEnergy(electrons, grid)
    actual = resolve_component_program(
        _library(source, depth), 0, grid_kinetic_factories()
    )
    reference = resolve_component_program(_library(source, 2), 0, _children())
    assert verify_program_record(actual) == ()
    # Compare every root operand/result/return SSA identity, ordered call,
    # nominal child signature and Free operation after names alone are normalized.
    assert _normalized(actual) == _normalized(reference)
    assert actual.functions[0].registers == tuple(native_signature(source.signature))
    assert actual.functions[0].registers == tuple(
        grid_kinetic_signature(electrons, grid)
    )
    original_calls = tuple(source.decompose_bloq().iter_bloqsoqs())
    assert [type(instance.bloq).__name__ for instance, _, _ in original_calls] == [
        "SumOfSquares",
        "QuantumVariableRotation",
        "Free",
    ] * electrons
    assert len(actual.functions[0].blocks[0].operations) == 3 * electrons
    with pytest.raises(ValueError, match="bodies"):
        bind_register_program(actual, ())


@pytest.mark.parametrize("grid", (0, 1, 3))
def test_empty_electron_array_retains_original_identity_without_unused_leaves(
    grid: int,
) -> None:
    from qualtran.bloqs.chemistry.trotter.grid_ham.kinetic import KineticEnergy

    source = KineticEnergy(0, grid)
    assert tuple(source.decompose_bloq().iter_bloqsoqs()) == ()
    result = resolve_component_program(_library(source, 0), 0, grid_kinetic_factories())
    assert verify_program_record(result) == ()
    assert len(result.functions) == 1
    assert result.functions[0].registers == tuple(native_signature(source.signature))
    assert not result.functions[0].blocks[0].operations
    assert result.functions[0].blocks[0].return_values == ()


def test_symbolic_electrons_stay_a_logical_declaration_and_symbolic_grid_refuses() -> (
    None
):
    from qualtran.bloqs.chemistry.trotter.grid_ham.kinetic import KineticEnergy

    n = sympy.Symbol("n")
    source = KineticEnergy(n, 3)
    with pytest.raises(TypeError):
        source.decompose_bloq()
    result = resolve_component_program(_library(source, 0), 0, grid_kinetic_factories())
    assert verify_program_record(result) == ()
    assert result.functions[0].registers == tuple(native_signature(source.signature))
    assert not result.functions[0].blocks
    with pytest.raises(AttributeError, match="bit_length"):
        _ = KineticEnergy(2, n).signature
    with pytest.raises(SerializationError, match="concrete grid"):
        grid_kinetic_factories()[_KEY]("bad", None, {"num_elec": 2, "num_grid": n})


def test_original_example_docspec_and_clean_free_are_not_native_erasure() -> None:
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.arithmetic import SumOfSquares
    from qualtran.bloqs.chemistry.trotter.grid_ham.kinetic import (
        _KINETIC_ENERGY,
        KineticEnergy,
        _kinetic_energy,
    )
    from qualtran.bloqs.chemistry.trotter.grid_ham.qvr import QuantumVariableRotation

    source = _kinetic_energy()
    assert source == KineticEnergy(12, 17)
    assert _KINETIC_ENERGY.bloq_cls is KineticEnergy and _KINETIC_ENERGY.examples == (
        _kinetic_energy,
    )
    for child in (SumOfSquares(6, 3), QuantumVariableRotation(14)):
        with pytest.raises(DecomposeNotImplementedError):
            child.decompose_bloq()
    program = resolve_component_program(
        _library(source, 0), 0, grid_kinetic_factories()
    )
    free = next(
        function
        for function in program.functions
        if function.registers and function.registers[0].name == "reg"
    )
    (operation,) = free.blocks[0].operations
    assert operation.opcode == "free_register" and operation.workspace_dirty is False
    # The independent mathematical operation uses square/phase/square. Source
    # transcription preserves square/phase/Free instead of rewriting its meaning.
    assert len(program.functions[0].blocks[0].operations) == 36


def test_invalid_metadata_and_sdk_free_native_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    factory = grid_kinetic_factories()[_KEY]
    for args in (
        {},
        {"num_elec": 2, "num_grid": True},
        {"num_elec": "2", "num_grid": 3},
        {"num_elec": 2, "num_grid": 3, "radians_per_unit": 0.1},
    ):
        with pytest.raises(SerializationError):
            factory("bad", None, args)
    with pytest.raises(SerializationError, match="signature"):
        factory(
            "bad",
            (Register("system", unsigned_register_type(3), (2, 3)),),
            {"num_elec": 2, "num_grid": 3},
        )
    original = builtins.__import__

    def no_source(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"qualtran", "cirq"}:
            raise ImportError(name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_source)
    program = factory("kinetic", None, {"num_elec": 2, "num_grid": 3})
    assert isinstance(program, ProgramRecord) and verify_program_record(program) == ()


def test_complete_original_source_and_test_assets_and_real_extension() -> None:
    import kickmix

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and callable(kickmix.Simulator)
    root = (
        Path(__file__).resolve().parents[3]
        / "ftqc-reference-env/qualtran/bloqs/chemistry/trotter/grid_ham"
    )
    for name, digest in (
        ("kinetic.py", "50dd860c59c0e29ba982654c544d06b75b7e464d"),
        ("kinetic_test.py", "efda833f6871240f348f0cf8f0027e1839158c61"),
    ):
        raw = (root / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest
