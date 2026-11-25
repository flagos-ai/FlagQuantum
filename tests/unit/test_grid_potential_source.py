"""Literal potential recipes preserve complete source graphs and default data."""

from dataclasses import replace
from hashlib import sha256
from typing import Any

import pytest

from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.registers.component_schedule import local_operation_order
from flagquantum.core.ir.program import ProgramRecord, ValueRef, verify_program_record
from flagquantum.ecosystem._grid_potential_component_library import (
    grid_potential_library_factories,
)
from tests.unit.test_legacy_arithmetic_factories import _library

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _children() -> dict[str, Any]:
    from flagquantum.compiler.formats.legacy_bookkeeping_factories import (
        bookkeeping_factories,
    )
    from flagquantum.compiler.formats.legacy_grid_inverse_sqrt_factories import (
        grid_inverse_sqrt_factories,
    )
    from flagquantum.compiler.formats.legacy_grid_qvr_factories import (
        grid_qvr_factories,
    )
    from flagquantum.compiler.formats.legacy_product_model_factories import (
        product_model_factories,
    )
    from flagquantum.compiler.formats.legacy_qrom_factories import qrom_factories
    from flagquantum.ecosystem._component_library import arithmetic_library_factories

    return {
        **bookkeeping_factories(),
        **grid_inverse_sqrt_factories(),
        **grid_qvr_factories(),
        **product_model_factories(),
        **qrom_factories(),
        **arithmetic_library_factories(),
    }


def _fingerprint(program: ProgramRecord) -> str:
    """Normalize names/SSA under the existing source logical scheduler only."""
    functions = {function.name: function for function in program.functions}
    memo: dict[str, str] = {}

    def visit(name: str) -> str:
        if name in memo:
            return memo[name]
        function = functions[name]
        blocks = []
        assert len(function.blocks) <= 1
        if function.blocks:
            block = function.blocks[0]
            refs: dict[ValueRef, ValueRef] = {}

            def define(value: ValueRef) -> ValueRef:
                assert value not in refs
                refs[value] = ValueRef("entry", len(refs), value.kind)
                return refs[value]

            arguments = tuple(define(value) for value in block.arguments)
            order = (
                local_operation_order(function, functions, logical_graph=True)
                if all(operation.opcode == "call" for operation in block.operations)
                else tuple(range(len(block.operations)))
            )
            operations = []
            for index in order:
                operation = block.operations[index]
                operands = tuple(refs[value] for value in operation.operands)
                results = tuple(define(value) for value in operation.results)
                operations.append(
                    replace(
                        operation,
                        operands=operands,
                        results=results,
                        parameters=tuple(
                            (key, refs[value]) for key, value in operation.parameters
                        ),
                        condition_clauses=tuple(
                            tuple(
                                replace(condition, value=refs[condition.value])
                                for condition in clause
                            )
                            for clause in operation.condition_clauses
                        ),
                        callee=visit(operation.callee) if operation.callee else None,
                    )
                )
            blocks.append(
                replace(
                    block,
                    name="entry",
                    arguments=arguments,
                    operations=tuple(operations),
                    return_values=tuple(
                        refs[value] for value in block.return_values or ()
                    ),
                )
            )
        canonical = replace(function, name="function", blocks=tuple(blocks))
        memo[name] = sha256(repr(canonical).encode()).hexdigest()
        return memo[name]

    return visit(program.functions[0].name)


def _pair(width: int = 1, poly: int = 3, target: int = 5) -> Any:
    from qualtran.bloqs.chemistry.trotter.grid_ham.inverse_sqrt import (
        build_qrom_data_for_poly_fit,
        get_inverse_square_root_poly_coeffs,
    )
    from qualtran.bloqs.chemistry.trotter.grid_ham.potential import PairPotential

    data = build_qrom_data_for_poly_fit(
        2 * width + 2, poly, get_inverse_square_root_poly_coeffs()
    )
    return PairPotential(
        width,
        tuple(tuple(int(v) for v in row) for row in data),
        poly_bitsize=poly,
        inv_sqrt_bitsize=target,
    )


@pytest.mark.parametrize("kind", ("pair", "total"))
@pytest.mark.parametrize("depth", (0, 1, 2))
def test_shallow_complete_recipe_equals_actual_source_graph(
    kind: str, depth: int
) -> None:
    from qualtran.bloqs.chemistry.trotter.grid_ham.potential import PotentialEnergy

    source = _pair() if kind == "pair" else PotentialEnergy(2, 1, poly_bitsize=3)
    actual = resolve_component_program(
        _library(source, depth), 0, grid_potential_library_factories()
    )
    reference = resolve_component_program(_library(source, 2), 0, _children())
    assert verify_program_record(actual) == ()
    assert _fingerprint(actual) == _fingerprint(reference)


@pytest.mark.parametrize("legacy_shims", (False, True))
@pytest.mark.parametrize("bank_length,zero", ((5, False), (16, True), (16, False)))
def test_complete_custom_banks_and_source_resource_categories(
    bank_length: int, zero: bool, legacy_shims: bool
) -> None:
    from qualtran.bloqs.chemistry.trotter.grid_ham.potential import PairPotential
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from flagquantum.algorithms.primitives.grid_potential_resource_model import (
        source_grid_pair_potential_cost,
    )
    from flagquantum.compiler._component_trace.resource_model import (
        estimate_component_resources,
    )

    data = tuple(
        tuple(0 if zero else (row + index) % 8 for index in range(bank_length))
        for row in range(4)
    )
    source = PairPotential(1, data, poly_bitsize=3, inv_sqrt_bitsize=5, label="Coulomb")
    actual = resolve_component_program(
        _library(source, 0), 0, grid_potential_library_factories()
    )
    reference = resolve_component_program(_library(source, 2), 0, _children())
    assert _fingerprint(actual) == _fingerprint(reference)
    native = estimate_component_resources(
        source_grid_pair_potential_cost(
            1, 3, 5, qrom_data=data, legacy_shims=legacy_shims
        )
    )
    expected = get_cost_value(source, QECGatesCost(legacy_shims=legacy_shims))
    for category in (
        "t",
        "toffoli",
        "and_bloq",
        "clifford",
        "rotation",
        "measurement",
        "cswap",
    ):
        assert getattr(native.gates, category) == getattr(expected, category)
    assert native.peak_logical_qubits == get_cost_value(source, QubitCount())


@pytest.mark.parametrize("example", ("_pair_potential", "_potential_energy"))
def test_complete_original_factories_and_docspecs(example: str) -> None:
    from qualtran.bloqs.chemistry.trotter.grid_ham import potential

    source = getattr(potential, example)()
    docs = (
        potential._PAIR_POTENTIAL
        if example == "_pair_potential"
        else potential._POTENTIAL_ENERGY
    )
    assert docs.bloq_cls is type(source)
    assert docs.examples == (getattr(potential, example),)
    # Full original example data and every nested native helper participate in
    # this comparison; a reduced coefficient table is not substituted.
    actual = resolve_component_program(
        _library(source, 0), 0, grid_potential_library_factories()
    )
    reference = resolve_component_program(_library(source, 2), 0, _children())
    assert verify_program_record(actual) == ()
    assert _fingerprint(actual) == _fingerprint(reference)


def test_source_graph_comparison_retains_bank_values_and_clean_free_semantics() -> None:
    from qualtran.bloqs.chemistry.trotter.grid_ham.potential import PairPotential

    source = _pair()
    original = resolve_component_program(
        _library(source, 0), 0, grid_potential_library_factories()
    )
    rows = [list(bank) for bank in source.qrom_data]
    rows[0][1] ^= 1
    changed = PairPotential(
        1, tuple(tuple(bank) for bank in rows), poly_bitsize=3, inv_sqrt_bitsize=5
    )
    mutated = resolve_component_program(
        _library(changed, 0), 0, grid_potential_library_factories()
    )
    assert _fingerprint(original) != _fingerprint(mutated)
    dirty = replace(
        original,
        functions=tuple(
            replace(
                function,
                blocks=tuple(
                    replace(
                        block,
                        operations=tuple(
                            (
                                replace(operation, workspace_dirty=True)
                                if operation.opcode == "free_register"
                                else operation
                            )
                            for operation in block.operations
                        ),
                    )
                    for block in function.blocks
                ),
            )
            for function in original.functions
        ),
    )
    assert _fingerprint(original) != _fingerprint(dirty)


def test_source_addition_is_retained_and_not_labeled_signed_distance() -> None:
    from qualtran.bloqs.chemistry.trotter.grid_ham.potential import PairPotential

    source = _pair()
    calls = tuple(source.decompose_bloq().iter_bloqsoqs())
    adders = [
        instance.bloq
        for instance, _, _ in calls
        if type(instance.bloq).__name__ == "OutOfPlaceAdder"
    ]
    assert len(adders) == 6
    assert sum(not adder.is_adjoint for adder in adders) == 3
    assert sum(adder.is_adjoint for adder in adders) == 3
    # The original sends both positive coordinates directly to addition. Equal
    # unit inputs therefore yield two, while native signed difference is zero.
    assert adders[0].call_classically(a=1, b=1)[2] == 2
    assert (
        PairPotential(1, source.qrom_data, poly_bitsize=3, label="").signature
        == source.signature
    )


@pytest.mark.parametrize("electrons", (0, 1))
def test_source_empty_pair_loop_keeps_identity_and_still_computes_default_data(
    electrons: int,
) -> None:
    from qualtran.bloqs.chemistry.trotter.grid_ham.potential import PotentialEnergy

    from flagquantum.compiler.formats.legacy_grid_potential_factories import (
        grid_potential_factories,
    )
    from flagquantum.errors import SerializationError

    source = PotentialEnergy(electrons, 1, poly_bitsize=3)
    assert tuple(source.decompose_bloq().iter_bloqsoqs()) == ()
    actual = resolve_component_program(
        _library(source, 0), 0, grid_potential_library_factories()
    )
    assert verify_program_record(actual) == () and len(actual.functions) == 1
    block = actual.functions[0].blocks[0]
    assert not block.operations and block.arguments == block.return_values
    factory = grid_potential_factories()[
        "qualtran.bloqs.chemistry.trotter.grid_ham.potential.PotentialEnergy"
    ]
    with pytest.raises(SerializationError, match="coefficient-data provider"):
        factory(
            "missing", None, {"num_elec": electrons, "num_grid": 1, "poly_bitsize": 3}
        )
    # The original evaluates defaults even with no pair calls. Preserve a codec
    # failure here instead of silently shortcutting the class to an identity.
    try:
        PotentialEnergy(electrons, 1, poly_bitsize=128).decompose_bloq()
    except (ValueError, OverflowError, TypeError) as error:
        with pytest.raises(type(error)):
            grid_potential_library_factories()[
                "qualtran.bloqs.chemistry.trotter.grid_ham.potential.PotentialEnergy"
            ]("bad", None, {"num_elec": electrons, "num_grid": 1, "poly_bitsize": 128})
    else:
        pytest.fail("The frozen wide-codec failure witness unexpectedly succeeded")


def test_source_factories_work_without_reference_sdk_and_keep_invalid_metadata_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    from flagquantum.errors import SerializationError

    source = _pair()
    factories = grid_potential_library_factories()
    pair = factories[
        "qualtran.bloqs.chemistry.trotter.grid_ham.potential.PairPotential"
    ]
    total = factories[
        "qualtran.bloqs.chemistry.trotter.grid_ham.potential.PotentialEnergy"
    ]
    for args in (
        {},
        {"bitsize": True, "qrom_data": source.qrom_data},
        {"bitsize": 1, "qrom_data": ()},
        {"bitsize": 1, "qrom_data": source.qrom_data, "label": 3},
    ):
        with pytest.raises(SerializationError):
            pair("bad", None, args)
    original = builtins.__import__

    def no_source(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"qualtran", "cirq"}:
            raise ImportError(name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_source)
    for factory, args in (
        (
            pair,
            {
                "bitsize": 1,
                "qrom_data": source.qrom_data,
                "poly_bitsize": 3,
                "inv_sqrt_bitsize": 5,
            },
        ),
        (total, {"num_elec": 2, "num_grid": 1, "poly_bitsize": 3}),
    ):
        result = factory("native", None, args)
        assert isinstance(result, ProgramRecord) and verify_program_record(result) == ()


def test_entire_frozen_source_test_assets_and_real_execution_extension() -> None:
    from hashlib import sha1
    from importlib.metadata import version
    from pathlib import Path

    import kickmix

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and callable(kickmix.Simulator)
    root = (
        Path(__file__).resolve().parents[3]
        / "ftqc-reference-env/qualtran/bloqs/chemistry/trotter/grid_ham"
    )
    for name, digest in (
        ("potential.py", "6fd2fae98f800302f658ff61dd1d66fd51df9334"),
        ("potential_test.py", "3312fae53b5f519934394fd5e7fcf5d0484e8f27"),
    ):
        raw = (root / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest


def test_symbolic_target_keeps_full_logical_source_recipe_and_width_poly_codec_failures_explicit() -> (
    None
):
    import sympy
    from qualtran.bloqs.chemistry.trotter.grid_ham.potential import PairPotential

    from flagquantum.compiler.formats.legacy_grid_potential_factories import (
        grid_potential_factories,
    )
    from flagquantum.compiler.registers.register_binding import bind_register_program
    from flagquantum.core.ir.program import FunctionRecord
    from tests.unit.test_subtraction_foreign_file import native_signature

    data = _pair().qrom_data
    target = sympy.Symbol("t")
    source = PairPotential(1, data, poly_bitsize=3, inv_sqrt_bitsize=target)
    actual = resolve_component_program(
        _library(source, 0), 0, grid_potential_library_factories()
    )
    reference = resolve_component_program(_library(source, 1), 0, _children())
    assert verify_program_record(actual) == ()
    assert _fingerprint(actual) == _fingerprint(reference)
    with pytest.raises(ValueError):
        bind_register_program(actual, ())
    factory = grid_potential_factories()[
        "qualtran.bloqs.chemistry.trotter.grid_ham.potential.PairPotential"
    ]
    for width, poly in ((sympy.Symbol("n"), 3), (1, sympy.Symbol("p"))):
        original = PairPotential(width, data, poly_bitsize=poly, inv_sqrt_bitsize=5)
        assert len(tuple(original.decompose_bloq().iter_bloqsoqs())) == 26
        with pytest.raises(ValueError, match="Object arrays cannot be saved"):
            _library(original, 1)
        declaration = factory(
            "logical",
            None,
            {
                "bitsize": width,
                "qrom_data": data,
                "poly_bitsize": poly,
                "inv_sqrt_bitsize": 5,
            },
        )
        assert isinstance(declaration, FunctionRecord) and not declaration.blocks
        assert declaration.registers == tuple(native_signature(original.signature))
