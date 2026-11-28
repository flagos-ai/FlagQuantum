"""Compose complete source-defined curve stages around actual symbolic refusals."""

import sys
from importlib import import_module
from typing import Any
from unittest.mock import patch

import pytest

from tests._reference_dependencies import unavailable_reference

try:
    import_module("numpy")
    import sympy

    import_module("cirq")
    import_module("kickmix")
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.cryptography.ecc import ec_add
except ModuleNotFoundError as error:
    unavailable_reference(error)

from flagquantum.compiler.formats.legacy_bookkeeping_factories import (
    bookkeeping_factories,
)
from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    ResolvedArgument,
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_modular_addition_factories import (
    _symbolic_addition,
    modular_addition_factories,
)
from flagquantum.compiler.formats.legacy_modular_inverse_factories import (
    modular_inverse_factories,
)
from flagquantum.compiler.formats.legacy_modular_multiplication_factories import (
    modular_multiplication_factories,
)
from flagquantum.compiler.formats.legacy_modular_subtraction_factories import (
    modular_subtraction_factories,
)
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.register_types import montgomery_register_type
from flagquantum.core.ir.program.registers import RegisterSide
from flagquantum.ecosystem._elliptic_curve_addition_library import (
    elliptic_curve_addition_factories,
)
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_ecc_addition_model import frozen_source as frozen_source
from tests.unit.test_ecc_symbolic_multiplication_window import _reachable
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_symbolic_modular_doubling_calls import (
    assert_unbound,
    source_body_program,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
_STAGES = {
    "_ECAddStepTwo",
    "_ECAddStepThree",
    "_ECAddStepFour",
    "_ECAddStepFive",
    "_ECAddStepSix",
    "ECAdd",
}


def _resolve(source: Any) -> ProgramRecord:
    return resolve_component_program(
        _library(source, 0), 0, elliptic_curve_addition_factories()
    )


def compare_stages(
    record: ProgramRecord, source: Any, finite: ProgramRecord, finite_source: Any
) -> None:
    """Certify complete unchanged/delegate closures, never infer semantics by ABI."""
    functions = {f.name: f for f in record.functions}
    original = {f.name: f for f in finite.functions}
    pending = [("main", "main", source, finite_source)]
    terminals: set[str] = set()
    seen: set[str] = set()
    refused = 0
    for current, previous, child, counterpart in pending:
        if current in seen:
            continue
        seen.add(current)
        actual_ops = functions[current].blocks[0].operations
        old_ops = original[previous].blocks[0].operations
        expected = sorted(
            child.decompose_bloq().iter_bloqsoqs(), key=lambda row: row[0].i
        )
        previous_children = sorted(
            counterpart.decompose_bloq().iter_bloqsoqs(), key=lambda row: row[0].i
        )
        assert (
            len(actual_ops) == len(old_ops) == len(expected) == len(previous_children)
        )
        for op, old, (instance, _, _), (old_instance, _, _) in zip(
            actual_ops, old_ops, expected, previous_children, strict=True
        ):
            name, old_name, bloq, earlier = (
                op.callee,
                old.callee,
                instance.bloq,
                old_instance.bloq,
            )
            assert name is not None and old_name is not None
            if type(bloq).__name__ in _STAGES:
                pending.append((name, old_name, bloq, earlier))
                continue
            actual = _reachable(record, name)
            if bloq == earlier:
                assert matches_component_definition(
                    actual, name, _reachable(finite, old_name)
                )
            elif type(bloq).__name__ == "DirtyOutOfPlaceMontgomeryModMul":
                assert not functions[name].blocks
                with pytest.raises(DecomposeNotImplementedError):
                    bloq.decompose_bloq()
                registers = functions[name].registers
                assert registers is not None
                assert tuple(r.name for r in registers) == (
                    "x",
                    "y",
                    "target",
                    "qrom_indices",
                    "reduced",
                )
                side = RegisterSide.LEFT if bloq.uncompute else RegisterSide.RIGHT
                assert all(
                    r.side is (RegisterSide.THRU if r.name in {"x", "y"} else side)
                    for r in registers
                )
                assert all(
                    dict(r.dtype.parameters).get("modulus") == bloq.mod
                    for r in registers
                    if r.dtype.encoding == "montgomery"
                )
                assert all(r.bitsize == bloq.bitsize for r in registers[:3])
                history = next(r for r in registers if r.name == "qrom_indices")
                assert (
                    history.bitsize
                    == ((bloq.bitsize + bloq.window_size - 1) // bloq.window_size)
                    * bloq.window_size
                )
                refused += 1
            else:
                wrapper = type(bloq).__name__ == "Adjoint"
                base = bloq.subbloq if wrapper else bloq
                kind = type(base).__name__
                if wrapper and kind in {"ModAdd", "CModAdd"}:
                    registers = functions[name].registers
                    assert registers is not None
                    dtype = (
                        montgomery_register_type(base.bitsize, base.mod)
                        if kind == "ModAdd"
                        else registers[1].dtype
                    )
                    delegated = _symbolic_addition(
                        dtype, base.mod, controlled=kind == "CModAdd", adjoint=True
                    )
                elif kind in {"Split", "Join", "Allocate", "Free"}:
                    # Original transport omits Montgomery's modulus; retain
                    # live source metadata when certifying atomic word views.
                    assert type(base.dtype).__name__ == "QMontgomeryUInt"
                    dtype = montgomery_register_type(
                        base.dtype.bitsize, base.dtype.modulus
                    )
                    arguments: dict[str, ResolvedArgument] = {"dtype": dtype}
                    if kind in {"Allocate", "Free"}:
                        arguments["dirty"] = base.dirty
                    function = bookkeeping_factories()[
                        "qualtran.bloqs.bookkeeping." + kind.lower() + "." + kind
                    ]("main", None, arguments)
                    assert isinstance(function, FunctionRecord)
                    delegated = ProgramRecord((function,))
                else:
                    factories = (
                        modular_inverse_factories()
                        if "Kaliski" in kind
                        else (
                            modular_multiplication_factories()
                            if kind == "ModDbl"
                            else (
                                modular_subtraction_factories()
                                if kind in {"ModNeg", "CModNeg", "ModSub", "CModSub"}
                                else (
                                    modular_addition_factories()
                                    if kind in {"ModAdd", "CModAdd"}
                                    else (
                                        elliptic_curve_addition_factories()
                                        if kind == "_ECAddStepOne"
                                        else {}
                                    )
                                )
                            )
                        )
                    )
                    assert factories, repr(bloq)
                    delegated = resolve_component_program(
                        _library(bloq, 0), 0, factories
                    )
                # Only the existing, separately certified codec bridge may
                # distinguish typed shared owners from their stored word view.
                if not matches_component_definition(actual, name, delegated):
                    delegated = source_body_program(delegated, bloq)
                    assert matches_component_definition(actual, name, delegated), (
                        current,
                        repr(bloq),
                    )
            terminals.add(name)
    assert refused == 0 if type(source).__name__ == "_ECAddStepSix" else refused >= 2
    compare_layers(record, source, terminals)


@pytest.mark.parametrize(
    "kind",
    (
        "_ECAddStepTwo",
        "_ECAddStepThree",
        "_ECAddStepFour",
        "_ECAddStepFive",
        "_ECAddStepSix",
        "ECAdd",
    ),
)
def test_all_available_symbolic_modulus_stage_calls_and_complete_dependencies(
    kind: str,
) -> None:
    p = sympy.Symbol("p")
    source = getattr(ec_add, kind)(5, p)
    record = _resolve(source)
    if kind == "ECAdd":
        # The six stage owners have their own source-layer proofs. Certify each
        # entire imported closure before reusing those proofs at the root.
        functions = {function.name: function for function in record.functions}
        operations = record.functions[0].blocks[0].operations
        children = sorted(
            source.decompose_bloq().iter_bloqsoqs(), key=lambda row: row[0].i
        )
        assert len(operations) == len(children) == 6
        terminals = set()
        for operation, (instance, _, _) in zip(operations, children, strict=True):
            name = operation.callee
            assert name is not None and name in functions
            assert matches_component_definition(
                _reachable(record, name), name, _resolve(instance.bloq)
            )
            terminals.add(name)
        compare_layers(record, source, terminals)
    else:
        concrete_source = getattr(ec_add, kind)(5, 7)
        finite = _resolve(concrete_source)
        compare_stages(record, source, finite, concrete_source)
    assert_unbound(record)
    encoded = _library_bytes(source)
    denied = {
        name: None
        for name in sys.modules
        if name.split(".", 1)[0] in {"qualtran", "cirq", "kickmix"}
    }
    with patch.dict(sys.modules, denied):
        assert _resolve_from_library(encoded) == record


def _library_bytes(source: Any) -> bytes:
    return serialize_component_library(_library(source, 0))


def _resolve_from_library(raw: bytes) -> ProgramRecord:
    return resolve_component_program(
        deserialize_component_library(raw), 0, elliptic_curve_addition_factories()
    )


@pytest.mark.parametrize(
    "kind",
    (
        "_ECAddStepTwo",
        "_ECAddStepThree",
        "_ECAddStepFour",
        "_ECAddStepFive",
        "_ECAddStepSix",
    ),
)
def test_symbolic_arithmetic_constructor_parameters_follow_original_children(
    kind: str,
) -> None:
    import flagquantum.ecosystem._elliptic_curve_addition_library as adapter
    from flagquantum.ecosystem._elliptic_curve_arithmetic_library import (
        curve_inverse_program,
        curve_modular_program,
        curve_multiplication_program,
    )

    p = sympy.Symbol("p")
    source = getattr(ec_add, kind)(5, p)
    with (
        patch.object(
            adapter, "curve_modular_program", wraps=curve_modular_program
        ) as modular,
        patch.object(
            adapter, "curve_inverse_program", wraps=curve_inverse_program
        ) as inverse,
        patch.object(
            adapter,
            "curve_multiplication_program",
            wraps=curve_multiplication_program,
        ) as multiply,
    ):
        _resolve(source)
    observed: set[tuple[object, ...]] = set()
    for call in modular.call_args_list:
        tag, dtype, modulus = call.args
        assert (
            dtype.encoding == "montgomery"
            and dtype.num_qubits == 5
            and dtype.parameters == ()
        )
        observed.add(("modular", tag, 5, modulus, call.kwargs.get("adjoint", False)))
    for call in inverse.call_args_list:
        width, modulus = call.args
        observed.add(("inverse", width, modulus, call.kwargs.get("adjoint", False)))
    for call in multiply.call_args_list:
        width, window, modulus = call.args
        observed.add(
            ("multiply", width, window, modulus, call.kwargs.get("adjoint", False))
        )
    expected: set[tuple[object, ...]] = set()
    for instance, _, _ in source.decompose_bloq().iter_bloqsoqs():
        child = instance.bloq
        direction = type(child).__name__ == "Adjoint"
        base = child.subbloq if direction else child
        tag = type(base).__name__
        if tag in {"ModAdd", "CModAdd", "ModSub", "CModSub", "CModNeg", "ModDbl"}:
            if hasattr(base, "cv"):
                assert base.cv == 1
            expected.add(("modular", tag, 5, base.mod, direction))
        elif tag == "KaliskiModInverse":
            expected.add(("inverse", base.bitsize, base.mod, base.uncompute))
        elif tag == "DirtyOutOfPlaceMontgomeryModMul":
            expected.add(
                ("multiply", base.bitsize, base.window_size, base.mod, base.uncompute)
            )
    assert observed == expected


def test_symbolic_modulus_and_multiplication_window_keep_exact_history_and_source_calls() -> (
    None
):
    p, window = sympy.symbols("p window")
    source = ec_add._ECAddStepFour(5, p, window)
    finite_source = ec_add._ECAddStepFour(5, 7, window)
    record = _resolve(source)
    compare_stages(record, source, _resolve(finite_source), finite_source)
    assert_unbound(record)
