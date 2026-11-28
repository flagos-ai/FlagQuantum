"""Close available ECC layers around the source-refused symbolic multiplier."""

import sys
from dataclasses import replace
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

from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.ecosystem._elliptic_curve_addition_library import (
    elliptic_curve_addition_factories,
)
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_ecc_addition_model import frozen_source as frozen_source
from tests.unit.test_legacy_arithmetic_factories import _library

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _compare_available_layers(
    record: ProgramRecord, source: Any, finite: ProgramRecord
) -> None:
    """Reuse a finite subtree only after proving its complete definition equal.

    The four changed stage bodies, ECAdd and ECWindowAddR are traversed. Every
    other finite child must match the full previously accepted dependency;
    declaration-only multipliers instead require an actual original refusal.
    """
    assert verify_program_record(record) == verify_program_record(finite) == ()
    functions = {f.name: f for f in record.functions}
    originals = {f.name: f for f in finite.functions}
    changed = {
        "ECAdd",
        "ECWindowAddR",
        "_ECAddStepTwo",
        "_ECAddStepThree",
        "_ECAddStepFour",
        "_ECAddStepFive",
    }
    pending = [("main", "main", source)]
    terminals: set[str] = set()
    visited: set[str] = set()
    refused = 0
    certified: set[tuple[str, str]] = set()
    while pending:
        current, previous, child = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        operations = functions[current].blocks[0].operations
        old_operations = originals[previous].blocks[0].operations
        instances = sorted(
            child.decompose_bloq().iter_bloqsoqs(), key=lambda row: row[0].i
        )
        assert len(operations) == len(old_operations) == len(instances)
        for operation, old_operation, (instance, _, _) in zip(
            operations, old_operations, instances, strict=True
        ):
            name, old_name = operation.callee, old_operation.callee
            assert name is not None and old_name is not None
            bloq = instance.bloq
            if type(bloq).__name__ in changed:
                pending.append((name, old_name, bloq))
            elif type(bloq).__name__ == "DirtyOutOfPlaceMontgomeryModMul":
                assert not functions[name].blocks
                with pytest.raises(
                    DecomposeNotImplementedError, match="symbolic decomposition"
                ):
                    bloq.decompose_bloq()
                registers = functions[name].registers
                assert registers is not None
                history = next(r for r in registers if r.name == "qrom_indices")
                assert history.dtype.num_qubits == (
                    ((bloq.bitsize + bloq.window_size - 1) // bloq.window_size)
                    * bloq.window_size
                )
                terminals.add(name)
                refused += 1
            else:
                if (name, old_name) not in certified:
                    actual = _reachable(record, name)
                    expected = _reachable(finite, old_name)
                    assert matches_component_definition(actual, name, expected), (
                        current,
                        type(bloq).__name__,
                        name,
                    )
                    certified.add((name, old_name))
                terminals.add(name)
    assert refused >= 2
    compare_layers(record, source, terminals)


def _reachable(record: ProgramRecord, entry: str) -> ProgramRecord:
    """Keep the complete reachable closure for a literal comparison certificate."""
    functions = {f.name: f for f in record.functions}
    pending = [entry]
    seen = {entry}
    for name in pending:
        for block in functions[name].blocks:
            for operation in block.operations:
                child = operation.callee
                if child is not None and child not in seen:
                    seen.add(child)
                    pending.append(child)
    return replace(
        record,
        functions=tuple(functions[name] for name in pending),
        canonical_payload=None,
    )


@pytest.mark.parametrize(
    "kind,width,modulus,depth",
    (
        ("root", 3, 3, 0),
        ("root", 5, 7, 0),
        ("Two", 5, 7, 1),
        ("Three", 5, 7, 1),
        ("Four", 5, 7, 1),
        ("Five", 5, 7, 1),
    ),
)
def test_complete_available_symbolic_window_graphs_reuse_exact_finite_dependencies(
    width: int, modulus: int, kind: str, depth: int
) -> None:
    window = sympy.Symbol("w")
    constructor = (
        ec_add.ECAdd if kind == "root" else getattr(ec_add, "_ECAddStep" + kind)
    )
    source = constructor(width, modulus, window)
    finite_source = constructor(width, modulus, 1)
    factories = elliptic_curve_addition_factories
    encoded = serialize_component_library(_library(source, depth))
    library = deserialize_component_library(encoded)
    assert serialize_component_library(library) == encoded
    record = resolve_component_program(library, 0, factories())
    finite = resolve_component_program(_library(finite_source, 0), 0, factories())
    _compare_available_layers(record, source, finite)
    expected_calls = {
        "Two": 18 + 2 * width,
        "Three": 18 + 2 * width,
        "Four": 19 + 3 * width,
        "Five": 13 + width,
        "root": 6,
    }[kind]
    assert len(record.functions[0].blocks[0].operations) == expected_calls
    with pytest.raises(ValueError, match="component bodies"):
        bind_register_program(record, ((0,),))
    denied = {
        name: None
        for name in sys.modules
        if name.split(".", 1)[0] in {"qualtran", "cirq", "kickmix"}
    }
    with patch.dict(sys.modules, denied):
        assert resolve_component_program(library, 0, factories()) == record
