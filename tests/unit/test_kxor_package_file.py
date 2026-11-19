"""Map both complete frozen optimization package files to native domain owners."""

import ast
from hashlib import sha1
from importlib import import_module
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]

_OWNERS = {
    "KikuchiNonZeroIndex": ("kikuchi_adjacency", "KikuchiNonZeroIndex"),
    "KikuchiMatrixEntry": ("kikuchi_matrix", "KikuchiMatrixEntry"),
    "KikuchiHamiltonian": ("kikuchi_block_encoding", "KikuchiHamiltonian"),
    "GuidingState": ("kikuchi_guiding_state", "GuidingState"),
    "SimpleGuidingState": ("kikuchi_guiding_state", "ScopeGuidingState"),
    "Constraint": ("kxor_instance", "Constraint"),
    "KXorInstance": ("kxor_instance", "KXorInstance"),
    "LoadConstraintScopes": ("kxor_loading", "LoadConstraintScopes"),
    "LoadUniqueScopeIndex": ("kxor_loading", "LoadUniqueScopeIndex"),
    "PRGAUniqueConstraintRHS": ("kxor_loading", "PRGAUniqueConstraintRHS"),
}


def test_empty_optimization_namespace_has_no_omitted_behavior() -> None:
    source = import_module("qualtran.bloqs.optimization")
    assert source.__file__ is not None
    raw = Path(source.__file__).read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "528a9789f62243bf6e64ad2ea3929dc4e663cbf7"
    )
    assert ast.parse(raw).body == []
    assert import_module("flagquantum.algorithms").__name__ == "flagquantum.algorithms"


def test_complete_kxor_package_imports_duplicates_and_native_ownership() -> None:
    source = import_module("qualtran.bloqs.optimization.k_xor_sat")
    assert source.__file__ is not None
    raw = Path(source.__file__).read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "79b00094a6874a9f8395876d6672201303951add"
    )
    statements = ast.parse(raw).body
    assert len(statements) == 6
    imports = []
    for statement in statements:
        assert isinstance(statement, ast.ImportFrom)
        assert statement.level == 1 and statement.module is not None
        owner = import_module(f"{source.__name__}.{statement.module}")
        for entry in statement.names:
            assert entry.asname is None
            assert getattr(source, entry.name) is getattr(owner, entry.name)
            imports.append(entry.name)
            module_name, native_name = _OWNERS[entry.name]
            native_owner = import_module(f"flagquantum.algorithms.{module_name}")
            native_type = getattr(native_owner, native_name)
            assert isinstance(native_type, type)
            assert native_type.__module__ == native_owner.__name__
            assert native_type is not getattr(source, entry.name)
    assert len(imports) == 12 and set(imports) == set(_OWNERS)
    assert (
        imports.count("KikuchiMatrixEntry") == imports.count("KikuchiNonZeroIndex") == 2
    )


@pytest.mark.parametrize(
    "source_name",
    tuple(name for name in _OWNERS if name not in ("Constraint", "KXorInstance")),
)
def test_every_imported_operation_constructs_its_source_signature(
    source_name: str,
) -> None:
    source = import_module("qualtran.bloqs.optimization.k_xor_sat")
    instances = import_module("flagquantum.algorithms.kxor_instance")
    module_name, native_name = _OWNERS[source_name]
    native_owner = import_module(f"flagquantum.algorithms.{module_name}")
    constraints = (((0, 1), 1), ((2, 3), -1), ((1, 2), 1))
    original_instance = source.KXorInstance(
        n=4,
        k=2,
        constraints=tuple(
            source.Constraint(scope, value) for scope, value in constraints
        ),
    )
    native_instance = instances.KXorInstance(
        n=4,
        k=2,
        constraints=tuple(
            instances.Constraint(scope, value) for scope, value in constraints
        ),
    )
    arguments = {
        "KikuchiNonZeroIndex": (4, 3),
        "KikuchiMatrixEntry": (4, 3),
        "KikuchiHamiltonian": (4, 3),
        "GuidingState": (4,),
        "SimpleGuidingState": (),
        "LoadConstraintScopes": (),
        "LoadUniqueScopeIndex": (),
        "PRGAUniqueConstraintRHS": (3,),
    }[source_name]
    original = getattr(source, source_name)(original_instance, *arguments)
    native = getattr(native_owner, native_name)(native_instance, *arguments)
    assert [
        (r.name, r.bitsize, r.shape_symbolic, r.side.name) for r in native.signature
    ] == [
        (r.name, r.bitsize, r.shape_symbolic, r.side.name) for r in original.signature
    ]
    assert native.inst.batched_scopes == original.inst.batched_scopes
