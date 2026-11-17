"""Certify the locked state-preparation exports and concrete append routes."""

import ast
from collections.abc import Callable
from hashlib import sha1
from importlib import import_module
from pathlib import Path
from unittest.mock import patch

import pytest

from examples.algorithms import uniform_preparation_catalog
from examples.algorithms.alias_preparation_catalog import (
    AliasPreparationExample,
    dense_alias_example,
    sparse_alias_example,
)
from examples.algorithms.rotation_preparation_catalog import RotationPreparationExample
from flagquantum.algorithms.primitives import alias_preparation
from flagquantum.algorithms.primitives.rotation_preparation import RotationPreparation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.preparation.uniform_preparation import (
    uniform_preparation_ir,
)
from flagquantum.dynamic import DynamicCircuit

pytestmark = pytest.mark.unit
_SOURCE_PACKAGE = "qualtran.bloqs.state_preparation"
_SOURCE_BLOB = "024a57c221ae73f95645dbaa4247f423b037a932"
_OWNERS = {
    "PrepareUniformSuperposition": "prepare_uniform_superposition",
    "SparseStatePreparationAliasSampling": "state_preparation_alias_sampling",
    "StatePreparationAliasSampling": "state_preparation_alias_sampling",
    "StatePreparationViaRotations": "state_preparation_via_rotation",
}


@pytest.mark.ftqc_reference
def test_locked_initializer_covers_every_state_preparation_export() -> None:
    package = pytest.importorskip(_SOURCE_PACKAGE)
    assert package.__file__ is not None
    content = Path(package.__file__).read_bytes()
    assert sha1(f"blob {len(content)}\0".encode() + content).hexdigest() == _SOURCE_BLOB
    seen = set()
    for node in ast.parse(content).body:
        assert isinstance(node, ast.ImportFrom)
        assert node.level == 0
        for alias in node.names:
            name = alias.name
            assert alias.asname is None and name not in seen
            seen.add(name)
            owner = f"{_SOURCE_PACKAGE}.{_OWNERS[name]}"
            assert node.module == owner
            assert getattr(package, name) is getattr(import_module(owner), name)
    assert seen == _OWNERS.keys()


def test_uniform_preparation_append_reaches_the_compiler_owner() -> None:
    op = uniform_preparation_catalog.UniformPreparationExample(count=5)
    assert (
        vars(uniform_preparation_catalog)["uniform_preparation_ir"]
        is uniform_preparation_ir
    )
    with patch.object(
        uniform_preparation_catalog,
        "uniform_preparation_ir",
        wraps=uniform_preparation_ir,
    ) as emit:
        DynamicCircuit(op.qubit_count).append(op)
    emit.assert_called_once()
    assert emit.call_args.args[0] == op.count


@pytest.mark.parametrize(
    ("make_example", "expected_sparse"),
    [(dense_alias_example, False), (sparse_alias_example, True)],
    ids=("dense", "sparse"),
)
def test_alias_preparation_append_reaches_the_shared_owner(
    make_example: Callable[[], AliasPreparationExample], expected_sparse: bool
) -> None:
    op = make_example()
    with patch.object(
        alias_preparation,
        "alias_preparation_ir",
        wraps=alias_preparation.alias_preparation_ir,
    ) as emit:
        Circuit(op.qubit_count).append(op)
    emit.assert_called_once()
    assert emit.call_args.args[0] is op.table
    assert bool(emit.call_args.kwargs["sparse_index"]) is expected_sparse


def test_rotation_preparation_append_reaches_the_rotation_owner() -> None:
    op = RotationPreparationExample()
    with patch.object(
        RotationPreparation, "emit", autospec=True, side_effect=RotationPreparation.emit
    ) as emit:
        Circuit(op.qubit_count).append(op)
    emit.assert_called_once()
    assert emit.call_args.args[0] == op.preparation
    assert emit.call_args.args[3] == op.gradient
