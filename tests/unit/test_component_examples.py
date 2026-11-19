"""Whole-file metadata semantics against the locked example-factory source."""

import runpy
from collections.abc import Callable, Iterator
from dataclasses import FrozenInstanceError, replace
from hashlib import sha1
from pathlib import Path
from typing import Any

import pytest

from examples.algorithms.register_component_tutorial import WordSwap
from flagquantum.testing.component_examples import (
    ComponentDocumentation,
    ComponentExample,
    component_example,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
pytest.importorskip("numpy")
pytest.importorskip("qualtran")
ROOT = Path(__file__).resolve().parents[3] / "ftqc-reference-env/qualtran/_infra"


@pytest.fixture(scope="module")
def source() -> dict[str, Any]:
    for name, expected in (
        ("bloq_example.py", "18abbba75b651863eadf164f73a61ed2670195e3"),
        ("bloq_example_test.py", "3075f7c508d6965bc4e52711818ab31c33e2d6ef"),
    ):
        raw = (ROOT / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == expected
    return runpy.run_path(str(ROOT / "bloq_example_test.py"))


def _swap_factory() -> WordSwap:
    """Keep this exact factory description, including its punctuation."""
    return WordSwap(3)


def test_all_unchanged_upstream_tests(source: dict[str, Any]) -> None:
    source["test_bloq_example_explicit"]()
    source["test_bloq_example_decorator"]()


def test_explicit_and_decorated_native_metadata_replace_source(
    source: dict[str, Any],
) -> None:
    original = source["BloqExample"](
        func=source["_tester_bloq_func"],
        name="tester_bloq",
        bloq_cls=source["TestAtom"],
    )
    native = ComponentExample(_swap_factory, "tester_bloq", WordSwap)
    assert original.name == native.name
    assert isinstance(original.make(), source["TestAtom"])
    assert isinstance(original(), source["TestAtom"])
    assert native.make() == native() == WordSwap(3)
    decorated = component_example(_swap_factory)
    assert decorated.name == "swap_factory"
    assert decorated.component_type is WordSwap
    assert decorated.description == _swap_factory.__doc__
    assert decorated.make() == WordSwap(3)
    assert source["_tester_bloq"].bloq_cls is source["TestAtom"]


@pytest.mark.parametrize("shape", ["callable", "list", "tuple", "empty"])
def test_generalizer_identity_and_sequence_snapshot(
    source: dict[str, Any], shape: str
) -> None:
    def keep(value: WordSwap) -> WordSwap:
        return value

    def drop(value: WordSwap) -> None:
        return None

    generalizer: (
        Callable[[WordSwap], WordSwap | None]
        | list[Callable[[WordSwap], WordSwap | None]]
        | tuple[Callable[[WordSwap], WordSwap | None], ...]
    )
    if shape == "callable":
        generalizer = keep
    elif shape == "list":
        generalizer = [keep, drop]
    elif shape == "tuple":
        generalizer = (keep, drop)
    else:
        generalizer = []
    native = component_example(generalizer=generalizer)(_swap_factory)
    original = source["bloq_example"](generalizer=generalizer)(
        source["_tester_bloq_func"]
    )
    assert native.generalizer == original.generalizer
    assert component_example(_swap_factory, generalizer=generalizer) == native
    if isinstance(generalizer, list):
        frozen = tuple(generalizer)
        generalizer.append(drop)
        assert native.generalizer == original.generalizer == frozen
    default = component_example(_swap_factory)
    assert callable(default.generalizer)
    assert default.generalizer(default.make()) == WordSwap(3)


def test_factory_is_lazy_fresh_and_preserves_exceptions() -> None:
    calls: list[int] = []

    def fresh() -> WordSwap:
        calls.append(len(calls))
        return WordSwap(len(calls))

    example = component_example(fresh)
    assert calls == []
    assert example().width == 1
    assert example.make().width == 2
    assert calls == [0, 1]
    failure = RuntimeError("factory failed")

    def broken() -> WordSwap:
        raise failure

    with pytest.raises(RuntimeError) as caught:
        component_example(broken).make()
    assert caught.value is failure


def test_annotation_and_leading_underscore_behavior(source: dict[str, Any]) -> None:
    def missing() -> WordSwap:
        return WordSwap(1)

    del missing.__annotations__["return"]

    with pytest.raises(ValueError, match="return type annotation"):
        component_example(missing)
    with pytest.raises(ValueError, match="return type annotation"):
        source["bloq_example"](missing)

    def forward() -> "WordSwap":
        return WordSwap(1)

    with pytest.raises(TypeError):
        component_example(forward)
    with pytest.raises(TypeError):
        source["bloq_example"](forward)

    def ___named() -> WordSwap:
        return WordSwap(1)

    assert component_example(___named).name == "named"
    # Native examples are metadata for dataclass operations without a common
    # base class. The original decorator's Bloq-only restriction is not retained.
    with pytest.raises(AssertionError):
        source["bloq_example"](___named)


def test_immutable_equality_hash_and_hidden_factory() -> None:
    example = component_example(_swap_factory)
    assert example == component_example(_swap_factory)
    assert hash(example) == hash(component_example(_swap_factory))
    assert "_swap_factory" not in repr(example)
    assert replace(example, name="other") != example
    with pytest.raises(FrozenInstanceError):
        setattr(example, "name", "changed")


@pytest.mark.parametrize("count", [0, 1, 3])
def test_documentation_defaults_and_single_iterable_consumption(
    source: dict[str, Any], count: int
) -> None:
    from qualtran._infra.bloq_example import BloqDocSpec

    consumed: list[int] = []
    example = component_example(_swap_factory)

    def examples() -> Iterator[ComponentExample[WordSwap]]:
        for index in range(count):
            consumed.append(index)
            yield example

    native = ComponentDocumentation(component_type=WordSwap, examples=examples())
    originals = tuple(source["_tester_bloq"] for _ in range(count))
    original = BloqDocSpec(bloq_cls=source["TestAtom"], examples=iter(originals))
    assert consumed == list(range(count))
    assert len(native.examples) == len(original.examples) == count
    assert native.call_graph_example == (example if count else None)
    assert original.call_graph_example == (originals[0] if count else None)
    expected_import = "from examples.algorithms import WordSwap"
    assert native.import_line == expected_import
    assert (
        original.import_line
        == f"from {source['TestAtom'].__module__.rsplit('.', 1)[0]} import TestAtom"
    )
    assert (
        ComponentDocumentation(
            component_type=WordSwap, examples=[example], call_graph_example=None
        ).call_graph_example
        is None
    )
    assert (
        ComponentDocumentation(
            component_type=WordSwap, import_line="explicit"
        ).import_line
        == "explicit"
    )


def test_explicit_graph_selection_and_source_membership_boundary(
    source: dict[str, Any],
) -> None:
    from qualtran._infra.bloq_example import BloqDocSpec

    example = component_example(_swap_factory)
    # The source documents membership but does not validate it. Preserve that
    # behavior instead of inventing an admission check during migration.
    assert (
        ComponentDocumentation(
            component_type=WordSwap, call_graph_example=example
        ).call_graph_example
        is example
    )
    original = source["_tester_bloq"]
    assert (
        BloqDocSpec(
            bloq_cls=source["TestAtom"], call_graph_example=original
        ).call_graph_example
        is original
    )


def test_real_documented_component_executes_full_complex_state() -> None:
    from examples.algorithms.component_example_catalog import main

    main()
