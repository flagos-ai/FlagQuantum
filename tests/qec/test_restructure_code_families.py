"""Code-family relocation against trusted pre-003 Python identifiers."""

from __future__ import annotations

import importlib
import inspect
import json
import pickle
from pathlib import Path

import pytest

import flagquantum.qec as qec

pytestmark = pytest.mark.unit
_FIXTURES = Path(__file__).with_name("fixtures")


def _resolve(path: str) -> object:
    parts = path.split(".")
    value = importlib.import_module(".".join(parts[:3]))
    for part in parts[3:]:
        value = getattr(value, part)
    return value


def test_pre_move_code_and_method_pickles_keep_exact_identities() -> None:
    # Repository-owned fixtures generated before relocation; never external input.
    data = (_FIXTURES / "restructure_003_codes.pkl").read_bytes()
    restored = pickle.loads(data)
    for path, value in restored.items():
        if path == "instances":
            for origin, instance in value.items():
                assert type(instance) is _resolve(origin)
                assert type(pickle.loads(pickle.dumps(instance))) is type(instance)
        else:
            assert value is _resolve(path), path
    serialized = pickle.dumps(restored, protocol=4)
    assert b"flagquantum.qec.codes.surface" not in serialized
    assert b"flagquantum.qec.codes.css" not in serialized
    round_trip = pickle.loads(serialized)
    for path, value in restored.items():
        if path != "instances":
            assert round_trip[path] is value


def test_definitions_and_class_members_keep_historical_metadata() -> None:
    expected = json.loads((_FIXTURES / "restructure_003_metadata.json").read_text())
    for path, metadata in expected.items():
        value = _resolve(path)
        if isinstance(value, property):
            value = value.fget
        assert value.__module__ == metadata["module"], path
        assert value.__qualname__ == metadata["qualname"], path


@pytest.mark.parametrize(
    ("old", "new"),
    (
        ("css_code", "css"),
        ("qldpc", "qldpc"),
        ("bicycle", "bicycle"),
        ("reed_muller", "reed_muller"),
        ("subsystem", "subsystem"),
        ("floquet", "floquet"),
    ),
)
def test_historical_family_imports_are_aliases(old: str, new: str) -> None:
    historical = importlib.import_module("flagquantum.qec." + old)
    owner = importlib.import_module("flagquantum.qec.codes." + new)
    assert historical.__all__ == owner.__all__
    for name in historical.__all__:
        value = getattr(historical, name)
        assert value is getattr(owner, name)
        assert value is getattr(qec, name)
        assert (
            Path(
                (
                    value.__post_init__ if inspect.isclass(value) else value
                ).__code__.co_filename
            ).resolve()
            == Path(owner.__file__).resolve()
        )


def test_both_surface_records_remain_distinct_through_all_imports() -> None:
    from flagquantum.qec import codes, surface
    from flagquantum.qec.codes import surface as families
    from flagquantum.qec.codes.surface import _rotated_frame, rotated, zxxz

    assert qec.RotatedSurfaceCode is codes.RotatedSurfaceCode
    assert families.RotatedSurfaceCode is rotated.RotatedSurfaceCode
    assert qec.RotatedSurfaceCode is rotated.RotatedSurfaceCode
    assert surface.RotatedSurfaceCode is _rotated_frame.RotatedSurfaceCode
    assert surface.RotatedSurfaceCode is not qec.RotatedSurfaceCode
    assert qec.ZxxzSurfaceCode is surface.ZxxzSurfaceCode is zxxz.ZxxzSurfaceCode
    assert families.ZxxzSurfaceCode is qec.ZxxzSurfaceCode
    assert type(qec.get_code("rotated_surface", distance=3)) is qec.RotatedSurfaceCode
    for distance in (3, 5, 7):
        registered = qec.RotatedSurfaceCode(distance)
        frame = surface.RotatedSurfaceCode(distance)
        assert registered != frame
        assert registered.checks == frame.checks
        assert registered.logical_observables == frame.logical_observables
        for record in (registered, frame, qec.ZxxzSurfaceCode(distance)):
            assert type(pickle.loads(pickle.dumps(record))) is type(record)
