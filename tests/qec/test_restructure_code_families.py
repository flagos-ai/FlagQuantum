"""Code-family relocation against trusted pre-003 Python identifiers."""

from __future__ import annotations

import importlib
import inspect
import json
import pickle
from pathlib import Path

import pytest
from _retirement import (
    assert_metadata,
    assert_retired_payload,
    resolve,
)

import flagquantum.qec as qec

pytestmark = pytest.mark.unit
_FIXTURES = Path(__file__).with_name("fixtures")


def _resolve(path: str) -> object:
    return resolve(path)


def test_old_compound_code_pickle_refuses_the_retired_identifiers() -> None:
    assert_retired_payload("restructure_003_codes.pkl")
    # The ten surviving individual code/record pickles remain covered by 001.
    for code in (
        qec.RepetitionCode(),
        qec.RotatedSurfaceCode(),
        qec.SteaneCode(),
        qec.ZxxzSurfaceCode(),
    ):
        restored = pickle.loads(pickle.dumps(code, protocol=4))
        assert type(restored) is type(code)
        assert restored.checks == code.checks
        assert restored.logical_observables == code.logical_observables


def test_code_metadata_preserves_surviving_identifiers_and_canonical_retired_owners() -> (
    None
):
    expected = json.loads((_FIXTURES / "restructure_003_metadata.json").read_text())
    for path, metadata in expected.items():
        assert_metadata(path, metadata["module"], metadata["qualname"])


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
def test_family_imports_use_their_canonical_owners(old: str, new: str) -> None:
    with pytest.raises(ModuleNotFoundError) as caught:
        importlib.import_module("flagquantum.qec." + old)
    assert caught.value.name == "flagquantum.qec." + old
    owner = importlib.import_module("flagquantum.qec.codes." + new)
    for name in owner.__all__:
        value = getattr(owner, name)
        assert value is getattr(qec, name)
        assert (
            Path(
                (
                    value.__post_init__ if inspect.isclass(value) else value
                ).__code__.co_filename
            ).resolve()
            == Path(owner.__file__).resolve()
        )


def test_one_registered_surface_record_serves_all_surviving_imports() -> None:
    from flagquantum.qec import codes
    from flagquantum.qec.codes import surface
    from flagquantum.qec.codes.surface import rotated, zxxz

    assert (
        qec.RotatedSurfaceCode
        is codes.RotatedSurfaceCode
        is surface.RotatedSurfaceCode
        is rotated.RotatedSurfaceCode
    )
    assert qec.ZxxzSurfaceCode is surface.ZxxzSurfaceCode is zxxz.ZxxzSurfaceCode
    assert type(qec.get_code("rotated_surface", distance=3)) is qec.RotatedSurfaceCode
    for distance in (3, 5, 7):
        for record in (qec.RotatedSurfaceCode(distance), qec.ZxxzSurfaceCode(distance)):
            assert type(pickle.loads(pickle.dumps(record))) is type(record)
    for module in (
        "flagquantum.qec.surface",
        "flagquantum.qec.codes.surface._rotated_frame",
        "flagquantum.qec.codes.surface.rotated_frame",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module)
