"""Assertions for the approved import break; never an import or pickle bridge."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import inspect
import json
import pickle
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = json.loads(
    (ROOT / "contracts/qec-historical-import-retirement-v0.4.json").read_text()
)
RETIRED = CONTRACT["module_replacements"]


def resolve(path: str) -> object:
    """Find the current owner to compare it with immutable historical metadata."""
    parts = path.split(".")
    old = ".".join(parts[:3])
    for owner in RETIRED.get(old, [old]):
        value = importlib.import_module(owner)
        if hasattr(value, parts[3]):
            for part in parts[3:]:
                value = getattr(value, part)
            return value
    raise AssertionError(f"No canonical owner for {path}")


def assert_retired_payload(name: str) -> None:
    path = ROOT / "tests/qec/fixtures" / name
    payload = path.read_bytes()
    assert (
        hashlib.sha256(payload).hexdigest()
        == CONTRACT["frozen_fixture_sha256"][path.relative_to(ROOT).as_posix()]
    )
    with pytest.raises(ModuleNotFoundError) as caught:
        pickle.loads(payload)
    assert caught.value.name in RETIRED


def assert_metadata(path: str, module: str, qualname: str) -> None:
    value = resolve(path)
    is_property = isinstance(value, property)
    if is_property:
        value = value.fget
    actual = getattr(value, "__module__", None)
    assert (
        getattr(value, "__qualname__", getattr(value, "__name__", None)) == qualname
    ), path
    if module not in RETIRED:
        assert actual == module, path
        if not is_property and "<locals>" not in qualname:
            # Protocol-4 GLOBAL also supports qualified class member names.
            payload = (
                b"\x80\x04c" + module.encode() + b"\n" + qualname.encode() + b"\n."
            )
            restored = pickle.loads(payload)
            assert (restored.__func__ if inspect.ismethod(restored) else restored) is (
                value.__func__ if inspect.ismethod(value) else value
            )
    else:
        assert actual not in RETIRED and actual.startswith("flagquantum.qec."), path
        owner = importlib.import_module(actual)
        for part in qualname.split("."):
            owner = getattr(owner, part)
        if isinstance(owner, property):
            owner = owner.fget
        assert (owner.__func__ if inspect.ismethod(owner) else owner) is (
            value.__func__ if inspect.ismethod(value) else value
        ), path


def assert_namespace(name: str, names: list[str]) -> None:
    module = "flagquantum.qec." + name
    if module in RETIRED:
        with pytest.raises(ModuleNotFoundError) as caught:
            importlib.import_module(module)
        assert caught.value.name == module
    else:
        owner = importlib.import_module(module)
        for attribute in names:
            assert hasattr(owner, attribute), (module, attribute)


def assert_surviving_objects() -> None:
    directory = Path(__file__).with_name("retirement_fixtures")
    expected = {
        "surviving_instances.pkl": "b92ca96f477120d3817002fd6c975aee0631b045fa03b8bc3fa3aa1b1b39f3f1",
        "pre_retirement_matrices.json": "3b88020b22e2dfaccae6de57e7040a049a68dada505157dcd2f36b047b7fec4c",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
    payload = pickle.loads((directory / "surviving_instances.pkl").read_bytes())
    assert {name: len(items) for name, items in payload.items()} == {
        "restructure_003_codes.pkl": 10,
        "restructure_005_globals.pkl": 4,
        "restructure_006_instances.pkl": 4,
        "restructure_002_matrices.pkl": 2,
    }

    def compare(old: object, new: object) -> None:
        assert type(new) is type(old)
        if isinstance(old, torch.Tensor):
            assert new.dtype == old.dtype
            assert torch.equal(new, old)
        elif dataclasses.is_dataclass(old):
            for field in dataclasses.fields(old):
                compare(getattr(old, field.name), getattr(new, field.name))
        elif isinstance(old, (tuple, list)):
            assert len(old) == len(new)
            for left, right in zip(old, new, strict=True):
                compare(left, right)
        elif isinstance(old, BaseException):
            assert new.args == old.args
        else:
            assert old == new

    for items in payload.values():
        for path, value in items.items():
            if path.startswith("flagquantum.qec."):
                assert type(value) is resolve(path)
            compare(value, pickle.loads(pickle.dumps(value, protocol=4)))
