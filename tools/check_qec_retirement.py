#!/usr/bin/env python3
"""Check the preparation or 0.4.0 retirement boundary without changing QEC.

The retired phase intentionally fails on the preparation implementation. CI
exercises that refusal as well as the current compatibility phase. This tool
checks import/serialization boundaries; existing QEC tests own numerical,
signature, streaming and registry conformance against immutable fixtures.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import inspect
import json
import pickle
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts/qec-historical-import-retirement-v0.4.json"


def load_contract() -> dict[str, Any]:
    # JSON is the external data boundary. Contract tests validate its exact scope.
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def import_errors(modules: Sequence[str], *, absent: bool) -> list[str]:
    errors = []
    for name in modules:
        try:
            import_module(name)
        except ModuleNotFoundError as exc:
            if not absent or exc.name != name:
                errors.append(f"{name}: unexpected missing module {exc.name}")
        else:
            if absent:
                errors.append(f"{name}: retired module is still importable")
    return errors


def fixture_errors(root: Path = ROOT) -> list[str]:
    errors = []
    for name, expected in load_contract()["frozen_fixture_sha256"].items():
        path = root / name
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            errors.append(f"{name}: frozen fixture changed or missing")
    return errors


def _metadata_errors(value: object, expected: str, label: str) -> list[str]:
    errors = []
    if getattr(value, "__module__", None) != expected:
        errors.append(f"{label}: expected pickle module {expected}")
    if inspect.isclass(value):
        for name, member in vars(value).items():
            if isinstance(member, property):
                member = member.fget
            elif isinstance(member, (classmethod, staticmethod)):
                member = member.__func__
            if (
                inspect.isfunction(member)
                and member.__qualname__.startswith(value.__qualname__ + ".")
                and member.__module__ != expected
            ):
                errors.append(f"{label}.{name}: expected method module {expected}")
    return errors


def _pickle_errors(
    baseline: dict[str, Any], retired: set[str], *, active: bool
) -> list[str]:
    errors = []
    for name, encoded in baseline["pickles"].items():
        origin, attribute = name.rsplit(".", 1)
        payload = base64.b64decode(encoded)
        try:
            value = pickle.loads(
                payload
            )  # Trusted, hash-protected repository fixtures.
        except ModuleNotFoundError as exc:
            if not active or origin not in retired or exc.name not in retired:
                errors.append(f"{name}: unexpected pickle lookup failure {exc.name}")
        else:
            if active and origin in retired:
                errors.append(f"{name}: retired pickle lookup still succeeds")
            elif type(value) is not getattr(import_module(origin), attribute):
                errors.append(f"{name}: surviving pickle changed class identity")
            elif pickle.dumps(value, protocol=4) != payload:
                errors.append(f"{name}: surviving pickle changed bytes")
    return errors


def check(phase: str) -> list[str]:
    if phase not in ("prepare", "retired"):
        raise ValueError("phase must be prepare or retired")
    import flagquantum.qec as qec

    contract = load_contract()
    active = phase == "retired"
    modules = contract["module_replacements"]
    retired = set(modules)
    baseline = json.loads(
        (ROOT / contract["preserved"]["root_api_fixture"]).read_text()
    )
    errors = fixture_errors()
    from flagquantum.version import __version__

    if active and tuple(int(part) for part in __version__.split(".")[:2]) < (0, 4):
        errors.append(
            "release: retirement requires 0.4.0 or later, never a 0.3.x patch"
        )
    if list(qec.__all__) != baseline["exports"] or len(qec.__all__) != 108:
        errors.append("flagquantum.qec: ordered 108 root exports changed")
    errors += import_errors(tuple(modules), absent=active)
    for name, replacements in modules.items():
        errors += import_errors(replacements, absent=False)
        path = ROOT / "flagquantum/qec" / (name.rsplit(".", 1)[1] + ".py")
        if active and path.exists():
            errors.append(f"{name}: retired file remains in the source tree")

    for name, origin_path in baseline["origins"].items():
        origin, attribute = origin_path.rsplit(".", 1)
        value = getattr(qec, name)
        expected = origin
        if active and origin in retired:
            owners = [
                module
                for module in modules[origin]
                if getattr(import_module(module), attribute, None) is value
            ]
            if not owners:
                errors.append(f"{name}: no canonical owner preserves root identity")
                continue
            expected = owners[0]
        elif getattr(import_module(origin), attribute, None) is not value:
            errors.append(f"{name}: surviving root/import identity changed")
        errors += _metadata_errors(value, expected, name)
        if active and (inspect.isclass(value) or inspect.isfunction(value)):
            if pickle.loads(pickle.dumps(value, protocol=4)) is not value:
                errors.append(f"{name}: canonical global pickle changed identity")

    registry = import_module("flagquantum.qec.decoders.registry")
    expected_registry = registry.__name__ if active else "flagquantum.qec.registry"
    errors += _metadata_errors(registry._DecoderT, expected_registry, "_DecoderT")
    errors += _metadata_errors(
        qec.register_decoder("retirement_metadata_probe"),
        expected_registry,
        "register_decoder decorator",
    )

    from flagquantum.qec.codes import RotatedSurfaceCode
    from flagquantum.qec.codes.surface.rotated import RotatedSurfaceCode as Owner

    if not (qec.RotatedSurfaceCode is RotatedSurfaceCode is Owner):
        errors.append("RotatedSurfaceCode: surviving imports disagree")
    if type(qec.get_code("rotated_surface", distance=3)) is not Owner:
        errors.append(
            "RotatedSurfaceCode: factory no longer selects the original class"
        )
    surface = contract["rotated_surface"]
    if active:
        definitions = [
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "flagquantum/qec").rglob("*.py")
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(node, ast.ClassDef) and node.name == "RotatedSurfaceCode"
        ]
        if definitions != ["flagquantum/qec/codes/surface/rotated.py"]:
            errors.append(
                f"RotatedSurfaceCode: expected one definition, found {definitions}"
            )
        if (ROOT / surface["retired_implementation"]).exists():
            errors.append("RotatedSurfaceCode: duplicate implementation remains")
        errors += import_errors(
            (
                surface["forbidden_new_public_module"],
                "flagquantum.qec.codes.surface._rotated_frame",
            ),
            absent=True,
        )
    else:
        historical = import_module("flagquantum.qec.surface").RotatedSurfaceCode
        if historical is Owner or historical(3).checks != Owner(3).checks:
            errors.append(
                "RotatedSurfaceCode: preparation changed the historical baseline"
            )
    errors += _pickle_errors(baseline, retired, active=active)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "retired"), default="prepare")
    args = parser.parse_args()
    errors = check(args.phase)
    if errors:
        print("QEC retirement contract failed:\n" + "\n".join(errors))
        return 1
    print(f"QEC retirement contract passed: {args.phase}, 23 modules, 108 exports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
