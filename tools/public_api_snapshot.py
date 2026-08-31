#!/usr/bin/env python3
"""Capture and validate the pre-open-source FlagQuantum public API baseline.

This baseline is a migration aid, not the final Stable Core contract. Updating
it requires an explicitly authorized API review; it must never be regenerated
merely to make CI pass.
"""

from __future__ import annotations

import argparse
import dataclasses
import difflib
import hashlib
import inspect
import json
import re
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "public_api_v1.json"
BASELINE = ROOT / "contracts" / "public-api-v0.2-baseline.json"
ADDRESS = re.compile(r"0x[0-9a-fA-F]+")


def qualified_name(value: Any) -> str:
    module = getattr(value, "__module__", type(value).__module__)
    name = getattr(value, "__qualname__", type(value).__qualname__)
    return f"{module}.{name}"


def annotation_text(annotation: Any) -> str | None:
    if annotation is inspect.Parameter.empty:
        return None
    if isinstance(annotation, str):
        return annotation
    return inspect.formatannotation(annotation)


def stable_value(value: Any) -> object:
    if value is inspect.Parameter.empty or value is dataclasses.MISSING:
        return {"kind": "missing"}
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"kind": "literal", "value": value}
    if isinstance(value, Enum):
        return {
            "kind": "enum",
            "type": qualified_name(type(value)),
            "name": value.name,
        }
    if isinstance(value, (tuple, list)):
        return {
            "kind": type(value).__name__,
            "items": [stable_value(item) for item in value],
        }
    if isinstance(value, dict):
        return {
            "kind": "mapping",
            "items": [
                [str(key), stable_value(item)]
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            ],
        }
    return {
        "kind": "object",
        "type": qualified_name(type(value)),
        "repr": ADDRESS.sub("<address>", repr(value)),
    }


def signature_record(value: Any) -> dict[str, object] | None:
    try:
        signature = inspect.signature(value, eval_str=False)
    except (TypeError, ValueError):
        return None
    return {
        "parameters": [
            {
                "name": parameter.name,
                "kind": parameter.kind.name,
                "annotation": annotation_text(parameter.annotation),
                "default": stable_value(parameter.default),
            }
            for parameter in signature.parameters.values()
        ],
        "return": annotation_text(signature.return_annotation),
    }


def dataclass_record(value: type[Any]) -> list[dict[str, object]] | None:
    if not dataclasses.is_dataclass(value):
        return None
    fields = []
    for field in dataclasses.fields(value):
        default_factory: object = {"kind": "missing"}
        if field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            default_factory = {
                "kind": "factory",
                "value": qualified_name(field.default_factory),
            }
        fields.append(
            {
                "name": field.name,
                "type": annotation_text(field.type),
                "default": stable_value(field.default),
                "default_factory": default_factory,
                "init": field.init,
                "keyword_only": field.kw_only,
            }
        )
    parameters = getattr(value, "__dataclass_params__")
    return [
        {
            "class_frozen": bool(parameters.frozen),
            "class_order": bool(parameters.order),
        },
        *fields,
    ]


def describe(value: Any) -> dict[str, object]:
    if isinstance(value, ModuleType):
        return {"kind": "module", "module": value.__name__}
    if inspect.isclass(value):
        record: dict[str, object] = {
            "kind": "class",
            "qualified_name": qualified_name(value),
            "constructor": signature_record(value),
        }
        dataclass_fields = dataclass_record(value)
        if dataclass_fields is not None:
            record["dataclass"] = dataclass_fields
        if issubclass(value, Enum):
            record["enum_members"] = [item.name for item in value]
        return record
    if callable(value):
        return {
            "kind": "callable",
            "qualified_name": qualified_name(value),
            "signature": signature_record(value),
        }
    return {
        "kind": "value",
        "qualified_type": qualified_name(type(value)),
        "value": stable_value(value),
    }


def generate() -> dict[str, object]:
    import flagquantum as fq

    manifest_bytes = MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    names = manifest["stable_exports"]
    return {
        "schema": "flagquantum_public_api_baseline_v1",
        "status": "pre_open_source_migration_baseline",
        "source_manifest": str(MANIFEST.relative_to(ROOT)),
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "package_version": fq.__version__,
        "exports": {name: describe(getattr(fq, name)) for name in names},
    }


def render(payload: dict[str, object]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def validate() -> tuple[str, ...]:
    expected = BASELINE.read_text(encoding="utf-8") if BASELINE.is_file() else ""
    actual = render(generate())
    if expected == actual:
        return ()
    difference = "".join(
        difflib.unified_diff(
            expected.splitlines(keepends=True),
            actual.splitlines(keepends=True),
            fromfile=str(BASELINE.relative_to(ROOT)),
            tofile="actual public API",
        )
    )
    return (
        "public API baseline changed; do not regenerate it without an approved "
        "API change proposal\n" + difference,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="write an explicitly authorized migration baseline",
    )
    args = parser.parse_args(argv)
    if args.write:
        BASELINE.write_text(render(generate()), encoding="utf-8")
        print(f"wrote {BASELINE.relative_to(ROOT)}")
        return 0
    errors = validate()
    if errors:
        print("\n".join(errors))
        return 1
    print("public API migration baseline passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
