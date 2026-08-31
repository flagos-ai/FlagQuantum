#!/usr/bin/env python3
"""Validate the implemented Stable Core against the migration baseline.

The v0.2 baseline remains an immutable audit record of the historical surface.
The current manifest selects the retained subset while the final frozen
contract is still being built. Updating either artifact requires an explicitly
authorized API review; neither may be regenerated merely to make CI pass.
"""

from __future__ import annotations

import argparse
import dataclasses
import difflib
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

    manifest = json.loads(MANIFEST.read_bytes())
    names = manifest["stable_exports"]
    return {
        "stable_exports": list(names),
        "exports": {name: describe(getattr(fq, name)) for name in names},
    }


def render(payload: dict[str, object]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def validate() -> tuple[str, ...]:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    actual = generate()
    names = actual["stable_exports"]
    assert isinstance(names, list)
    historical_exports = baseline["exports"]
    assert isinstance(historical_exports, dict)
    missing = sorted(set(names) - set(historical_exports))
    if missing:
        return (
            "current stable exports are absent from the reviewed baseline and "
            f"require an approved additive API contract: {', '.join(missing)}",
        )
    expected = {
        "stable_exports": names,
        "exports": {name: historical_exports[name] for name in names},
    }
    if expected == actual:
        return ()
    difference = "".join(
        difflib.unified_diff(
            render(expected).splitlines(keepends=True),
            render(actual).splitlines(keepends=True),
            fromfile="reviewed retained API",
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
        help="unsupported: retained API contracts must be reviewed manually",
    )
    args = parser.parse_args(argv)
    if args.write:
        parser.error("automatic API contract regeneration is disabled")
    errors = validate()
    if errors:
        print("\n".join(errors))
        return 1
    print("public API migration baseline passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
