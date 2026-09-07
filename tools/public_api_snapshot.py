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
EXECUTION_OPTIONS_CONTRACT = ROOT / "contracts" / "execution-options-v1-candidate.json"
EXECUTION_PLAN_CONTRACT = ROOT / "contracts" / "execution-plan-v1-candidate.json"
EXECUTION_RESULT_CONTRACT = ROOT / "contracts" / "execution-result-v1-candidate.json"
MODULE_TRAINING_CONTRACT = ROOT / "contracts" / "module-training-v1-candidate.json"
ERRORS_MODULE_CONTRACT = ROOT / "contracts" / "errors-module-boundary-v1-candidate.json"
EXTENSION_PROTOCOL_CONTRACT = (
    ROOT / "contracts" / "extension-protocol-v1-candidate.json"
)
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
    options_contract = json.loads(
        EXECUTION_OPTIONS_CONTRACT.read_text(encoding="utf-8")
    )
    plan_contract = json.loads(EXECUTION_PLAN_CONTRACT.read_text(encoding="utf-8"))
    result_contract = json.loads(EXECUTION_RESULT_CONTRACT.read_text(encoding="utf-8"))
    module_training_contract = json.loads(
        MODULE_TRAINING_CONTRACT.read_text(encoding="utf-8")
    )
    errors_module_contract = json.loads(
        ERRORS_MODULE_CONTRACT.read_text(encoding="utf-8")
    )
    extension_protocol_contract = json.loads(
        EXTENSION_PROTOCOL_CONTRACT.read_text(encoding="utf-8")
    )
    actual = generate()
    names = actual["stable_exports"]
    assert isinstance(names, list)
    historical_exports = baseline["exports"]
    assert isinstance(historical_exports, dict)
    authorized_changes: set[str] = set()
    if (
        options_contract.get("implementation_authorized") is True
        and options_contract.get("root_manifest_authorized") is True
    ):
        authorized_changes = {
            str(options_contract["root_addition"]),
            *(
                name
                for name in options_contract.get("public_signatures", {})
                if "." not in name
            ),
        }
    if plan_contract.get("implementation_authorized") is True:
        authorized_changes.update(
            name
            for name in plan_contract.get("proposed_signatures", {})
            if "." not in name
        )
    if plan_contract.get("root_manifest_authorized") is True:
        authorized_changes.add(str(plan_contract["root_addition"]))
    if result_contract.get("implementation_authorized") is True:
        authorized_changes.update(result_contract.get("protected_root_changes", ()))
    if errors_module_contract.get("implementation_authorized") is True:
        authorized_changes.update(
            errors_module_contract.get("protected_root_changes", ())
        )
    missing = sorted(set(names) - set(historical_exports) - authorized_changes)
    if missing:
        return (
            "current stable exports are absent from the reviewed baseline and "
            f"require an approved additive API contract: {', '.join(missing)}",
        )
    compared_names = [name for name in names if name not in authorized_changes]
    expected = {
        "stable_exports": compared_names,
        "exports": {name: historical_exports[name] for name in compared_names},
    }
    compared_actual = {
        "stable_exports": compared_names,
        "exports": {name: actual["exports"][name] for name in compared_names},
    }
    errors: list[str] = []
    if expected != compared_actual:
        difference = "".join(
            difflib.unified_diff(
                render(expected).splitlines(keepends=True),
                render(compared_actual).splitlines(keepends=True),
                fromfile="reviewed retained API",
                tofile="actual public API",
            )
        )
        errors.append(
            "public API baseline changed; do not regenerate it without an approved "
            "API change proposal\n" + difference
        )
    expected_signatures = dict(options_contract.get("public_signatures", {}))
    if plan_contract.get("implementation_authorized") is True:
        expected_signatures.update(plan_contract.get("proposed_signatures", {}))
    if result_contract.get("implementation_authorized") is True:
        expected_signatures.update(result_contract.get("public_signatures", {}))
    if module_training_contract.get("implementation_authorized") is True:
        expected_signatures.update(
            module_training_contract.get("public_signatures", {})
        )
    if errors_module_contract.get("implementation_authorized") is True:
        expected_signatures.update(errors_module_contract.get("public_signatures", {}))
    errors.extend(
        _validate_authorized_execution_options(
            options_contract,
            names,
            expected_signatures=expected_signatures,
        )
    )
    if plan_contract.get("root_manifest_authorized") is True:
        errors.extend(_validate_authorized_execution_plan(plan_contract, names))
    if result_contract.get("implementation_authorized") is True:
        errors.extend(_validate_authorized_execution_result(result_contract))
    if module_training_contract.get("implementation_authorized") is True:
        errors.extend(_validate_authorized_module_training(module_training_contract))
    if errors_module_contract.get("implementation_authorized") is True:
        errors.extend(_validate_authorized_errors_module(errors_module_contract))
    if extension_protocol_contract.get("implementation_authorized") is True:
        errors.extend(
            _validate_authorized_extension_protocol(extension_protocol_contract)
        )
    return tuple(errors)


def _validate_authorized_execution_options(
    contract: dict[str, Any],
    names: list[str],
    *,
    expected_signatures: dict[str, str],
) -> list[str]:
    import flagquantum as fq

    errors: list[str] = []
    root_name = str(contract["root_addition"])
    if root_name not in names or root_name not in fq.__all__:
        errors.append(f"authorized root API {root_name} is missing")
    expected_fields = [item["name"] for item in contract["contract"]["fields"]]
    actual_fields = [item.name for item in dataclasses.fields(fq.ExecutionOptions)]
    if actual_fields != expected_fields:
        errors.append("ExecutionOptions fields differ from the approved proposal")
    if not fq.ExecutionOptions.__dataclass_params__.frozen:
        errors.append("ExecutionOptions must remain frozen")
    if hasattr(fq.ExecutionOptions(), "__dict__"):
        errors.append("ExecutionOptions must remain slots-based")
    objects = {
        "plan": fq.plan,
        "run": fq.run,
        "Circuit.plan": fq.Circuit.plan,
        "Circuit.run": fq.Circuit.run,
        "RuntimePolicy": fq.RuntimePolicy,
        "Module": fq.Module,
        "Module.forward": fq.Module.forward,
        "Module.execute": fq.Module.execute,
        "Module.save_checkpoint": fq.Module.save_checkpoint,
        "Module.load_checkpoint": fq.Module.load_checkpoint,
        "train": fq.train,
    }
    for name, expected_signature in expected_signatures.items():
        actual_signature = str(inspect.signature(objects[name], eval_str=False))
        if actual_signature != expected_signature:
            errors.append(
                f"authorized API signature changed: {name}: "
                f"{actual_signature} != {expected_signature}"
            )
    return errors


def _validate_authorized_execution_plan(
    contract: dict[str, Any], names: list[str]
) -> list[str]:
    import flagquantum as fq
    from flagquantum.runtime.execution_plan import ExecutionPlan

    errors: list[str] = []
    root_name = str(contract["root_addition"])
    if root_name not in names or root_name not in fq.__all__:
        errors.append(f"authorized root API {root_name} is missing")
        return errors
    if fq.ExecutionPlan is not ExecutionPlan:
        errors.append("fq.ExecutionPlan must be the canonical compilation model")
    dataclass_fields = {field.name for field in dataclasses.fields(ExecutionPlan)}
    for entry in contract["contract"]["properties"]:
        name = str(entry["name"])
        descriptor = getattr(ExecutionPlan, name, None)
        if name not in dataclass_fields and not isinstance(descriptor, property):
            errors.append(f"ExecutionPlan stable property is missing: {name}")
    for name in contract["contract"]["methods"]:
        if not callable(getattr(ExecutionPlan, str(name), None)):
            errors.append(f"ExecutionPlan stable method is missing: {name}")
    return errors


def _validate_authorized_execution_result(contract: dict[str, Any]) -> list[str]:
    import flagquantum as fq

    errors: list[str] = []
    result_contract = contract["result_contract"]
    annotations = result_contract["field_annotations"]
    result_fields = {
        field.name: str(field.type) for field in dataclasses.fields(fq.ExecutionResult)
    }
    measurement_fields = {
        field.name: str(field.type)
        for field in dataclasses.fields(fq.MeasurementResult)
    }
    if result_fields["plan"] != annotations["ExecutionResult.plan"]:
        errors.append("ExecutionResult.plan annotation changed")
    if measurement_fields["value"] != annotations["MeasurementResult.value"]:
        errors.append("MeasurementResult.value annotation changed")
    for name in result_contract["accessors"]:
        if not callable(getattr(fq.ExecutionResult, str(name), None)):
            errors.append(f"ExecutionResult stable accessor is missing: {name}")
    if "__getattr__" in fq.ExecutionResult.__dict__:
        errors.append("ExecutionResult must not delegate backend-native attributes")
    summary = fq.ExecutionResult().summary()
    if summary.get("schema") != result_contract["summary_schema"]:
        errors.append("ExecutionResult summary schema changed")
    if summary.get("version") != result_contract["summary_version"]:
        errors.append("ExecutionResult summary version changed")
    return errors


def _validate_authorized_module_training(contract: dict[str, Any]) -> list[str]:
    import flagquantum as fq
    import flagquantum.training as training

    errors: list[str] = []
    extension = contract["stable_extension"]
    additions = set(extension["additions"])
    if set(training.__all__) != additions:
        errors.append("flagquantum.training exports differ from Proposal 005")
    if hasattr(fq.Module, "run"):
        errors.append("Module must not expose a run method")
    for name in ("require_value", "diagnostics"):
        if not callable(getattr(fq.ExecutionResult, name, None)):
            errors.append(f"ExecutionResult stable accessor is missing: {name}")
    diagnostics = fq.ExecutionResult().diagnostics()
    result_contract = contract["result_semantics"]
    if diagnostics.get("schema") != result_contract["diagnostics_schema"]:
        errors.append("ExecutionResult diagnostics schema changed")
    if diagnostics.get("version") != result_contract["diagnostics_version"]:
        errors.append("ExecutionResult diagnostics version changed")
    fields = [field.name for field in dataclasses.fields(fq.TrainingResult)]
    if fields != contract["training_result"]["fields"]:
        errors.append("TrainingResult fields differ from Proposal 005")
    if not isinstance(getattr(fq.TrainingResult, "final_loss", None), property):
        errors.append("TrainingResult.final_loss property is missing")
    return errors


def _validate_authorized_errors_module(contract: dict[str, Any]) -> list[str]:
    import flagquantum as fq
    import flagquantum.errors as errors_module
    import flagquantum.training as training
    from flagquantum.runtime.execution_plan_contract import (
        ExecutionPlanContractError,
    )

    errors: list[str] = []
    extension = contract["stable_extension"]
    if set(errors_module.__all__) != set(extension["additions"]):
        errors.append("flagquantum.errors exports differ from Proposal 006")
    mappings = {
        fq.IRValidationError: errors_module.ValidationError,
        fq.IRSerializationError: errors_module.SerializationError,
        ExecutionPlanContractError: errors_module.PlanningError,
        training.TrainingStateError: errors_module.ExecutionError,
    }
    for specific, category in mappings.items():
        if not issubclass(specific, category):
            errors.append(f"{specific.__name__} is outside its approved error category")
    if "deployment_binding" in inspect.signature(fq.Module).parameters:
        errors.append("Module must not own deployment_binding")
    if (
        "deployment_binding"
        in fq.Module(lambda p: fq.Circuit(1).ry(0, p[0]), 1).get_extra_state()
    ):
        errors.append("Module extra state must not own deployment_binding")
    return errors


def _validate_authorized_extension_protocol(contract: dict[str, Any]) -> list[str]:
    import flagquantum as fq
    import flagquantum.ecosystem.extensions as extensions
    from flagquantum.errors import CapabilityError, ExecutionError, FlagQuantumError

    errors: list[str] = []
    declared = contract["stable_extensions"]
    expected = {
        str(name)
        for section in declared
        if section["namespace"] == "flagquantum.ecosystem.extensions"
        for name in section["additions"]
    }
    if set(extensions.__all__) != expected:
        errors.append(
            "flagquantum.ecosystem.extensions exports differ from Proposal 007"
        )
    leaked = sorted(expected & set(fq.__all__))
    if leaked:
        errors.append(
            "extension protocol names must not enter the stable root: "
            + ", ".join(leaked)
        )
    if extensions.SDK_API_VERSION != contract["protocol_semantics"]["sdk_api_version"]:
        errors.append("extension SDK API version differs from Proposal 007")
    mappings = {
        extensions.ExtensionError: FlagQuantumError,
        extensions.ExtensionCompatibilityError: CapabilityError,
        extensions.ExtensionLifecycleError: ExecutionError,
    }
    for specific, category in mappings.items():
        if not issubclass(specific, category):
            errors.append(
                f"{specific.__name__} is outside its Proposal 007 error category"
            )
    return errors


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
