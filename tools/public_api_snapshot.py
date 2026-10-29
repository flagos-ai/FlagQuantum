#!/usr/bin/env python3
"""Validate the implemented Stable Core against the migration baseline.

The v0.2 baseline remains an immutable audit record of the historical surface.
The current manifest selects the retained subset while the final frozen
contract is still being built. Updating either artifact requires an explicitly
authorized API review; neither may be regenerated merely to make CI pass.

The presence and type of ``fq.__version__`` remain part of the stable surface,
but its release-specific literal is intentionally not compared with the v0.2
value. Package metadata and runtime version identity derive from the same source
in ``flagquantum.version`` and are verified by the distribution checks.
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
OBSERVABLE_OUTPUTS_CONTRACT = (
    ROOT / "contracts" / "observable-outputs-v1-candidate.json"
)
TWIN_CONTRACT = ROOT / "contracts" / "twin-v1-candidate.json"
GRADIENT_CONTRACT = ROOT / "contracts" / "gradient-api-v1-candidate.json"
OPENQASM_IMPORT_CONTRACT = ROOT / "contracts" / "openqasm-import-v1-candidate.json"
EXPERIMENT_CONTRACT = ROOT / "contracts" / "experiment-file-v1-candidate.json"
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
    observable_outputs_contract = json.loads(
        OBSERVABLE_OUTPUTS_CONTRACT.read_text(encoding="utf-8")
    )
    jobs_contract = json.loads(
        (ROOT / "contracts/remote-jobs-v1-candidate.json").read_text()
    )
    twin_contract = json.loads(TWIN_CONTRACT.read_text(encoding="utf-8"))
    gradient_contract = json.loads(GRADIENT_CONTRACT.read_text(encoding="utf-8"))
    openqasm_import_contract = json.loads(
        OPENQASM_IMPORT_CONTRACT.read_text(encoding="utf-8")
    )
    experiment_contract = json.loads(EXPERIMENT_CONTRACT.read_text(encoding="utf-8"))
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
    if (
        extension_protocol_contract.get("implementation_authorized") is True
        and extension_protocol_contract.get("root_manifest_change") is True
    ):
        authorized_changes.update(extension_protocol_contract.get("root_additions", ()))
        authorized_changes.update(
            extension_protocol_contract.get("protected_root_changes", ())
        )
    authorized_changes.update(observable_outputs_contract.get("root_additions", ()))
    authorized_changes.update(observable_outputs_contract.get("root_removals", ()))
    if jobs_contract.get("implementation_authorized") is True:
        authorized_changes.update(jobs_contract["root_additions"])
    if (
        twin_contract.get("implementation_authorized") is True
        and twin_contract.get("root_manifest_authorized") is True
    ):
        authorized_changes.add(str(twin_contract["root_addition"]))
    if gradient_contract.get("implementation_authorized") is True:
        authorized_changes.update(gradient_contract.get("root_additions", ()))
        authorized_changes.update(gradient_contract.get("root_removals", ()))
    if openqasm_import_contract.get("implementation_authorized") is True:
        authorized_changes.update(openqasm_import_contract.get("root_additions", ()))
        authorized_changes.update(openqasm_import_contract.get("root_removals", ()))
    if experiment_contract.get("implementation_authorized") is True:
        authorized_changes.update(experiment_contract["root_additions"])
    missing = sorted(set(names) - set(historical_exports) - authorized_changes)
    if missing:
        return (
            "current stable exports are absent from the reviewed baseline and "
            f"require an approved additive API contract: {', '.join(missing)}",
        )
    # A version export cannot retain the historical release's literal while a
    # new distribution is cut. Keep the name in the manifest and baseline, but
    # leave its release-specific value to the package artifact checks.
    compared_names = [
        name
        for name in names
        if name not in authorized_changes and name != "__version__"
    ]
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
    if extension_protocol_contract.get("implementation_authorized") is True:
        expected_signatures.update(
            extension_protocol_contract.get("root_signatures", {})
        )
    if experiment_contract.get("implementation_authorized") is True:
        expected_signatures.update(experiment_contract["signatures"])
    expected_signatures.update(observable_outputs_contract.get("signatures", {}))
    if observable_outputs_contract["rules"].get(
        "tensor_request_equality_by_object_identity"
    ):
        import flagquantum as fq

        request_contract = observable_outputs_contract.get("output_request", {})
        if dataclass_record(fq.OutputRequest) != request_contract.get("dataclass"):
            errors.append("OutputRequest fields differ from approved proposal 080")
        if list(fq.OutputRequest.__slots__) != request_contract.get("slots"):
            errors.append("OutputRequest slots differ from approved proposal 080")
    if gradient_contract.get("implementation_authorized") is True:
        expected_signatures.update(gradient_contract.get("signatures", {}))
    if openqasm_import_contract.get("implementation_authorized") is True:
        expected_signatures.update(openqasm_import_contract.get("signatures", {}))
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
    if jobs_contract.get("implementation_authorized") is True:
        from flagquantum.remote import jobs

        for name, expected_signature in jobs_contract["public_signatures"].items():
            obj = jobs
            for component in name.split("."):
                obj = getattr(obj, component)
            if str(inspect.signature(obj)) != expected_signature:
                errors.append(f"authorized remote job API signature changed: {name}")
        import flagquantum as fq

        for name in jobs_contract["root_additions"]:
            if name not in names or name not in fq.__all__:
                errors.append(f"authorized remote job root API missing: {name}")
    if twin_contract.get("implementation_authorized") is True:
        import flagquantum as fq

        name = str(twin_contract["root_addition"])
        value = getattr(fq, name)
        if name not in names or name not in fq.__all__:
            errors.append(f"authorized Twin root namespace missing: {name}")
        if not isinstance(value, ModuleType) or value.__name__ != "flagquantum.twin":
            errors.append("authorized Twin root object must be flagquantum.twin")
        else:
            missing_symbols = sorted(
                set(twin_contract["public_symbols"]) - set(value.__all__)
            )
            if missing_symbols:
                errors.append(
                    "authorized Twin public API missing: " + ", ".join(missing_symbols)
                )
            for symbol, expected_signature in twin_contract[
                "public_signatures"
            ].items():
                if str(inspect.signature(getattr(value, symbol))) != expected_signature:
                    errors.append(f"authorized Twin API signature changed: {symbol}")
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
        "compile": fq.compile,
        "counts": fq.counts,
        "density_matrix": fq.density_matrix,
        "expectation": fq.expectation,
        "from_openqasm": fq.from_openqasm,
        "gradient": fq.gradient,
        "jacobian": fq.jacobian,
        "jvp": fq.jvp,
        "probabilities": fq.probabilities,
        "samples": fq.samples,
        "save": fq.save,
        "load": fq.load,
        "train": fq.train,
        "vjp": fq.vjp,
        "vn_entropy": fq.vn_entropy,
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
    expected_presence = {
        "proposal": "docs/development/API_CHANGE_PROPOSAL_092_RYDBERG_PRESENCE_TRAJECTORY.md",
        "record": "docs/api-changes/FQ-RYDBERG-PRESENCE-TRAJECTORY-20261025.md",
        "implementation_authorized": True,
        "delivery_stage": "runtime",
        "production_implemented": True,
        "operation": "rydberg.measure_presence",
        "source_version": "1.4",
        "lindblad_plan_version": "1.8",
        "root_admission": "fq.run(LindbladPlan), sampled 1.8 only",
        "result_plan_annotation_addition": "LindbladPlan",
        "additional_result_fields": [],
        "additional_result_classes": [],
        "environment_protocol_version": "1.1",
        "rng_requirement": "rng_torch_version",
        "consumer_tests": "tests/unit/test_rydberg_presence_contract.py",
        "compatibility_tests": "tests/unit/test_rydberg_presence_compatibility.py",
    }
    if contract.get("native_rydberg_presence_admission") != expected_presence:
        errors.append(
            "native Rydberg presence admission differs from approved proposal 092"
        )
    approved_run_signature = "(program_or_plan: 'Circuit | CircuitIR | ExecutionPlan | IsingBraidProgram | LindbladPlan', *, options: 'ExecutionOptions | None' = None, outputs: 'OutputRequest | Sequence[OutputRequest] | None' = None, noise_model: 'NoiseModel | None' = None, compiler: 'str | None' = None, target: 'str | None' = None, target_qubits: 'Sequence[int] | None' = None, shots: 'int | None' = None, name: 'str | None' = None) -> 'ExecutionResult'"
    if (
        str(inspect.signature(fq.run)) != approved_run_signature
        or contract["proposed_signatures"]["run"] != approved_run_signature
    ):
        errors.append("proposal 092 root input union differs from approval")
    from flagquantum.core.rydberg import RydbergProgram

    presence = getattr(RydbergProgram, "measure_presence", None)
    if not callable(presence) or str(inspect.signature(presence)) != (
        "(self, atom: 'str', *, at: 'float', efficiency: 'float') -> 'RydbergProgram'"
    ):
        errors.append("proposal 092 Core presence signature differs from approval")
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
    native = contract.get("native_ising_admission")
    expected_native = {
        "proposal": "docs/development/API_CHANGE_PROPOSAL_085_ISING_BRAID_PROGRAMS.md",
        "record": "docs/api-changes/FQ-ISING-BRAID-PROGRAMS-20261023.md",
        "frontend": "flagquantum.experimental.simulation.IsingBraidProgram",
        "plan_version": "1.1",
        "analysis_annotation": "CircuitAnalysis | None",
        "additional_payload_field": "outputs",
        "unread_observation": {
            "proposal": "docs/development/API_CHANGE_PROPOSAL_089_ISING_UNREAD_OBSERVATION.md",
            "record": "docs/api-changes/FQ-ISING-UNREAD-OBSERVATION-20261024.md",
            "implementation_authorized": True,
            "delivery_stage": "implemented",
            "operation": "ising.dephase_pair",
            "new_operation_source_and_plan_version": "1.2",
            "braid_only_source_and_plan_version": "1.1",
            "production_implemented": True,
            "consumer_tests": "tests/unit/test_ising_unread_contract.py",
            "compatibility_tests": "tests/unit/test_ising_unread_compatibility.py",
        },
    }
    expected_native["sampled_observation"] = {
        "proposal": "docs/development/API_CHANGE_PROPOSAL_091_ISING_SAMPLED_OBSERVATION.md",
        "record": "docs/api-changes/FQ-ISING-SAMPLED-OBSERVATION-20261024.md",
        "implementation_authorized": True,
        "delivery_stage": "implemented",
        "operation": "ising.measure_pair",
        "new_operation_source_and_plan_version": "1.3",
        "environment_protocol_version": "1.1",
        "rng_requirement": "rng_torch_version",
        "production_implemented": True,
        "consumer_tests": "tests/unit/test_ising_sampled_contract.py",
        "compatibility_tests": "tests/unit/test_ising_sampled_compatibility.py",
    }
    if native != expected_native:
        errors.append("native Ising admission differs from approved proposals 085/089")
    else:
        from flagquantum.core.ising import IsingBraidProgram
        from flagquantum.experimental import simulation
        from flagquantum.runtime._ising_plan import plan_ising

        if simulation.IsingBraidProgram is not IsingBraidProgram:
            errors.append("native Ising frontend must resolve to its Core authority")
        sampled = getattr(IsingBraidProgram, "measure_pair", None)
        if not callable(sampled) or str(inspect.signature(sampled)) != (
            "(self) -> 'IsingBraidProgram'"
        ):
            errors.append(
                "native Ising sampled source must expose measure_pair(self) -> IsingBraidProgram"
            )
        annotations = {
            field.name: str(field.type) for field in dataclasses.fields(ExecutionPlan)
        }
        if annotations["analysis"] != native["analysis_annotation"]:
            errors.append("ExecutionPlan.analysis differs from proposal 085")
        selected = plan_ising(IsingBraidProgram())
        if (
            selected.schema_version != native["plan_version"]
            or selected.analysis is not None
        ):
            errors.append("native Ising plan must use reviewed 1.1 semantics")
        if native["additional_payload_field"] not in selected.to_dict():
            errors.append("native Ising plan must seal ordered outputs")
        if "IsingBraidProgram" in fq.__all__:
            errors.append("IsingBraidProgram must remain an experimental export")
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
    approved_plan_annotation = (
        "ExecutionPlan | RuntimePlanContract | MPSProductionPlan | LindbladPlan | None"
    )
    if annotations["ExecutionResult.plan"] != approved_plan_annotation:
        errors.append("proposal 092 result-plan annotation differs from approval")
    if result_fields["plan"] != annotations["ExecutionResult.plan"]:
        errors.append("ExecutionResult.plan annotation changed")
    if measurement_fields["value"] != annotations["MeasurementResult.value"]:
        errors.append("MeasurementResult.value annotation changed")
    for name in result_contract["accessors"]:
        if not callable(getattr(fq.ExecutionResult, str(name), None)):
            errors.append(f"ExecutionResult stable accessor is missing: {name}")
    for name in result_contract.get("properties", ()):
        if not isinstance(getattr(fq.ExecutionResult, str(name), None), property):
            errors.append(f"ExecutionResult stable property is missing: {name}")
    if "__getattr__" in fq.ExecutionResult.__dict__:
        errors.append("ExecutionResult must not delegate backend-native attributes")
    summary = fq.ExecutionResult().summary()
    if summary.get("schema") != result_contract["summary_schema"]:
        errors.append("ExecutionResult summary schema changed")
    if summary.get("version") != result_contract["summary_version"]:
        errors.append("ExecutionResult summary version changed")
    errors.extend(
        _validate_terminal_instrument_admission(
            contract.get("terminal_instrument_sampling")
        )
    )
    return errors


def _validate_terminal_instrument_admission(admission: Any) -> list[str]:
    """Protect the approved Runtime-only entry without exposing a pending stub."""

    import flagquantum as fq
    import flagquantum.runtime as runtime

    fields = {
        "proposal",
        "approval_record",
        "namespace",
        "entry",
        "signature",
        "implementation_required",
        "namespace_prefix",
        "result",
        "result_metadata",
    }
    if not isinstance(admission, dict) or set(admission) != fields:
        return ["terminal instrument admission fields differ from proposal 083"]
    errors: list[str] = []
    if (
        admission["namespace"] != "flagquantum.runtime"
        or admission["entry"] != "sample_terminal"
        or admission["proposal"]
        != "docs/development/API_CHANGE_PROPOSAL_083_TERMINAL_INSTRUMENT_SAMPLING.md"
        or admission["approval_record"]
        != "docs/api-changes/FQ-TERMINAL-INSTRUMENT-SAMPLING-20261023.md"
    ):
        errors.append(
            "terminal instrument admission must retain its approved owner and decision"
        )
    required = admission["implementation_required"]
    if type(required) is not bool or not isinstance(admission["signature"], str):
        return [*errors, "terminal instrument activation/signature has an invalid type"]
    prefix = admission["namespace_prefix"]
    if not isinstance(prefix, list) or not all(isinstance(x, str) for x in prefix):
        return [*errors, "terminal instrument Runtime prefix must be a name list"]
    entry = "sample_terminal"
    expected = tuple(prefix) + ((entry,) if required else ())
    if entry in prefix or tuple(runtime.__all__) != expected:
        errors.append(
            "Runtime exports differ from the approved terminal instrument stage"
        )
    if entry in fq.__all__ or hasattr(fq, entry):
        errors.append("terminal instrument sampling must not enter the stable root")
    present = hasattr(runtime, entry)
    function = getattr(runtime, entry, None)
    if required:
        if not callable(function):
            errors.append("activated terminal instrument Runtime entry is missing")
        elif str(inspect.signature(function, eval_str=False)) != admission["signature"]:
            errors.append(
                "terminal instrument Runtime signature differs from proposal 083"
            )
    elif present:
        errors.append(
            "pending terminal instrument admission must not publish an API stub"
        )
    if admission["result"] != {
        "kind": "sample",
        "qubits": [],
        "value_dtype": "int64",
        "value_device": "cpu",
        "statistics": {},
    }:
        errors.append(
            "terminal instrument result must reuse the approved MeasurementResult shape"
        )
    metadata = admission["result_metadata"]
    expected_fields = {
        "fields": {
            "outcome_labels",
            "probabilities",
            "probability_correction",
            "terminal_instrument",
        },
        "instrument_fields": {
            "schema",
            "version",
            "dimension",
            "dtype",
            "branches",
            "requested_seed",
            "effective_seed",
            "torch_version",
        },
        "branch_fields": {"label", "kraus"},
    }
    if not isinstance(metadata, dict) or set(metadata) != {
        *expected_fields,
        "schema",
        "version",
    }:
        errors.append("terminal instrument metadata descriptor has unapproved fields")
        return errors
    if (
        metadata["schema"] != "flagquantum.terminal_instrument"
        or metadata["version"] != "1.0"
    ):
        errors.append("terminal instrument metadata identity differs from proposal 083")
    for name, approved in expected_fields.items():
        value = metadata[name]
        if (
            not isinstance(value, list)
            or not all(isinstance(x, str) for x in value)
            or len(value) != len(approved)
            or set(value) != approved
        ):
            errors.append(
                f"terminal instrument metadata {name} differs from proposal 083"
            )
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
            "flagquantum.ecosystem.extensions exports differ from its approved contract"
        )
    leaked = sorted(expected & set(fq.__all__))
    if leaked:
        errors.append(
            "extension protocol names must not enter the stable root: "
            + ", ".join(leaked)
        )
    if contract["protocol_semantics"]["sdk_api_version"] != extensions.SDK_API_VERSION:
        errors.append("extension SDK API version differs from its approved contract")
    mappings = {
        extensions.ExtensionError: FlagQuantumError,
        extensions.ExtensionCompatibilityError: CapabilityError,
        extensions.ExtensionLifecycleError: ExecutionError,
    }
    for specific, category in mappings.items():
        if not issubclass(specific, category):
            errors.append(f"{specific.__name__} is outside its approved error category")
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
