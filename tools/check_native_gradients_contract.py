#!/usr/bin/env python3
"""Validate the native gradients contract by re-deriving every declaration it records.

`contracts/native-gradients-contract.toml` registers FlagQuantum's native gradient
capability in one place. Before this file, that capability was described in three
places that could not see each other: the split real/imag statevector executor
summaries, the P5 autograd bridge summary, and the reversible adjoint behind the
distributed statevector reverse executor. None of them was compared with the
vocabulary the public entry point accepts, and that comparison is where the defect is.

Two things are checked here rather than read back as text.

First, the census. Every `gradient_method` / `backward_method` dict entry in
`flagquantum/**` is found by parsing the package, together with the scope that
declares it. The contract must have exactly one row per site, each classified onto
an axis, and a value that the gate did not find is a failure. A declaration added
anywhere in the package therefore turns this gate red until it is registered, which
is what keeps the registration from decaying into a snapshot.

Second, the correspondence. Every differentiation-route value that appears in the
census or in the benchmark engine table must have an `[[alias]]` row naming the
public method it corresponds to, and the gate drives `fq.gradient` for every public
method name and for every declared name that is not requestable. So the two failure
directions are both covered: a declared identifier that no caller can request, and a
served public method that nothing declares. Neither was checked before.

The gate does not change the implementation or the public surface. It records that
they disagree, which is the honest state of the repository and the input to a
separate decision about which vocabulary should win.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 and older
    import tomli as tomllib

import torch

import flagquantum as fq
from flagquantum import gradients as gradient_module
from flagquantum.benchmarking.differentiable_support import build_support_matrix
from flagquantum.errors import CapabilityError, ValidationError

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts" / "native-gradients-contract.toml"

#: The exception class each contracted `exception` name denotes. A name absent here is a
#: contracted class this gate cannot observe, which is a failure rather than a pass.
EXCEPTIONS: dict[str, type[BaseException]] = {
    "CapabilityError": CapabilityError,
    "ValidationError": ValidationError,
}

#: The package directories the census walks, relative to the repository root.
CENSUS_SOURCE = "flagquantum"

#: The field names the shared identifier appears under. Both are the same concept at the
#: call site and are only distinguishable by which axis their value belongs to.
CENSUS_FIELDS = ("backward_method", "gradient_method")

#: Every `[rules]` flag, named so that a flag nobody reads cannot be added unnoticed.
RULE_FLAGS = (
    "one_row_per_declaration_site",
    "every_declaration_is_axis_classified",
    "both_axes_share_one_identifier_and_that_is_recorded",
    "every_differentiation_route_value_has_an_alias",
    "a_new_declaration_fails_the_gate_until_registered",
    "a_new_engine_row_fails_the_gate_until_registered",
    "a_new_public_method_fails_the_gate_until_measured",
    "declared_names_are_driven_not_trusted",
    "no_public_surface_change",
    "no_silent_fallback",
)

#: Every `[verification]` flag, named for the same reason.
VERIFICATION_FLAGS = (
    "census_is_rederived_from_the_package_source",
    "public_vocabulary_is_rederived_from_the_implementation",
    "engine_table_is_rederived_from_the_benchmarking_module",
    "hybrid_phases_are_rederived_from_the_hybrid_contract",
    "every_refusal_is_driven_by_the_gate",
)

AXIS_ROUTE = "differentiation_route"
AXIS_SVD = "svd_truncation_rule"
AXIS_PASSTHROUGH = "field_passthrough"
AXES = (AXIS_ROUTE, AXIS_SVD, AXIS_PASSTHROUGH)

#: Sites whose `gradient_method` value is another field rather than a declaration, so the
#: gate reads the axis instead of a value. Named individually because the point of the
#: axis is that these two are the only ones.
PASSTHROUGH_SITES = (
    "flagquantum/benchmarking/differentiable_support.py",
    "flagquantum/simulation/mps/dense_island.py",
)

#: The values the truncation axis is allowed to take. A third value is a contract change.
SVD_VALUES = ("none", "projected_stop_subspace")

#: The identifier of the reference evaluation point, recorded in `[reference]`.
REFERENCE_PARAMETERS = torch.tensor([0.3, 0.7, -0.4], dtype=torch.float64)

#: The declared names the gate drives in addition to the public method vocabulary, because
#: each is asserted by something in the repository and must be observed to be refused.
PROBED_DECLARED_NAMES = (
    "adjoint",
    "statevector_adjoint",
    "torch_reverse_mode_autograd",
)


def _load_contract() -> dict[str, Any]:
    return tomllib.loads(CONTRACT.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Re-derivation from the package source.
# --------------------------------------------------------------------------- #
def _literal_values(source: str) -> tuple[tuple[str, ...], bool] | None:
    """Return the constant strings a value source can take, and whether it can be absent.

    ``None`` means the gate could not resolve the source to constants, which the census
    records as a dynamic value rather than guessing one.
    """
    try:
        node = ast.parse(source, mode="eval").body
    except SyntaxError:
        return None
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return (node.value,), False
        if node.value is None:
            return (), True
        return None
    if isinstance(node, ast.IfExp):
        left = _literal_values(ast.unparse(node.body))
        right = _literal_values(ast.unparse(node.orelse))
        if left is None or right is None:
            return None
        return left[0] + right[0], left[1] or right[1]
    return None


class _ScopeVisitor(ast.NodeVisitor):
    """Collect gradient-method dict entries together with the scope that produces them."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.stack: list[str] = []
        self.rows: list[dict[str, Any]] = []

    def _scope(self) -> str:
        head = self.path.name.removesuffix(".py")
        return f"{head}." + ".".join(self.stack) if self.stack else head

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815

    def visit_Dict(self, node: ast.Dict) -> None:
        for key, value in zip(node.keys, node.values, strict=True):
            if isinstance(key, ast.Constant) and key.value in CENSUS_FIELDS:
                self.rows.append(
                    {
                        "site": self.path.relative_to(ROOT).as_posix(),
                        "field": key.value,
                        "declared_by": self._scope(),
                        "value_source": ast.unparse(value),
                    }
                )
        self.generic_visit(node)


def _axis_of(site: str, values: tuple[str, ...] | None) -> str:
    """Classify one declaration onto the axis its value belongs to."""
    if values is None or site in PASSTHROUGH_SITES:
        return AXIS_PASSTHROUGH
    if values and all(value in SVD_VALUES for value in values):
        return AXIS_SVD
    return AXIS_ROUTE


#: The measurement functions below read the repository and never write it, so their
#: results are cached for the life of this process. The gate is a single-shot check
#: against a fixed working tree, and the contract test drives many mutated copies of one
#: contract through it, so re-walking the package and re-running the reference program per
#: call would multiply the cost without adding a measurement. Nothing here may be mutated
#: by a caller; the checks only read.
@functools.cache
def _census() -> tuple[dict[str, Any], ...]:
    """Every gradient-method declaration in the package, with its resolved value."""
    rows: list[dict[str, Any]] = []
    for path in sorted((ROOT / CENSUS_SOURCE).rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        visitor = _ScopeVisitor(path)
        visitor.visit(tree)
        for row in visitor.rows:
            parsed = _literal_values(str(row["value_source"]))
            values, can_be_absent = parsed if parsed is not None else (None, False)
            row["values"] = values
            row["value_can_be_absent"] = can_be_absent
            row["axis"] = _axis_of(str(row["site"]), values)
            rows.append(row)
    return tuple(rows)


@functools.cache
def _engine_rows() -> tuple[dict[str, Any], ...]:
    """The benchmark engine table's differentiation declaration, re-read from it."""
    rows: list[dict[str, Any]] = []
    for name, row in build_support_matrix(()).items():
        if "gradient_method" not in row:
            continue
        value = str(row["gradient_method"])
        rows.append(
            {
                "engine": name,
                "gradient_method": value,
                "identifier_shaped": bool(re.fullmatch(r"[a-z0-9_]+", value)),
            }
        )
    return tuple(rows)


@functools.cache
def _hybrid_rows() -> tuple[dict[str, Any], ...]:
    """The hybrid compilation contract's gradient-method declarations, re-read from it."""
    source = ROOT / "contracts" / "hybrid-compilation-private-v0-candidate.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for phase in ("phase5", "phase6"):
        block = payload[phase]
        for field in CENSUS_FIELDS:
            if field in block:
                rows.append(
                    {
                        "phase": phase,
                        "field": field,
                        "value": block[field],
                        "gradient_executor": block.get("gradient_executor"),
                    }
                )
    return tuple(rows)


@functools.cache
def _public_vocabulary() -> tuple[str, ...]:
    """The public method vocabulary, read from the module that validates it."""
    methods = getattr(gradient_module, "_GRADIENT_METHODS", None)
    if methods is None:
        return ()
    return tuple(str(method) for method in methods)


# --------------------------------------------------------------------------- #
# Re-measurement by driving the implementation.
# --------------------------------------------------------------------------- #
def _reference_circuit(parameters: torch.Tensor) -> Any:
    return (
        fq.Circuit(2, dtype=torch.complex128)
        .ry(0, theta=parameters[0])
        .rx(0, theta=parameters[1])
        .ry(1, theta=parameters[2])
        .cx(1, 0)
    )


def _reference_loss(circuit: Any) -> torch.Tensor:
    return fq.run(circuit, outputs=fq.expectation(fq.Z(0))).expectations[0]


@functools.cache
def _drive(method: str) -> dict[str, Any]:
    """Ask the public entry point for one method and record what it did."""
    outcome: dict[str, Any] = {"method": method}
    try:
        result = fq.gradient(
            _reference_circuit, REFERENCE_PARAMETERS, _reference_loss, method=method
        )
    except Exception as exc:
        outcome.update(
            {"served": False, "exception": type(exc).__name__, "message": str(exc)}
        )
    else:
        outcome.update(
            {
                "served": True,
                "reported_method": str(result.method),
                "exact": bool(result.exact),
                "step": result.step,
            }
        )
    return outcome


# --------------------------------------------------------------------------- #
# Checks.
# --------------------------------------------------------------------------- #
def _header_errors(contract: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if contract.get("schema") != "flagquantum_native_gradients_contract_v1":
        errors.append(f"schema is {contract.get('schema')!r}, not the v1 schema")
    if contract.get("maturity") != "development_evidence":
        errors.append(
            f"maturity is {contract.get('maturity')!r}; this file records evidence "
            "rather than a promise, so it must stay development_evidence"
        )
    expected_implementation = "flagquantum/gradients.py"
    if contract.get("implementation") != expected_implementation:
        errors.append(
            f"implementation is {contract.get('implementation')!r}, expected "
            f"{expected_implementation!r}"
        )
    if contract.get("census_source") != f"{CENSUS_SOURCE}/**/*.py":
        errors.append(
            f"census_source is {contract.get('census_source')!r}; the census walks "
            f"{CENSUS_SOURCE}/**/*.py"
        )
    if tuple(contract.get("census_fields", ())) != tuple(sorted(CENSUS_FIELDS)):
        errors.append(
            f"census_fields is {contract.get('census_fields')!r}, expected "
            f"{sorted(CENSUS_FIELDS)!r}"
        )
    authorization = contract.get("authorization")
    if not isinstance(authorization, list) or not authorization:
        errors.append("authorization must name the change record that admits this file")
    else:
        for record in authorization:
            if not (ROOT / str(record)).is_file():
                errors.append(f"authorization record {record!r} does not exist")
    if contract.get("shared_identifier") != "gradient_method":
        errors.append(
            f"shared_identifier is {contract.get('shared_identifier')!r}; the two axes "
            "share the identifier gradient_method, which is the finding"
        )
    return errors


def _axis_errors(
    contract: Mapping[str, Any], census: list[dict[str, Any]]
) -> list[str]:
    errors: list[str] = []
    declared = contract.get("axis", ())
    names = tuple(str(row.get("name")) for row in declared)
    if sorted(names) != sorted(AXES):
        errors.append(f"[[axis]] names {list(names)}, expected exactly {sorted(AXES)}")
    for row in declared:
        name = str(row.get("name"))
        if not str(row.get("meaning", "")).strip():
            errors.append(
                f"axis {name!r} states no meaning, so the axis has no justification"
            )
        if name not in AXES:
            continue
        measured = sum(1 for entry in census if entry["axis"] == name)
        if row.get("row_count") != measured:
            errors.append(
                f"axis {name!r} records row_count {row.get('row_count')!r}, "
                f"the census measures {measured}"
            )
    for name in AXES:
        if name not in names:
            continue
        if (
            not any(
                str(entry["axis"]) == name and str(entry["values"]) != "None"
                for entry in census
            )
            and name != AXIS_PASSTHROUGH
        ):
            errors.append(f"axis {name!r} has no census row, so it is a claim only")
    return errors


def _declaration_errors(
    contract: Mapping[str, Any], census: tuple[dict[str, Any], ...]
) -> list[str]:
    """One contracted row per measured site, with the measured value and axis."""
    errors: list[str] = []
    rows = contract.get("declaration", ())
    index: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("site")), str(row.get("declared_by")))
        index.setdefault(key, []).append(row)
    for measured in census:
        key = (str(measured["site"]), str(measured["declared_by"]))
        candidates = index.get(key)
        if not candidates:
            errors.append(
                f"{measured['site']} declares {measured['field']} in "
                f"{measured['declared_by']} with value source "
                f"{measured['value_source']!r}, which the contract does not record"
            )
            continue
        match = next(
            (
                row
                for row in candidates
                if str(row.get("value_source")) == str(measured["value_source"])
            ),
            None,
        )
        if match is None:
            errors.append(
                f"{measured['site']} declares {measured['field']} in "
                f"{measured['declared_by']} as {measured['value_source']!r}, which the "
                "contract does not record at that site"
            )
            continue
        if row_axis := str(match.get("axis")):
            if row_axis != measured["axis"]:
                errors.append(
                    f"{measured['site']}:{measured['declared_by']} is contracted on "
                    f"axis {row_axis!r}, the census classifies it as {measured['axis']!r}"
                )
        contracted_values = tuple(str(value) for value in match.get("values", ()))
        measured_values = (
            tuple(measured["values"]) if measured["values"] else ("dynamic",)
        )
        if contracted_values != measured_values:
            errors.append(
                f"{measured['site']}:{measured['declared_by']} records values "
                f"{list(contracted_values)}, the source resolves to "
                f"{list(measured_values)}"
            )
        if bool(match.get("value_can_be_absent")) != bool(
            measured["value_can_be_absent"]
        ):
            errors.append(
                f"{measured['site']}:{measured['declared_by']} records "
                f"value_can_be_absent={match.get('value_can_be_absent')!r}, measured "
                f"{measured['value_can_be_absent']!r}"
            )
    measured_keys = {
        (str(row["site"]), str(row["declared_by"]), str(row["value_source"]))
        for row in census
    }
    for row in rows:
        key = (
            str(row.get("site")),
            str(row.get("declared_by")),
            str(row.get("value_source")),
        )
        if key not in measured_keys:
            errors.append(
                f"contract records {key[0]}:{key[1]} = {key[2]!r}, which the census "
                "does not find"
            )
    return errors


def _alias_errors(
    contract: Mapping[str, Any],
    census: tuple[dict[str, Any], ...],
    engines: tuple[dict[str, Any], ...],
) -> list[str]:
    """Every differentiation-route value must be registered, and only those may be."""
    errors: list[str] = []
    aliases = contract.get("alias", ())
    registered: dict[str, Mapping[str, Any]] = {}
    for row in aliases:
        value = str(row.get("declared_value"))
        if value in registered:
            errors.append(f"[[alias]] declares {value!r} more than once")
        registered[value] = row
        requestable = row.get("requestable")
        if not isinstance(requestable, bool):
            errors.append(f"alias {value!r} must state requestable as a boolean")
        if requestable and not row.get("public_method"):
            errors.append(f"alias {value!r} is requestable but names no public method")
        if row.get("public_method") == "" and requestable:
            errors.append(
                f"alias {value!r} has an empty public method and cannot be requestable"
            )
    seen_route: set[str] = set()
    seen_off_route: set[str] = set()
    for row in census:
        target = seen_route if row["axis"] == AXIS_ROUTE else seen_off_route
        for value in row["values"] or ():
            target.add(value)
    for row in engines:
        seen_route.add(str(row["gradient_method"]))
    for value in sorted(seen_route):
        if value not in registered:
            errors.append(
                f"declared differentiation route {value!r} has no [[alias]] row; a new "
                "declaration must be registered before this gate can pass"
            )
    for value in sorted(registered):
        if value not in seen_route and value not in seen_off_route:
            errors.append(
                f"[[alias]] declares {value!r}, which no declaration or engine row uses"
            )
        if registered[value].get("requestable") and value in seen_off_route:
            errors.append(
                f"alias {value!r} belongs to another axis and cannot be a requestable "
                "gradient method"
            )
    # The two vocabularies must stay distinguishable: at least one declared route value
    # is not a public method name, which is the defect the contract records.
    if not any(
        value not in _public_vocabulary() and not registered[value].get("requestable")
        for value in sorted(registered)
        if value in seen_route
    ):
        errors.append(
            "contract implies every declared differentiation route is a public method "
            "name, which the implementation does not"
        )
    return errors


def _engine_errors(
    contract: Mapping[str, Any], engines: tuple[dict[str, Any], ...]
) -> list[str]:
    errors: list[str] = []
    rows = contract.get("engine", ())
    contracted = {str(row.get("engine")): row for row in rows}
    measured = {str(row["engine"]): row for row in engines}
    for name in sorted(set(measured) - set(contracted)):
        errors.append(
            f"engine {name!r} declares a gradient method the contract does not record"
        )
    for name in sorted(set(contracted) - set(measured)):
        errors.append(
            f"contract records engine {name!r}, which the table does not serve"
        )
    aliases = {str(row["declared_value"]): row for row in contract.get("alias", ())}
    for name in sorted(set(measured) & set(contracted)):
        row = contracted[name]
        if str(row.get("gradient_method")) != measured[name]["gradient_method"]:
            errors.append(
                f"engine {name!r} records gradient_method "
                f"{row.get('gradient_method')!r}, measured "
                f"{measured[name]['gradient_method']!r}"
            )
        if bool(row.get("identifier_shaped")) != measured[name]["identifier_shaped"]:
            errors.append(
                f"engine {name!r} records identifier_shaped "
                f"{row.get('identifier_shaped')!r}, measured "
                f"{measured[name]['identifier_shaped']!r}"
            )
        alias = aliases.get(measured[name]["gradient_method"])
        expected_public = "" if alias is None else str(alias.get("public_method", ""))
        if str(row.get("public_method", "")) != expected_public:
            errors.append(
                f"engine {name!r} records public_method {row.get('public_method')!r}, "
                f"the alias for {measured[name]['gradient_method']!r} says "
                f"{expected_public!r}"
            )
    return errors


def _hybrid_errors(contract: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    measured = _hybrid_rows()
    rows = contract.get("hybrid_phase", ())
    contracted = {(str(row.get("phase")), str(row.get("field"))): row for row in rows}
    for row in measured:
        key = (str(row["phase"]), str(row["field"]))
        if key not in contracted:
            errors.append(
                f"hybrid contract {key[0]} declares {key[1]} = {row['value']!r}, which "
                "this contract does not record"
            )
            continue
        contracted_row = contracted[key]
        if str(contracted_row.get("value")) != str(row["value"]):
            errors.append(
                f"hybrid {key[0]}.{key[1]} records {contracted_row.get('value')!r}, "
                f"measured {row['value']!r}"
            )
        if str(contracted_row.get("gradient_executor", "")) != str(
            row["gradient_executor"] or ""
        ):
            errors.append(
                f"hybrid {key[0]} records gradient_executor "
                f"{contracted_row.get('gradient_executor')!r}, measured "
                f"{row['gradient_executor']!r}"
            )
    for key in sorted(
        set(contracted) - {(str(r["phase"]), str(r["field"])) for r in measured}
    ):
        errors.append(
            f"contract records hybrid {key[0]}.{key[1]}, which the hybrid contract "
            "does not declare"
        )
    return errors


def _public_errors(contract: Mapping[str, Any]) -> list[str]:
    """Every public method name is driven, and the vocabulary is read from the module."""
    errors: list[str] = []
    vocabulary = _public_vocabulary()
    if not vocabulary:
        errors.append(
            "flagquantum.gradients._GRADIENT_METHODS is not readable, so the public "
            "method vocabulary cannot be re-derived"
        )
        return errors
    if tuple(contract.get("declared_public_methods", ())) != vocabulary:
        errors.append(
            f"declared_public_methods is "
            f"{list(contract.get('declared_public_methods', ()))}, the implementation "
            f"declares {list(vocabulary)}"
        )
    rows = contract.get("public_method", ())
    contracted = {str(row.get("method")): row for row in rows}
    for method in vocabulary:
        row = contracted.get(method)
        if row is None:
            errors.append(f"public method {method!r} is not measured by this contract")
            continue
        outcome = _drive(method)
        if bool(row.get("served")) != bool(outcome["served"]):
            errors.append(
                f"public method {method!r} records served={row.get('served')!r}, "
                f"measured {outcome['served']!r}"
            )
            continue
        if not outcome["served"]:
            errors.append(
                f"public method {method!r} is declared by the implementation but not "
                f"served: {outcome['message']}"
            )
            continue
        if str(row.get("reported_method")) != outcome["reported_method"]:
            errors.append(
                f"public method {method!r} records reported_method "
                f"{row.get('reported_method')!r}, measured {outcome['reported_method']!r}"
            )
        if bool(row.get("exact")) != bool(outcome["exact"]):
            errors.append(
                f"public method {method!r} records exact={row.get('exact')!r}, "
                f"measured {outcome['exact']!r}"
            )
        recorded_step = row.get("step")
        measured_step = outcome["step"]
        if measured_step is None:
            if recorded_step is not None:
                errors.append(
                    f"public method {method!r} records step {recorded_step!r}, measured "
                    "no step"
                )
        elif recorded_step is None or abs(
            float(recorded_step) - float(measured_step)
        ) > (1e-12 * max(1.0, abs(float(measured_step)))):
            errors.append(
                f"public method {method!r} records step {recorded_step!r}, measured "
                f"{measured_step!r}"
            )
    for method in sorted(set(contracted) - set(vocabulary)):
        errors.append(
            f"contract measures public method {method!r}, which the implementation does "
            "not declare"
        )
    # The declared-by-a-capability-block flag is a census fact, so it is re-derived here
    # rather than trusted.
    census = _census()
    declared_route_values = {
        value
        for entry in census
        if entry["axis"] == AXIS_ROUTE
        for value in (entry["values"] or ())
    }
    for method in vocabulary:
        row = contracted.get(method)
        if row is None:
            continue
        expected = method in declared_route_values
        if bool(row.get("declared_by_any_capability_block")) != expected:
            errors.append(
                f"public method {method!r} records "
                f"declared_by_any_capability_block="
                f"{row.get('declared_by_any_capability_block')!r}, measured {expected!r}"
            )
    return errors


def _refusal_errors(contract: Mapping[str, Any]) -> list[str]:
    """Every declared name that is not requestable must still be refused, and how."""
    errors: list[str] = []
    rows = contract.get("refusal", ())
    contracted = {str(row.get("requested")): row for row in rows}
    for name in PROBED_DECLARED_NAMES:
        row = contracted.get(name)
        if row is None:
            errors.append(
                f"declared name {name!r} is not recorded as a refusal, and it is "
                "asserted by the repository"
            )
            continue
        exception_name = str(row.get("exception"))
        expected = EXCEPTIONS.get(exception_name)
        if expected is None:
            errors.append(
                f"refusal {name!r} names exception {exception_name!r}, which this gate "
                "cannot observe"
            )
            continue
        try:
            fq.gradient(
                _reference_circuit,
                REFERENCE_PARAMETERS,
                _reference_loss,
                method=name,
            )
        except expected as exc:
            if str(exc) != str(row.get("message")):
                errors.append(
                    f"refusal {name!r} records a message that differs from the one "
                    f"raised; measured {str(exc)!r}"
                )
        except Exception as exc:
            errors.append(
                f"refusal {name!r} raises {type(exc).__name__}, the contract records "
                f"{exception_name}"
            )
        else:
            errors.append(
                f"refusal {name!r} is served, so the contract's refusal is stale"
            )
    for name in sorted(set(contracted) - set(PROBED_DECLARED_NAMES)):
        errors.append(
            f"contract records refusal {name!r}, which this gate does not drive"
        )
    return errors


def _reference_errors(
    contract: Mapping[str, Any],
    census: tuple[dict[str, Any], ...],
    engines: tuple[dict[str, Any], ...],
    public: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    reference = contract.get("reference", {})
    route_values = sorted(
        {
            value
            for entry in census
            if entry["axis"] == AXIS_ROUTE
            for value in (entry["values"] or ())
        }
    )
    engine_values = sorted({str(row["gradient_method"]) for row in engines})
    svd_values = sorted(
        {
            value
            for entry in census
            if entry["axis"] == AXIS_SVD
            for value in (entry["values"] or ())
        }
    )
    declared_route = set(route_values) | set(engine_values)
    aliases = {str(row["declared_value"]): row for row in contract.get("alias", ())}
    derived = {
        "declaration_sites": len(census),
        "declaration_files": len({str(row["site"]) for row in census}),
        "differentiation_route_rows": sum(
            1 for row in census if row["axis"] == AXIS_ROUTE
        ),
        "svd_truncation_rule_rows": sum(1 for row in census if row["axis"] == AXIS_SVD),
        "field_passthrough_rows": sum(
            1 for row in census if row["axis"] == AXIS_PASSTHROUGH
        ),
        "engine_rows": len(engines),
        "engine_rows_using_statevector_adjoint": sum(
            1 for row in engines if row["gradient_method"] == "statevector_adjoint"
        ),
        "engine_rows_with_a_prose_value": sum(
            1 for row in engines if not row["identifier_shaped"]
        ),
        "hybrid_declaration_rows": len(_hybrid_rows()),
        "public_methods": len(public),
        "public_methods_declared_by_a_capability_block": len(
            {
                str(row["method"])
                for row in public
                if str(row["method"]) in declared_route
            }
        ),
    }
    for key, measured in derived.items():
        if reference.get(key) != measured:
            errors.append(
                f"[reference] {key} records {reference.get(key)!r}, measured {measured!r}"
            )
    # The `[reference]` figures above are re-derived from the implementation, so they
    # would still read correctly while the contract's own rows disagreed with them. These
    # cross-checks close that gap: the recorded rows are counted against the same figures.
    contracted_rows = {
        "declaration_sites": len(contract.get("declaration", ())),
        "differentiation_route_rows": sum(
            1
            for row in contract.get("declaration", ())
            if row.get("axis") == AXIS_ROUTE
        ),
        "svd_truncation_rule_rows": sum(
            1 for row in contract.get("declaration", ()) if row.get("axis") == AXIS_SVD
        ),
        "field_passthrough_rows": sum(
            1
            for row in contract.get("declaration", ())
            if row.get("axis") == AXIS_PASSTHROUGH
        ),
        "engine_rows": len(contract.get("engine", ())),
        "engine_rows_using_statevector_adjoint": sum(
            1
            for row in contract.get("engine", ())
            if row.get("gradient_method") == "statevector_adjoint"
        ),
        "engine_rows_with_a_prose_value": sum(
            1 for row in contract.get("engine", ()) if not row.get("identifier_shaped")
        ),
        "hybrid_declaration_rows": len(contract.get("hybrid_phase", ())),
        "public_methods": len(contract.get("public_method", ())),
    }
    for key, counted in contracted_rows.items():
        if reference.get(key) != counted:
            errors.append(
                f"[reference] {key} records {reference.get(key)!r}, but the rows it "
                f"counts number {counted}"
            )
    for key, measured in (
        ("declared_differentiation_route_values", route_values),
        ("declared_svd_truncation_values", svd_values),
    ):
        if list(reference.get(key, ())) != measured:
            errors.append(
                f"[reference] {key} records {reference.get(key)!r}, measured {measured!r}"
            )
    not_declared = sorted(
        str(row["method"]) for row in public if str(row["method"]) not in declared_route
    )
    if list(
        reference.get("public_methods_not_declared_by_any_capability_block", ())
    ) != (not_declared):
        errors.append(
            "[reference] public_methods_not_declared_by_any_capability_block records "
            f"{reference.get('public_methods_not_declared_by_any_capability_block')!r}, "
            f"measured {not_declared!r}"
        )
    not_requestable = sorted(
        value
        for value in declared_route
        if not aliases.get(value, {}).get("requestable")
    )
    if (
        list(reference.get("declared_route_values_not_requestable", ()))
        != not_requestable
    ):
        errors.append(
            "[reference] declared_route_values_not_requestable records "
            f"{reference.get('declared_route_values_not_requestable')!r}, measured "
            f"{not_requestable!r}"
        )
    return errors


def _rule_errors(contract: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    rules = contract.get("rules", {})
    for flag in RULE_FLAGS:
        if rules.get(flag) is not True:
            errors.append(f"rule {flag!r} is not contracted as true")
    for flag in sorted(set(rules) - set(RULE_FLAGS)):
        errors.append(f"rule {flag!r} is not read by this gate")
    return errors


def _verification_errors(contract: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    verification = contract.get("verification", {})
    contract_path = str(verification.get("contract", ""))
    if not contract_path:
        errors.append("[verification] does not name a contract test")
    elif not (ROOT / contract_path).is_file():
        errors.append(f"[verification] contract test {contract_path!r} does not exist")
    gate_path = str(verification.get("gate", ""))
    if gate_path != "tools/check_native_gradients_contract.py":
        errors.append(
            f"[verification] gate is {gate_path!r}, expected this file; a contract that "
            "names another gate is not enforced by it"
        )
    for flag in VERIFICATION_FLAGS:
        if verification.get(flag) is not True:
            errors.append(f"verification flag {flag!r} is not contracted as true")
    for flag in sorted(set(verification) - set(VERIFICATION_FLAGS)):
        if flag in {"contract", "gate"}:
            continue
        errors.append(f"verification flag {flag!r} is not read by this gate")
    # The gate must be reachable from both the CI workflow and the pre-push runner, the
    # way every other contract gate in this repository is.
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    if "tools/check_native_gradients_contract.py" not in workflow:
        errors.append(
            "the gate is not invoked by .github/workflows/ci.yml, so it does not run "
            "in CI"
        )
    pre_push = (ROOT / "tools" / "pre_push.py").read_text(encoding="utf-8")
    if "tools/check_native_gradients_contract.py" not in pre_push:
        errors.append(
            "the gate is not invoked by tools/pre_push.py, so it does not run before "
            "a push"
        )
    return errors


def contract_errors(contract: dict[str, Any]) -> tuple[str, ...]:
    """Return every way the contract disagrees with the repository it describes."""
    census = _census()
    engines = _engine_rows()
    public = contract.get("public_method", ())
    errors = _header_errors(contract)
    errors.extend(_axis_errors(contract, census))
    errors.extend(_declaration_errors(contract, census))
    errors.extend(_alias_errors(contract, census, engines))
    errors.extend(_engine_errors(contract, engines))
    errors.extend(_hybrid_errors(contract))
    errors.extend(_public_errors(contract))
    errors.extend(_refusal_errors(contract))
    errors.extend(
        _reference_errors(
            contract, census, engines, list(public) if isinstance(public, list) else []
        )
    )
    errors.extend(_rule_errors(contract))
    errors.extend(_verification_errors(contract))
    return tuple(errors)


def _contract_digest() -> str:
    return hashlib.sha256(CONTRACT.read_bytes()).hexdigest()[:16]


def main() -> int:
    contract = _load_contract()
    errors = contract_errors(contract)
    if errors:
        print("\n".join(errors))
        return 1
    reference = contract.get("reference", {})
    print(
        "Native gradients contract passed: "
        f"{reference.get('declaration_sites')} declaration sites in "
        f"{reference.get('declaration_files')} files across "
        f"{len(contract.get('axis', ()))} axes, "
        f"{reference.get('engine_rows')} engine rows, "
        f"{reference.get('public_methods')} public methods driven, "
        f"{len(contract.get('alias', ()))} aliases and "
        f"{len(contract.get('refusal', ()))} refusals re-measured "
        f"(contract {_contract_digest()})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
