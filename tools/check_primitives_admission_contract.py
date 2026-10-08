#!/usr/bin/env python3
"""Validate the fail-closed admission contract of the algorithm primitives package.

``flagquantum.algorithms.primitives`` is the one package in the repository whose public
surface is admitted by a stated rule rather than by a public-API decision. The rule was prose
in two files, and prose cannot be wrong in a way a check notices: an export could lose its
last consumer, or gain a distribution-semantics claim, and both READMEs would keep saying what
they always said.

This gate turns the rule into three measurements and compares them with
``contracts/primitives-admission-contract.toml``:

``distribution_semantics``
    The package may not name a non-default distribution semantic unless the contract records
    it for the module that defines the export. Rule 1 forbids reading replicated execution as
    scalability, and a primitive that mentions a semantic is making that claim.

``differentiability``
    An export may take a ``torch.Tensor`` without being differentiable through it -- today two
    do, because the amplitude vector is read as classical data -- but it must record that and
    say why. A recorded ``differentiable = true`` must name a gradient test that exists.

``consumers``
    The admission basis is derived from the measured import graph, not declared: ``shared``
    needs two algorithm-module consumers, ``confirmed`` exactly one, ``grounded_expectation``
    none but a consumer inside the package, and ``public_unit`` a caller outside the package
    with no consumer inside it. An export with no consumer anywhere is ``unadmitted``: the
    rule has teeth only if the gate refuses to record that as a basis, and the four grounds
    are then exhaustive by construction rather than by good intentions.

Every dimension also has to be *stated* where a contributor reads it, so the contract names
the files that state each one and the marker phrase they state it with. A rule whose prose and
whose check can drift apart is two rules, and the second one is the one that ships.

Run ``--measure`` to print the measured record for the current tree.
"""

from __future__ import annotations

import ast
import functools
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts" / "primitives-admission-contract.toml"
PACKAGE = "flagquantum/algorithms/primitives"
PACKAGE_DIR = ROOT / "flagquantum" / "algorithms" / "primitives"
FRAMEWORK_ROOT = "flagquantum"
CONSUMER_ROOTS = ("flagquantum", "tests", "examples", "benchmarks")
# The four grounds the prose states. `unadmitted` is not a fifth: it is what the measurement
# returns when none of the four holds, and the gate refuses a contract that records it.
ADMISSION_BASES = (
    "shared",
    "confirmed",
    "grounded_expectation",
    "public_unit",
)
UNADMITTED = "unadmitted"

# The three dimensions of the rule. The marker phrase each one is stated with, and the files
# that state it, belong to the contract: they are prose a contributor reads, and the gate's job
# is to check that the prose and the contract agree rather than to be a second place the rule is
# written down. Because it is one rule stated in several places, every dimension is required in
# every stating file -- a file that states two of the three is stating a different rule.
DIMENSIONS = ("distribution_semantics", "differentiability", "consumers")
# The one shape that explains a tensor argument with no gradient behind it. It is named in the
# gate as well as in the contract because the gate is what refuses the alternative.
CLASSICAL_TENSOR_SHAPE = "classical_tensor_input"


def _absolute_imports(path: Path) -> list[tuple[str, tuple[str, ...]]]:
    """Resolve every ``from ... import ...`` in ``path`` to an absolute module and its names.

    Relative imports are resolved against the file's own package, so a primitive imported as
    ``from .primitives.oracle import qft`` and one imported as
    ``from flagquantum.algorithms.primitives.oracle import qft`` are measured the same way.
    """
    parts = path.resolve().relative_to(ROOT).with_suffix("").parts
    package = ".".join(parts[:-1])
    resolved: list[tuple[str, tuple[str, ...]]] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            base = package.split(".")
            base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
            module = ".".join((*base, node.module)) if node.module else ".".join(base)
        else:
            module = node.module or ""
        resolved.append((module, tuple(alias.name for alias in node.names)))
    return resolved


def _in_package(module: str) -> bool:
    return module == "flagquantum.algorithms.primitives" or module.startswith(
        "flagquantum.algorithms.primitives."
    )


def _references(path: Path, name: str) -> bool:
    """Whether ``path`` mentions ``name`` anywhere in its code.

    An import graph cannot see a sibling that uses a name it defines itself, and the package's
    own modules are exactly where a primitive's first in-package consumer lives. Reading the
    module's names rather than its imports is what makes the ``grounded_expectation`` basis
    measurable instead of declared.
    """
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Name) and node.id == name:
            return True
        if isinstance(node, ast.Attribute) and node.attr == name:
            return True
        if isinstance(node, ast.ImportFrom) and any(
            alias.name == name for alias in node.names
        ):
            return True
    return False


def measure_consumers() -> dict[str, dict[str, list[str]]]:
    """Measure, per exported name, the files that consume it.

    ``framework_consumers`` are the files under ``flagquantum/`` outside this package that
    import the name -- a cross-module use must import it. ``package_consumers`` are the
    package's own modules, other than the one that defines the name, that mention it.
    """
    measured: dict[str, dict[str, list[str]]] = {}
    for root_name in CONSUMER_ROOTS:
        root = ROOT / root_name
        if not root.exists():
            continue
        bucket = "framework" if root_name == FRAMEWORK_ROOT else root_name
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(ROOT).as_posix()
            inside_package = relative.startswith(f"{PACKAGE}/")
            if inside_package:
                if path.name == "__init__.py":
                    # The package's own re-export list is the surface, not a consumer of it.
                    continue
                for name in measure_surface():
                    if relative != _module_of(name) and _references(path, name):
                        measured.setdefault(name, {}).setdefault("package", []).append(
                            relative
                        )
                continue
            for module, names in _absolute_imports(path):
                if not _in_package(module):
                    continue
                for name in names:
                    measured.setdefault(name, {}).setdefault(bucket, []).append(
                        relative
                    )
    return {
        name: {key: sorted(set(value)) for key, value in row.items()}
        for name, row in measured.items()
    }


@functools.lru_cache(maxsize=1)
def measure_surface() -> tuple[str, ...]:
    """Read the package's declared export list without importing it."""
    tree = ast.parse((PACKAGE_DIR / "__init__.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            return tuple(sorted(ast.literal_eval(node.value)))
    raise SystemExit(f"{PACKAGE}/__init__.py declares no __all__")


def _defining_module(name: str) -> Path | None:
    """The module that defines ``name``, or ``None`` when the package defines it nowhere."""
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name == name:
                    return path
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id == name:
                    return path
            elif isinstance(node, ast.Assign):
                if any(
                    isinstance(target, ast.Name) and target.id == name
                    for target in node.targets
                ):
                    return path
    return None


@functools.lru_cache(maxsize=None)
def _module_of(name: str) -> str:
    """The repository-relative path of the module that defines ``name``."""
    module = _defining_module(name)
    if module is None:
        raise SystemExit(f"{name} is declared in __all__ but defined nowhere")
    return module.relative_to(ROOT).as_posix()


def measure_semantics() -> tuple[dict[str, list[str]], str, tuple[str, ...]]:
    """Measure the distribution semantics each package module names.

    Returns the per-module non-default values, the declared default, and the full vocabulary,
    the last two read from the authoritative ``flagquantum.core.contracts`` declaration.
    """
    sys.path.insert(0, str(ROOT))
    from flagquantum.core.contracts import DistributionSemantics

    vocabulary = tuple(DistributionSemantics.__args__)
    default = vocabulary[0]
    named: dict[str, list[str]] = {}
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        found: list[str] = []
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in vocabulary and node.value != default:
                    found.append(node.value)
        if found:
            named[path.relative_to(ROOT).as_posix()] = sorted(set(found))
    return named, default, vocabulary


def measure_accepts_tensor(name: str, module: Path) -> bool:
    """Whether the export annotates any of its own parameters as a tensor."""
    tree = ast.parse(module.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if node.name != name:
            continue
        if isinstance(node, ast.ClassDef):
            return any(
                "Tensor" in ast.unparse(item.annotation)
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                for item in (
                    *child.args.posonlyargs,
                    *child.args.args,
                    *child.args.kwonlyargs,
                )
                if item.annotation is not None
            )
        return any(
            item.annotation is not None and "Tensor" in ast.unparse(item.annotation)
            for item in (
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            )
        )
    return False


def derive_basis(row: dict[str, list[str]]) -> str:
    """The admission basis the measured import graph implies."""
    framework = row.get("framework", [])
    package = row.get("package", [])
    if len(framework) >= 2:
        return "shared"
    if len(framework) == 1:
        return "confirmed"
    if package:
        return "grounded_expectation"
    if row.get("tests", []) or row.get("examples", []) or row.get("benchmarks", []):
        return "public_unit"
    return UNADMITTED


def measure() -> dict[str, Any]:
    consumers = measure_consumers()
    named, default, vocabulary = measure_semantics()
    rows = []
    for name in measure_surface():
        module = _defining_module(name)
        if module is None:
            raise SystemExit(f"{name} is declared in __all__ but defined nowhere")
        relative = module.relative_to(ROOT).as_posix()
        row = consumers.get(name, {})
        rows.append(
            {
                "name": name,
                "module": relative,
                "admission_basis": derive_basis(row),
                "framework_consumers": row.get("framework", []),
                "package_consumers": row.get("package", []),
                "test_consumers": row.get("tests", []),
                "example_consumers": row.get("examples", [])
                + row.get("benchmarks", []),
                "distribution_semantics": (
                    named[relative][0] if relative in named else default
                ),
                "accepts_tensor": measure_accepts_tensor(name, module),
            }
        )
    return {"default_semantics": default, "vocabulary": list(vocabulary), "rows": rows}


def contract_errors(
    contract: dict[str, Any], measured: dict[str, Any]
) -> tuple[str, ...]:
    errors: list[str] = []
    expected = {
        "schema": "flagquantum_primitives_admission_contract_v1",
        "package": PACKAGE,
    }
    if any(contract.get(name) != value for name, value in expected.items()):
        errors.append("primitives admission contract identity drifted")

    vocabulary = measured["vocabulary"]
    default = measured["default_semantics"]
    if contract.get("default_semantics") != default:
        errors.append(
            "primitives admission contract declares the wrong default distribution "
            f"semantic: contract={contract.get('default_semantics')!r}, core={default!r}"
        )
    if list(contract.get("distribution_semantics_vocabulary", [])) != vocabulary:
        errors.append(
            "primitives admission contract distribution-semantics vocabulary drifted "
            f"from flagquantum.core.contracts: contract="
            f"{contract.get('distribution_semantics_vocabulary')!r}, core={list(vocabulary)!r}"
        )

    errors.extend(_dimension_errors(contract))

    declared_shapes = _declared_shapes(contract, errors)
    rows = contract.get("primitive", [])
    recorded_names = [row.get("name") for row in rows]
    measured_names = [row["name"] for row in measured["rows"]]
    if recorded_names != measured_names:
        errors.append(
            "primitives admission contract surface drifted from the package: "
            f"contract={recorded_names}, package={measured_names}"
        )
        return tuple(errors)

    by_name = {row["name"]: row for row in measured["rows"]}
    for row in rows:
        name = row["name"]
        actual = by_name[name]
        basis = row.get("admission_basis")
        if actual["admission_basis"] == UNADMITTED:
            errors.append(
                f"primitive {name} has no consumer in the repository, so no admission "
                "basis holds for it; delete it, or give it the consumer that admits it"
            )
            continue
        if basis not in ADMISSION_BASES:
            errors.append(
                f"primitive {name} records an unknown admission basis {basis!r}"
            )
            continue
        if basis != actual["admission_basis"]:
            errors.append(
                f"primitive {name} admission basis drifted from its import graph: "
                f"contract={basis!r}, measured={actual['admission_basis']!r}, "
                f"framework={actual['framework_consumers']}, "
                f"package={actual['package_consumers']}"
            )
        for field, key in (
            ("framework_consumers", "framework_consumers"),
            ("package_consumers", "package_consumers"),
        ):
            recorded = sorted(row.get(field, []))
            if recorded != actual[key]:
                errors.append(
                    f"primitive {name} {field} drifted from the import graph: "
                    f"contract={recorded}, measured={actual[key]}"
                )
        semantics = row.get("distribution_semantics")
        if semantics not in vocabulary:
            errors.append(
                f"primitive {name} records an unknown distribution semantic {semantics!r}"
            )
        elif semantics != actual["distribution_semantics"]:
            errors.append(
                f"primitive {name} distribution semantic drifted from its module: "
                f"contract={semantics!r}, measured={actual['distribution_semantics']!r}"
            )
        if bool(row.get("accepts_tensor")) != actual["accepts_tensor"]:
            errors.append(
                f"primitive {name} tensor-parameter annotation drifted: "
                f"contract={row.get('accepts_tensor')!r}, measured={actual['accepts_tensor']!r}"
            )
        errors.extend(_differentiability_errors(name, row, actual, declared_shapes))

    errors.extend(_retired_export_errors(contract, recorded_names))
    for module, values in measure_semantics()[0].items():
        if module not in {row["module"] for row in rows}:
            errors.append(
                f"{module} names the distribution semantic(s) {values} but defines no "
                "exported primitive, so the claim is recorded nowhere"
            )
    return tuple(errors)


def _fold(text: str) -> str:
    """Case-fold ``text`` and collapse whitespace so a marker survives prose rewrapping."""
    return " ".join(text.split()).casefold()


def _declared_shapes(contract: dict[str, Any], errors: list[str]) -> set[str]:
    """Read the differentiability shape table, refusing an empty or unnamed one."""
    declared: set[str] = set()
    for shape in contract.get("differentiability", {}).get("shape", ()):
        name = shape.get("name")
        statement = shape.get("statement")
        if not isinstance(name, str) or not name:
            errors.append("primitives admission contract declares an unnamed shape")
            continue
        if name in declared:
            errors.append(
                f"primitives admission contract declares the shape {name} twice"
            )
        declared.add(name)
        if not isinstance(statement, str) or not statement.strip():
            errors.append(
                f"primitives admission shape {name} does not state what it means"
            )
    if CLASSICAL_TENSOR_SHAPE not in declared:
        errors.append(
            "primitives admission contract must declare the "
            f"{CLASSICAL_TENSOR_SHAPE!r} shape that a tensor-taking export with no "
            "gradient has to record"
        )
    return declared


def _differentiability_errors(
    name: str, row: dict[str, Any], actual: dict[str, Any], declared: set[str]
) -> list[str]:
    """Reconcile one export's differentiability record with its signature.

    A tensor in the signature is where the question is live. Such an export is either proved
    differentiable -- by a gradient test that exists and is named function by function -- or it
    is declared to read its tensor as classical data, with the command that measured that. The
    third answer, silence, is what the gate exists to refuse: a trainable-looking argument with
    no gradient behind it is the exact shape of a claim a user pays for and does not receive.
    """
    errors: list[str] = []
    shape = row.get("differentiability_shape")
    if shape not in declared:
        return [
            f"primitive {name} names the differentiability shape {shape!r}, which the "
            "contract does not declare"
        ]
    differentiate = row.get("differentiable")
    if differentiate not in (True, False):
        return errors + [
            f"primitive {name} does not state whether it is differentiable"
        ]
    if not actual["accepts_tensor"]:
        if differentiate:
            errors.append(
                f"primitive {name} takes no tensor and still claims differentiability"
            )
        if shape == CLASSICAL_TENSOR_SHAPE:
            errors.append(
                f"primitive {name} takes no tensor so it cannot read one as classical data"
            )
        return errors
    if differentiate:
        errors.extend(_gradient_test_errors(name, row.get("gradient_test")))
    elif shape != CLASSICAL_TENSOR_SHAPE:
        errors.append(
            f"primitive {name} takes a tensor, claims no gradient, and records the shape "
            f"{shape!r}; only {CLASSICAL_TENSOR_SHAPE!r} states why no gradient arrives"
        )
    command = row.get("differentiability_reproduce")
    if not isinstance(command, str) or not command.strip():
        errors.append(
            f"primitive {name} takes a tensor without recording how its differentiability "
            "was measured"
        )
    return errors


def _gradient_test_errors(name: str, test: Any) -> list[str]:
    """Resolve a ``differentiable = true`` claim to a test that exists.

    A claim of differentiability is the one admission dimension a user acts on with money and
    time, so it may not rest on a file that merely exists. ``path::test_name`` is resolved to
    the function, and a claim that resolves nowhere fails.
    """
    if not isinstance(test, str) or "::" not in test:
        return [
            f"primitive {name} claims differentiability without naming "
            "path::test_name for the gradient test"
        ]
    path_text, _, function = test.partition("::")
    path = ROOT / path_text
    if not path.is_file():
        return [f"primitive {name} names a missing gradient test file: {path_text!r}"]
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            return []
    return [
        f"primitive {name} names a gradient test that {path_text} does not define: "
        f"{function!r}"
    ]


def _dimension_errors(contract: dict[str, Any]) -> list[str]:
    """Reconcile the three dimensions with the files that are supposed to state them.

    The rule lived in prose in two files and in a docstring, and prose cannot be wrong in a way
    a check notices. Naming the stating files in the contract and requiring the dimension's own
    marker phrase in each one is what turns ``stated in three places`` from a claim into a
    measurement: a prose rewrite that drops a dimension now fails the same run that measures it.

    The comparison folds case and collapses whitespace, because prose wraps and sentences start
    with a capital: a marker that only matched an unwrapped lowercase occurrence would fail on
    the next paragraph edit for a reason that has nothing to do with the rule.
    """
    errors: list[str] = []
    stated_in = contract.get("stated_in")
    if not isinstance(stated_in, list) or not stated_in:
        errors.append(
            "primitives admission contract names no file that states the rule"
        )
        return errors
    texts: list[tuple[str, str]] = []
    for entry in stated_in:
        path = ROOT / str(entry)
        if not path.is_file():
            errors.append(
                f"primitives admission contract names a missing stating file {entry!r}"
            )
            continue
        texts.append((str(entry), _fold(path.read_text(encoding="utf-8"))))
    declared = [row.get("name") for row in contract.get("dimension", [])]
    if declared != list(DIMENSIONS):
        errors.append(
            "primitives admission contract dimensions must be the three the gate measures: "
            f"contract={declared}, measured={list(DIMENSIONS)}"
        )
        return errors
    for row in contract["dimension"]:
        name = row["name"]
        marker = row.get("marker")
        if not isinstance(marker, str) or not marker:
            errors.append(
                f"primitive admission dimension {name} does not say what phrase states it"
            )
            continue
        for entry, text in texts:
            if _fold(marker) not in text:
                errors.append(
                    f"{entry} does not state the {name} dimension: it never writes "
                    f"{marker!r}, so the rule a contributor reads there is a different rule"
                )
    return errors


def _retired_export_errors(
    contract: dict[str, Any], recorded_names: list[str]
) -> list[str]:
    """Check that an export the rule refused is contracted, not merely gone.

    The rule's first real finding was an export that no module, test, or example consumed. The
    repair is a deletion, and a deletion that is not written down is indistinguishable from an
    accident, so the contract has to carry the name, the consumer that was missing, and what a
    caller should use instead.
    """
    errors: list[str] = []
    live = set(recorded_names)
    for row in contract.get("retired_export", []):
        name = row.get("name")
        if not isinstance(name, str) or not name:
            errors.append("primitives admission contract retires an unnamed export")
            continue
        if name in live:
            errors.append(
                f"primitive {name} is recorded as retired and still on the export list"
            )
        for field in ("was_ground", "reason", "replaced_by"):
            if not isinstance(row.get(field), str) or not row[field]:
                errors.append(f"retired primitive {name} is missing its {field}")
    return errors


def main(argv: list[str]) -> int:
    measured = measure()
    if "--measure" in argv:
        import json

        print(json.dumps(measured, indent=2))
        return 0
    contract = tomllib.loads(CONTRACT.read_text(encoding="utf-8"))
    errors = contract_errors(contract, measured)
    if errors:
        print("\n".join(errors))
        return 1
    print(
        "Primitives admission contract passed: "
        f"{len(measured['rows'])} exports, "
        f"{sum(1 for row in measured['rows'] if row['admission_basis'] == 'public_unit')} "
        "admitted as public units"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
