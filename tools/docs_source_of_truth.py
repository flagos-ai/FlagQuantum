#!/usr/bin/env python3
"""Generate and validate documentation derived from executable manifests."""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

try:
    from tools.check_capability_maturity import (
        aggregate_claim_value,
        claim_values,
        maturity_errors,
    )
except (ImportError, ModuleNotFoundError):  # direct execution beside an older install
    from check_capability_maturity import (  # type: ignore[no-redef]
        aggregate_claim_value,
        claim_values,
        maturity_errors,
    )

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "docs/source_of_truth.json"
API_REF = re.compile(r"\bfq\.([A-Za-z_]\w*)")
LOCAL_LINK = re.compile(r"\[[^]]+\]\((?!https?://|#)([^)]+)\)")


def load(path: Path) -> object:
    display_path = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{display_path}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys
    )


def test_node_exists(target: str) -> bool:
    parts = target.split("::")
    if len(parts) < 2:
        return False
    path = ROOT / parts[0]
    if not path.is_file():
        return False
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes: list[ast.AST] = list(tree.body)
    for name in parts[1:]:
        match = next(
            (
                node
                for node in nodes
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                )
                and node.name == name
            ),
            None,
        )
        if match is None:
            return False
        nodes = list(match.body) if isinstance(match, ast.ClassDef) else []
    return True


def render_api(data: dict[str, object]) -> str:
    exports = data["stable_exports"]
    verification = data["verification"]
    assert isinstance(exports, list)
    assert isinstance(verification, dict)
    rows = "\n".join(
        f"| `fq.{name}` | Stable | executable contract |" for name in exports
    )
    return (
        "# Generated Stable API\n\nDo not edit. Source: `docs/public_api_v1.json`.\n\n| API | Stability | Verification |\n| --- | --- | --- |\n"
        + rows
        + "\n"
    )


def render_operators(data: dict[str, object]) -> str:
    lowerings = data["lowerings"]
    assert isinstance(lowerings, dict)
    names = sorted(lowerings)
    ops = sorted({op for value in lowerings.values() for op in value["supported"]})
    head = "| Operator | " + " | ".join(names) + " |"
    rule = "| --- | " + " | ".join("---" for _ in names) + " |"
    rows = [head, rule]
    for op in ops:
        rows.append(
            "| `"
            + op
            + "` | "
            + " | ".join(
                "yes" if op in lowerings[name]["supported"] else "no" for name in names
            )
            + " |"
        )
    return (
        "# Generated Operator Capabilities\n\nDo not edit. Source: `docs/operator_manifest.json`; `yes` means an executable registered lowering, not release or scalability evidence.\n\n"
        + "\n".join(rows)
        + "\n"
    )


CAPABILITY_CATEGORY_LABELS = {
    "build_and_compile": "Build and compile",
    "simulation_and_training": "Simulation and training",
    "distributed_execution": "Distributed execution",
    "deployment_and_extension": "Deployment and extension",
}

MATURITY_LABELS = {
    "experimental": "Experimental",
    "development_evidence": "Development evidence",
    "production_supported": "Production supported",
    "release_certified": "Release certified",
}

GENERATED_REGION = re.compile(
    r"(?P<begin><!-- BEGIN GENERATED (?P<name>[A-Z_]+) -->)\n.*?\n"
    r"(?P<end><!-- END GENERATED (?P=name) -->)",
    re.DOTALL,
)


def root_link(path: str, label: str) -> str:
    return f"[{label}](../../{path})"


def format_claim_value(value: object, value_format: str) -> str:
    if value_format == "integer":
        return f"{int(value):,}"
    if value_format == "bytes":
        byte_count = int(value)
        if byte_count >= 1024**3:
            return f"{byte_count:,} bytes ({byte_count / 1024**3:,.2f} GiB)"
        if byte_count >= 1024**2:
            return f"{byte_count:,} bytes ({byte_count / 1024**2:,.2f} MiB)"
        return f"{byte_count:,} bytes"
    if value_format == "seconds":
        return f"{float(value):,.2f} s"
    if value_format == "scientific":
        return f"{float(value):.2e}"
    if value_format == "boolean":
        return "true" if value else "false"
    return str(value)


def claim_measurements(claim: dict[str, object], payload: object, field: str) -> str:
    rendered: list[str] = []
    measurements = claim[field]
    assert isinstance(measurements, list)
    for measurement in measurements:
        assert isinstance(measurement, dict)
        values = claim_values(payload, str(measurement["selector"]))
        value = aggregate_claim_value(
            values, str(measurement.get("aggregate", "single"))
        )
        formatted = format_claim_value(value, str(measurement["format"]))
        rendered.append(f"**{measurement['label']}:** {formatted}")
    return "<br>".join(rendered).replace("|", "\\|")


def performance_claims(
    data: dict[str, object],
) -> list[tuple[dict[str, object], dict[str, object], object]]:
    result: list[tuple[dict[str, object], dict[str, object], object]] = []
    capabilities = data["capabilities"]
    assert isinstance(capabilities, dict)
    for capability in capabilities.values():
        assert isinstance(capability, dict)
        for claim in capability.get("performance_claims", []):
            assert isinstance(claim, dict)
            artifact = ROOT / str(claim["artifact"])
            result.append((capability, claim, load(artifact)))
    return result


def render_performance_claims(data: dict[str, object], link_prefix: str) -> str:
    rows = [
        "| Claim | Maturity and scope | Artifact-derived result | Recorded environment | Evidence identity |",
        "| --- | --- | --- | --- | --- |",
    ]
    for capability, claim, payload in performance_claims(data):
        artifact = str(claim["artifact"])
        evidence = (
            f"[raw JSON]({link_prefix}{artifact})<br>"
            f"SHA-256 `{claim['artifact_sha256']}`<br>"
            f"code `{claim['code_version']}`"
        )
        scope = f"`{claim['maturity']}`<br>{claim['scope']}".replace("|", "\\|")
        environment = (
            claim_measurements(claim, payload, "environment")
            + f"<br>**Metadata boundary:** {claim['environment_limitations']}"
        ).replace("|", "\\|")
        rows.append(
            f"| **{claim['title']}**<br>`{claim['id']}` | {scope} | "
            f"{claim_measurements(claim, payload, 'metrics')} | {environment} | {evidence} |"
        )
    if len(rows) == 2:
        rows.append("| No publishable performance claim | — | — | — | — |")
    return "\n".join(rows)


def render_readme_summary(data: dict[str, object]) -> str:
    capabilities = data["capabilities"]
    assert isinstance(capabilities, dict)
    rows = [
        "Maturity applies to each capability—not to the package as a whole.",
        "",
        "| Capability | Maturity | Current boundary |",
        "| --- | --- | --- |",
    ]
    for capability in capabilities.values():
        assert isinstance(capability, dict)
        boundary = str(capability["limitations"]).replace("|", "\\|")
        rows.append(f"| {capability['title']} | `{capability['level']}` | {boundary} |")
    rows.extend(
        [
            "",
            "The machine-validated [capability matrix](capability-maturity.toml) is the authority.",
            "The generated [capability catalog](docs/generated/CAPABILITIES.md) maps user goals",
            "to APIs, runtime modes, evidence, examples, and known boundaries.",
        ]
    )
    return "\n".join(rows)


def render_known_limitations(data: dict[str, object]) -> str:
    capabilities = data["capabilities"]
    assert isinstance(capabilities, dict)
    rows = [
        "# Known limitations",
        "",
        "Do not edit this generated region. Capability boundaries come from",
        "[`capability-maturity.toml`](../../capability-maturity.toml).",
        "",
        "| Capability | Current level | Limitation |",
        "| --- | --- | --- |",
    ]
    for capability in capabilities.values():
        assert isinstance(capability, dict)
        limitation = str(capability["limitations"]).replace("|", "\\|")
        rows.append(
            f"| {capability['title']} | `{capability['level']}` | {limitation} |"
        )
    rows.extend(
        [
            "",
            "## Validated public performance claims",
            "",
            render_performance_claims(data, "../../"),
            "",
            "These statements are support boundaries, not a development backlog. Promotion",
            "requires all evidence for the target level to pass",
            "`python tools/check_capability_maturity.py`.",
        ]
    )
    return "\n".join(rows)


def replace_generated_region(
    path: Path, name: str, body: str, source: str | None = None
) -> str:
    text = path.read_text(encoding="utf-8") if source is None else source
    matches = [
        match
        for match in GENERATED_REGION.finditer(text)
        if match.group("name") == name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{path.relative_to(ROOT)}: expected one generated region {name}"
        )
    match = matches[0]
    replacement = f"{match.group('begin')}\n{body.rstrip()}\n{match.group('end')}"
    return text[: match.start()] + replacement + text[match.end() :]


def capability_api_is_available(
    module: object,
    stable: set[str],
    name: str,
    maturity: str,
) -> bool:
    """Accept Stable Core names or importable canonical namespace paths."""

    if name in stable:
        return True
    if name.startswith("flagquantum."):
        parts = name.split(".")
        for boundary in range(len(parts), 0, -1):
            try:
                target = importlib.import_module(".".join(parts[:boundary]))
            except ModuleNotFoundError:
                continue
            try:
                for part in parts[boundary:]:
                    target = getattr(target, part)
            except AttributeError:
                return False
            return True
        return False
    if maturity != "experimental" or not name.startswith("experimental."):
        return False
    module_name = getattr(module, "__name__", None)
    if not isinstance(module_name, str):
        return False
    try:
        target = importlib.import_module(f"{module_name}.experimental")
        for part in name.split(".")[1:]:
            target = getattr(target, part)
    except (AttributeError, ModuleNotFoundError):
        return False
    return True


def public_api_label(name: str) -> str:
    if name.startswith("flagquantum."):
        return name
    return f"fq.{name}"


def render_capabilities(data: dict[str, object]) -> str:
    capabilities = data["capabilities"]
    assert isinstance(capabilities, dict)
    sections: list[str] = [
        "# FlagQuantum Capabilities",
        "",
        "Choose a supported workflow by user goal, runtime, hardware, and evidence level.",
        "This catalog is generated from the machine-validated",
        "[`capability-maturity.toml`](../../capability-maturity.toml) source of truth.",
        "",
        "> Maturity applies only to the scope stated in each row. A local, replicated,",
        "> sliced, or planned execution path is not distributed scalability evidence.",
        "",
        "## How to read maturity",
        "",
        "| Level | Meaning |",
        "| --- | --- |",
        "| **Release certified** | Release-gated with audited, reproducible evidence and no unresolved release blocker. |",
        "| **Production supported** | Supported path with compatibility, operational guidance, and target-hardware evidence. |",
        "| **Development evidence** | Executable and tested development result; not a production or general scalability claim. |",
        "| **Experimental** | Research surface without compatibility or production guarantees. |",
        "",
        "## Find a capability by goal",
        "",
        "| Goal | Capability | Maturity | Start here |",
        "| --- | --- | --- | --- |",
    ]
    for capability in capabilities.values():
        assert isinstance(capability, dict)
        title = str(capability["title"])
        maturity = MATURITY_LABELS[str(capability["level"])]
        start = root_link(str(capability["quick_start"]), "Run example")
        for goal in capability["user_goals"]:
            sections.append(f"| {goal} | {title} | {maturity} | {start} |")

    for category, category_label in CAPABILITY_CATEGORY_LABELS.items():
        sections.extend(["", f"## {category_label}", ""])
        for name, capability in capabilities.items():
            assert isinstance(capability, dict)
            if capability["category"] != category:
                continue
            apis = ", ".join(
                f"`{public_api_label(str(api))}`" for api in capability["public_apis"]
            )
            runtime_modes = ", ".join(
                f"`{mode}`" for mode in capability["runtime_modes"]
            )
            hardware = ", ".join(f"`{item}`" for item in capability["hardware"])
            sections.extend(
                [
                    f"### {capability['title']}",
                    "",
                    str(capability["summary"]),
                    "",
                    f"- **Maturity:** {MATURITY_LABELS[str(capability['level'])]}",
                    f"- **Public API:** {apis}",
                    f"- **Runtime modes:** {runtime_modes}",
                    f"- **Hardware:** {hardware}",
                    f"- **Gradient support:** `{capability['gradient_support']}`",
                    f"- **Distribution semantics:** `{capability['distribution_semantics']}`",
                    f"- **Start:** {root_link(str(capability['quick_start']), 'quick example')}",
                    f"- **Documentation:** {root_link(str(capability['documentation']), 'guide')}",
                    f"- **Known boundary:** {capability['limitations']}",
                    "",
                ]
            )
    sections.extend(
        [
            "",
            "## Validated public performance claims",
            "",
            "Every value below is read from a hash-bound raw artifact. Missing or changed",
            "evidence makes the source-of-truth check fail closed.",
            "",
            render_performance_claims(data, "../../"),
        ]
    )
    return "\n".join(sections).rstrip() + "\n"


def generated() -> dict[Path, str]:
    config = load(CONFIG)
    assert isinstance(config, dict)
    auth = config["authorities"]
    assert isinstance(auth, dict)
    api = load(ROOT / str(auth["stable_api"]))
    assert isinstance(api, dict)
    ops = load(ROOT / str(auth["operator_capability"]))
    assert isinstance(ops, dict)
    maturity_path = ROOT / str(auth["capability_maturity"])
    maturity = tomllib.loads(maturity_path.read_text(encoding="utf-8"))
    errors = maturity_errors(maturity, ROOT)
    if errors:
        raise ValueError(
            "invalid capability maturity matrix:\n- " + "\n- ".join(errors)
        )
    readme = replace_generated_region(
        ROOT / "README.md", "CAPABILITY_SUMMARY", render_readme_summary(maturity)
    )
    readme_claims = (
        "Every value below is read from a hash-bound raw artifact. Missing or changed\n"
        "evidence makes the source-of-truth check fail closed.\n\n"
        + render_performance_claims(maturity, "")
    )
    readme = replace_generated_region(
        ROOT / "README.md", "PERFORMANCE_CLAIMS", readme_claims, source=readme
    )
    return {
        ROOT / "docs/generated/STABLE_API.md": render_api(api),
        ROOT / "docs/generated/OPERATOR_CAPABILITIES.md": render_operators(ops),
        ROOT / "docs/generated/CAPABILITIES.md": render_capabilities(maturity),
        ROOT / "README.md": readme,
        ROOT
        / "docs/reference/KNOWN_LIMITATIONS.md": replace_generated_region(
            ROOT / "docs/reference/KNOWN_LIMITATIONS.md",
            "KNOWN_LIMITATIONS",
            render_known_limitations(maturity),
        ),
    }


def validate() -> list[str]:
    errors: list[str] = []
    config = load(CONFIG)
    assert isinstance(config, dict)
    authorities = config["authorities"]
    assert isinstance(authorities, dict)
    externally_validated = config.get("externally_validated_authorities", {})
    assert isinstance(externally_validated, dict)
    for authority, relative in authorities.items():
        path = ROOT / str(relative)
        if not path.exists():
            errors.append(
                f"declared authority {authority!r} does not exist: {relative}"
            )
            continue
        if authority in externally_validated:
            # Large benchmark trees have their own schema and release audits.
            # Documentation validation checks the declared boundary exists but
            # must not duplicate a multi-gigabyte evidence scan.
            continue
        authority_files = tuple(path.rglob("*.json")) if path.is_dir() else (path,)
        for authority_file in authority_files:
            if authority_file.suffix != ".json":
                continue
            try:
                load(authority_file)
            except (ValueError, json.JSONDecodeError) as error:
                errors.append(f"declared authority {authority!r} is invalid: {error}")
        if path.suffix == ".toml":
            try:
                tomllib.loads(path.read_text(encoding="utf-8"))
            except tomllib.TOMLDecodeError as error:
                errors.append(f"declared authority {authority!r} is invalid: {error}")

    api = load(ROOT / str(authorities["stable_api"]))
    assert isinstance(api, dict)
    stable = set(api["stable_exports"])
    verification = api.get("verification", {})
    assert isinstance(verification, dict)
    module = importlib.import_module(str(api["package"]))
    for name in stable:
        if not hasattr(module, name):
            errors.append(f"stable API is not importable: fq.{name}")
    for name in stable:
        test_path = verification.get(name)
        if not isinstance(test_path, str) or not test_path:
            errors.append(f"stable API has no executable contract mapping: fq.{name}")
        elif not test_node_exists(test_path):
            errors.append(
                f"stable API contract node does not exist for fq.{name}: {test_path}"
            )
    extra_verification = set(verification) - stable
    if extra_verification:
        errors.append(
            "stable API verification contains non-stable exports: "
            + ", ".join(sorted(extra_verification))
        )
    maturity_path = ROOT / str(authorities["capability_maturity"])
    maturity = tomllib.loads(maturity_path.read_text(encoding="utf-8"))
    for capability_name, capability in maturity["capabilities"].items():
        for name in capability["public_apis"]:
            if not capability_api_is_available(
                module,
                stable,
                name,
                capability["level"],
            ):
                errors.append(
                    f"{capability_name}: capability API is neither Stable Core nor "
                    f"an importable canonical namespace path: {public_api_label(name)}"
                )
    for doc in config["stable_documents"]:
        path = ROOT / doc
        text = path.read_text(encoding="utf-8")
        for name in API_REF.findall(text):
            if name not in stable:
                errors.append(f"{doc}: fq.{name} is not in the stable API manifest")
        for target in LOCAL_LINK.findall(text):
            clean = target.split("#", 1)[0]
            if clean and not (path.parent / clean).exists():
                errors.append(f"{doc}: broken local link {target}")
    for doc in config["intent_documents"]:
        text = (ROOT / doc).read_text(encoding="utf-8").lower()
        if "future intent" not in text[:1000]:
            errors.append(f"{doc}: must identify itself as future intent near the top")
    release = (ROOT / "docs/reference/RELEASE_NOTES.md").read_text(encoding="utf-8")
    if re.search(r"reviewer.approved ISSUE-|Recorded .*ISSUE-", release, re.I):
        errors.append(
            "release notes duplicate issue/review logs instead of user-visible changes"
        )
    for path, expected in generated().items():
        if not path.exists() or path.read_text(encoding="utf-8") != expected:
            errors.append(f"generated documentation is stale: {path.relative_to(ROOT)}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check:
        for path, content in generated().items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    errors = validate()
    if errors:
        raise SystemExit(
            "documentation source-of-truth check failed:\n- " + "\n- ".join(errors)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
