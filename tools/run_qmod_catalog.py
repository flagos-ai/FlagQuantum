"""Run classified Qmod originals and explicitly selected approved revisions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools.synthesis_catalog import (
    Case,
    Catalog,
    asset_path,
    load_catalog,
    verify_catalog,
)


def run_case(
    case: Case, catalog: Catalog, *, revision: str, memory_mib: int = 512
) -> dict[str, object]:
    """Execute one verified catalog entry through the existing corpus audit path."""
    from tools.audit_qmod_corpus import probe

    if revision not in {"original", "approved"}:
        raise ValueError("revision must be original or approved")
    if memory_mib < 1:
        raise ValueError("memory_mib must be positive")
    source = case["approved"] if revision == "approved" else case["original"]
    path = asset_path(source["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
        raise ValueError(f"source digest mismatch: {case['id']}")
    result = probe(
        path,
        catalog["upstream_commit"],
        memory_mib * 1024**2,
        True,
        case["classical_inputs"],
        basis_mps=case["basis_mps"],
    )
    return {
        "id": case["id"],
        "revision": revision,
        "source_sha256": source["sha256"],
        **result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "case", nargs="?", help="original upstream path listed in CATALOG.md"
    )
    parser.add_argument(
        "--revision", choices=("original", "approved"), default="original"
    )
    parser.add_argument("--all", action="store_true", help="execute all 216 entries")
    parser.add_argument(
        "--verify", action="store_true", help="verify catalog assets without execution"
    )
    parser.add_argument("--memory-mib", type=int, default=512)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.memory_mib < 1 or bool(args.case) + args.all + args.verify != 1:
        parser.error(
            "choose exactly one case, --all or --verify and a positive memory limit"
        )
    catalog = load_catalog()
    verify_catalog(catalog)
    if args.verify:
        print(
            "Verified 216 original cases and six separately approved source revisions"
        )
        return
    cases = [case for case in catalog["cases"] if args.all or case["id"] == args.case]
    if not cases:
        parser.error(
            "unknown original case; use a path from tests/fixtures/classiq_qmod/corpus_catalog.json"
        )
    rows = []
    for case in cases:
        row = run_case(
            case, catalog, revision=args.revision, memory_mib=args.memory_mib
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    if args.output:
        args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(
        0 if all(row["stage"] == "execution_smoke_passed" for row in rows) else 1
    )


if __name__ == "__main__":
    main()
