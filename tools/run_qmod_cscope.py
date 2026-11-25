"""Run an original Qmod QAOA/VQE workflow and print its saved values."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from flagquantum.ecosystem._qmod.source import load_qmod
from flagquantum.services._qmod_workflow import VqeRun, run_qaoa_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--memory-mib", type=int, default=512)
    args = parser.parse_args()
    if args.memory_mib < 1:
        parser.error("--memory-mib must be positive")
    imported = load_qmod(args.source)
    if imported.workflow is None:
        parser.error("source has no admitted QAOA/VQE workflow")
    result = run_qaoa_workflow(
        imported.workflow, memory_limit_bytes=args.memory_mib * 1024**2
    )
    print(
        json.dumps(
            {
                name: (
                    {
                        "best": asdict(value.best),
                        "evaluation_count": len(value.evaluations),
                        "termination": value.termination,
                    }
                    if isinstance(value, VqeRun)
                    else value
                )
                for name, value in result.saved.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
