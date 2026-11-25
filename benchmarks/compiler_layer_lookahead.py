"""Compare native layer lookahead with retained routing on development inputs."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from unittest.mock import patch

from benchmarks.compiler_sabre_trials import _metrics, _program
from flagquantum.compiler import CouplingMap
from flagquantum.compiler.topology import sabre


def main() -> None:
    rows: list[dict[str, object]] = []
    for width in (6, 8, 12, 16):
        for topology in ("line", "grid"):
            coupling = (
                CouplingMap.line(width)
                if topology == "line"
                else CouplingMap.grid(2, width // 2)
            )
            for seed in range(10):
                program = _program(width, seed)
                measurements: dict[str, dict[str, object]] = {}
                for label in ("baseline", "cirq_proposal", "selected"):
                    start = time.perf_counter()
                    if label == "baseline":
                        with patch.object(
                            sabre, "choose_layer_swaps", return_value=None
                        ):
                            plan = sabre.plan_sabre_swaps(program, coupling)
                    elif label == "cirq_proposal":
                        trial = sabre._plan_sabre_trial(
                            program, coupling, layer_lookahead=True
                        )
                        if trial is None:
                            raise RuntimeError("unbounded layer trial returned no plan")
                        plan = trial
                    else:
                        plan = sabre.plan_sabre_swaps(program, coupling)
                    seconds = time.perf_counter() - start
                    measurements[label] = {
                        **_metrics(program, coupling, plan),
                        "seconds": seconds,
                    }
                rows.append(
                    {
                        "width": width,
                        "topology": topology,
                        "seed": seed,
                        "measurements": measurements,
                    }
                )
    print(
        json.dumps(
            {
                "evidence_class": "development",
                "python": sys.version,
                "platform": platform.platform(),
                "source_sha256": hashlib.sha256(
                    Path(sabre.__file__).read_bytes()
                ).hexdigest(),
                "layer_source_sha256": hashlib.sha256(
                    Path(sabre.__file__).with_name("_layer_lookahead.py").read_bytes()
                ).hexdigest(),
                "scope": "four-trial baseline, standalone Cirq proposal, five-trial selection",
                "acceptance": "exact complete GF(2) operator, restored layout, every interaction on a coupling edge",
                "timing_limit": "shared host, one sample, no warmup; diagnostic only",
                "depth_definition": "dependency depth counting each logical CX and SWAP as one; not target CZ depth",
                "rows": rows,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
