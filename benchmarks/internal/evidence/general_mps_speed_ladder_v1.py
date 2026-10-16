"""The frozen matched-speed MPS ladder: the release workload at reduced shapes.

The MPS release contract rests on a capacity premise no single device can hold,
so the speedup that contract publishes cannot be a ratio of the premise against
itself: a ratio needs a denominator one device can run. This module names the
workload that denominator is measured on.

The timed circuit is built by the same body the frozen capacity premise uses
(``general_mps_capacity_16``), at the same target world size, with the same
trainable-leaf-per-gate parameterization, the same optimizer, the same gradient
policy and the same reverse-tape rule. Only the site count and the bond
dimension differ, and they are the two numbers that set the size of that
workload rather than which workload it is. A rung is therefore a smaller copy of
the premise and not a second, unrelated circuit, which is what lets the release
gate read the sharded leg as the premise's own construction sharded.

``LADDER`` is the single source of the rung shapes and ``ACCEPTANCE`` names the
rung the released speedup is published at. ``speed_workload`` in
``benchmarks/manifests/mps_release_v1.json`` freezes a copy so the release gate
can read the ladder without executing this file, and both the producer and
``tests/benchmark_contract/test_mps_release_evidence.py`` fail if the two copies
disagree. The file is also the identity of the timed protocol: the sealer records
its digest as the workload a matched-speed payload measured, because the digest
it records for a capacity payload names a launcher whose site count is the
premise's and not the rung's.
"""

from __future__ import annotations

import hashlib
import json

LADDER: tuple[dict[str, object], ...] = (
    {"name": "matched_speed_512sites_chi64", "n_sites": 512, "trained_max_bond": 64},
    {"name": "matched_speed_1024sites_chi64", "n_sites": 1_024, "trained_max_bond": 64},
    {"name": "matched_speed_2048sites_chi64", "n_sites": 2_048, "trained_max_bond": 64},
    {"name": "matched_speed_4096sites_chi64", "n_sites": 4_096, "trained_max_bond": 64},
    {"name": "matched_speed_8192sites_chi64", "n_sites": 8_192, "trained_max_bond": 64},
)
ACCEPTANCE = "matched_speed_8192sites_chi64"
STEPS = 1


def rung(name: str) -> dict[str, object]:
    """Return the frozen rung called ``name``."""

    for entry in LADDER:
        if entry["name"] == name:
            return dict(entry)
    raise ValueError(f"the frozen matched-speed ladder names no rung {name!r}")


def acceptance_rung() -> dict[str, object]:
    """Return the rung the released speedup is published at."""

    return rung(ACCEPTANCE)


def ladder_fingerprint() -> str:
    """Return the digest of the frozen ladder, rung order included."""

    content = {"acceptance": ACCEPTANCE, "ladder": list(LADDER), "steps": STEPS}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
