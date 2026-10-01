"""The capability manifest states the routing strategies the code accepts.

`capability-maturity.toml` is where a user reads what routing supports, and
`ROUTING_STRATEGIES` is where the code decides it. The two went out of step
once: `sabre` and `sabre_layout` were added while the manifest still said
routing offers two strategies, so the two implementations that matter most on a
real device were invisible in the capability statement.

A prose field cannot be derived, so it is checked instead. Every strategy the
code accepts has to appear in that sentence, which means the next strategy
cannot ship undocumented.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flagquantum.compiler.routing import ROUTING_STRATEGIES

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

pytestmark = pytest.mark.unit

MANIFEST = Path(__file__).resolve().parents[3] / "capability-maturity.toml"


def test_the_manifest_names_every_routing_strategy_the_code_accepts() -> None:
    data = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    limitations = data["capabilities"]["program_compilation"]["limitations"]
    missing = [name for name in ROUTING_STRATEGIES if name not in limitations]

    assert not missing, (
        f"capability-maturity.toml names no routing strategy for {missing}; "
        "extend the program_compilation limitations sentence to cover every "
        "entry in flagquantum.compiler.routing.ROUTING_STRATEGIES"
    )
