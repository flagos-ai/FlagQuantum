"""What the wire-vocabulary census counts, and what it deliberately does not.

The census produces the number every progress report about the `wire` to `qubit`
migration quotes, so the three ways it can be wrong are pinned here as fixtures
rather than described in prose: a variadic parameter that a naive scan misses, a
package `__init__.py` that a private-path filter wrongly drops, and a method of a
private class that an `ast.walk` wrongly promotes to the public surface. Each
fixture is a tiny package written into `tmp_path`, so the assertion is about the
scanner and not about the repository's current state.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_CENSUS_PATH = _ROOT / "tools" / "census_wire_vocabulary.py"

_SPEC = importlib.util.spec_from_file_location("census_wire_vocabulary", _CENSUS_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_CENSUS = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _CENSUS
_SPEC.loader.exec_module(_CENSUS)


def _package(tmp_path: Path, files: dict[str, str]) -> Path:
    """Write a throwaway package and return its root, so identifiers stay short."""

    for relative, source in files.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
    return tmp_path / "flagquantum"


def _identifiers(sites: tuple[Any, ...]) -> set[str]:
    return {site.identifier for site in sites}


def test_variadic_parameters_are_counted_with_their_prefix(tmp_path: Path) -> None:
    root = _package(
        tmp_path,
        {
            "flagquantum/circuit.py": (
                "def any(*wires, unitary=None):\n" "    return (wires, unitary)\n"
            )
        },
    )
    scanned = _CENSUS.census(root)
    assert _identifiers(scanned.canonical) == {"flagquantum/circuit.py::any::*wires"}


def test_the_package_namespace_is_public_even_though_it_starts_with_underscore(
    tmp_path: Path,
) -> None:
    root = _package(
        tmp_path,
        {
            "flagquantum/observables/__init__.py": "def X(qubit=None, *, wire=None):\n    return qubit, wire\n",
            "flagquantum/observables/_internal.py": "def hidden(wire=None):\n    return wire\n",
        },
    )
    scanned = _CENSUS.census(root)
    assert _identifiers(scanned.aliases) == {
        "flagquantum/observables/__init__.py::X::wire"
    }
    assert scanned.canonical == ()


def test_a_name_inside_private_code_is_private_all_the_way_down(tmp_path: Path) -> None:
    root = _package(
        tmp_path,
        {
            "flagquantum/algorithms/thing.py": (
                "class _Helper:\n"
                "    def apply(self, wires):\n"
                "        return wires\n"
                "\n"
                "class _Outer:\n"
                "    class _Inner:\n"
                "        def run(self, n_wires):\n"
                "            return n_wires\n"
                "\n"
                "def _module_helper(wire):\n"
                "    return wire\n"
                "\n"
                "def public(wires):\n"
                "    return wires\n"
            )
        },
    )
    scanned = _CENSUS.census(root)
    assert _identifiers(scanned.canonical) == {
        "flagquantum/algorithms/thing.py::public::wires"
    }
    assert len(scanned.internal) == 3


def test_a_dunder_is_a_public_entry_point_even_though_it_starts_with_underscore(
    tmp_path: Path,
) -> None:
    root = _package(
        tmp_path,
        {
            "flagquantum/circuit.py": (
                "class Circuit:\n"
                "    def __init__(self, n_qubits=None, *, n_wires=None):\n"
                "        self.n_qubits = n_qubits if n_wires is None else n_wires\n"
                "\n"
                "    def _cache(self, wires):\n"
                "        return wires\n"
                "\n"
                "class _Hidden:\n"
                "    def __init__(self, n_wires=None):\n"
                "        self.n_wires = n_wires\n"
            )
        },
    )
    scanned = _CENSUS.census(root)
    assert _identifiers(scanned.aliases) == {
        "flagquantum/circuit.py::Circuit.__init__::n_wires"
    }
    assert scanned.canonical == ()
    # `_cache` and the private class stay excluded: only dunders are promoted.
    assert len(scanned.internal) == 2


def test_a_signature_with_both_spellings_is_an_alias(tmp_path: Path) -> None:
    root = _package(
        tmp_path,
        {
            "flagquantum/api.py": (
                "def a(qubit=None, *, wire=None):\n"
                "    return qubit, wire\n"
                "\n"
                "def b(wires=None):\n"
                "    return wires\n"
            )
        },
    )
    scanned = _CENSUS.census(root)
    assert _identifiers(scanned.aliases) == {"flagquantum/api.py::a::wire"}
    assert _identifiers(scanned.canonical) == {"flagquantum/api.py::b::wires"}


@pytest.mark.parametrize(
    ("parameter", "expected"),
    [
        ("wire", "qubit"),
        ("wires", "qubits"),
        ("*wires", "*qubits"),
        ("n_wires", "n_qubits"),
        ("n_counting_wires", "n_counting_qubits"),
        ("observable_wires", "observable_qubits"),
        ("wire_order", "qubit_order"),
        ("show_all_wires", "show_all_qubits"),
        ("left_wire", "left_qubit"),
        ("preferred_local_wires", "preferred_local_qubits"),
    ],
)
def test_the_replacement_rule_substitutes_the_root(
    parameter: str, expected: str
) -> None:
    assert _CENSUS.replacement_name(parameter) == expected


def test_the_repository_scan_separates_the_three_populations() -> None:
    scanned = _CENSUS.census(_ROOT / "flagquantum")
    assert scanned.canonical and scanned.aliases and scanned.internal
    # The three sizes are recorded once, in the contract the gate reads, and are
    # not repeated here. A pinned copy of a measurement states the number instead
    # of checking anything about the scanner, and it fails on every legitimate
    # change to the package -- including a file that stops existing -- which is
    # the opposite of what this test is for. What belongs to the scanner is that
    # it separates the populations and that the boundary sees the aliases.
    kinds = {
        "canonical": {site.kind for site in scanned.canonical},
        "aliases": {site.kind for site in scanned.aliases},
        "internal": {site.kind for site in scanned.internal},
    }
    assert kinds["canonical"] == {"canonical"}
    assert kinds["aliases"] == {"deprecated_alias"}
    assert kinds["internal"] == {"internal"}
    identifiers = {
        "canonical": {site.identifier for site in scanned.canonical},
        "aliases": {site.identifier for site in scanned.aliases},
        "internal": {site.identifier for site in scanned.internal},
    }
    assert not identifiers["canonical"] & identifiers["aliases"]
    assert not identifiers["canonical"] & identifiers["internal"]
    aliases = {site.identifier: site.replacement for site in scanned.aliases}
    assert aliases["flagquantum/observables/__init__.py::Z::wire"] == "qubit"
    assert (
        aliases[
            "flagquantum/deployment/cloud.py::CloudBackendProfile.simulator::n_wires"
        ]
        == "n_qubits"
    )
    # The constructor is the most user-visible signature in the package, so it is
    # the one a private-by-underscore rule silently dropped.
    assert aliases["flagquantum/circuit.py::Circuit.__init__::n_wires"] == "n_qubits"


def test_the_supersedes_helper_agrees_with_the_replacement_rule() -> None:
    assert _CENSUS.supersedes("wire", "qubit")
    assert _CENSUS.supersedes("n_wires", "n_qubits")
    assert not _CENSUS.supersedes("wire", "qubits")
    assert not _CENSUS.supersedes("qubit", "qubit")
