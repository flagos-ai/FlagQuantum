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

import ast
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
    # The scan is live, not frozen: `WQ-2` and `WQ-3` renamed 30 of the 341
    # baseline sites, so 311 wire-named parameters are still on screen. The frozen
    # number lives in the contract's `[ledger]`, and the two agree through
    # `[retirement]`.
    assert len(scanned.canonical) == 311
    assert len(scanned.aliases) == 11
    assert len(scanned.internal) == 397
    assert scanned.canonical and scanned.aliases and scanned.internal
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


# ------------------------------------------------------------- the split rule


_PAYLOAD_FIXTURE = """
from dataclasses import asdict, dataclass


@dataclass
class Inner:
    wires: tuple = ()


@dataclass
class Outer:
    n_wires: int = 0
    inner: tuple[Inner, ...] = ()

    def to_dict(self):
        payload = asdict(self)
        return payload


class Plain:
    wire_pairs: tuple = ()


class _Private:
    wires: tuple = ()
"""


def test_an_attribute_stays_on_the_surface_when_nothing_serializes_it(
    tmp_path: Path,
) -> None:
    """The default is a rename, not an exemption.

    A rule that exempted anything it could not prove was a key would exempt the
    whole surface, since the proof never appears as a string literal.
    """

    root = _package(tmp_path, {"flagquantum/a.py": _PAYLOAD_FIXTURE})
    scanned = _CENSUS.attribute_census(root)
    assert _identifiers(scanned.ledgered) == {"flagquantum/a.py::Plain::wire_pairs"}
    assert scanned.ledgered[0].verdict == "canonical"
    assert scanned.ledgered[0].evidence == ""
    assert scanned.ledgered[0].replacement == "qubit_pairs"


def test_a_class_that_serializes_itself_excludes_its_own_fields(tmp_path: Path) -> None:
    root = _package(tmp_path, {"flagquantum/a.py": _PAYLOAD_FIXTURE})
    scanned = _CENSUS.attribute_census(root)
    outer = next(
        site
        for site in scanned.excluded
        if site.identifier.endswith("::Outer::n_wires")
    )
    assert outer.evidence == "serialization_method"
    assert outer.witness == "Outer"
    assert "`Outer`" in outer.reason
    assert outer.removal_condition


def test_a_field_of_a_serializing_class_excludes_the_contained_class(
    tmp_path: Path,
) -> None:
    """Containment is an annotation edge, not "mentioned in the same method".

    `asdict` recurses, so a field whose type is another dataclass makes that
    class's field names keys too. The witness has to be the container, because
    that is where the key actually appears.
    """

    root = _package(tmp_path, {"flagquantum/a.py": _PAYLOAD_FIXTURE})
    scanned = _CENSUS.attribute_census(root)
    inner = next(
        site for site in scanned.excluded if site.identifier.endswith("::Inner::wires")
    )
    assert inner.evidence == "contained"
    assert inner.witness == "Outer"
    assert "`Outer`" in inner.reason


def test_a_private_class_attribute_is_outside_the_boundary(tmp_path: Path) -> None:
    root = _package(tmp_path, {"flagquantum/a.py": _PAYLOAD_FIXTURE})
    scanned = _CENSUS.attribute_census(root)
    assert not any(
        "Private" in site.identifier for site in (*scanned.ledgered, *scanned.excluded)
    )


def test_the_payload_rule_disagrees_with_a_literal_scan_both_ways(
    tmp_path: Path,
) -> None:
    """The measured reason the rule is structural rather than textual.

    A literal scan is the cheap way to decide "is this name a payload key", and
    it is wrong in both directions on a package this size: it misses keys that
    `dataclasses.asdict` produces with no literal to find, and it exempts names
    that merely share a word with an unrelated literal. The fixture contains one
    of each, so the scanner's verdicts cannot be reproduced by a literal scan.
    """

    root = _package(
        tmp_path,
        {
            "flagquantum/a.py": (
                "from dataclasses import asdict, dataclass\n"
                "\n"
                "\n"
                "GREETING = 'wires'\n"
                "\n"
                "\n"
                "@dataclass\n"
                "class Payload:\n"
                "    wires: tuple = ()\n"
                "\n"
                "    def to_dict(self):\n"
                "        return asdict(self)\n"
                "\n"
                "\n"
                "class Unrelated:\n"
                "    wires: tuple = ()\n"
            )
        },
    )
    source = (root / "a.py").read_text(encoding="utf-8")
    literals = {
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    scanned = _CENSUS.attribute_census(root)
    assert literals == {"wires"}
    # The literal scan finds the key by accident and misses the real distinction:
    # both names share the spelling, so it cannot separate them at all.
    assert _identifiers(scanned.excluded) == {"flagquantum/a.py::Payload::wires"}
    assert _identifiers(scanned.ledgered) == {"flagquantum/a.py::Unrelated::wires"}
    assert {site.attribute for site in (*scanned.excluded, *scanned.ledgered)} == {
        "wires"
    }


def test_a_class_name_alone_is_not_evidence_of_containment(tmp_path: Path) -> None:
    """A serializer that never treats a class as a field cannot key it.

    The over-broad version of the rule collected every identifier appearing
    anywhere in a serializer body, which swept in helpers the method merely
    called. `Bystander` is mentioned in `Outer.to_dict` and must stay a rename.
    """

    root = _package(
        tmp_path,
        {
            "flagquantum/a.py": (
                "from dataclasses import asdict, dataclass\n"
                "\n"
                "\n"
                "@dataclass\n"
                "class Bystander:\n"
                "    wires: tuple = ()\n"
                "\n"
                "\n"
                "def asdict_of(value):\n"
                "    return asdict(value)\n"
                "\n"
                "\n"
                "@dataclass\n"
                "class Outer:\n"
                "    n_wires: int = 0\n"
                "\n"
                "    def to_dict(self):\n"
                "        Bystander()\n"
                "        return asdict_of(self)\n"
            )
        },
    )
    scanned = _CENSUS.attribute_census(root)
    assert _identifiers(scanned.ledgered) == {"flagquantum/a.py::Bystander::wires"}
    assert _identifiers(scanned.excluded) == {"flagquantum/a.py::Outer::n_wires"}


def test_the_repository_split_accounts_for_every_wire_named_attribute() -> None:
    scanned = _CENSUS.attribute_census(_ROOT / "flagquantum")
    # Live, like the parameter scan: `WQ-2` and `WQ-3` retired 27 of the 122
    # ledgered names and kept 3 of them as deprecated forwarders, so 95 are still
    # on screen.
    assert len(scanned.ledgered) == 95
    assert len(scanned.excluded) == 22
    assert len(_CENSUS.public_attribute_names(_ROOT / "flagquantum")) == 117
    assert _CENSUS.public_attribute_names(_ROOT / "flagquantum") == tuple(
        sorted(site.identifier for site in (*scanned.ledgered, *scanned.excluded))
    )


def test_every_declaration_kind_is_on_the_attribute_ledger() -> None:
    """A class puts a name on the screen three ways, and each is a site.

    `fq.Circuit.n_wires` is a property, so the annotated-assignment scan could not
    see it; `TextDrawer().wire_order` was assigned to `self` inside `__init__`, so
    neither of the other two doors saw it. The parameter ledger sees none of the
    three, so every missing kind leaves the spelling on screen.
    """

    scanned = _CENSUS.attribute_census(_ROOT / "flagquantum")
    kinds = {site.declaration for site in (*scanned.ledgered, *scanned.excluded)}
    assert kinds == set(_CENSUS.DECLARATION_KINDS)
    owned = {
        kind: {site.identifier for site in scanned.ledgered if site.declaration == kind}
        for kind in _CENSUS.DECLARATION_KINDS
    }
    assert "flagquantum/circuit.py::Circuit::n_wires" in owned["member"]
    # `WQ-2` renamed the instance example this test used to name. The live
    # instance kind is now the one site a slice still owes, and the renamed one
    # is gone from the scan rather than from the record.
    assert (
        "flagquantum/runtime/executors/mps/distributed_state.py"
        "::ShardedMPSState::n_wires"
    ) in owned["instance"]
    assert (
        "flagquantum/drawer/text_drawer.py::TextDrawer::wire_order"
        not in owned["instance"]
    )
    assert (
        "flagquantum/algorithms/amplitude_estimation.py"
        "::AmplitudeEstimationResult::n_counting_wires"
    ) in owned["field"]
    # The exclusion list is split across two of the three kinds, and an instance
    # attribute is the one kind that has never reached a payload.
    assert {site.declaration for site in scanned.excluded} == {"field", "member"}
    # A property plus its setter is one user-visible name, not two sites.
    identifiers = [site.identifier for site in (*scanned.ledgered, *scanned.excluded)]
    assert len(identifiers) == len(set(identifiers))


def test_an_instance_attribute_is_a_site_and_a_bare_assignment_is_not(
    tmp_path: Path,
) -> None:
    """The third door, on a fixture, so the rule is not only a repository fact.

    `self.n_wires = ...` inside a method is read by callers and is a site. A local
    variable that happens to be called `n_wires` is not: it never reaches an
    instance. And an instance attribute whose name is private is not a site either,
    which is what keeps `Circuit._n_wires` out of the ledger.
    """

    root = _package(
        tmp_path,
        {
            "flagquantum/a.py": (
                "class Drawer:\n"
                "    def __init__(self, wire_order=None):\n"
                "        self.wire_order = wire_order\n"
                "        self._wire_map = {}\n"
                "        n_wires = 3\n"
                "        self.n_wires = n_wires\n"
                "\n"
                "    def width(self):\n"
                "        return self.n_wires\n"
            )
        },
    )
    scanned = _CENSUS.attribute_census(root)
    assert _identifiers(scanned.ledgered) == {
        "flagquantum/a.py::Drawer::n_wires",
        "flagquantum/a.py::Drawer::wire_order",
    }
    assert {site.declaration for site in scanned.ledgered} == {"instance"}
    assert not scanned.excluded


def test_a_member_of_a_private_class_is_not_an_instance_site(tmp_path: Path) -> None:
    """Privacy is private all the way down, on the third door too."""

    root = _package(
        tmp_path,
        {
            "flagquantum/a.py": (
                "class _Hidden:\n"
                "    def __init__(self):\n"
                "        self.n_wires = 2\n"
            )
        },
    )
    scanned = _CENSUS.attribute_census(root)
    assert not scanned.ledgered
    assert not scanned.excluded


def test_every_exclusion_names_a_witness_and_a_removal_condition() -> None:
    """An exemption has to be arguable: what proves it, and what ends it."""

    scanned = _CENSUS.attribute_census(_ROOT / "flagquantum")
    for site in scanned.excluded:
        assert site.evidence in {
            "serialization_method",
            "contained",
            "payload_literal",
        }
        if site.evidence == "payload_literal":
            # A member's or instance attribute's spelling can only leave the
            # process because a method of its own class reads it, so the witness is
            # that method.
            assert site.witness.startswith(f"{site.owner}.")
            assert site.declaration in {"member", "instance"}
        elif site.evidence == "serialization_method":
            assert site.witness == site.owner
        else:
            assert site.witness and site.witness != site.owner
        assert site.reason
        assert site.removal_condition
        assert site.replacement != site.attribute


def test_a_member_is_not_excluded_by_its_class_serializing(tmp_path: Path) -> None:
    """The measured reason members need their own evidence rule.

    `dataclasses.asdict` walks fields, so it can never turn a property's name into
    a payload key. Reusing the class-level witness for members excluded
    `KrausChannel.n_wires` and `MPSState.n_wires` in this repository, neither of
    which can leave the process -- `MPSState` is not even a dataclass. A member is
    excluded only when the owning class's own serialization method spells it.
    """

    root = _package(
        tmp_path,
        {
            "flagquantum/a.py": (
                "from dataclasses import asdict, dataclass\n"
                "\n"
                "\n"
                "@dataclass\n"
                "class Payload:\n"
                "    n_wires: int = 0\n"
                "\n"
                "    def to_dict(self):\n"
                "        return asdict(self)\n"
                "\n"
                "    @property\n"
                "    def wire_count(self):\n"
                "        return self.n_wires\n"
            )
        },
    )
    scanned = _CENSUS.attribute_census(root)
    assert _identifiers(scanned.excluded) == {"flagquantum/a.py::Payload::n_wires"}
    assert _identifiers(scanned.ledgered) == {"flagquantum/a.py::Payload::wire_count"}


def test_a_member_is_excluded_when_its_own_class_reads_the_spelling(
    tmp_path: Path,
) -> None:
    """The one member-shaped payload key in the repository, on a fixture.

    `RuntimePolicy.from_dict` still accepts `observable_wires`, so that spelling
    is an input key. The witness is the method, not the class, because the class
    also emits the canonical name: scoping the literal to the owning class's own
    serialization methods is the difference between one exclusion and two false
    ones.
    """

    root = _package(
        tmp_path,
        {
            "flagquantum/a.py": (
                "class Policy:\n"
                "    observable_qubits: tuple = ()\n"
                "\n"
                "    @property\n"
                "    def observable_wires(self):\n"
                "        return self.observable_qubits\n"
                "\n"
                "    @classmethod\n"
                "    def from_dict(cls, payload):\n"
                "        return cls(payload.get('observable_wires', payload))\n"
            )
        },
    )
    scanned = _CENSUS.attribute_census(root)
    site = next(iter(scanned.excluded))
    assert site.identifier == "flagquantum/a.py::Policy::observable_wires"
    assert site.evidence == "payload_literal"
    assert site.witness == "Policy.from_dict"
    assert "`Policy.from_dict`" in site.reason
    assert site.removal_condition


def test_the_attribute_namespace_is_broader_than_the_wire_named_attributes() -> None:
    """Why the forbidden-alternative rule cannot reuse the wire-named scan."""

    every = _CENSUS.all_public_attribute_names(_ROOT / "flagquantum")
    wire_named = _CENSUS.public_attribute_names(_ROOT / "flagquantum")
    assert len(every) > len(wire_named)
    assert set(wire_named) <= set(every)
    # `qubit_indices` contains no `wire`, so only the whole namespace can see it.
    assert all("wire" in name.rsplit("::", 1)[1] for name in wire_named)
