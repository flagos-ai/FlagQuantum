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

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 and older
    import tomli as tomllib

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_CENSUS_PATH = _ROOT / "tools" / "census_wire_vocabulary.py"
_CONTRACT_PATH = _ROOT / "contracts" / "qubit-vocabulary-contract.toml"

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


def _contract() -> dict[str, Any]:
    """The frozen baseline, so a live count can be read against its record."""

    return tomllib.loads(_CONTRACT_PATH.read_text(encoding="utf-8"))


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
    # The scan is live, not frozen: the eight slices renamed all 341 baseline
    # sites, and the 11 names still on screen are aliases rather than unfinished
    # work -- each is a forwarding keyword that only the 0.4.0 removal deletes.
    # The frozen numbers live in the contract's `[ledger]` and `[aliases]`, and
    # the two agree through `[retirement]`.
    assert len(scanned.canonical) == 0
    assert len(scanned.aliases) == 11
    # The private bucket moves only as a side effect: `WQ-5` renamed the 41,
    # `WQ-6` the 23, `WQ-7` the 18 and `WQ-8` the 49 `wire`-named parameters of
    # their own private helpers, which no ledger counts.
    assert len(scanned.internal) == 226
    assert scanned.aliases and scanned.internal
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
    # Live, like the parameter scan: the eight slices retired 117 of the 122
    # ledgered names and kept 5 of them as deprecated forwarders, so 5 are still
    # on screen.
    assert len(scanned.ledgered) == 5
    assert len(scanned.excluded) == 22
    assert len(_CENSUS.public_attribute_names(_ROOT / "flagquantum")) == 27
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
    owned = {
        kind: {site.identifier for site in scanned.ledgered if site.declaration == kind}
        for kind in _CENSUS.DECLARATION_KINDS
    }
    # Every kind is proved on a fixture rather than on the repository, because
    # `WQ-8` renamed the last live `field` and `instance` example the scan could
    # name and `WQ-2`, `WQ-4`, `WQ-5`, `WQ-6` and `WQ-7` had already taken the
    # ones before it. The retired examples are gone from the scan rather than
    # from the record: `[attribute_retirement]` still carries each of them with
    # the declaration kind it discharged.
    assert set(owned) == set(_CENSUS.DECLARATION_KINDS)
    assert {site.declaration for site in scanned.ledgered} == {"member"}
    assert owned["field"] == set()
    assert owned["instance"] == set()
    assert "flagquantum/circuit.py::Circuit::n_wires" in owned["member"]
    assert (
        "flagquantum/compiler/openqasm_import.py::OpenQASMImport::n_wires"
        in owned["member"]
    )
    # The retirement table is where the exhausted kinds still live: `WQ-8`'s own
    # rows discharge a `field` and an `instance`, so both kinds are exercised end
    # to end and the ledger is not merely declared to cover them.
    contract = _contract()
    retired = {
        (row["site"], row["declaration"])
        for row in contract["attribute_retirement"]["sites"]
    }
    assert (
        "flagquantum/compiler/routing.py::CouplingMap::n_wires",
        "field",
    ) in retired
    assert (
        "flagquantum/remote/qpu/http.py::HttpQuantumProvider::default_n_wires",
        "instance",
    ) in retired
    assert "flagquantum/compiler/routing.py::CouplingMap::n_wires" not in owned["field"]
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


def test_a_documented_keyword_counts_wherever_wire_sits_in_its_name() -> None:
    """`wires`, `n_wires` and `terminal_wires` are one question, not three.

    The first matcher required the literal `wire` to *start* the identifier, which
    reported zero sites for `wires=(0,)` -- the single most common spelling in the
    documentation -- and every later count inherited the omission.
    """

    for name in ("wire", "wires", "n_wires", "terminal_wires", "show_wire_labels"):
        assert _CENSUS.documentation_keywords(f"{name}=1") == (f"{name}=",)


def test_the_documentation_matcher_ignores_every_other_way_wire_appears() -> None:
    """A keyword argument is an instruction to a reader; nothing else is.

    Counting the bare word would forbid the sentences that have to keep it: a
    comparison against CUDA-Q's wire order, a CLI flag, and the English verb are
    all correct as written, so a matcher that flagged them would be turned off
    rather than satisfied.
    """

    for text in (
        "info.wires=1",  # an attribute read on a caller's object
        "--n-wires 12",  # a command-line flag
        "n_wires == 2",  # a comparison
        "n_wires >= 2",
        "wire count",  # the bare noun
        "a five-wire circuit",  # a compound noun
        "wired into the selector",  # the English verb
        "n_qubits=2",  # the replacement vocabulary
        "qubits=(0,)",
    ):
        assert _CENSUS.documentation_keywords(text) == ()


def test_the_documentation_matcher_reports_each_occurrence_not_each_name() -> None:
    """A count, not a set: dropping one of two sites must move the number."""

    text = "build_workload(n_wires=22, layers=2)\nrecommend(n_wires=22)\n"
    assert _CENSUS.documentation_keywords(text) == ("n_wires=", "n_wires=")


def test_the_documentation_walk_skips_dated_records_and_keeps_the_live_docs() -> None:
    """The boundary is which documents speak about *now*, not which are clean."""

    files = _CENSUS.documentation_files(_ROOT)
    assert "docs/reference/API.md" in files
    assert "docs/guides/SIMULATOR_ADVISOR.md" in files
    assert "README.md" in files
    # A record states what the tree said on its date, so it must be able to quote
    # the retired spelling without the gate calling that a regression.
    for excluded in _CENSUS.DOCUMENTATION_EXCLUDED_PREFIXES:
        assert not [name for name in files if name.startswith(excluded)]
    for excluded in _CENSUS.DOCUMENTATION_EXCLUDED_FILES:
        assert excluded not in files
    # The naming reference is excluded by name rather than by directory, so the
    # exclusion has to be exact: its directory stays in scope.
    assert "docs/reference/KNOWN_LIMITATIONS.md" in files


def test_every_documented_wire_keyword_names_a_file_that_exists() -> None:
    scanned = _CENSUS.documentation_census(_ROOT)
    assert scanned.sites
    for site in scanned.sites:
        relative = site.split("::", 1)[0]
        assert (_ROOT / relative).is_file()
        assert "wire" in site.rsplit("::", 1)[1]


def test_a_backtick_quoted_keyword_is_still_an_instruction_to_a_reader() -> None:
    """The exclusion that hid the one occurrence which had actually gone stale.

    A document that writes ``(`wires=`)`` in a code span is naming the keyword it
    tells the reader to pass, exactly as a code block is. The matcher used to
    require that the name not follow a backtick, and the single occurrence that rule
    hid was a stale keyword in a contract whose scoreboard is generated from it.
    """

    assert _CENSUS.documentation_keywords("per-qubit (`wires=`) injection") == (
        "wires=",
    )
    assert _CENSUS.documentation_keywords("`n_wires=2` remains compatible") == (
        "n_wires=",
    )


def test_the_documentation_exemptions_cover_exactly_the_live_sites() -> None:
    """The live reading, compared against the record rather than a hard-coded list.

    Two documents may show a wire-named keyword and each for a reason recorded in
    the contract: the hybrid plan quotes the capture layer's source language, and
    the API reference documents a deprecated alias that still works. Everything else
    is a FlagQuantum instruction and must use the qubit vocabulary. Comparing
    against the contract means a new site cannot be absorbed by editing the test.
    """

    contract = _contract()
    exempt = {
        str(row["file"]): sorted(str(keyword) for keyword in row.get("keywords", ()))
        for row in contract["documentation"]["exempt"]
    }
    observed: dict[str, list[str]] = {}
    for site in _CENSUS.documentation_census(_ROOT).sites:
        relative, _, keyword = site.partition("::")
        observed.setdefault(relative, []).append(keyword)
    assert observed, "the documentation census found nothing to reconcile"
    assert {name: sorted(values) for name, values in observed.items()} == exempt
    assert int(contract["documentation"]["measured_keywords"]) == sum(
        len(values) for values in exempt.values()
    )


def test_the_generated_scoreboard_no_longer_teaches_a_retired_keyword() -> None:
    """`NoiseModel.add` takes `qubits=`, so the page rendered from its contract must say so."""

    page = (_ROOT / "docs/reference/CUDAQ_PARITY_MATRIX.md").read_text(encoding="utf-8")
    assert "per-qubit (`qubits=`) static injection" in page
    assert "(`wires=`) static injection" not in page
    parity = (_ROOT / "contracts/cudaq-parity-matrix.toml").read_text(encoding="utf-8")
    assert "(`wires=`)" not in parity


def test_a_docstring_token_is_an_identifier_not_a_line() -> None:
    """`help()` prints a sentence, but the unit is the name inside it.

    A line-based count would let one reworded name hide behind another on the same
    line. Splitting into identifier-shaped tokens means `n_wires - 1 - wire` is two
    sites that have to be accounted for separately, and nothing inside a code span
    can be reworded on the strength of the prose around it.
    """

    assert _CENSUS._wire_tokens("bit ``n_wires - 1 - wire``") == ("n_wires", "wire")
    assert _CENSUS._wire_tokens("a five-wire circuit") == ("wire",)
    assert _CENSUS._wire_tokens("no spelling here") == ()
    # The hyphen does not start an identifier, so the compound adjective is one
    # token rather than two.
    assert _CENSUS._wire_tokens("single-wire term") == ("wire",)


def test_a_comment_is_attributed_to_its_innermost_definition(tmp_path: Path) -> None:
    """The key has to survive a line moving, so it names the scope and not the line."""

    root = _package(
        tmp_path,
        {
            "flagquantum/alpha.py": (
                '"""Module prose about qubits."""\n'
                "\n"
                "\n"
                "# a module-level note about a wire\n"
                "def outer(n_wires: int) -> int:\n"
                '    """Outer."""\n'
                "    # a note inside outer about a wire\n"
                "    def inner() -> int:\n"
                '        """Inner."""\n'
                "        # a note inside inner about a wire\n"
                "        return n_wires\n"
                "    return inner()\n"
            )
        },
    )
    assert _CENSUS.docstring_census(root) == (
        ("flagquantum/alpha.py::<module>", "wire"),
        ("flagquantum/alpha.py::outer", "wire"),
        ("flagquantum/alpha.py::outer.inner", "wire"),
    )


def test_a_docstring_is_scanned_even_in_a_private_module(tmp_path: Path) -> None:
    """`[boundary]` stops at the public surface; `help()` does not.

    A docstring reaches whoever imports the module that holds it, so the scan reads
    the whole package rather than the modules the keyword ledgers call public. A
    private module's prose is still published.
    """

    root = _package(
        tmp_path,
        {
            "flagquantum/_helper.py": '"""A helper that talks about a wire."""\n',
            "flagquantum/deep/_also.py": '"""Nested, and also about wires."""\n',
        },
    )
    assert _CENSUS.docstring_census(root) == (
        ("flagquantum/_helper.py::<module>", "wire"),
        ("flagquantum/deep/_also.py::<module>", "wires"),
    )


def test_a_wire_token_is_reported_each_time_it_occurs(tmp_path: Path) -> None:
    """A multiset, not a set: dropping one of two sites must move the reading."""

    root = _package(
        tmp_path,
        {
            "flagquantum/twice.py": (
                '"""One wire here."""\n'
                "\n"
                "\n"
                "def f() -> None:\n"
                '    """And another wire here."""\n'
            )
        },
    )
    assert _CENSUS.docstring_census(root) == (
        ("flagquantum/twice.py::<module>", "wire"),
        ("flagquantum/twice.py::f", "wire"),
    )


def test_two_definitions_sharing_a_bare_name_are_two_containers(tmp_path: Path) -> None:
    """The key is a dotted qualname, and the exemption is per container.

    `A._prepare` and `B._prepare` are different docstrings that reach different
    readers. Keyed by the bare `_prepare` they would be one container holding two
    tokens, and a single recorded row would then cover an occurrence in the other
    class -- so the gate would pass on a docstring nobody had looked at.
    """

    root = _package(
        tmp_path,
        {
            "flagquantum/two.py": (
                "class A:\n"
                '    """A."""\n'
                "\n"
                "    def _prepare(self) -> None:\n"
                '        """Bind each wire to one index."""\n'
                "\n"
                "\n"
                "class B:\n"
                '    """B."""\n'
                "\n"
                "    def _prepare(self) -> None:\n"
                '        """Bind each wire to one index."""\n'
            )
        },
    )
    assert _CENSUS.docstring_census(root) == (
        ("flagquantum/two.py::A._prepare", "wire"),
        ("flagquantum/two.py::B._prepare", "wire"),
    )


def test_a_nested_definition_is_keyed_by_its_whole_path(tmp_path: Path) -> None:
    """A closure is not the method that holds it, and a reader can reach both."""

    root = _package(
        tmp_path,
        {
            "flagquantum/nested.py": (
                "def outer() -> None:\n"
                '    """Outer."""\n'
                "\n"
                "    def inner() -> None:\n"
                '        """Reindex every wire."""\n'
            )
        },
    )
    assert _CENSUS.docstring_census(root) == (
        ("flagquantum/nested.py::outer.inner", "wire"),
    )


def test_the_docstring_matcher_finds_wire_anywhere_in_the_name() -> None:
    """`terminal_wires` is the same question as `wires`, as the keyword matcher is."""

    assert _CENSUS._wire_tokens("terminal_wires and shardable_wires") == (
        "terminal_wires",
        "shardable_wires",
    )
    assert _CENSUS._wire_tokens("qubits only") == ()


def test_the_docstring_exemptions_cover_exactly_the_live_tokens() -> None:
    """The live reading, compared against the record rather than a hard-coded list.

    Every wire-named token a docstring or comment still publishes is either gone or
    named in `[docstring]` with a reason, and the comparison is per container: a
    second occurrence inside an already-exempt docstring needs its own row rather
    than being absorbed by the first. Comparing against the contract means a new site
    cannot be absorbed by editing the test.
    """

    contract = _contract()
    exempt: dict[str, list[str]] = {}
    for row in contract["docstring"]["keep"]:
        for identifier in row["identifier"]:
            container, _, token = str(identifier).rpartition("::")
            exempt.setdefault(container, []).append(token)
    observed: dict[str, list[str]] = {}
    for container, token in _CENSUS.docstring_census(_ROOT / "flagquantum"):
        observed.setdefault(container, []).append(token)
    assert observed, "the docstring census found nothing to reconcile"
    assert {name: sorted(tokens) for name, tokens in observed.items()} == {
        name: sorted(tokens) for name, tokens in exempt.items()
    }
    assert int(contract["docstring"]["measured_tokens"]) == sum(
        len(tokens) for tokens in exempt.values()
    )


def test_the_docstring_example_surface_is_read_apart_from_the_prose(
    tmp_path: Path,
) -> None:
    """A `>>>` block is code the doctest runner executes, not prose a reader reads.

    Both live in a docstring, so a scan that read the whole string would hand a
    rewriting pass a code token and call it a sentence. That is not hypothetical: it
    turned `Instruction.wires` into `.qubits` in the `Circuit.compose` example, and
    the example started raising `AttributeError`. So the two readings have to be
    disjoint, and their union has to be the whole docstring.
    """

    root = _package(
        tmp_path,
        {
            "flagquantum/example.py": (
                '"""Prose about a wire.\n'
                "\n"
                "Examples:\n"
                "    >>> Circuit(2).n_wires\n"
                "    2\n"
                '"""\n'
            )
        },
    )
    assert _CENSUS.docstring_census(root) == (
        ("flagquantum/example.py::<module>", "wire"),
    )
    assert _CENSUS.docstring_example_tokens(root) == (
        ("flagquantum/example.py::<module>", "n_wires"),
    )


def test_the_live_example_tokens_match_the_pinned_record() -> None:
    """The count is small, so an entry appearing here is a signal rather than noise.

    Unlike the prose ledger, a token here is not debt the migration owes: the example
    would still work if it had always been written in the qubit vocabulary. It is
    pinned so that a pass cannot move one without saying so.
    """

    contract = _contract()
    pinned: dict[str, list[str]] = {}
    for row in contract["docstring_example"]["keep"]:
        for identifier in row["identifier"]:
            container, _, token = str(identifier).rpartition("::")
            pinned.setdefault(container, []).append(token)
    observed: dict[str, list[str]] = {}
    for container, token in _CENSUS.docstring_example_tokens(_ROOT / "flagquantum"):
        observed.setdefault(container, []).append(token)
    assert observed, "the docstring example census found nothing to reconcile"
    assert {name: sorted(tokens) for name, tokens in observed.items()} == {
        name: sorted(tokens) for name, tokens in pinned.items()
    }
    assert int(contract["docstring_example"]["measured_tokens"]) == sum(
        len(tokens) for tokens in pinned.values()
    )


def _importable_package(tmp_path: Path, name: str, files: dict[str, str]) -> Path:
    """Write a package that the scan can actually import, and return its root.

    The call-site scan resolves a keyword against the callee's real signature, so a
    fixture has to be importable rather than merely parseable. The name is a
    parameter because every fixture has to be distinct in `sys.modules`: two tests
    writing `flagquantum` would make the second one resolve the first one's callee.
    """

    root = tmp_path / name
    for relative, source in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
    sys.path.insert(0, str(tmp_path))
    importlib.invalidate_caches()
    return root


def test_a_call_site_that_passes_a_retired_keyword_is_reported(tmp_path: Path) -> None:
    """The defect class no ledger can hold: the declaration moved, the caller did not.

    `WQ-4` renamed `execute_torch_distributed_statevector_reverse`'s `observable_wire`
    to `observable_qubit` and left `flagquantum/runtime/module.py` writing the old
    keyword. Every other surface here reads a *declaration*, so all of them passed
    while that call was a live `TypeError` on the distributed-statevector path.
    """

    root = _importable_package(
        tmp_path,
        "fqpkg_retired_caller",
        {
            "__init__.py": "",
            "callee.py": "def run(*, observable_qubit=0):\n    return observable_qubit\n",
            "caller.py": "from .callee import run\n\n\ndef go():\n    return run(observable_wire=0)\n",
        },
    )
    found = _CENSUS.qubit_keyword_mismatches(root)
    assert len(found) == 1
    assert "caller.py" in found[0]
    assert "observable_wire" in found[0]


def test_a_callee_that_takes_kwargs_is_skipped_rather_than_guessed(
    tmp_path: Path,
) -> None:
    """A `**kwargs` callee accepts any keyword, so the question is undecidable.

    Reporting it would make the gate fire on correct code, and a gate that fires on
    correct code is a gate that gets an exemption added to it.
    """

    root = _importable_package(
        tmp_path,
        "fqpkg_kwargs_callee",
        {
            "__init__.py": "",
            "callee.py": "def run(**kwargs):\n    return kwargs\n",
            "caller.py": "from .callee import run\n\n\ndef go():\n    return run(observable_wire=0)\n",
        },
    )
    assert _CENSUS.qubit_keyword_mismatches(root) == ()


def test_the_opposite_lag_is_reported_too(tmp_path: Path) -> None:
    """A caller written against a spelling that has not landed is the same defect.

    Checking only `wire`-named keywords would make the scan blind to the direction
    the package is moving in, which is the direction its own new code is written.
    """

    root = _importable_package(
        tmp_path,
        "fqpkg_qubit_caller",
        {
            "__init__.py": "",
            "callee.py": "def run(*, observable_wire=0):\n    return observable_wire\n",
            "caller.py": "from .callee import run\n\n\ndef go():\n    return run(observable_qubit=0)\n",
        },
    )
    found = _CENSUS.qubit_keyword_mismatches(root)
    assert len(found) == 1
    assert "observable_qubit" in found[0]


def test_a_relative_import_inside_a_package_namespace_resolves_to_its_own_package(
    tmp_path: Path,
) -> None:
    """`from .x import f` in an `__init__.py` means *this* package's `x`.

    An `__init__.py`'s module name already is its package, so resolving it the way a
    regular module resolves (`rpartition` the last component) would look one level
    too high and silently skip the call rather than misreport it -- the failure mode
    that makes a gate look green.
    """

    root = _importable_package(
        tmp_path,
        "fqpkg_namespace",
        {
            "__init__.py": (
                "from .sub import run\n\n\ndef go():\n    return run(observable_wire=0)\n"
            ),
            "sub.py": "def run(*, observable_qubit=0):\n    return observable_qubit\n",
        },
    )
    found = _CENSUS.qubit_keyword_mismatches(root)
    assert len(found) == 1
    assert "observable_wire" in found[0]


def test_the_live_package_passes_no_rejected_keyword_to_any_callee() -> None:
    """The gate's own assertion, read from the tree rather than from the contract."""

    assert _CENSUS.qubit_keyword_mismatches(_ROOT / "flagquantum") == ()


def test_the_call_site_invariant_is_pinned_at_zero() -> None:
    """The allowance is not a measurement, so a slice may not raise it to pass."""

    surfaces = _contract()["other_surfaces"]
    assert surfaces["call_site_mismatches"] == 0
    assert surfaces["call_site_surface"]
    assert surfaces["call_site_owner"]
    assert surfaces["call_site_condition"]


def test_the_prose_noun_reading_separates_the_adr_bucket() -> None:
    """An ADR is scanned but left alone, and the reading has to show both facts.

    Treating the decisions directory as either excluded or ordinary would hide one of
    them: as excluded, a future ADR that writes out a `wires=` example would be
    invisible to the gate; as ordinary, its 34 occurrences of the noun would read as
    unmigrated work in a document whose job is to record what was decided.
    """

    rows = dict(
        (bucket, (occurrences, files))
        for bucket, occurrences, files in _CENSUS.prose_noun_census(_ROOT)
    )
    assert "docs/architecture/decisions/" in rows
    assert "everything else" in rows
    for bucket in _CENSUS.DOCUMENTATION_EXCLUDED_PREFIXES:
        assert bucket in rows
    occurrences, files = rows["everything else"]
    assert occurrences > 0
    assert 0 < files <= occurrences


def test_the_prose_noun_reading_agrees_with_a_file_by_file_recount() -> None:
    """The bucket totals are a partition, so an independent recount must match.

    The figures this reports are quoted in prose about the migration, and the first
    attempt at them came from a shell loop that kept only its last iteration and was
    wrong by a factor of two. So the reading is re-derived here from the file list
    rather than trusted, and the two must agree exactly.
    """

    import subprocess

    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=_ROOT,
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8")
    occurrences = 0
    files = 0
    for name in sorted(set(listed.split("\0"))):
        if not name.endswith(".md"):
            continue
        if name.startswith(
            _CENSUS.DOCUMENTATION_EXCLUDED_PREFIXES
            + _CENSUS.PROSE_SCANNED_ONLY_PREFIXES
        ):
            continue
        if name in _CENSUS.DOCUMENTATION_EXCLUDED_FILES:
            continue
        text = (_ROOT / name).read_text(encoding="utf-8", errors="replace")
        count = text.lower().count("wire")
        occurrences += count
        files += 1 if count else 0
    rows = {
        bucket: (found, seen)
        for bucket, found, seen in _CENSUS.prose_noun_census(_ROOT)
    }
    assert rows["everything else"] == (occurrences, files)
