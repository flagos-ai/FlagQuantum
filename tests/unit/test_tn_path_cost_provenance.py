"""Provenance of an exported tensor-network contraction cost.

``TensorNetworkContractionProfile`` reports the cost of one contraction order
for one network. The same network can be ordered by this package's own search or
by an external planner behind an optional extra, and the two costs are not
comparable without knowing which search produced them: a number recorded as
evidence is only traceable if the record names its planner.

``TensorNetworkSlicingPlan`` already carried that identity in
``contraction_path_source``. The profile, which is the object a caller exports a
bare cost from, did not, so these tests pin that an exported cost names its
planner and inherits the name from the slicing plan it was built on. They also
pin that the module README documents exactly the cost surface that exists, so
the export contract cannot drift away from the prose a caller reads.

The slicing route is the only route that can reach an external planner today, so
the inheritance test substitutes the planner behind the module-level seam rather
than requiring the optional extra to be installed.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

import flagquantum as fq
from flagquantum.simulation import tensor_network
from flagquantum.simulation.tensor_network import build_shape_only_tensor_network
from flagquantum.simulation.tensor_network import contraction as contraction_module
from flagquantum.simulation.tensor_network import path_search as path_search_module
from flagquantum.simulation.tensor_network.models import (
    TensorNetworkContractionProfile,
    TensorNetworkExpectationPlan,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "flagquantum/simulation/tensor_network/README.md"
README_SECTION = "## Exporting a path cost"

NATIVE_STRATEGIES = ("greedy", "memory_greedy", "quality_greedy", "beam", "optimal")
SLICED_STRATEGIES = ("sliced", "auto_sliced", "beam_sliced", "quality_sliced")
ALL_STRATEGIES = NATIVE_STRATEGIES + SLICED_STRATEGIES + ("einsum",)
EXTERNAL_SOURCE = "cotengra:0.8.2"


def _circuit(n_wires: int):
    circuit = fq.Circuit(n_wires)
    for wire in range(n_wires):
        circuit.h(wire)
    for wire in range(n_wires - 1):
        circuit.cx(wire, wire + 1)
    for wire in range(n_wires):
        circuit.ry(wire, 0.1 * (wire + 1))
    return circuit


def _plan(n_wires: int):
    """A shape-only plan whose labels are unique to ``n_wires``.

    The contraction profile is cached per network and strategy, so a test that
    wants to observe a fresh computation must build a network no other test in
    this module has profiled.
    """

    return build_shape_only_tensor_network(_circuit(n_wires))


def _readme_section() -> str:
    text = README.read_text(encoding="utf-8")
    assert README_SECTION in text, "the export section is missing from the README"
    return text.split(README_SECTION, 1)[1].split("\n## ", 1)[0]


def test_every_exported_cost_names_its_planner():
    plan = _plan(3)

    for strategy in ALL_STRATEGIES:
        profile = plan.contraction_profile(strategy)
        assert profile.contraction_path_source == "native", strategy


def test_the_summary_carries_the_planner_it_reports():
    plan = _plan(4)

    for strategy in ALL_STRATEGIES:
        profile = plan.contraction_profile(strategy)
        summary = profile.summary()
        assert summary["contraction_path_source"] == profile.contraction_path_source
        assert summary["contraction_path_source"].strip(), strategy


def test_a_native_planner_is_the_declared_default():
    field = TensorNetworkContractionProfile.__dataclass_fields__[
        "contraction_path_source"
    ]

    assert field.default == "native"


def test_the_new_key_is_added_without_dropping_the_exported_ones():
    """Existing consumers read the summary by key, so none may disappear."""

    plan = _plan(7)
    summary = plan.contraction_profile("greedy").summary()

    assert set(summary) == {
        "strategy",
        "estimated_cost",
        "peak_size",
        "n_steps",
        "output_size",
        "n_slices",
        "sliced_labels",
        "total_intermediate_size",
        "contraction_path_source",
    }


def test_the_profile_inherits_the_planner_of_the_slicing_plan(monkeypatch):
    plan = _plan(5)
    seen: list[dict[str, object]] = []
    real = contraction_module._build_slicing_plan

    def _marked(*args, **kwargs):
        seen.append(kwargs)
        slicing = real(*args, **kwargs)
        return dataclasses.replace(slicing, contraction_path_source=EXTERNAL_SOURCE)

    monkeypatch.setattr(contraction_module, "_build_slicing_plan", _marked)

    profile = plan.contraction_profile("quality_sliced")

    assert seen, "the sliced profile must be built on a slicing plan"
    assert profile.contraction_path_source == EXTERNAL_SOURCE
    assert profile.summary()["contraction_path_source"] == EXTERNAL_SOURCE


def test_a_native_sliced_profile_still_agrees_with_its_slicing_plan():
    plan = _plan(6)

    profile = plan.contraction_profile("beam_sliced")
    slicing = plan.slicing_plan(contraction_strategy="beam")

    assert profile.contraction_path_source == slicing.contraction_path_source


def test_both_plan_kinds_export_the_same_planner_name():
    plan = _plan(3)
    expectation = TensorNetworkExpectationPlan(
        n_wires=plan.n_wires,
        bsz=plan.bsz,
        nodes=plan.nodes,
        output_labels=plan.output_labels,
        observable_wires=(0,),
        path=(),
    )

    assert plan.contraction_profile("greedy").contraction_path_source == (
        expectation.contraction_profile("greedy").contraction_path_source
    )


def test_the_module_readme_documents_every_exported_summary_key():
    """The README's key list must be the summary's key set.

    The check reads the enumerating clause rather than the whole section: the
    section names the provenance key a second time in prose, so a looser check
    would keep passing after the key was dropped from the list a caller reads.
    """

    section = _readme_section()
    marker = "whose `summary()` reports"
    assert marker in section
    enumerated = section.split(marker, 1)[1].split(".", 1)[0]
    documented = set(re.findall(r"`([a-z_]+)`", enumerated))
    summary = _plan(8).contraction_profile("greedy").summary()

    assert documented == set(summary)


def test_the_module_readme_names_the_exported_surfaces():
    section = _readme_section()
    plan = _plan(8)

    for name in ("contraction_cost", "contraction_profile"):
        assert name in section, name
        assert callable(getattr(plan, name)), name

    peak_bytes = "tensor_network_contraction_peak_bytes"
    assert peak_bytes in section
    assert peak_bytes in tensor_network.__all__
    assert callable(getattr(tensor_network, peak_bytes))


def test_the_optimal_objective_delegates_to_a_wider_beam_above_its_ceiling(
    monkeypatch,
):
    """The ceiling the README documents must be behaviour, not a comment.

    An exported `optimal` cost is a claim about the order, so a reader has to be
    able to tell an exhaustive search from a bounded approximation. The two
    searches are indistinguishable by cost on many small networks, so this pins
    the call itself rather than comparing two numbers.
    """

    plan = _plan(1)
    assert len(plan.nodes) > 1
    widths: list[int] = []
    real = path_search_module._contract_nodes_beam

    def _spy(nodes, output_labels, *, dry_run=False, beam_width=8):
        widths.append(int(beam_width))
        return real(nodes, output_labels, dry_run=dry_run, beam_width=beam_width)

    monkeypatch.setattr(path_search_module, "_contract_nodes_beam", _spy)

    plan.optimal_path(max_nodes=1)
    assert widths == [16], "the delegation must not reuse the default beam width"

    widths.clear()
    plan.optimal_path(max_nodes=len(plan.nodes))
    assert widths == [], "an exhaustive search must not run a beam search"
