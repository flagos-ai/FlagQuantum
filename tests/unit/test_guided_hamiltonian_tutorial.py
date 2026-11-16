"""Compare every native guided tutorial leaf with the frozen notebook."""

import json
from hashlib import sha1
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest
from sympy import simplify

from examples.algorithms.guided_hamiltonian_query_costs import (
    tutorial_query_graph,
    tutorial_query_upper_bound,
)
from flagquantum.algorithms.primitives.guided_hamiltonian_resource_model import (
    GuidedHamiltonianModel,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module")
def source_tutorial() -> dict[str, Any]:
    import kickmix
    from qualtran.bloqs import hamiltonian_simulation as source

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    folder = Path(source.__file__).parent
    path = folder / "tutorial_guided_hamiltonian.ipynb"
    raw = path.read_bytes()
    assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == (
        "268867bfe0a964c7ef204230feb72eea9c455ff3"
    )
    scope: dict[str, Any] = {"__name__": "locked_guided_tutorial"}
    for index, cell in enumerate(json.loads(raw)["cells"]):
        if cell["cell_type"] == "code":
            exec(
                compile("".join(cell["source"]), f"{path.name}:{index}", "exec"), scope
            )
        if index == 2:
            # Comparisons use the original calculations; rendering runs in Jupyter.
            for display in ("show_bloq", "show_counts_sigma"):
                scope[display] = lambda *_args, **_kwargs: None
    return scope


def _source_leaf_names(scope: dict[str, Any]) -> dict[Any, str]:
    from qualtran.bloqs.basic_gates import Toffoli
    from qualtran.bloqs.mcmt.and_bloq import And

    guided = scope["bloq"]
    position, entry = guided.hamiltonian.O_F, guided.hamiltonian.O_H
    arbitrary = scope["ArbitraryGate"]
    return {
        arbitrary(): "ArbitraryGate",
        arbitrary(n_ctrls=1): "ControlledArbitraryGate",
        And(): "And",
        And(uncompute=True): "And†",
        Toffoli(): "Toffoli",
        guided.hamiltonian.signal_state.prepare: "PrepareIdentity[walk]",
        guided._refl_guide_ancilla.prepare_gate: "PrepareIdentity[guide]",
        guided._refl_all.prepare_gate: "PrepareIdentity[all]",
        position: "O_F",
        position.adjoint(): "O_F†",
        position.controlled(): "ControlledO_F",
        position.adjoint().controlled(): "ControlledO_F†",
        entry: "O_H",
        entry.controlled(): "ControlledO_H",
    }


@pytest.mark.parametrize("amplify,leaf_count", [(False, 10), (True, 14)])
def test_every_exact_and_bounded_leaf(
    source_tutorial: dict[str, Any], amplify: bool, leaf_count: int
) -> None:
    scope = source_tutorial
    original = scope["bloq"]
    guide = original.guiding_state
    block = original.hamiltonian
    model = GuidedHamiltonianModel(
        block.system_bitsize,
        block.ancilla_bitsize,
        guide.selection_bitsize,
        guide.junk_bitsize,
        block.alpha,
        original.lambd,
        original.alpha,
        original.gamma,
    )
    exact = tutorial_query_graph(model, guide.G, amplify=amplify).leaf_counts
    if amplify:
        source_exact = scope["sigma"]
        source_bound = scope["sigma_simpl"]
    else:
        generalizers = [
            scope[name]
            for name in (
                "ignore_alloc_free",
                "ignore_split_join",
                "generalize_1_2_qubit_gates",
            )
        ]
        _, source_exact = original.qpe_bloq.call_graph(generalizer=generalizers)
        source_bound = scope["sigma_pe"]
    names = _source_leaf_names(scope)
    expected = {names[leaf]: count for leaf, count in source_exact.items()}
    assert exact.keys() == expected.keys() and len(exact) == leaf_count
    for leaf, count in exact.items():
        assert simplify(count - expected[leaf]) == 0, leaf
    bound = tutorial_query_upper_bound(model, exact)
    expected_bound = {names[leaf]: count for leaf, count in source_bound.items()}
    assert bound.keys() == expected_bound.keys()
    for leaf, upper in bound.items():
        assert simplify(upper - expected_bound[leaf]) == 0, leaf
    assert bound["O_F"] != exact["O_F"]
    assert bound["ControlledO_H"] == exact["ControlledO_H"]
