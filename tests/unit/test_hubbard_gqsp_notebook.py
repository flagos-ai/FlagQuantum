"""Compare the locked Hubbard GQSP notebook's complete declaration tables."""

from hashlib import sha1
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest

from examples.algorithms.hubbard_gqsp import (
    hubbard_gqsp_graph,
    hubbard_gqsp_scale,
    hubbard_gqsp_signature,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _key(node: Any, source: Any) -> str:
    from qualtran import CtrlSpec
    from qualtran.bloqs.bookkeeping import Always

    walk = source.walk_operator
    keys = {
        source: "HamiltonianSimulationByGQSP",
        Always(walk.prepare): "Always(PREPARE)",
        Always(walk.prepare.adjoint()): "Always(PREPARE†)",
        walk.controlled(CtrlSpec(cvs=0)): "ZeroControlledWalk",
        walk.controlled(): "PositiveControlledWalk",
        walk.adjoint().controlled(): "PositiveControlledWalk†",
        walk.block_encoding.adjoint().controlled(): "ControlledSelectBlock†",
        walk.reflect.controlled(): "PositiveControlledReflection",
    }
    if node in keys:
        return keys[node]
    name = type(node).__name__
    if name == "SU2RotationGate":
        return "SU2Rotation"
    if name == "Controlled" and type(node.subbloq).__name__ == "Ry":
        return "ControlledRy" if node.subbloq.angle > 0 else "ControlledRy†"
    if name == "CSwap":
        return f"CSwap({node.bitsize})"
    if name == "PrepareUniformSuperposition":
        return f"PrepareUniformSuperposition({node.n})"
    if name == "Adjoint":
        return f"{_key(node.subbloq, source)}†"
    assert name in {
        "Ry",
        "Hadamard",
        "XGate",
        "CNOT",
        "CHadamard",
        "ModAddK",
        "MultiAnd",
        "GlobalPhase",
        "Rz",
        "Rx",
    }, node
    return name


@pytest.mark.parametrize("symbolic", [False, True])
def test_original_notebook_signatures_and_both_call_depths(symbolic: bool) -> None:
    import kickmix
    from qualtran.bloqs.hamiltonian_simulation import (
        hamiltonian_simulation_by_gqsp as upstream,
    )
    from qualtran.resource_counting.generalizers import (
        generalize_rotation_angle,
        ignore_alloc_free,
        ignore_split_join,
    )
    from sympy import symbols

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    notebook = Path(upstream.__file__).with_suffix(".ipynb")
    raw = notebook.read_bytes()
    assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == (
        "ddaba12ea254562ca27a2477b6009c49b12bf591"
    )
    if symbolic:
        hopping, time, inv_eps = symbols(r"\tau t \epsilon^{-1}", positive=True)
        interaction, precision = 4 * hopping, 1 / inv_eps
        original = upstream._symbolic_hamsim_by_gqsp.make()
    else:
        hopping, interaction, time, precision = 1, 1, 5, 1e-7
        original = upstream._hubbard_time_evolution_by_gqsp.make()
    assert hubbard_gqsp_scale(hopping, interaction) == original.alpha

    signature = hubbard_gqsp_signature()
    # Upstream takes the set union of selection words, so compare by name.
    assert {r.name: (r.bitsize, r.shape, r.side.value) for r in signature} == {
        r.name: (r.bitsize, r.shape, r.side.value) for r in original.signature
    }
    assert signature[0].name == "signal" and signature[-1].name == "target"
    native_by_name = {r.name: r for r in signature}
    for r in tuple(original.signature)[1:-1]:
        dtype = native_by_name[r.name].dtype
        assert dtype.encoding == "uint"
        assert dict(dtype.parameters)["bound"] == r.dtype.iteration_length == 2

    for depth in (1, 2):
        graph = hubbard_gqsp_graph(hopping, interaction, time, precision, depth=depth)
        expected, counts = original.call_graph(
            max_depth=depth,
            generalizer=[
                ignore_split_join,
                ignore_alloc_free,
                generalize_rotation_angle,
            ],
        )
        assert graph.direct_calls == {
            (_key(a, original), _key(b, original)): data["n"]
            for a, b, data in expected.edges(data=True)
        }
        assert graph.leaf_counts == {
            _key(leaf, original): count for leaf, count in counts.items()
        }
