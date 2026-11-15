"""Check frozen permutation factories, declared child keys and full signatures."""

from collections.abc import Hashable
from hashlib import sha1
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest

from examples.algorithms.permutation_catalog import (
    permutation_example_graph,
    permutation_examples,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
SOURCE = (
    Path(__file__).resolve().parents[3] / "ftqc-reference-env/qualtran/bloqs/arithmetic"
)


def test_locked_assets() -> None:
    import kickmix

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    for name, digest in (
        ("permutation.py", "8c294d59344ef64fa5dc5e2aa22f34c599e1243a"),
        ("permutation_test.py", "b5d84a16ecdb931a9db265f0f982180afaf7b607"),
        ("permutation.ipynb", "3f5d690004a801afc238d966bc199749aebc575d"),
    ):
        data = (SOURCE / name).read_bytes()
        assert sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == digest


def _cycle(cycle: Any) -> Any:
    from qualtran.symbolics import Shaped

    return cycle.shape[0] if isinstance(cycle, Shaped) else cycle


def _key(child: Any) -> Hashable:
    from qualtran import QBit
    from qualtran._infra.controlled import Controlled
    from qualtran.bloqs.arithmetic import EqualsAConstant, XorK
    from qualtran.bloqs.arithmetic.permutation import PermutationCycle
    from qualtran.bloqs.bookkeeping import Allocate, Free

    if isinstance(child, PermutationCycle):
        return ("cycle", child.N, child.bitsize, _cycle(child.cycle))
    if isinstance(child, EqualsAConstant):
        return (
            "equals",
            child.bitsize,
            child.val if isinstance(child.val, int) else "unknown",
        )
    if isinstance(child, Controlled) and isinstance(child.subbloq, XorK):
        assert child.ctrl_spec.cvs[0].item() == 1
        xor = child.subbloq
        return (
            "controlled_xor",
            xor.dtype.bitsize,
            xor.k if isinstance(xor.k, int) else "unknown",
        )
    if isinstance(child, (Allocate, Free)):
        assert child.dtype == QBit() and not child.dirty
        return ("allocate_marker" if isinstance(child, Allocate) else "free_marker",)
    raise AssertionError(f"Unexpected child {child}")


@pytest.mark.parametrize("index", range(8))
def test_complete_factory_signature_cycles_and_calls(index: int) -> None:
    from qualtran import BQUInt
    from qualtran.bloqs.arithmetic import permutation as p

    source = (
        p._permutation,
        p._permutation_symb,
        p._permutation_symb_with_cycles,
        p._sparse_permutation,
        p._sparse_permutation_with_symbolic_N,
        p._permutation_cycle_symb_N,
        p._permutation_cycle_symb,
        p._permutation_cycle,
    )[index].make()
    example = permutation_examples()[index]
    name, signature, cycles, _ = example
    expected = source.cycles if index < 5 else source.cycle
    if index < 5 and isinstance(expected, tuple):
        assert cycles == tuple(_cycle(cycle) for cycle in expected)
    else:
        assert cycles == _cycle(expected)
    assert all(isinstance(r.dtype, BQUInt) for r in source.signature)
    assert [
        (
            r.name,
            r.bitsize,
            r.shape_symbolic,
            r.side.value,
            dict(r.dtype.parameters)["bound"],
        )
        for r in signature
    ] == [
        (r.name, r.bitsize, r.shape_symbolic, r.side.value, r.dtype.iteration_length)
        for r in source.signature
    ]
    graph, leaves = source.call_graph(max_depth=1)
    native = permutation_example_graph(example)
    assert native.direct_calls == {
        (name, _key(child)): data["n"] for _, child, data in graph.edges(data=True)
    }
    assert native.leaf_counts == {_key(child): count for child, count in leaves.items()}
