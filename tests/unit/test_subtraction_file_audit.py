"""Validate all eight original subtraction declarations and symbolic bindings."""

from collections.abc import Hashable
from hashlib import sha1
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest
from sympy import Symbol

from examples.algorithms.subtraction_catalog import (
    subtraction_example_graph,
    subtraction_examples,
    symbolic_subtraction_steps,
)
from flagquantum.compiler._component_trace.arithmetic_call import ArithmeticCall
from flagquantum.compiler._component_trace.resource_generalization import (
    CliffordCall,
    OnEachCall,
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
        ("subtraction.py", "f062557083599ffe7532f0ae9125b9bf83f64ed7"),
        ("subtraction_test.py", "fe9a5cb48ed30016eb214174f1c84a63e06aaa28"),
        ("subtraction.ipynb", "ce9e322114ccb4bb52a553c2291fa69a5f847121"),
    ):
        data = (SOURCE / name).read_bytes()
        assert sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == digest


def _key(child: Any) -> Hashable:
    from qualtran import QInt
    from qualtran.bloqs.arithmetic import Add, BitwiseNot
    from qualtran.bloqs.basic_gates import OnEach, XGate
    from qualtran.bloqs.bookkeeping import Allocate, Cast, Free
    from qualtran.bloqs.mcmt import MultiTargetCNOT

    if isinstance(child, Add):
        return ArithmeticCall(
            "add",
            (child.a_dtype.bitsize, child.b_dtype.bitsize),
            (
                ("twos_complement", "twos_complement")
                if isinstance(child.a_dtype, QInt)
                else ()
            ),
        )
    if isinstance(child, BitwiseNot):
        assert isinstance(child.dtype, QInt)
        return ArithmeticCall(
            "bitwise_not", (child.dtype.bitsize,), ("twos_complement",)
        )
    if isinstance(child, OnEach):
        assert isinstance(child.gate, XGate)
        return OnEachCall(CliffordCall("x", 1), child.n)
    if isinstance(child, Cast):
        return (
            "cast",
            "signed" if isinstance(child.inp_dtype, QInt) else "unsigned",
            "signed" if isinstance(child.out_dtype, QInt) else "unsigned",
            child.inp_dtype.bitsize,
        )
    if isinstance(child, MultiTargetCNOT):
        return ("multi_target_cx", child.bitsize)
    if isinstance(child, (Allocate, Free)):
        return (
            "allocate" if isinstance(child, Allocate) else "free",
            child.dtype.bitsize,
        )
    raise AssertionError(f"Unexpected subtraction child {child}")


@pytest.mark.parametrize("index", range(8))
def test_complete_example_signatures_and_declarations(index: int) -> None:
    from qualtran import QInt
    from qualtran.bloqs.arithmetic import subtraction

    factories = (
        subtraction._sub_symb,
        subtraction._sub_small,
        subtraction._sub_large,
        subtraction._sub_diff_size_regs,
        subtraction._sub_symp_decomposition,
        subtraction._sub_from_symb,
        subtraction._sub_from_small,
        subtraction._sub_from_large,
    )
    source = factories[index].make()
    example = subtraction_examples()[index]
    name, signature, _ = example
    assert all(isinstance(r.dtype, QInt) for r in source.signature)
    assert [(r.name, r.bitsize, r.shape_symbolic, r.side.value) for r in signature] == [
        (r.name, r.bitsize, r.shape_symbolic, r.side.value) for r in source.signature
    ]
    graph, leaves = source.call_graph(max_depth=1)
    native = subtraction_example_graph(example)
    assert native.direct_calls == {
        (name, _key(child)): data["n"] for _, child, data in graph.edges(data=True)
    }
    assert native.leaf_counts == {_key(child): count for child, count in leaves.items()}


def test_symbolic_composite_order_and_wire_bindings() -> None:
    from qualtran.bloqs.arithmetic.subtraction import _sub_symp_decomposition

    source = _sub_symp_decomposition.make()
    from qualtran import LeftDangle, Soquet

    active = {r.name: Soquet(LeftDangle, r) for r in source.signature.lefts()}
    steps = sorted(source.iter_bloqsoqs(), key=lambda step: step[0].i)
    for (instance, inputs, outputs), (child, ports) in zip(
        steps, symbolic_subtraction_steps(Symbol("n")), strict=True
    ):
        assert _key(instance.bloq) == child
        assert tuple(inputs.values()) == tuple(active[port] for port in ports)
        active.update(zip(ports, outputs, strict=True))
    assert active == source.final_soqs()
