"""Verify the complete frozen negation teaching examples."""

from hashlib import sha1
from importlib.metadata import version
from pathlib import Path

import pytest

from examples.algorithms.negation_catalog import (
    negation_example_graph,
    negation_examples,
)
from flagquantum.compiler._component_trace.arithmetic_call import ArithmeticCall

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
SOURCE = (
    Path(__file__).resolve().parents[3] / "ftqc-reference-env/qualtran/bloqs/arithmetic"
)


def test_locked_assets() -> None:
    import kickmix

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    for name, digest in (
        ("negate.py", "0d42dcb3f1edd0554273991c9cfce0aa05d2400a"),
        ("negate_test.py", "b6ee06978ae5805fec57af478a5777ce6548a84a"),
        ("negate.ipynb", "832bdbaf6629d27515a3a889fbdf88e200ebc594"),
    ):
        data = (SOURCE / name).read_bytes()
        assert sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == digest


@pytest.mark.parametrize("index", [0, 1])
def test_original_signature_graph_and_order(index: int) -> None:
    from qualtran import QInt
    from qualtran.bloqs.arithmetic import AddK, BitwiseNot
    from qualtran.bloqs.arithmetic.negate import _negate, _negate_symb

    source = (_negate, _negate_symb)[index].make()
    example = negation_examples()[index]
    name, signature, calls = example
    assert all(isinstance(r.dtype, QInt) for r in source.signature)
    assert [(r.name, r.bitsize, r.shape_symbolic, r.side.value) for r in signature] == [
        (r.name, r.bitsize, r.shape_symbolic, r.side.value) for r in source.signature
    ]
    native = negation_example_graph(example)
    expected = {}
    for child, count in source.bloq_counts().items():
        assert isinstance(child, (AddK, BitwiseNot)) and isinstance(child.dtype, QInt)
        key = ArithmeticCall(
            "constant_add" if isinstance(child, AddK) else "bitwise_not",
            (child.dtype.bitsize,),
            ("twos_complement",),
            constant=child.k if isinstance(child, AddK) else None,
        )
        expected[key] = count
    assert native.direct_calls == {
        (name, child): count for child, count in expected.items()
    }
    assert native.leaf_counts == expected
    decomposition = source.decompose_bloq()
    instances = sorted(decomposition.iter_bloqsoqs(), key=lambda item: item[0].i)
    assert [type(item[0].bloq) for item in instances] == [BitwiseNot, AddK]
    assert [
        child.operation for child in calls if isinstance(child, ArithmeticCall)
    ] == ["bitwise_not", "constant_add"]
    (_, first_ports, first_out), (_, second_ports, second_out) = instances
    assert list(first_ports) == ["x"]
    assert second_ports == {"x": first_out[0]}
    assert decomposition.final_soqs() == {"x": second_out[0]}
