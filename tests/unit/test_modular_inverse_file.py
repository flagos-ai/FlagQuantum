"""Whole-file audit of the locked Kaliski modular inverse tutorial and suite."""

from collections import Counter
from hashlib import sha1
from importlib.metadata import version
from math import gcd, sqrt
from pathlib import Path
from typing import Any

import pytest
from sympy import Symbol, expand

from examples.algorithms.modular_inverse_catalog import modular_inverse_examples
from examples.algorithms.register_modular_inverse import ModularInverse
from flagquantum.compiler.registers.modular.modular_inverse import (
    modular_inverse_ir,
    modular_inverse_outputs,
    source_modular_inverse_t_count,
)
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_inverse_algorithm import pack
from tests.unit.test_modular_inverse import layout, original_public
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
EXAMPLES = modular_inverse_examples()


@pytest.fixture(scope="module")
def source_module() -> Any:
    import kickmix
    from qualtran.bloqs.mod_arithmetic import mod_division

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    folder = Path(mod_division.__file__).parent
    for name, digest in (
        ("mod_division.ipynb", "fd34e136d02bd80d927945123f9a65f428c38e2a"),
        ("mod_division.py", "947f40eb5d17101ad91ff4d4432ac5287f867bd5"),
        ("mod_division_test.py", "2b8a7fbca15e4b5e7143f4b50d002334165b4c5d"),
    ):
        raw = (folder / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest
    return mod_division


def _child(node: Any) -> str:
    name = type(node).__name__
    if name == "_KaliskiIteration":
        return f"KaliskiIteration({node.bitsize}, {node.mod})"
    if name == "BitwiseNot":
        return f"BitwiseNot({node.dtype.bitsize})"
    if name in ("AddK", "XorK"):
        return f"{name}({node.dtype.bitsize}, {node.k})"
    if name == "Controlled":
        nested = node.subbloq
        assert type(nested).__name__ == "XorK"
        return f"C[XorK]({nested.dtype.bitsize}, {nested.k}, {node.ctrl_spec.get_single_ctrl_val()})"
    assert name == "XGate"
    return str(node)


@pytest.mark.parametrize("name", tuple(EXAMPLES))
def test_original_fixture_signatures_and_direct_children(
    name: str, source_module: Any
) -> None:
    source = getattr(source_module, "_" + name).make()
    native = EXAMPLES[name]
    for adjoint in (False, True):
        source_view = source.adjoint() if adjoint else source
        signature = native.signature(adjoint=adjoint)
        assert [
            (r.name, r.bitsize, r.shape_symbolic, r.side.value) for r in signature
        ] == [
            (r.name, r.bitsize, r.shape_symbolic, r.side.value)
            for r in source_view.signature
        ]
        assert dict(signature[0].dtype.parameters)["modulus"] == source_view.mod
    _, children = source.call_graph(max_depth=1)
    assert native.call_graph().leaf_counts == {
        _child(node): count for node, count in children.items()
    }


def test_original_symbolic_and_exact_swap_cost(source_module: Any) -> None:
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    n, p = Symbol("n"), Symbol("p")
    source = source_module.KaliskiModInverse(n, p)
    qec = get_cost_value(source, QECGatesCost())
    total = 26 * n**2 + 9 * n - 1
    assert expand(qec.total_toffoli_only()) == total
    assert expand(source_modular_inverse_t_count(n)) == 4 * total
    assert (
        expand(
            source_modular_inverse_t_count(n, exact_swaps=True)
            - qec.total_t_count(ts_per_cswap=7)
        )
        == 0
    )


@pytest.mark.parametrize("n", (5, 6))
@pytest.mark.parametrize("p", (3, 5, 7, 11, 13, 15))
def test_original_unit_zero_and_history_grid(
    n: int, p: int, source_module: Any
) -> None:
    x, junk, work, _ = layout(n)
    output, returned = modular_inverse_outputs(x, work)
    assert modular_inverse_outputs(output, returned, adjoint=True) == (x, work)
    width = 10 * n + 8
    source = source_module.KaliskiModInverse(n, p)
    residue = pow(2, n, p)
    inputs = tuple(
        [0, *(value * residue % p for value in range(1, p) if gcd(value, p) == 1)]
    )
    before = tuple(pack((value, 0), (x, junk), width) for value in inputs)
    expected = []
    for value in inputs:
        y, history = source.call_classically(x=value)
        assert y == (pow(value, -1, p) * residue**2 % p if value else 0)
        if not value:
            assert history == (1 << (4 * n - 1)) + (1 << (2 * n - 1))
        assert source.adjoint().call_classically(x=y, junk=history) == (value,)
        expected.append(pack((int(y), int(history)), (output, junk), width))

    def words(labels: tuple[int, ...] | list[int]) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(label >> j & 1 for j in reversed(range(width))) for label in labels
        )

    forward = DynamicCircuit(width).append(ModularInverse(n, p)).to_ir()
    reverse = DynamicCircuit(width).append(ModularInverse(n, p, adjoint=True)).to_ir()
    rows, events = certified_and_channel_map(forward.instructions, words(before))
    assert rows == words(expected)
    assert certified_and_channel_map(reverse.instructions, rows) == (
        words(before),
        events,
    )
    counts = Counter(g.name for g in forward.instructions)
    assert counts["measure"] == counts["reset"] == events
    assert counts["t"] + counts["tdg"] == source_modular_inverse_t_count(
        n, exact_swaps=True
    )

    # Two original-grid witnesses retain a coherent full-support state as
    # well as the all-history certificate for the exact input/output map.
    if p == 15:
        amplitudes = tuple(1j**i / sqrt(len(before)) for i in range(len(before)))
        initial = dict(zip(before, amplitudes, strict=True))
        final = dict(zip(expected, amplitudes, strict=True))
        for actual, desired in (
            (execute_sparse_register(forward, initial, seed=73), final),
            (execute_sparse_register(reverse, final, seed=94), initial),
        ):
            assert (
                max(
                    abs(actual.get(i, 0) - desired.get(i, 0))
                    for i in actual.keys() | desired.keys()
                )
                < 2e-12
            )


@pytest.mark.parametrize("n,p", ((5, 7), (6, 15)))
@pytest.mark.parametrize("adjoint", (False, True))
def test_original_grid_gate_order_feedback_and_physical_roles(
    n: int, p: int, adjoint: bool, source_module: Any
) -> None:
    from qualtran.resource_counting import QubitCount, get_cost_value

    x, junk, work, classical = layout(n)
    if adjoint:
        x, work = modular_inverse_outputs(x, work)
    ir = modular_inverse_ir(x, junk, p, work, classical, adjoint=adjoint)
    assert_source(ir, original_public(n, p, adjoint))
    assert (
        get_cost_value(
            source_module.KaliskiModInverse(n, p, uncompute=adjoint), QubitCount()
        )
        == 10 * n + 8
    )
