"""Whole-file checks for the locked modular multiplication tutorial and tests."""

from collections import Counter
from hashlib import sha1
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest
from sympy import Rational, symbols

from examples.algorithms.modular_multiplication_catalog import (
    modular_multiplication_examples,
)
from examples.algorithms.register_modular_doubling import ModularDoubling
from examples.algorithms.register_modular_multiplication import ModularMultiplication
from examples.algorithms.register_windowed_mod_mul import (
    WindowedMontgomeryMultiplication,
)
from flagquantum.compiler.registers.modular.modular_doubling import (
    modular_double_requirements,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.simulation.classical_modular_multiplication import (
    controlled_modular_multiply_constant_classical_values,
    modular_double_classical_value,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module")
def source() -> Any:
    import kickmix
    from qualtran.bloqs.mod_arithmetic import mod_multiplication

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    for suffix, digest in (
        (".ipynb", "d8f7d6166fece00ea1ec07759ad8907bf9a41b35"),
        (".py", "c0242f9735bf269564c9480c14c008d5f9621fd7"),
        ("_test.py", "d6a1caa6231a0dcf4769b3f8c2b747bd15e8212c"),
    ):
        path = Path(mod_multiplication.__file__).with_name(
            f"mod_multiplication{suffix}"
        )
        raw = path.read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest
    return mod_multiplication


def _label(node: Any) -> str:
    name = type(node).__name__
    if name in ("AddK", "Controlled"):
        controlled = name == "Controlled"
        if controlled:
            assert node.ctrl_spec.get_single_ctrl_val() == 1
            node = node.subbloq
            assert type(node).__name__ == "AddK"
        label = "C[AddK]" if controlled else "AddK"
        return f"{label}({type(node.dtype).__name__}({node.dtype.bitsize}), {node.k})"
    if name == "CtrlScaleModAdd":
        return f"CtrlScaleModAdd({node.bitsize}, {node.mod}, {node.k})"
    if name == "CSwap":
        return f"CSwap({node.bitsize})"
    if name == "LessThanConstant":
        return f"LessThanConstant({node.bitsize}, {node.less_than_val})"
    if name == "SingleWindowModMul":
        return f"SingleWindowModMul({node.window_size}, {node.bitsize}, {node.mod})"
    return str({"CNOT": "CNOT", "XGate": "X"}[name])


@pytest.mark.parametrize("name", tuple(modular_multiplication_examples()))
def test_complete_notebook_fixture_signatures_and_calls(name: str, source: Any) -> None:
    ex = modular_multiplication_examples()[name]
    original = getattr(source, "_" + name).make()
    assert ex.modulus == original.mod
    if ex.constant is not None:
        assert ex.constant == original.k
    for uncompute in ((False, True) if ex.window_size is not None else (False,)):
        view = original.adjoint() if uncompute else original
        native = ex.signature(uncompute=uncompute)
        assert [
            (r.name, r.bitsize, r.shape_symbolic, r.side.value) for r in native
        ] == [
            (r.name, r.bitsize, r.shape_symbolic, r.side.value) for r in view.signature
        ]
        for lhs, rhs in zip(native, view.signature, strict=True):
            if hasattr(rhs.dtype, "modulus"):
                assert dict(lhs.dtype.parameters)["modulus"] == rhs.dtype.modulus
    _, leaves = original.call_graph(max_depth=1)
    assert ex.call_graph().leaf_counts == {
        _label(node): count for node, count in leaves.items()
    }


@pytest.mark.parametrize("modulus", (13, 17, 23))
def test_original_full_controlled_value_grid(modulus: int, source: Any) -> None:
    from qualtran import QMontgomeryUInt, QUInt

    for dtype in (QUInt, QMontgomeryUInt):
        for n in range(modulus.bit_length(), 10):
            double = source.ModDbl(dtype(n), modulus)
            for x in range(modulus):
                assert (
                    modular_double_classical_value(x, modulus)
                    == double.call_classically(x=x)[0]
                )
            for k in range(1, modulus):
                multiply = source.CModMulK(dtype(n), k, modulus)
                for c in (0, 1):
                    for x in range(modulus):
                        expected = multiply.call_classically(ctrl=c, x=x)
                        assert (
                            controlled_modular_multiply_constant_classical_values(
                                c, x, k, modulus
                            )
                            == expected
                        )


def test_original_symbolic_cost_regressions(source: Any) -> None:
    from qualtran import QMontgomeryUInt
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    n, m, p = symbols("n m p", integer=True)
    cost = get_cost_value(source.ModDbl(QMontgomeryUInt(n), p), QECGatesCost())
    assert cost.total_t_and_ccz_count() == {"n_t": 0, "n_ccz": 2 * n + 1}
    for uncompute in (False, True):
        original = source.DirtyOutOfPlaceMontgomeryModMul(4 * m, 4, p, uncompute)
        cost = get_cost_value(original, QECGatesCost()).total_t_and_ccz_count()
        assert cost["n_t"] == 0
        assert (
            cost["n_ccz"].subs(m, n / 4).expand()
            == Rational(9, 4) * n**2 + Rational(29, 4) * n - 1
        )


def test_append_uses_original_notebook_parameters_and_physical_models(
    source: Any,
) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    for name, ex in modular_multiplication_examples().items():
        n, p = ex.width, ex.modulus
        if not isinstance(n, int) or not isinstance(p, int):
            continue
        original = getattr(source, "_" + name).make()
        model = get_cost_value(original, QECGatesCost())
        if ex.window_size is not None:
            w = ex.window_size
            op = WindowedMontgomeryMultiplication(width=n, window=w, modulus=p)
            width = op.qubit_count
            circuit = DynamicCircuit(width).append(op)
            assert (
                len(set((*op.target_wires, *op.history_wires, op.reduction_wire)))
                == 2 * n + 1
            )
            assert op.target_wires[-w:] == tuple(range(4 * n + 1, 4 * n + 1 + w))
            lookup = original._mod_mul_impl._window.qrom
            load = get_cost_value(lookup, QECGatesCost())
            erase = get_cost_value(lookup.adjoint(), QECGatesCost())
            t_delta = n // w * (load.total_t_count() - erase.total_t_count())
            m_delta = n // w * (load.measurement - erase.measurement)
        elif isinstance(ex.constant, int):
            multiply = ModularMultiplication(ex.constant, width=n, modulus=p)
            width = multiply.qubit_count
            circuit = DynamicCircuit(width).append(multiply)
            t_delta, m_delta = 3 * n, 0
        else:
            work, _ = modular_double_requirements(n)
            width = n + work
            double = ModularDoubling(tuple(range(n)), tuple(range(n, width)), modulus=p)
            circuit = DynamicCircuit(width).append(double)
            t_delta, m_delta = 0, 0
        counts = Counter(gate.name for gate in circuit.to_ir().instructions)
        assert width == get_cost_value(original, QubitCount())
        assert counts["t"] + counts["tdg"] == model.total_t_count() + t_delta
        assert counts["measure"] == model.measurement + m_delta
