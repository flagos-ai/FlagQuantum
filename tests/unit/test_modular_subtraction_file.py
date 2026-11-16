"""Whole-file checks for modular subtraction and the source package exports."""

import ast
from collections import Counter
from dataclasses import replace
from hashlib import sha1
from importlib import import_module
from importlib.metadata import version
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sympy import symbols

from examples.algorithms.modular_subtraction_catalog import modular_subtraction_examples
from flagquantum.dynamic import DynamicCircuit
from flagquantum.simulation.classical_modular_negation import (
    controlled_modular_negate_classical_values,
    modular_negate_classical_value,
)
from flagquantum.simulation.classical_modular_subtraction import (
    controlled_modular_subtract_classical_values,
    modular_subtract_classical_values,
)
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.sparse_register_execution import execute_sparse_register

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module")
def source() -> Any:
    import kickmix
    from qualtran.bloqs.mod_arithmetic import mod_subtraction

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    folder = Path(mod_subtraction.__file__).parent
    for name, digest in (
        ("__init__.py", "774aa44d6c868ebc8eaa08386ac9b318b0d898de"),
        ("mod_subtraction.ipynb", "e925323025d5e221d32d61c6974680b95c759d07"),
        ("mod_subtraction.py", "e0d0fbf81a81ec2ed55e02b4b31ad3e75ab85324"),
        ("mod_subtraction_test.py", "3d013fdd1c43b5853681c2920fd7296ba00bdedc"),
    ):
        data = (folder / name).read_bytes()
        assert sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == digest
    return mod_subtraction


def _dtype_label(dtype: Any) -> str:
    return f"{type(dtype).__name__}({dtype.bitsize})"


def _label(node: Any) -> str:
    kind = type(node).__name__
    if kind == "Adjoint":
        return _label(node.subbloq) + "†"
    if kind == "XGate":
        return "X"
    if kind == "BitwiseNot":
        return f"BitwiseNot({_dtype_label(node.dtype)})"
    if kind == "AddK":
        return f"AddK({_dtype_label(node.dtype)}, {node.k})"
    if kind in ("Controlled", "ControlledViaAnd"):
        assert len(node.ctrl_spec.cvs) == 1
        cv = node.ctrl_spec.cvs[0].item()
        sub = _label(node.subbloq)
        if sub == "X":
            return f"C[X]({_dtype_label(node.ctrl_spec.qdtypes[0])}, {cv})"
        return f"C[{sub}]({cv})"
    if kind == "MultiTargetCNOT":
        return f"MultiTargetCNOT({node.bitsize})"
    if kind == "MultiControlX":
        assert all(cv == 0 for cv in node.cvs)
        return f"MultiControlX({len(node.cvs)}, 0)"
    if kind == "And":
        return f"And({node.cv1}, {node.cv2})" + ("†" if node.uncompute else "")
    if kind == "ModAdd":
        return f"ModAdd({node.bitsize}, {node.mod})"
    if kind == "CModAdd":
        return f"CModAdd({_dtype_label(node.dtype)}, {node.mod}, {node.cv})"
    raise AssertionError(kind)


@pytest.mark.parametrize("name", tuple(modular_subtraction_examples()))
def test_original_fixture_signature_and_calls(name: str, source: Any) -> None:
    ex = modular_subtraction_examples()[name]
    original = getattr(source, "_" + name).make()
    assert ex.modulus == original.mod
    assert [
        (r.name, r.bitsize, r.side.value, r.shape_symbolic) for r in ex.signature
    ] == [
        (r.name, r.bitsize, r.side.value, r.shape_symbolic) for r in original.signature
    ]
    _, leaves = original.call_graph(max_depth=1)
    assert ex.call_graph().leaf_counts == {
        _label(node): count for node, count in leaves.items()
    }


@pytest.mark.parametrize("modulus", (13, 17, 23))
def test_original_prime_residue_grids(modulus: int, source: Any) -> None:
    from qualtran import QMontgomeryUInt, QUInt

    for dtype, n in product((QUInt, QMontgomeryUInt), range(modulus.bit_length(), 10)):
        word = dtype(n)
        neg = source.ModNeg(word, modulus)
        sub = source.ModSub(word, modulus)
        for x in range(modulus):
            assert (
                modular_negate_classical_value(x, modulus),
            ) == neg.call_classically(x=x)
            for y in range(modulus):
                assert modular_subtract_classical_values(
                    x, y, modulus
                ) == sub.call_classically(x=x, y=y)
        for cv in (0, 1):
            cneg = source.CModNeg(word, modulus, cv)
            csub = source.CModSub(word, modulus, cv)
            for c, x in product((0, 1), range(modulus)):
                assert controlled_modular_negate_classical_values(
                    c, x, modulus, control_value=cv
                ) == cneg.call_classically(ctrl=c, x=x)
                for y in range(modulus):
                    assert controlled_modular_subtract_classical_values(
                        c, x, y, modulus, control_value=cv
                    ) == csub.call_classically(ctrl=c, x=x, y=y)


@pytest.mark.parametrize(
    "width,modulus", tuple(product((32, 33), (10**9 + 7, 10**9 + 9)))
)
def test_original_large_seeded_grid(width: int, modulus: int, source: Any) -> None:
    from qualtran import QMontgomeryUInt

    dtype = QMontgomeryUInt(width)
    sub = source.ModSub(dtype, modulus)
    csub = source.CModSub(dtype, modulus)
    rng = np.random.default_rng(13324)
    for x, y in product(
        rng.choice(modulus, 5).tolist(), rng.choice(modulus, 5).tolist()
    ):
        assert modular_subtract_classical_values(x, y, modulus) == sub.call_classically(
            x=x, y=y
        )
        assert controlled_modular_subtract_classical_values(
            1, x, y, modulus
        ) == csub.call_classically(ctrl=1, x=x, y=y)


@pytest.mark.parametrize("name", tuple(modular_subtraction_examples()))
@pytest.mark.parametrize("cv", (0, 1))
def test_symbolic_cost_and_native_append_channels(
    name: str, cv: int, source: Any
) -> None:
    from qualtran import QMontgomeryUInt
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    n, p = symbols("n p")
    ex = modular_subtraction_examples()[name]
    controlled = ex.kind.startswith("C")
    root = getattr(source, ex.kind)
    model = root(QMontgomeryUInt(n), p, *([cv] if controlled else []))
    cost = get_cost_value(model, QECGatesCost()).total_t_and_ccz_count()
    symbolic = replace(ex, width=n, modulus=p, control_value=cv)
    assert cost == {"n_t": 0, "n_ccz": symbolic.source_ccz_count()}

    op = replace(symbolic, width=2, modulus=2)
    circuit = DynamicCircuit(op.qubit_count)
    circuit.append(op)
    forward = circuit.to_ir()
    counts = Counter(g.name for g in forward.instructions)
    assert counts["t"] + counts["tdg"] == 4 * op.source_ccz_count()
    native_reference = root(QMontgomeryUInt(2), 2, *([cv] if controlled else []))
    width = sum(r.bitsize for r in op.signature)
    padding = (0,) * (op.qubit_count - width)
    initial, expected = [], []
    for values in product(range(2), repeat=len(op.signature)):
        args = dict(zip((r.name for r in op.signature), values, strict=True))
        outputs = native_reference.call_classically(**args)
        words = tuple(r.bitsize for r in op.signature)
        initial.append(
            tuple(
                int(bit)
                for v, w in zip(values, words, strict=True)
                for bit in f"{v:0{w}b}"
            )
            + padding
        )
        expected.append(
            tuple(
                int(bit)
                for v, w in zip(outputs, words, strict=True)
                for bit in f"{v:0{w}b}"
            )
            + padding
        )
    assert certified_and_channel_map(forward.instructions, tuple(initial)) == (
        tuple(expected),
        counts["measure"],
    )

    def amplitudes(rows: list[tuple[int, ...]]) -> dict[int, complex]:
        return {
            int("".join(map(str, row)), 2): complex(i + 1, 2 * i - 1)
            for i, row in enumerate(rows)
        }

    start, end = amplitudes(initial), amplitudes(expected)
    norm = sum(abs(amp) ** 2 for amp in start.values()) ** 0.5
    start, end = ({k: v / norm for k, v in state.items()} for state in (start, end))
    circuit.append(replace(op, adjoint=True))
    for ir, target in ((forward, end), (circuit.to_ir(), start)):
        actual = execute_sparse_register(ir, start, seed=41)
        assert all(
            abs(actual.get(k, 0) - target.get(k, 0)) < 2e-12
            for k in actual.keys() | target.keys()
        )


def test_source_initializer_maps_every_export_to_an_owned_emitter(source: Any) -> None:
    package = import_module("qualtran.bloqs.mod_arithmetic")
    assert package.__file__ is not None
    tree = ast.parse(Path(package.__file__).read_text())
    exports = {
        alias.name: getattr(
            import_module(f".{node.module}", package.__name__), alias.name
        )
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    mapping = {
        "CModAdd": ("controlled_modular_addition", "controlled_modular_add_ir"),
        "CModAddK": ("modular_constant_addition", "controlled_modular_add_constant_ir"),
        "CtrlScaleModAdd": (
            "modular_scaled_addition",
            "controlled_modular_scaled_add_ir",
        ),
        "ModAdd": ("modular_addition", "modular_add_ir"),
        "ModAddK": ("bounded_modular_addition", "bounded_modular_add_ir"),
        "KaliskiModInverse": ("modular_inverse", "modular_inverse_ir"),
        "CModMulK": (
            "modular_multiplication",
            "controlled_modular_multiply_constant_ir",
        ),
        "DirtyOutOfPlaceMontgomeryModMul": ("windowed_mod_mul", "windowed_mod_mul_ir"),
        "ModDbl": ("modular_doubling", "modular_double_ir"),
        "CModNeg": ("controlled_modular_negation", "controlled_modular_negate_ir"),
        "CModSub": ("controlled_modular_subtraction", "controlled_modular_subtract_ir"),
        "ModNeg": ("modular_negation", "modular_negate_ir"),
        "ModSub": ("modular_subtraction", "modular_subtract_ir"),
    }
    assert mapping.keys() == exports.keys()
    for name, (owner, entry) in mapping.items():
        assert getattr(package, name) is exports[name]
        module = import_module(f"flagquantum.compiler.registers.modular.{owner}")
        emitter = getattr(module, entry)
        assert callable(emitter) and emitter.__module__ == module.__name__
