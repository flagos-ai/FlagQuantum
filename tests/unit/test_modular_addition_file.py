"""Frozen modular-addition notebook and the native full-domain ModAddK binding."""

from hashlib import sha1
from importlib.metadata import version
from itertools import product
from pathlib import Path
from typing import Any

import pytest

from examples.algorithms.modular_addition_catalog import (
    ModularAdditionExample,
    modular_addition_examples,
)
from flagquantum.compiler.registers.modular.bounded_modular_addition import (
    bounded_modular_add_ir,
    bounded_modular_add_requirements,
)
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.test_sparse_chemistry_prepare import _branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
EXAMPLES = modular_addition_examples()


@pytest.fixture(scope="module")
def source_module() -> Any:
    import kickmix
    from qualtran.bloqs.mod_arithmetic import mod_addition

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    folder = Path(mod_addition.__file__).parent
    for name, digest in (
        ("mod_addition.ipynb", "b081513d2c657fb069e0da81f2b4914df531ba2f"),
        ("mod_addition.py", "0a167e84fe00d5df50488c4b28925ea6114728f7"),
        ("mod_addition_test.py", "7786a0ce6e07bf5ee7f7de410a0a41536e22ff3e"),
    ):
        raw = (folder / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest
    return mod_addition


def _child_label(node: Any) -> str:
    kind = type(node).__name__
    if kind == "Add":
        return f"Add({node.a_dtype.bitsize}, {node.b_dtype.bitsize})"
    if kind == "CAdd":
        return f"CAdd({node.a_dtype.bitsize}, {node.b_dtype.bitsize}, {node.cv})"
    if kind == "AddK":
        return f"AddK({node.dtype.bitsize}, {node.k})"
    if kind == "Controlled":
        child = node.subbloq
        return f"C[AddK]({child.dtype.bitsize}, {child.k}, {node.ctrl_spec.get_single_ctrl_val()})"
    if kind == "CModAdd":
        return f"CModAdd({node.dtype.bitsize}, {node.mod}, {node.cv})"
    if kind == "CModAddK":
        return f"CModAddK({node.bitsize}, {node.mod}, {node.k})"
    if kind == "CLinearDepthGreaterThan":
        return f"CLinearDepthGreaterThan({node.dtype.bitsize}, {node.cv})"
    if kind == "LinearDepthGreaterThan":
        assert node.signed is False
        return f"LinearDepthGreaterThan({node.bitsize})"
    assert kind in ("And", "XGate")
    return str(node)


@pytest.mark.parametrize("name", tuple(EXAMPLES))
def test_all_original_examples_signatures_and_declared_children(
    name: str, source_module: Any
) -> None:
    original = getattr(source_module, "_" + name).make()
    native = EXAMPLES[name]
    assert [
        (r.name, r.bitsize, r.shape_symbolic, r.side.value) for r in native.signature
    ] == [
        (r.name, r.bitsize, r.shape_symbolic, r.side.value) for r in original.signature
    ]
    assert [
        {"bit": "QBit", "uint": "QUInt", "montgomery": "QMontgomeryUInt"}[
            r.dtype.encoding
        ]
        for r in native.signature
    ] == [type(r.dtype).__name__ for r in original.signature]
    _, children = original.call_graph(max_depth=1)
    assert native.call_graph().leaf_counts == {
        _child_label(child): count for child, count in children.items()
    }


def _bits(word: int, width: int) -> tuple[int, ...]:
    return tuple(word >> j & 1 for j in reversed(range(width)))


@pytest.mark.parametrize(
    "n,p,k,values",
    (
        (1, 2, -3, ()),
        (2, 3, 2, ()),
        (3, 5, -8, (0, 1)),
        (3, 8, 11, (0,)),
        (3, 8, -1, (1, 0)),
        (3, 1, 8, (1,)),
        (3, 7, 0, ()),
        (3, 7, 15, ()),
    ),
)
def test_bounded_mod_add_preserves_padding_and_all_feedback_branches(
    n: int, p: int, k: int, values: tuple[int, ...], source_module: Any
) -> None:
    control_width = len(values)
    work, slots = bounded_modular_add_requirements(n, p, control_width)
    end = n + control_width
    gates = bounded_modular_add_ir(
        tuple(range(control_width, end)),
        k,
        p,
        tuple(range(end, end + work)),
        tuple(range(slots)),
        controls=tuple(range(control_width)),
        control_values=values,
    )
    source = source_module.ModAddK(n, p, k, cvs=values)
    before, after = [], []
    for control, x in product(range(1 << control_width), range(1 << n)):
        actual = source.on_classical_vals(x=x, **({"ctrl": control} if values else {}))[
            "x"
        ]
        prefix = _bits(control, control_width)
        before.append((*prefix, *_bits(x, n), *(0,) * work))
        after.append((*prefix, *_bits(actual, n), *(0,) * work))
    events = sum(gate.name == "measure" for gate in gates)
    assert certified_and_channel_map(gates, tuple(before)) == (tuple(after), events)
    inverse = bounded_modular_add_ir(
        tuple(range(control_width, end)),
        -k,
        p,
        tuple(range(end, end + work)),
        tuple(range(slots)),
        controls=tuple(range(control_width)),
        control_values=values,
    )
    inverse_events = sum(gate.name == "measure" for gate in inverse)
    assert certified_and_channel_map(inverse, tuple(after)) == (
        tuple(before),
        inverse_events,
    )


def test_large_original_mod_add_constant_preserves_boundary_states(
    source_module: Any,
) -> None:
    instance = EXAMPLES["mod_add_k_large"]
    original = source_module._mod_add_k_large.make()
    n = 64
    work, _ = instance.workspace_requirements()
    gates = DynamicCircuit(n + work).append(instance).to_ir().instructions
    boundary = (0, 1, 476, 477, 499, 500, 501, (1 << n) - 1)
    before = tuple((*_bits(x, n), *(0,) * work) for x in boundary)
    after = tuple(
        (*_bits(original.on_classical_vals(x=x)["x"], n), *(0,) * work)
        for x in boundary
    )
    events = sum(gate.name == "measure" for gate in gates)
    assert certified_and_channel_map(gates, before) == (after, events)


@pytest.mark.parametrize(
    "kind", ("ModAdd", "ModAddK", "CModAdd", "CModAddK", "CtrlScaleModAdd")
)
def test_native_append_coherent_port_layout(kind: Any, source_module: Any) -> None:
    n, p, k = 2, 3, 2
    native = ModularAdditionExample(kind, n, p, k)
    source = {
        "ModAdd": lambda: source_module.ModAdd(n, p),
        "ModAddK": lambda: source_module.ModAddK(n, p, k),
        "CModAdd": lambda: source_module.CModAdd(source_module.QUInt(n), p),
        "CModAddK": lambda: source_module.CModAddK(k=k, mod=p, bitsize=n),
        "CtrlScaleModAdd": lambda: source_module.CtrlScaleModAdd(k=k, mod=p, bitsize=n),
    }[kind]()
    live = int(native.signature.n_qubits())
    work, _ = native.workspace_requirements()
    ir = DynamicCircuit(live + work).append(native).to_ir()
    count = sum(gate.name == "measure" for gate in ir.instructions)
    initial, expected = {}, {}
    has_control = kind.startswith("C")
    has_y = kind in ("ModAdd", "CModAdd", "CtrlScaleModAdd")
    for c, x, y in product(
        range(2) if has_control else (0,),
        range((1 << n) if kind == "ModAddK" else p),
        range(p) if has_y else (0,),
    ):
        kwargs = (
            {"x": x}
            | ({"ctrl": c} if has_control else {})
            | ({"y": y} if has_y else {})
        )
        result = source.on_classical_vals(**kwargs)
        in_word = ((c << n | x) << n | y) if has_y else (c << n | x)
        out_word = (
            ((c << n | result["x"]) << n | result["y"])
            if has_y
            else (c << n | result["x"])
        )
        amplitude = complex(1 + in_word, 1 - in_word) / (1 << live)
        initial[in_word << work] = amplitude
        expected[out_word << work] = amplitude * 2 ** (-count / 2)
    for history in ((0,) * count, (1,) * count, tuple(i % 2 for i in range(count))):
        assert _branch(ir.instructions, ir.n_wires, history, initial) == pytest.approx(
            expected, abs=3e-12
        )


def test_original_large_controlled_classical_grid(source_module: Any) -> None:
    from flagquantum.simulation.classical_modular_addition import (
        controlled_modular_add_classical_values,
    )

    for prime, activation in product((11, 13, 31), (0, 1)):
        for width, dtype in product(
            range(1 + prime.bit_length(), 8),
            (source_module.QUInt, source_module.QMontgomeryUInt),
        ):
            operation = source_module.CModAdd(dtype(width), prime, cv=activation)
            for control, left, right in product((0, 1), range(prime + 1), range(prime)):
                source = operation.on_classical_vals(ctrl=control, x=left, y=right)
                assert controlled_modular_add_classical_values(
                    control, left, right, prime, control_value=activation
                ) == (source["ctrl"], source["x"], source["y"])


@pytest.mark.parametrize("prime", (13, 17, 23))
def test_original_large_constant_and_scaled_classical_grids(
    prime: int, source_module: Any
) -> None:
    from flagquantum.simulation.classical_modular_constant_addition import (
        controlled_modular_add_constant_classical_values,
    )
    from flagquantum.simulation.classical_modular_scaled_addition import (
        controlled_modular_scaled_add_classical_values,
    )

    for width, constant in product(range(prime.bit_length(), 8), range(1, prime)):
        add = source_module.CModAddK(k=constant, mod=prime, bitsize=width)
        scale = source_module.CtrlScaleModAdd(k=constant, mod=prime, bitsize=width)
        for control, x in product((0, 1), range(prime)):
            src_add = add.on_classical_vals(ctrl=control, x=x)
            assert controlled_modular_add_constant_classical_values(
                control, x, constant, prime
            ) == (src_add["ctrl"], src_add["x"])
            for y in range(prime):
                src_scale = scale.on_classical_vals(ctrl=control, x=x, y=y)
                assert controlled_modular_scaled_add_classical_values(
                    control, x, y, constant, prime
                ) == (src_scale["ctrl"], src_scale["x"], src_scale["y"])


def test_original_controlled_constant_tensor_blocks(source_module: Any) -> None:
    import numpy as np

    original = source_module._cmod_add_k_small.make()
    instance = EXAMPLES["cmod_add_k_small"]
    work, _ = instance.workspace_requirements()
    ir = DynamicCircuit(5 + work).append(instance).to_ir()
    labels = tuple(product((0, 1), range(7)))
    states = tuple((c, *_bits(x, 4), *(0,) * work) for c, x in labels)
    actual, events = certified_and_channel_map(ir.instructions, states)
    assert events == 21
    matrix = np.zeros((32, 32))
    for (control, value), output in zip(labels, actual, strict=True):
        assert not any(output[5:])
        target = sum(bit << i for i, bit in enumerate(reversed(output[1:5])))
        matrix[16 * output[0] + target, 16 * control + value] = 1
    reference = original.tensor_contract()
    for start in (0, 16):
        subset = slice(start, start + 7)
        np.testing.assert_allclose(
            matrix[subset, subset], reference[subset, subset], atol=1e-14, rtol=0
        )
