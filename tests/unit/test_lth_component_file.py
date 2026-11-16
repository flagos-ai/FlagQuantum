"""Whole-file tests for the locked shared-target indexed component."""

import ast
from collections import Counter
from hashlib import sha1
from importlib.metadata import version
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from examples.algorithms.lth_component_catalog import (
    IndexedComponentExample,
    indexed_component_example_call_graph,
)
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_relative_controlled_swap import assert_source_dependencies

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module")
def source() -> Any:
    import kickmix
    from qualtran.bloqs.multiplexers import apply_lth_bloq

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    directory = Path(apply_lth_bloq.__file__).parent
    for suffix, digest in (
        (".ipynb", "ff8d6499f96132421124c9a8a3b643d064cbd8fd"),
        (".py", "438ae361b6805c2f707849ada7c0a4eed04e460f"),
        ("_test.py", "b98e5bcdff8a4a1b338bd0f51227f5a754ec780f"),
    ):
        data = (directory / f"apply_lth_bloq{suffix}").read_bytes()
        assert sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == digest
    return apply_lth_bloq


def test_original_multiplexer_initializer_is_documentation_only(source: Any) -> None:
    data = Path(source.__file__).with_name("__init__.py").read_bytes()
    assert (
        sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        == "25c17b6a77979aac442a70b226c59c138db931a6"
    )
    tree = ast.parse(data)
    assert len(tree.body) == 1
    documentation = ast.get_docstring(tree)
    assert documentation is not None and "via unary iteration" in documentation


def _label(node: Any) -> str:
    kind = type(node).__name__
    if kind == "And":
        return f"And({node.cv1},{node.cv2})" + ("†" if node.uncompute else "")
    if getattr(node, "subbloq", None) and type(node.subbloq).__name__ == "TGate":
        assert node.ctrl_spec.get_single_ctrl_val() == 1
        assert not node.subbloq.is_adjoint
        return "C[T]"
    return kind


def test_notebook_declarations_and_source_costs(source: Any) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from qualtran.resource_counting.generalizers import ignore_split_join

    original = source._apply_lth_bloq.make()
    _, leaves = original.call_graph(generalizer=ignore_split_join)
    assert indexed_component_example_call_graph().leaf_counts == {
        _label(child): count for child, count in leaves.items()
    }
    op = IndexedComponentExample()
    circuit = DynamicCircuit(op.qubit_count)
    circuit.append(op)
    emitted = Counter(g.name for g in circuit.to_ir().instructions)
    assert (emitted["t"] + emitted["tdg"], emitted["cx"], emitted["measure"]) == (
        12,
        22,
        3,
    )
    model = get_cost_value(original, QECGatesCost())
    modeled = estimate_component_resources(op.resource_model())
    assert (
        (
            modeled.gates.and_bloq,
            modeled.gates.clifford,
            modeled.gates.rotation,
            modeled.gates.measurement,
        )
        == (model.and_bloq, model.clifford, model.rotation, model.measurement)
        == (3, 12, 3, 3)
    )
    assert modeled.gates.total_t_count() == model.total_t_count() == 45
    assert (
        modeled.peak_logical_qubits
        == get_cost_value(original, QubitCount())
        == op.qubit_count
    )


@pytest.mark.parametrize("shape", ((2,), (3,), (4,), (2, 2)))
@pytest.mark.parametrize("adjoint", (False, True))
def test_original_tensor_grid_with_atomic_phase_and_feedback(
    shape: tuple[int, ...], adjoint: bool, source: Any
) -> None:
    from qualtran.bloqs.basic_gates import Hadamard, TGate, XGate, ZGate

    ops = np.array((TGate(), Hadamard(), ZGate(), XGate()), dtype=object)
    count = int(np.prod(shape))
    original = source.ApplyLthBloq(ops[:count].reshape(shape), control_val=1)
    op = IndexedComponentExample(adjoint=adjoint, shape=shape)
    assert [(r.name, r.bitsize, r.side.value) for r in op.signature] == [
        (r.name, r.bitsize, r.side.value) for r in original.signature
    ]
    for i, register in enumerate(original.selection_registers, start=1):
        assert (
            dict(op.signature[i].dtype.parameters)["bound"]
            == register.dtype.iteration_length
        )
    ports: dict[str, tuple[int, ...]] = {}
    start = 0
    for register in op.signature:
        end = start + int(register.bitsize)
        ports[register.name] = tuple(range(start, end))
        start = end

    circuit = DynamicCircuit(op.qubit_count)
    circuit.append(op)
    actual = circuit.to_ir().instructions
    work = tuple(range(start, op.qubit_count))
    reference = reference_arithmetic_gates(
        original.adjoint().decompose_bloq() if adjoint else original,
        ports,
        work,
        dict(zip(work, range(len(work)), strict=True)),
        cirq_style=True,
        preserve_components=True,
        keep_pauli_leaves=True,
        keep_exchange_leaves=True,
    )
    assert_source_dependencies(actual, reference)

    # Use every legal control/selection row and both target amplitudes.
    state = torch.zeros(1 << op.qubit_count, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    for ctrl in (0, 1):
        for index, address in enumerate(product(*(range(extent) for extent in shape))):
            prefix = ctrl
            for digit, width in zip(address, op.axis_widths, strict=True):
                prefix = (prefix << width) | digit
            positions = torch.tensor([((prefix << 1) | q) << len(work) for q in (0, 1)])
            vector = torch.tensor(
                [complex(1 + prefix, 7 - index), complex(3 + index, prefix - 2)],
                dtype=torch.complex128,
            )
            unitary = torch.as_tensor(
                ops[index].tensor_contract(), dtype=torch.complex128
            )
            state[positions] = vector
            expected[positions] = (
                (unitary.mH if adjoint else unitary) @ vector if ctrl else vector
            )

    norm = state.norm()
    state /= norm
    expected /= norm
    events = sum(g.name == "measure" for g in actual)
    assert events == count - 1
    for history in product((0, 1), repeat=events):
        observed = forced_branch(state, actual, history)
        torch.testing.assert_close(
            observed, expected / 2 ** (events / 2), rtol=0, atol=2e-13
        )
