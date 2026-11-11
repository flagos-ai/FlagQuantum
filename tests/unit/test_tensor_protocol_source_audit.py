"""Complete original tensor-protocol tests with native boundary replacements."""

import ast
import inspect
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.core.ir.program import FunctionRecord
from flagquantum.simulation.tensor_network.classical_action_tensor import (
    classical_action_tensor_node,
)
from flagquantum.simulation.tensor_network.component_tensor_providers import (
    ComponentTensorProviders,
)
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from tests.unit.test_classical_action_tensor import (
    decode,
    encode,
    native,
    signature_from_source,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
Ports = Mapping[str, tuple[tuple[int, ...], ...]]


def native_classical_supplier(
    bloq: Any, incoming: dict[str, Any], outgoing: dict[str, Any]
) -> list[Any]:
    """Transport native node results into the original test's Quimb boundary.

    Connections stay confined to this test-only adapter. Table generation,
    bit codecs, data-axis binding and validation all use existing native owners.
    The source action is the callback input, not the source table generator.
    """
    import quimb.tensor as qtn

    labels: dict[Any, int] = {}
    native_ports = []
    for registers, connections in (
        (bloq.signature.lefts(), incoming),
        (bloq.signature.rights(), outgoing),
    ):
        ports = {}
        for register in registers:
            words = []
            for connection in np.asarray(connections[register.name]).flat:
                word = []
                for bit in range(register.dtype.num_qubits):
                    index = (connection, bit)
                    if index not in labels:
                        labels[index] = len(labels)
                    word.append(labels[index])
                words.append(tuple(word))
            ports[register.name] = tuple(words)
        native_ports.append(ports)
    node = classical_action_tensor_node(
        signature_from_source(bloq.signature),
        native_ports[0],
        native_ports[1],
        bloq.call_classically,
        decode,
        encode,
        torch.empty((), dtype=torch.float64),
        operation_name=str(bloq),
    )
    indices = {label: index for index, label in labels.items()}
    return [
        qtn.Tensor(node.tensor.numpy(), tuple(indices[label] for label in node.labels))
    ]


@pytest.mark.parametrize("kind", ["x", "swap", "toffoli", "add", "xor"])
def test_complete_original_classical_dense_tests_with_native_table(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    from qualtran import QAny, QUInt
    from qualtran.bloqs.arithmetic import Add, Xor
    from qualtran.bloqs.basic_gates import Toffoli, TwoBitCSwap, XGate
    from qualtran.simulation.tensor import _tensor_from_classical_test as original

    bloq = {
        "x": XGate(),
        "swap": TwoBitCSwap(),
        "toffoli": Toffoli(),
        "add": Add(QUInt(3)),
        "xor": Xor(QAny(3)),
    }[kind]
    monkeypatch.setattr(original, "bloq_to_dense_via_classical_action", native)
    original.test_tensor_consistent_with_classical(bloq)


def test_complete_original_classical_supplier_test_with_native_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qualtran.simulation.tensor import _tensor_from_classical_test as original

    monkeypatch.setattr(
        original, "my_tensors_from_classical_action", native_classical_supplier
    )
    original.test_my_tensors_from_classical_action()


def test_complete_original_classifier_test_with_native_explicit_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qualtran.bloqs.basic_gates import CNOT
    from qualtran.bloqs.mcmt.and_bloq import MultiAnd
    from qualtran.simulation.tensor import _flattening_test as original

    def unexpected_supply(
        function: FunctionRecord, incoming: Ports, outgoing: Ports
    ) -> tuple[TensorNetworkNode, ...]:
        raise AssertionError("classification must not execute numerical suppliers")

    providers = ComponentTensorProviders({"cnot": unexpected_supply})
    names = {CNOT: "cnot", MultiAnd: "multi_and"}

    def has_tensor(bloq: Any) -> bool:
        return providers.has_custom_tensor(names[type(bloq)])

    monkeypatch.setattr(original, "bloq_has_custom_tensors", has_tensor)
    original.test_bloq_has_custom_tensors()


def test_original_tensor_namespace_is_only_explicit_reexports() -> None:
    import qualtran.simulation.tensor as source

    tree = ast.parse(inspect.getsource(source))
    imports = {
        (statement.module, alias.name)
        for statement in tree.body
        if isinstance(statement, ast.ImportFrom)
        for alias in statement.names
    }
    assert imports == {
        ("_dense", "bloq_to_dense"),
        ("_dense", "quimb_to_dense"),
        ("_flattening", "bloq_has_custom_tensors"),
        ("_flattening", "flatten_for_tensor_contraction"),
        ("_quimb", "cbloq_to_quimb"),
        ("_quimb", "cbloq_to_superquimb"),
        ("_quimb", "DiscardInd"),
        ("_quimb", "initialize_from_zero"),
        ("_tensor_data_manipulation", "active_space_for_ctrl_spec"),
        ("_tensor_data_manipulation", "eye_tensor_for_signature"),
        ("_tensor_data_manipulation", "tensor_data_from_unitary_and_signature"),
        ("_tensor_data_manipulation", "tensor_out_inp_shape_from_signature"),
        ("_tensor_data_manipulation", "tensor_shape_from_signature"),
        ("_tensor_from_classical", "bloq_to_dense_via_classical_action"),
        ("_tensor_from_classical", "my_tensors_from_classical_action"),
    }
    assert all(
        isinstance(statement, ast.ImportFrom)
        or (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
        )
        for statement in tree.body
    )


def test_original_simulation_namespace_is_documentation_only() -> None:
    import qualtran.simulation as source

    tree = ast.parse(inspect.getsource(source))
    assert len(tree.body) == 1
    assert isinstance(tree.body[0], ast.Expr)
    assert isinstance(tree.body[0].value, ast.Constant)
    assert isinstance(tree.body[0].value.value, str)
