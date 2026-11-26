"""Close the two frozen Quimb tests through existing native wiring owners."""

import builtins
import os
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from hashlib import sha1
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.register_bindings import RegisterBinding
from flagquantum.core.ir.program.register_types import opaque_register_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.component_tensor_nodes import (
    component_tensor_nodes,
)
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.register_identity import (
    register_bit_wiring_nodes,
)
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    contract_register_tensors,
)
from flagquantum.simulation.tensor_network.register_tensor_indices import (
    register_tensor_outer_indices,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
Bindings = Mapping[str, tuple[tuple[int, ...], ...]]


@pytest.fixture(scope="module")
def source() -> ModuleType:
    pytest.importorskip("qualtran")
    pytest.importorskip("quimb")
    import cirq
    import kickmix

    assert version("qualtran") == "0.7.0"
    assert cirq.__version__.startswith(
        os.environ.get("FQ_FIXED_GATE_SOURCE_CIRQ", "1.7") + "."
    )
    assert getattr(kickmix, "__version__", None) == "0.1" and hasattr(
        kickmix, "Simulator"
    )
    module = import_module("qualtran.simulation.tensor._quimb_test")
    assert module.__file__ is not None
    data = Path(module.__file__).read_bytes()
    assert (
        sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        == "ceb9a347a024e85317bb11eb8befd43cf2cba5c3"
    )
    return module


def _graphs(source: ModuleType, scenario: str) -> tuple[ProgramRecord, Any]:
    from qualtran._infra.composite_bloq import BloqBuilder
    from qualtran._infra.registers import Signature as SourceSignature

    if scenario == "custom":
        signature = Signature((Register("x", opaque_register_type(1)),))
        leaf = ComponentProgramBuilder(signature, name="identity_tensor")
        function = leaf.finalize(leaf.inputs).functions[0]
        native = ComponentProgramBuilder(signature, (function,))
        values = native.inputs
        foreign, original = BloqBuilder.from_signature(SourceSignature.build(x=1))
        for _ in range(4):
            values = native.call_dict(function.name, values)
            original["x"] = foreign.add(source.TensorAdderSimple(), x=original["x"])
    else:
        widths = (
            {"selection": 2, "target": 1} if scenario == "idle_word" else {"target": 1}
        )
        signature = Signature(
            Register(name, opaque_register_type(width))
            for name, width in widths.items()
        )
        native = ComponentProgramBuilder(signature)
        values = native.inputs
        foreign, original = BloqBuilder.from_signature(SourceSignature.build(**widths))
        if scenario == "idle_word":
            from qualtran._infra.data_types import QAny
            from qualtran.bloqs.bookkeeping.join import Join
            from qualtran.bloqs.bookkeeping.split import Split

            word = values["selection"]
            assert isinstance(word, RegisterBinding)
            values["selection"] = native.join(native.split(word), word.register.dtype)
            original["selection"] = foreign.add(
                Join(QAny(2)),
                reg=foreign.add(Split(QAny(2)), reg=original["selection"]),
            )
    return native.finalize(values), foreign.finalize(**original)


def _supply(
    function: FunctionRecord, incoming: Bindings, outgoing: Bindings
) -> tuple[TensorNetworkNode, ...]:
    like = torch.empty((), dtype=torch.complex128)
    if function.name == "identity_tensor":
        # The original custom class deliberately orders input before output.
        return (
            TensorNetworkNode(
                torch.eye(2, dtype=like.dtype),
                (*incoming["x"][0], *outgoing["x"][0]),
                name=function.name,
            ),
        )
    assert function.registers is not None
    return register_bit_wiring_nodes(
        Signature(function.registers), incoming, outgoing, like
    )


def _canonical_edges(
    rows: Iterable[tuple[Any, ...]], external: Mapping[Any, object]
) -> tuple[tuple[object, ...], ...]:
    internal: dict[object, int] = {}
    return tuple(
        tuple(
            (
                external[label]
                if label in external
                else ("internal", internal.setdefault(label, len(internal)))
            )
            for label in row
        )
        for row in rows
    )


def test_complete_unchanged_original_tests(source: ModuleType) -> None:
    source.test_cbloq_to_quimb()
    source.test_cbloq_to_quimb_with_no_ops_on_register()


@pytest.mark.parametrize("scenario", ("custom", "idle_word", "idle_bit"))
def test_original_nodes_connectivity_and_source_free_native_execution(
    source: ModuleType, scenario: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qualtran.simulation.tensor._dense import quimb_to_dense
    from qualtran.simulation.tensor._quimb import cbloq_to_quimb

    record, foreign = _graphs(source, scenario)
    original = cbloq_to_quimb(foreign)
    expected = quimb_to_dense(original, foreign.signature)
    actual_import: Callable[..., Any] = builtins.__import__

    def native_only(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"qualtran", "quimb"}:
            raise AssertionError("Native tensor wiring must not import source SDKs")
        return actual_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", native_only)
    order = tuple(position for position, _, _ in iter_component_calls(record))
    nodes, incoming, outgoing = component_tensor_nodes(
        record, order, _supply, torch.empty((), dtype=torch.complex128)
    )
    signature = Signature(record.functions[0].registers or ())
    dense = contract_register_tensors(nodes, signature, incoming, outgoing)
    np.testing.assert_array_equal(dense.numpy(), expected)
    assert (
        len(nodes)
        == len(original.tensors)
        == {
            "custom": 4,
            "idle_word": 5,
            "idle_bit": 1,
        }[scenario]
    )
    names = register_tensor_outer_indices(signature, incoming, outgoing)
    occurrences = Counter(label for node in nodes for label in node.labels)
    assert {label for label, count in occurrences.items() if count == 1} == names.keys()
    assert set(names.values()) == set(original.outer_inds())
    assert all(
        count == (1 if label in names else 2) for label, count in occurrences.items()
    )
    for node, tensor in zip(nodes, original.tensors, strict=True):
        np.testing.assert_array_equal(node.tensor.numpy(), tensor.data)
    source_rows = [tensor.inds for tensor in original.tensors]
    # Source idle placeholders use input/output; the native owner uses output/input.
    # Active Split/Join tensors and the explicit custom supplier retain source axes.
    if scenario != "custom":
        source_rows[-1] = source_rows[-1][::-1]
    assert _canonical_edges((node.labels for node in nodes), names) == _canonical_edges(
        source_rows, {label: label for label in original.outer_inds()}
    )
    if scenario == "custom":
        assert len(order) == 4
        assert sorted(names.values()) == [("x", (), 0, "l"), ("x", (), 0, "r")]
    width = sum(reg.total_bits() for reg in signature)
    bindings = []
    offset = 0
    for reg in signature:
        assert isinstance(reg.bitsize, int)
        bindings.append(tuple(range(offset, offset + reg.bitsize)))
        offset += reg.bitsize
    bound = bind_register_program(record, tuple(bindings), clean=(width,))
    assert bound.circuit.instructions == ()
    import flagquantum as fq
    from examples.algorithms.binary_field_arithmetic import FieldOperation

    vector = torch.complex(
        torch.arange(1, 2**width + 1, dtype=torch.float64),
        torch.arange(2**width, 0, -1, dtype=torch.float64),
    )
    vector /= vector.norm()
    initial = torch.zeros(2 ** (width + 1), dtype=torch.complex128)
    initial[::2] = vector
    state = (
        fq.Circuit(width + 1, inputs=initial, dtype=torch.complex128)
        .append(FieldOperation(width + 1, bound.circuit.instructions))
        .state()
        .flatten()
    )
    torch.testing.assert_close(state[::2], dense @ vector, atol=1e-15, rtol=0)
    assert torch.count_nonzero(state[1::2]) == 0
