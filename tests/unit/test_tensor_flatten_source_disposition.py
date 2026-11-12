"""Unmodified flatten source boundaries and explicit native disposition."""

from functools import wraps
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.compiler.registers.component_tensor_expansion import (
    expand_component_root_for_tensor,
    expand_components_for_tensor,
)
from flagquantum.core.ir.program import verify_program_record
from flagquantum.simulation.tensor_network.component_tensor_providers import (
    ComponentTensorProviders,
)
from tests.unit.test_component_expansion import _hierarchy
from tests.unit.test_component_tensor_expansion import source_custom
from tests.unit.test_component_tensor_nodes import h_supplier

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("base", False),
        ("register_gate", False),
        ("custom", True),
        ("inherited", True),
        ("wrapped_base", False),
    ],
)
def test_complete_source_classifier_and_explicit_declaration(
    kind: str, expected: bool
) -> None:
    from qualtran import Bloq, GateWithRegisters
    from qualtran.simulation.tensor._flattening import bloq_has_custom_tensors

    parent = type(source_custom())
    calls: list[str] = []

    @wraps(Bloq.my_tensors)
    def wrapped(self: object, *args: Any, **kwargs: Any) -> Any:
        calls.append("tensor callback")
        raise AssertionError("classification must not execute tensors")

    methods = {
        "base": Bloq.my_tensors,
        "register_gate": GateWithRegisters.my_tensors,
        "custom": parent.my_tensors,
        "wrapped_base": wrapped,
    }
    cls = type(
        "InheritedLeaf",
        (parent,),
        {} if kind == "inherited" else {"my_tensors": methods[kind]},
    )
    assert bloq_has_custom_tensors(cls()) is expected
    assert calls == []
    # Declarations replace Python method-name inference; wrappers cannot hide them.
    providers = ComponentTensorProviders({"h": h_supplier})
    assert providers.has_custom_tensor("h")
    assert not providers.has_custom_tensor("H")
    record, _ = _hierarchy(1, 1)
    selected, labels = expand_components_for_tensor(
        record, providers.has_custom_tensor, full_flatten=False
    )
    assert selected == record and labels == (0,)


@pytest.mark.parametrize("failure_kind", ["atomic", "missing", "runtime", "value"])
@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("full", [False, True])
def test_source_lazy_decomposition_failures_have_explicit_native_boundary(
    failure_kind: str, custom: bool, full: bool
) -> None:
    from qualtran import Bloq, DecomposeNotImplementedError, DecomposeTypeError
    from qualtran.simulation.tensor._flattening import flatten_for_tensor_contraction

    error_types: dict[str, type[Exception]] = {
        "atomic": DecomposeTypeError,
        "missing": DecomposeNotImplementedError,
        "runtime": RuntimeError,
        "value": ValueError,
    }
    failure = error_types[failure_kind]("lazy decomposition sentinel")
    calls: list[str] = []

    def decompose(self: object) -> Any:
        calls.append("decompose")
        raise failure

    parent = type(source_custom())
    cls = type(
        "DeferredLeaf",
        (parent,),
        {
            "decompose_bloq": decompose,
            "my_tensors": parent.my_tensors if custom else Bloq.my_tensors,
        },
    )
    source = cls()
    attempted = full or not custom
    swallowed = failure_kind in {"atomic", "missing"}
    if attempted and not swallowed:
        with pytest.raises(type(failure)) as caught:
            flatten_for_tensor_contraction(source, full_flatten=full)
        assert caught.value is failure
    else:
        flattened = flatten_for_tensor_contraction(source, full_flatten=full)
        instances = list(flattened.iter_bloqsoqs())
        assert len(instances) == 1
        assert instances[0][0].i == 0 and instances[0][0].bloq is source
    assert calls == (["decompose"] if attempted else [])

    # Native definitions are already materialized. No foreign lazy factory runs.
    record, _ = _hierarchy(1, 1)
    leaf = record.functions[-1]
    atomic_record = type(record)(functions=(leaf,))
    seen: list[str] = []

    def classify(name: str) -> bool:
        seen.append(name)
        return custom

    selected, labels = expand_component_root_for_tensor(
        atomic_record, classify, entry="h", full_flatten=full
    )
    assert verify_program_record(selected) == ()
    assert labels == (0,)
    assert selected.functions[0].blocks[0].operations[0].callee == "h"
    assert seen == ([] if full else ["h"])

    # Propagation is verified at the actual native callback boundary, not falsely
    # equated to a foreign lazy-decomposition callback that does not exist here.
    def fail(name: str) -> bool:
        raise failure

    if not full:
        with pytest.raises(type(failure)) as caught_native:
            expand_component_root_for_tensor(
                atomic_record, fail, entry="h", full_flatten=False
            )
        assert caught_native.value is failure
    else:
        assert expand_component_root_for_tensor(atomic_record, fail, entry="h")[1] == (
            0,
        )


@pytest.mark.parametrize("full", [False, True])
def test_empty_decomposition_is_eliminated_with_identity_outputs(full: bool) -> None:
    from qualtran import Bloq, BloqBuilder
    from qualtran import Signature as SourceSignature
    from qualtran.simulation.tensor._flattening import flatten_for_tensor_contraction

    from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
    from flagquantum.compiler.registers.component_graph import iter_component_calls
    from flagquantum.core.ir.program.signatures import Signature
    from flagquantum.simulation.tensor_network.component_dense import (
        contract_component_tensors,
    )

    def build(self: object, bb: Any, q: Any) -> dict[str, Any]:
        return {"q": q}

    cls = type(
        "EmptyIdentity",
        (Bloq,),
        {
            "signature": property(lambda self: SourceSignature.build(q=1)),
            "build_composite_bloq": build,
        },
    )
    source = cls()
    flattened = flatten_for_tensor_contraction(source, full_flatten=full)
    assert len(flattened.bloq_instances) == 0
    # Source identity path is retained by its direct left-to-right connection.
    bb, inputs = BloqBuilder.from_signature(source.signature)
    expected = bb.finalize(**inputs).tensor_contract()
    record, _ = _hierarchy(1, 1)
    builder = ComponentProgramBuilder(Signature(record.functions[0].registers or ()))
    empty = builder.finalize(dict(builder.inputs))
    selected, labels = expand_component_root_for_tensor(
        empty, lambda name: False, full_flatten=full
    )
    assert labels == ()
    calls: list[str] = []

    def unexpected(*args: Any) -> Any:
        calls.append("supply")
        raise AssertionError("empty graph must use existing idle identity wiring")

    order = tuple(i for i, _, _ in iter_component_calls(selected, "tensor_root"))
    actual = contract_component_tensors(
        selected,
        order,
        unexpected,
        torch.empty((), dtype=torch.complex128),
        "tensor_root",
    )
    np.testing.assert_array_equal(actual.numpy(), expected)
    assert calls == []
