"""Source outer-index names remain inspection metadata on native integer graphs."""

from itertools import product
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.conjugate_pairs import tensor_conjugate_pairs
from flagquantum.simulation.tensor_network.register_superoperator_binding import (
    contract_register_superoperator,
)
from flagquantum.simulation.tensor_network.register_tensor_indices import (
    register_tensor_outer_indices,
)
from tests.unit.test_register_tensor import native_signature
from tests.unit.test_register_tensor_binding import source_bindings

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("sides", list(product(("LEFT", "RIGHT", "THRU"), repeat=2)))
@pytest.mark.parametrize("shape", [(), (2,), (1, 2), (0,)])
@pytest.mark.parametrize("superoperator", [False, True])
def test_original_tuple_and_friendly_names_with_classical_ports(
    sides: tuple[str, ...], shape: tuple[int, ...], superoperator: bool
) -> None:
    import quimb.tensor as qtn
    from qualtran import CBit, Connection, LeftDangle, QAny, RightDangle, Side, Soquet
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.simulation.tensor._quimb import (
        _get_outer_indices,
        _get_outer_superindices,
    )

    source = SourceSignature(
        [
            SourceRegister("word", QAny(2), shape=shape, side=Side[sides[0]]),
            SourceRegister("classical", CBit(), side=Side[sides[1]]),
            SourceRegister("zero", QAny(0)),
        ]
    )
    sig = native_signature(source)
    labels: dict[tuple[str, int], int] = {}
    _, inc = source_bindings(source.lefts(), "in", labels)
    _, out = source_bindings(source.rights(), "out", labels)
    nodes = []
    source_to_native = {}
    # Actual source Connection/Soquet boundary objects feed the unmodified helper.
    for registers, bindings, side in (
        (source.lefts(), inc, "l"),
        (source.rights(), out, "r"),
    ):
        for reg in registers:
            for index, word in zip(reg.all_idxs(), bindings[reg.name], strict=True):
                endpoint = Soquet(
                    LeftDangle if side == "l" else RightDangle, reg, index
                )
                middle = Soquet(None, reg, index)
                cxn = (
                    Connection(endpoint, middle)
                    if side == "l"
                    else Connection(middle, endpoint)
                )
                for bit, label in enumerate(word):
                    for backward in ((False, True) if superoperator else (False,)):
                        ind = (cxn, bit, not backward) if superoperator else (cxn, bit)
                        native = 2 * label + int(backward) if superoperator else label
                        source_to_native[ind] = native
                        nodes.append(qtn.Tensor(np.array([1, 1j]), inds=(ind,)))
    tn = qtn.TensorNetwork(nodes[::-1])
    helper = _get_outer_superindices if superoperator else _get_outer_indices
    for friendly in (False, True):
        expected = {
            source_to_native[key]: value
            for key, value in helper(tn, friendly_indices=friendly).items()
        }
        actual = register_tensor_outer_indices(
            sig, inc, out, superoperator=superoperator, friendly=friendly
        )
        assert actual == expected
    assert all(
        type(label) is int
        for label in register_tensor_outer_indices(
            sig, inc, out, superoperator=superoperator
        )
    )


@pytest.mark.parametrize("superoperator", [False, True])
def test_complete_source_phase_network_inspection_does_not_change_execution(
    superoperator: bool,
) -> None:
    from qualtran.simulation.tensor._quimb import cbloq_to_quimb, cbloq_to_superquimb

    from flagquantum.compiler.registers.component_graph import iter_component_calls
    from flagquantum.simulation.tensor_network.component_tensor_nodes import (
        component_tensor_nodes,
    )
    from flagquantum.simulation.tensor_network.register_tensor_binding import (
        contract_register_tensors,
    )
    from tests.unit.test_component_zero_initialization import phase_fixture
    from tests.unit.test_register_zero_state import supplier

    record, source = phase_fixture(1, (2,))
    order = tuple(i for i, _, _ in iter_component_calls(record))
    nodes, inc, out = component_tensor_nodes(
        record, order, supplier, torch.empty((), dtype=torch.complex128)
    )
    sig = Signature(record.functions[0].registers or ())
    selected = tensor_conjugate_pairs(nodes) if superoperator else nodes
    contract = (
        contract_register_superoperator if superoperator else contract_register_tensors
    )
    before = contract(selected, sig, inc, out)
    snapshot = tuple(
        (node.labels, node.tensor, node.tensor.clone()) for node in selected
    )
    friendly = register_tensor_outer_indices(
        sig, inc, out, superoperator=superoperator, friendly=True
    )
    structured = register_tensor_outer_indices(
        sig, inc, out, superoperator=superoperator
    )
    original = (cbloq_to_superquimb if superoperator else cbloq_to_quimb)(
        source, friendly_indices=True
    )
    assert set(friendly.values()) == set(original.outer_inds())
    original_structured = (cbloq_to_superquimb if superoperator else cbloq_to_quimb)(
        source
    )
    assert set(structured.values()) == set(original_structured.outer_inds())
    assert {
        label
        for node in selected
        for label in node.labels
        if sum(label in other.labels for other in selected) == 1
    } == friendly.keys()
    for node, (labels, data, copy) in zip(selected, snapshot, strict=True):
        assert node.labels == labels and node.tensor is data
        torch.testing.assert_close(node.tensor, copy)
    torch.testing.assert_close(contract(selected, sig, inc, out), before)
    # Diagnostics are replaceable: consumer keeps integer edges and tensor data.
    friendly.clear()
    torch.testing.assert_close(contract(selected, sig, inc, out), before)


@pytest.mark.parametrize("superoperator", [False, True])
def test_display_name_collisions_preserve_integer_identity(superoperator: bool) -> None:
    from flagquantum.core.ir.program.registers import (
        Register,
        RegisterSide,
        RegisterType,
    )

    sig = Signature(
        [
            Register("q", RegisterType("uint", 1), (1,), side=RegisterSide.RIGHT),
            Register("q(0,)", RegisterType("uint", 1), side=RegisterSide.RIGHT),
        ]
    )
    result = register_tensor_outer_indices(
        sig,
        {},
        {"q": ((101,),), "q(0,)": ((205,),)},
        superoperator=superoperator,
        friendly=True,
    )
    assert len(result) == (4 if superoperator else 2)
    assert len(set(result.values())) == (2 if superoperator else 1)


@pytest.mark.parametrize(
    "binding", [{}, {"q": ()}, {"q": ((-1,),)}, {"q": ((True,),)}, {"q": ((1, 2),)}]
)
def test_invalid_connections_are_not_coerced(binding: Any) -> None:
    with pytest.raises((KeyError, ValueError)):
        register_tensor_outer_indices(Signature.build(q=1), binding, {"q": ((9,),)})


@pytest.mark.parametrize("superoperator", [False, True])
def test_shared_external_edge_is_ambiguous(superoperator: bool) -> None:
    with pytest.raises(ValueError, match="distinct"):
        register_tensor_outer_indices(
            Signature.build(q=1),
            {"q": ((5,),)},
            {"q": ((5,),)},
            superoperator=superoperator,
        )


@pytest.mark.parametrize("mode", ["superoperator", "friendly"])
def test_nonboolean_modes_are_refused(mode: str) -> None:
    flags: Any = {mode: 1}
    with pytest.raises(ValueError, match="booleans"):
        register_tensor_outer_indices(Signature([]), {}, {}, **flags)


def test_scalar_empty_network_has_no_names() -> None:
    assert register_tensor_outer_indices(Signature([]), {}, {}) == {}
