"""Pinned dense grouping and native numerical-boundary replacement audit."""

from collections.abc import Callable
from itertools import product
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.register_axis_groups import (
    group_register_outer_indices,
)
from flagquantum.simulation.tensor_network.register_superoperator_binding import (
    contract_register_superoperator,
)
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    contract_register_tensors,
)
from tests.unit.test_register_tensor_source_audit import native_signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def native_dense(
    network: Any,
    signature: Any,
    superoperator: bool = False,
    *,
    simplification_atol: float | None = None,
) -> NDArray[Any]:
    """Replace only source quimb_to_dense, retaining the original graph and tests."""
    assert not superoperator  # The complete original file contains pure-state tests.
    labels = {index: i for i, index in enumerate(network.ind_map)}
    nodes = tuple(
        TensorNetworkNode(
            torch.from_numpy(np.asarray(tensor.data).copy()),
            tuple(labels[index] for index in tensor.inds),
        )
        for tensor in network.tensors
    )
    bindings = []
    for side, registers in (("l", signature.lefts()), ("r", signature.rights())):
        bindings.append(
            {
                register.name: tuple(
                    tuple(
                        labels[(register.name, index, bit, side)]
                        for bit in range(register.dtype.num_bits)
                    )
                    for index in register.all_idxs()
                )
                for register in registers
            }
        )
    return contract_register_tensors(
        nodes,
        native_signature(signature),
        bindings[0],
        bindings[1],
        simplification_atol=simplification_atol,
    ).numpy()


@pytest.mark.parametrize(
    "name",
    [
        "test_bloq_to_dense",
        "test_nest",
        "test_double_nest",
        "test_bloq_with_non_trivial_inds",
        "test_bloq_stop_flattening",
    ],
)
@pytest.mark.parametrize("simplification_atol", [None, 1e-12])
def test_complete_original_tests_with_native_numerical_replacement(
    monkeypatch: pytest.MonkeyPatch, name: str, simplification_atol: float | None
) -> None:
    from functools import partial

    from qualtran.simulation.tensor import _dense, _dense_test

    monkeypatch.setattr(
        _dense,
        "quimb_to_dense",
        partial(native_dense, simplification_atol=simplification_atol),
    )
    original: Callable[[], None] = getattr(_dense_test, name)
    original()


@pytest.mark.parametrize("sides", list(product(("LEFT", "RIGHT", "THRU"), repeat=2)))
@pytest.mark.parametrize("shape", [(), (1, 2), (0,)])
@pytest.mark.parametrize("mixed", [False, True])
def test_actual_outer_groups_preserve_source_order_and_ignore_extras(
    sides: tuple[str, ...], shape: tuple[int, ...], mixed: bool
) -> None:
    import quimb.tensor as qtn
    from qualtran import CBit, QAny, Register, Side
    from qualtran import Signature as SourceSignature
    from qualtran.simulation.tensor._dense import _group_outer_inds

    source = SourceSignature(
        [
            Register("word", QAny(2), shape=shape, side=Side[sides[0]]),
            Register("classical", CBit(), side=Side[sides[1]]),
        ]
    )
    descriptors = [
        (register.name, index, bit, side + direction)
        for side, registers in (("l", source.lefts()), ("r", source.rights()))
        for register in registers
        for index in register.all_idxs()
        for bit in range(register.dtype.num_bits)
        for direction in (("f", "b") if mixed else ("",))
    ]
    descriptors.reverse()
    descriptors.append(("unused", (), 0, "other"))
    network = qtn.TensorNetwork(
        [qtn.Tensor(np.ones((2,) * len(descriptors)), descriptors)]
    )
    labels = {index: 3 * i + 10 for i, index in enumerate(network.outer_inds())}
    expected = tuple(
        tuple(labels[index] for index in group)
        for group in _group_outer_inds(network, source, superoperator=mixed)
    )
    assert (
        group_register_outer_indices(
            native_signature(source), labels, superoperator=mixed
        )
        == expected
    )
    for removed in descriptors[:-1]:
        remaining = [index for index in descriptors if index != removed]
        broken = qtn.TensorNetwork(
            [qtn.Tensor(np.ones((2,) * len(remaining)), remaining)]
        )
        with pytest.raises(KeyError) as original:
            _group_outer_inds(broken, source, superoperator=mixed)
        with pytest.raises(KeyError) as native:
            group_register_outer_indices(
                native_signature(source),
                {index: label for index, label in labels.items() if index != removed},
                superoperator=mixed,
            )
        assert native.value.args == original.value.args


@pytest.mark.parametrize("mixed", [False, True])
def test_missing_input_is_reported_before_missing_output(mixed: bool) -> None:
    from flagquantum.core.ir.program.registers import (
        Register,
        RegisterSide,
        RegisterType,
    )

    signature = Signature(
        [
            Register("output", RegisterType("opaque", 1), side=RegisterSide.RIGHT),
            Register("input", RegisterType("opaque", 1), side=RegisterSide.LEFT),
        ]
    )
    with pytest.raises(KeyError) as error:
        group_register_outer_indices(signature, {}, superoperator=mixed)
    assert error.value.args == (("input", (), 0),)
    with pytest.raises(KeyError):
        contract = (
            contract_register_superoperator if mixed else contract_register_tensors
        )
        contract(
            (TensorNetworkNode(torch.tensor(1j), ()),),
            signature,
            {"input": ((0,),)},
            {"output": ((1,),)},
        )


@pytest.mark.parametrize("mixed", [False, True])
def test_zero_port_scalar_and_gradient_reuse_native_engine(mixed: bool) -> None:
    scalar = torch.tensor(1 + 2j, dtype=torch.complex128, requires_grad=True)
    contract = contract_register_superoperator if mixed else contract_register_tensors
    actual = contract((TensorNetworkNode(scalar, ()),), Signature([]), {}, {})
    assert actual.shape == () and actual == scalar
    (gradient,) = torch.autograd.grad(actual.real, scalar)
    assert gradient == 1


@pytest.mark.parametrize("width", [12.0, 12.001])
def test_source_simplification_threshold_is_an_explicit_remaining_gap(
    width: float,
) -> None:
    from qualtran import Signature as SourceSignature
    from qualtran.simulation.tensor._dense import quimb_to_dense

    calls: list[str] = []

    class Network:
        def outer_inds(self) -> tuple[()]:
            return ()

        def contraction_width(self) -> float:
            calls.append("width")
            return width

        def full_simplify(self, *, inplace: bool) -> None:
            assert inplace
            calls.append("simplify")

        def contract(self) -> complex:
            calls.append("contract")
            return 1j

    assert quimb_to_dense(Network(), SourceSignature([])) == 1j
    assert calls == (
        ["width", "simplify", "contract"] if width > 12 else ["width", "contract"]
    )
