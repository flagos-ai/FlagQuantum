"""Open-system external axes match the unmodified pinned dense protocol."""

from itertools import product
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.conjugate_pairs import tensor_conjugate_pairs
from flagquantum.simulation.tensor_network.measurement_copy import (
    z_measurement_copy_node,
)
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.register_superoperator_binding import (
    contract_register_superoperator,
    register_superoperator_label_groups,
)
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    bind_register_tensor,
)
from flagquantum.simulation.tensor_network.trace_indices import trace_conjugate_indices
from tests.unit.test_register_tensor import native_signature
from tests.unit.test_register_tensor_binding import source_bindings

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("sides", list(product(("LEFT", "RIGHT", "THRU"), repeat=2)))
@pytest.mark.parametrize("shape", [(), (2,), (0,)])
def test_original_mixed_quantum_classical_axis_order_and_contraction(
    sides: tuple[str, ...], shape: tuple[int, ...]
) -> None:
    import quimb.tensor as qtn
    from qualtran import CBit, QAny, Side
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.simulation.tensor._dense import _group_outer_inds, quimb_to_dense

    source = SourceSignature(
        [
            SourceRegister("q", QAny(1), shape=shape, side=Side[sides[0]]),
            SourceRegister("c", CBit(), side=Side[sides[1]]),
            SourceRegister("zero", QAny(0)),
        ]
    )
    sig = native_signature(source)
    labels: dict[tuple[str, int], int] = {}
    _, inc = source_bindings(source.lefts(), "in", labels)
    _, out = source_bindings(source.rights(), "out", labels)
    groups = register_superoperator_label_groups(sig, inc, out)
    reference_indices = []
    native_labels = []
    for registers, ports, side in (
        (tuple(source.rights()), out, "r"),
        (tuple(source.lefts()), inc, "l"),
    ):
        for backward in (False, True):
            for reg in registers:
                for word, idx in zip(
                    ports.get(reg.name, ()), reg.all_idxs(), strict=True
                ):
                    for bit, label in enumerate(word):
                        reference_indices.append(
                            (reg.name, idx, bit, side + ("b" if backward else "f"))
                        )
                        native_labels.append(2 * label + int(backward))
    rng = np.random.default_rng(64)
    shape_bits = (2,) * len(native_labels)
    data = np.asarray(rng.normal(size=shape_bits) + 1j * rng.normal(size=shape_bits))
    node = TensorNetworkNode(torch.from_numpy(data), tuple(native_labels))
    original = qtn.Tensor(data=data, inds=tuple(reference_indices))
    original.transpose_(*reversed(original.inds))
    tn = qtn.TensorNetwork([original])
    source_groups = _group_outer_inds(tn, source, superoperator=True)
    mapping = dict(zip(reference_indices, native_labels, strict=True))
    assert groups == tuple(tuple(mapping[i] for i in group) for group in source_groups)
    expected = quimb_to_dense(tn, source, superoperator=True)
    actual = contract_register_superoperator((node,), sig, inc, out)
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-14, rtol=0)
    assert actual.ndim == len(source_groups)


def test_measurement_channel_classical_output_and_discard_scalar() -> None:
    from qualtran import BloqBuilder, CBit, QBit, Side
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.basic_gates import Discard, MeasureZ
    from qualtran.simulation.tensor._dense import quimb_to_dense
    from qualtran.simulation.tensor._quimb import cbloq_to_superquimb

    source_sig = SourceSignature(
        [
            SourceRegister("q", QBit(), side=Side.LEFT),
            SourceRegister("c", CBit(), side=Side.RIGHT),
        ]
    )
    bb, inputs = BloqBuilder.from_signature(source_sig)
    c = bb.add(MeasureZ(), q=inputs["q"])
    measured = bb.finalize(c=c)
    node = z_measurement_copy_node(3, 17, 41, torch.empty((), dtype=torch.complex128))
    nodes = tensor_conjugate_pairs((node,))
    traced = trace_conjugate_indices(nodes, (41,))
    channel = contract_register_superoperator(
        traced, native_signature(source_sig), {"q": ((3,),)}, {"c": ((17,),)}
    )
    assert channel.shape == (2, 2, 2, 2)
    np.testing.assert_array_equal(
        channel.numpy(),
        quimb_to_dense(
            cbloq_to_superquimb(measured), measured.signature, superoperator=True
        ),
    )
    rho = torch.tensor([[0.4, 0.2 + 0.3j], [0.2 - 0.3j, 0.6]], dtype=torch.complex128)
    torch.testing.assert_close(
        torch.einsum("abij,ij->ab", channel, rho), torch.diag(rho.diag())
    )
    sig = SourceSignature([SourceRegister("q", QBit(), side=Side.LEFT)])
    bb, inputs = BloqBuilder.from_signature(sig)
    bb.add(Discard(), c=bb.add(MeasureZ(), q=inputs["q"]))
    discarded = bb.finalize()
    effect = contract_register_superoperator(
        trace_conjugate_indices(traced, (17,)),
        native_signature(sig),
        {"q": ((3,),)},
        {},
    )
    assert effect.shape == (2, 2)
    np.testing.assert_array_equal(
        effect.numpy(),
        quimb_to_dense(cbloq_to_superquimb(discarded), sig, superoperator=True),
    )
    torch.testing.assert_close(torch.einsum("ij,ij->", effect, rho), torch.trace(rho))


def test_density_autograd_replacement_and_scalar() -> None:
    theta = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    state = torch.stack((torch.cos(theta).to(torch.complex128), 1j * torch.sin(theta)))
    sig = Signature([Register("q", RegisterType("qubit", 1), side=RegisterSide.RIGHT)])

    def consume(data: torch.Tensor) -> torch.Tensor:
        node = bind_register_tensor(data, sig, {}, {"q": ((13,),)})
        return contract_register_superoperator(
            tensor_conjugate_pairs((node,)), sig, {}, {"q": ((13,),)}
        )

    rho = consume(state)
    torch.testing.assert_close(rho, state[:, None] * state.conj()[None, :])
    torch.testing.assert_close(consume(state * 1j), rho)
    (gradient,) = torch.autograd.grad(rho[0, 1].imag, (theta,))
    torch.testing.assert_close(gradient, -torch.cos(2 * theta))
    empty = Signature(
        [
            Register("zero", RegisterType("opaque", 0)),
            Register("empty", RegisterType("opaque", 2), (0,)),
        ]
    )
    scalar = TensorNetworkNode(torch.tensor(2 + 3j, dtype=torch.complex128), ())
    result = contract_register_superoperator(
        tensor_conjugate_pairs((scalar,)), empty, {}, {}
    )
    assert result.shape == () and result == 13
    with pytest.raises(ValueError, match="at least one"):
        contract_register_superoperator((), empty, {}, {})


@pytest.mark.parametrize(
    "binding", [{}, {"q": ()}, {"q": ((-1,),)}, {"q": ((True,),)}, {"q": ((1, 2),)}]
)
def test_invalid_bindings_fail_before_contraction(binding: Any) -> None:
    with pytest.raises((KeyError, ValueError)):
        contract_register_superoperator(
            (), Signature.build(q=1), binding, {"q": ((5,),)}
        )
