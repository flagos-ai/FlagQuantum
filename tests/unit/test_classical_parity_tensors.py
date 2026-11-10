"""Parity networks preserve original value tables without claiming source tensors."""

from itertools import count, product
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.core.ir.program.register_types import classical_bit_type
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.classical_parity import (
    classical_parity_nodes,
)
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    contract_register_tensors,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", [0, 1, 2, 3, 5, 8])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128]
)
def test_entire_original_value_table_and_linear_shared_storage(
    width: int, dtype: torch.dtype
) -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import Parity

    inc = {"x": tuple((i,) for i in range(width))}
    out = {"x": tuple((width + i,) for i in range(width)), "parity": ((2 * width,),)}
    fresh = count(2 * width + 1)
    nodes = classical_parity_nodes(
        width, inc, out, fresh.__next__, torch.empty((), dtype=dtype)
    )
    sig = Signature(
        (
            Register("x", classical_bit_type(), (width,)),
            Register("parity", classical_bit_type(), side=RegisterSide.RIGHT),
        )
    )
    actual = contract_register_tensors(nodes, sig, inc, out)
    expected = (
        torch.zeros((1 << (width + 1), 1 << width), dtype=dtype)
        if width
        else torch.tensor([1, 0], dtype=dtype)
    )
    original = Parity(width)
    if width:
        for index, bits in enumerate(product((0, 1), repeat=width)):
            x, parity = original.call_classically(x=np.array(bits))
            np.testing.assert_array_equal(x, bits)
            expected[2 * index + int(parity), index] = 1
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert len(nodes) == width + 1
    assert sum(n.tensor.numel() for n in nodes) == 16 * width + 2
    storages = {
        n.tensor.untyped_storage().data_ptr(): n.tensor.untyped_storage().nbytes()
        for n in nodes
    }
    assert (
        sum(storages.values())
        == (18 if width else 2) * torch.empty((), dtype=dtype).element_size()
    )
    # No original tensor exists to compare gate-by-gate: retain that negative contract.
    with pytest.raises(NotImplementedError, match="tensor simulation"):
        original.my_tensors(
            {"x": np.arange(width)}, {"x": np.arange(width), "parity": "p"}
        )


@pytest.mark.parametrize("width", [-1, True, 1.0])
def test_invalid_count_rejected_before_allocator(width: Any) -> None:
    def fail() -> int:
        raise AssertionError("must not allocate")

    with pytest.raises(ValueError, match="count"):
        classical_parity_nodes(width, {}, {}, fail, torch.empty(()))


@pytest.mark.parametrize("labels", [(0,), (-1,), (True,), (1, 1)])
def test_allocator_collisions_invalid_values_and_duplicates(
    labels: tuple[Any, ...],
) -> None:
    width = len(labels)
    seq = iter(labels)
    with pytest.raises(ValueError, match="internal labels"):
        classical_parity_nodes(
            width,
            {"x": tuple((i,) for i in range(width))},
            {"x": tuple((width + i,) for i in range(width)), "parity": ((2 * width,),)},
            seq.__next__,
            torch.empty(()),
        )


def test_external_binding_errors_do_not_consume_allocator() -> None:
    def fail() -> int:
        raise AssertionError("must not allocate")

    with pytest.raises(ValueError, match="external labels"):
        classical_parity_nodes(
            1, {"x": ((0,),)}, {"x": ((0,),), "parity": ((2,),)}, fail, torch.empty(())
        )
    with pytest.raises(ValueError, match="element/bit"):
        classical_parity_nodes(
            1,
            {"x": ((0, 1),)},
            {"x": ((2,),), "parity": ((3,),)},
            fail,
            torch.empty(()),
        )
    with pytest.raises(KeyError):
        classical_parity_nodes(
            1, {}, {"x": ((2,),), "parity": ((3,),)}, fail, torch.empty(())
        )
