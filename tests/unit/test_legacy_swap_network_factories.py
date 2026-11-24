"""Source swap constructors retain exact exchange calls and measured inverses."""

from hashlib import sha1
from importlib.metadata import version
from math import prod
from pathlib import Path

import pytest
import torch

from flagquantum import Circuit
from flagquantum.algorithms.primitives._block_binding import _BoundBlockOperation
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_swap_network_factories import (
    swap_network_factories,
)
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_multiplexed_source_binding import logical_operator

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def pinned_swap_sources() -> None:
    import kickmix
    from qualtran.bloqs.swap_network import multiplexed_cswap

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    for filename, digest in (
        ("multiplexed_cswap.py", "d41e2bb20c122be67e14542411eca4d65be06e9a"),
        ("multiplexed_cswap_test.py", "5d0ff34eb3667cc12acc2402b945b0b04cbf3971"),
        ("swap_with_zero.py", "756704c796ab999b949fd11903d881734304ac2e"),
        ("swap_with_zero_test.py", "ca49bb77d61d71c68b9ca68dce9aa79997bad7b0"),
    ):
        raw = (Path(multiplexed_cswap.__file__).parent / filename).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest


@pytest.mark.parametrize(
    "example", ("_multiplexed_cswap", "_swz", "_swz_small", "_swz_multi_dimensional")
)
@pytest.mark.parametrize("inverse", (False, True))
def test_complete_original_factories_and_depths(example: str, inverse: bool) -> None:
    from qualtran.bloqs.swap_network import multiplexed_cswap, swap_with_zero

    module = multiplexed_cswap if example == "_multiplexed_cswap" else swap_with_zero
    forward = getattr(module, example).make()
    source = forward.adjoint() if inverse else forward
    record = resolve_component_program(_library(source, 0), 0, swap_network_factories())
    assert verify_program_record(record) == ()
    assert record.functions[0].blocks
    deeper = resolve_component_program(_library(source, 1), 0, swap_network_factories())
    assert matches_component_definition(record, "main", deeper)
    compare_layers(record, source, atomic_leaves(record))


@pytest.mark.parametrize(
    "controls,multiaxis", ((0, False), (1, False), (2, False), (3, False), (1, True))
)
@pytest.mark.parametrize("inverse", (False, True))
def test_multiplexed_full_complex_operators_all_histories_and_append(
    controls: int, multiaxis: bool, inverse: bool
) -> None:
    from qualtran import BQUInt, QAny, QBit, Register
    from qualtran.bloqs.swap_network.multiplexed_cswap import MultiplexedCSwap

    forward = MultiplexedCSwap(
        (
            (Register("row", BQUInt(1, 2)), Register("column", BQUInt(1, 2)))
            if multiaxis
            else Register("address", BQUInt(1, 2))
        ),
        1,
        (
            (Register("enable", QAny(3)),)
            if controls == 3
            else (Register("enable", QBit(), shape=(controls,)),) if controls else ()
        ),
    )
    source = forward.adjoint() if inverse else forward
    record = resolve_component_program(_library(source, 0), 0, swap_network_factories())
    compare_layers(record, source, atomic_leaves(record))
    logical = controls + (7 if multiaxis else 4)
    cursor = 0
    inputs = []
    for register in record.functions[0].registers or ():
        for _ in register.all_idxs():
            width = int(register.bitsize)
            inputs.append(tuple(range(cursor, cursor + width)))
            cursor += width
    bound = bind_register_program(
        record,
        tuple(inputs),
        clean=tuple(range(logical, logical + 12)),
    )
    assert bound.outputs == tuple(inputs)
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=3e-11,
        rtol=0,
    )
    appended = (
        Circuit(bound.circuit.n_wires)
        .append(_BoundBlockOperation(bound.circuit.instructions, bound.circuit.n_wires))
        .to_ir()
    )
    assert appended.instructions == bound.circuit.instructions
    assert bound.circuit.from_json(bound.circuit.to_json()) == bound.circuit


@pytest.mark.parametrize(
    "widths,shape",
    (((0,), (1,)), ((1,), (2,)), ((2,), (3,)), ((1, 1), (2, 2)), ((1, 0), (2, 1))),
)
@pytest.mark.parametrize("inverse", (False, True))
def test_relative_network_full_complex_operator(
    widths: tuple[int, ...], shape: tuple[int, ...], inverse: bool
) -> None:
    from qualtran.bloqs.swap_network.swap_with_zero import SwapWithZero

    source = SwapWithZero(widths, 1, shape, uncompute=inverse)
    record = resolve_component_program(_library(source, 0), 0, swap_network_factories())
    compare_layers(record, source, atomic_leaves(record))
    logical = sum(widths) + prod(shape)
    cursor = 0
    inputs = []
    for register in record.functions[0].registers or ():
        for _ in register.all_idxs():
            width = int(register.bitsize)
            inputs.append(tuple(range(cursor, cursor + width)))
            cursor += width
    bound = bind_register_program(record, tuple(inputs))
    assert not bound.events
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=3e-11,
        rtol=0,
    )


@pytest.mark.parametrize("kind", ("multiplexed", "relative"))
def test_source_free_resolution_and_symbolic_word_declarations(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    import sympy
    from qualtran import BQUInt, Register
    from qualtran.bloqs.swap_network.multiplexed_cswap import MultiplexedCSwap
    from qualtran.bloqs.swap_network.swap_with_zero import SwapWithZero

    for symbolic in (False, True):
        word = sympy.Symbol("b", integer=True, positive=True) if symbolic else 1
        forward = (
            MultiplexedCSwap(Register("address", BQUInt(1, 2)), word)
            if kind == "multiplexed"
            else SwapWithZero((1,), word, (2,))
        )
        for source in (forward, forward.adjoint()):
            library = _library(source, 0)
            with monkeypatch.context() as patch:
                patch.setitem(sys.modules, "qualtran", None)
                record = resolve_component_program(library, 0, swap_network_factories())
            assert verify_program_record(record) == ()
            assert bool(record.functions[0].blocks) is not symbolic


def test_original_leading_zero_and_uncontrolled_singleton_refusals() -> None:
    from qualtran import BQUInt, Register
    from qualtran.bloqs.swap_network.multiplexed_cswap import MultiplexedCSwap
    from qualtran.bloqs.swap_network.swap_with_zero import SwapWithZero

    leading_zero = SwapWithZero((0, 1), 1, (1, 2))
    with pytest.raises(IndexError):
        leading_zero.decompose_bloq()
    with pytest.raises(ValueError, match="earlier zero-bit"):
        resolve_component_program(
            _library(leading_zero, 0), 0, swap_network_factories()
        )
    singleton = MultiplexedCSwap(Register("address", BQUInt(1, 1)), 1)
    with pytest.raises(IndexError):
        singleton.decompose_bloq()
    with pytest.raises(ValueError, match="singleton first axis"):
        resolve_component_program(_library(singleton, 0), 0, swap_network_factories())
